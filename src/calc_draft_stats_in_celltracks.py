"""
Calculates 3D W core statistics from gridded Met data for tracked convective cells.
The 3D core statistics are written to netCDF file matching the cell track statistics file format.
"""
import numpy as np
import os, sys, glob
import time
import traceback
import warnings
import gc
from datetime import datetime
from pytz import utc
import yaml
import xarray as xr
from scipy import ndimage
from skimage.segmentation import expand_labels
from scipy.ndimage import generate_binary_structure, binary_dilation, iterate_structure
import dask
import dask.array as da
from dask.distributed import Client, LocalCluster
import psutil
import concurrent.futures
# import matplotlib.pyplot as plt

def log_memory_usage(stage_name):
    """Log current memory usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"{stage_name}: Memory usage: {memory_info.rss / 1024**3:.2f} GB")

def cleanup_memory():
    """Enhanced memory cleanup function"""
    gc.collect()
    # Force garbage collection multiple times
    gc.collect()
    gc.collect()

def monitor_memory_and_cleanup(threshold_gb=40):
    """Monitor memory usage and cleanup if threshold exceeded"""
    process = psutil.Process()
    memory_gb = process.memory_info().rss / 1024**3
    if memory_gb > threshold_gb:
        print(f"Memory usage ({memory_gb:.2f} GB) exceeds threshold, cleaning up...")
        cleanup_memory()
        return True
    return False

#-----------------------------------------------------------------------
def calc_basetime(filelist, filebase):
    """
    Calculates basetime (Epoch Time) and a filename dictionary from a list of filenames.
    The basetime uses year, month, day, hour, minute but set all seconds to 0.

    Args:
        filelist: list
            A list of input file names
        filebase: string
            Basename of the files

    Returns:
        file_basetime: int array
            Epoch time corresponding to the input files.
        file_dict: dictionary
            Direction key by basetime and value is the file names
    """
    nfiles = len(filelist)
    prelength = len(filebase)
    file_basetime = np.full(nfiles, -999, dtype=int)
    file_dict = {}
    for ifile in range(nfiles):
        fname = os.path.basename(filelist[ifile])
        # File name format: basename_yyyymmdd.hhmmss
        TEMP_filetime = datetime(
            int(fname[prelength:(prelength+4)]), 
            int(fname[prelength+4:(prelength+6)]), 
            int(fname[prelength+6:(prelength+8)]),
            int(fname[prelength+9:(prelength+11)]),
            int(fname[prelength+11:(prelength+13)]),
            int(fname[prelength+13:(prelength+15)]),
            tzinfo=utc,
            # int(fname[prelength+11:(prelength+13)]), 0, tzinfo=utc
        )
        # file_basetime[ifile] = calendar.timegm(TEMP_filetime.timetuple())
        file_basetime[ifile] = TEMP_filetime.timestamp()
        file_dict[file_basetime[ifile]] = filelist[ifile]
    return file_basetime, file_dict


#-----------------------------------------------------------------------
def label_cores(W, W_thresh, Q, Q_thresh, VMF, ncores_min, min_core_npix, method='>'):
    """
    Label up/down draft cores using threshold and connectivity.

    Args:
        W: np.array
            Vertical velocity array
        W_thresh: float
            Vertical velocity threshold
        Q: np.array
            Mixing ratio array
        Q_thresh: float
            Mixing ratio threshold
        VMF: np.array
            Vertical mass flux array
        ncores_min: int
            Minimum number of cores to save
        min_core_npix: int
            Minimum number of pixels to define a core
        method: string
            Method to define cores

    Returns:
        ncores_save: int
            Number of cores
        npix_core_sorted: np.array
            Number of pixels for each core
        core_numbers_sorted: np.array
            Labeled core numbers (1D)
        core_label: np.array
            Labeled core numbers map (2D)
    """
    # Label cores at a given vertical level
    # struct = generate_binary_structure(2, 1).astype(int)
    if method == '>':
        core_label, ncores = ndimage.label((W > W_thresh) & (Q > Q_thresh))
    elif method == '<':
        core_label, ncores = ndimage.label((W < W_thresh) & (Q < Q_thresh))
    else:
        print(f'Error: Undefined method to label cores: {method}!')
    
    # Get core sizes
    core_numbers, npix_core = np.unique(core_label, return_counts=True)
    
    # Remove 0 label result since that is the background
    npix_core = npix_core[core_numbers > 0]
    core_numbers = core_numbers[core_numbers > 0]
    
    # Find the small cores (EJ) 
    mask = npix_core <= min_core_npix
    rm_core_numbers = core_numbers[mask]

    # Remove small cores
    mask = npix_core > min_core_npix
    npix_core = npix_core[mask]
    core_numbers = core_numbers[mask]

    # Calculate VMF of cores
    zVMF = np.zeros_like(npix_core)
    for i in range(0,len(core_numbers)):
        zVMF[i] = np.nansum(VMF[core_label == core_numbers[i]])

    # Sort the core size by descending order
    # sort_idx = npix_core.argsort()[::-1]
    # Sort the cores by VMF
    sort_idx = zVMF.argsort()[::-1] 
    npix_core_sorted = npix_core[sort_idx]
    core_numbers_sorted = core_numbers[sort_idx]
   
    # Save the largest X cores
    ncores_all = len(npix_core)
    ncores_save = np.nanmin([ncores_all, ncores_min])

    # Remove small cores in the 2D mask (EJ)
    for i in rm_core_numbers:
        core_label[core_label == i] = 0
    
    # Put output variables in a dictionary
    out_dict = {
        'ncores_all': ncores_all,
        'ncores_save': ncores_save,
        'core_npix': npix_core_sorted[:ncores_save+1],
        'core_numbers': core_numbers_sorted[:ncores_save+1],
        'core_label': core_label,
    }
    return out_dict

#--------------------------------------------------------------------------
def make_dilation_structure(dilate_radius, DX, DY):
    """
    Make a circular dilation structure

    Args:
        dilate_radius: float
            Dilation radius [kilometer].
        DX: float
            Grid spacing in x-direction [meter].
        DY: float
            Grid spacing in y-direction [meter]. 
    
    Returns:
        struc: np.array
            Dilation structure array.
    """
    # Convert radius to number grids
    rad_gridx = int(dilate_radius * 1000 / DX)
    rad_gridy = int(dilate_radius * 1000 / DY)
    xgrd, ygrd = np.ogrid[-rad_gridx:rad_gridx+1, -rad_gridy:rad_gridy+1]
    # Make dilation structure
    strc = xgrd*xgrd + ygrd*ygrd <= (dilate_radius * 1000 / DX) * (dilate_radius * 1000 / DY)
    return strc

#-----------------------------------------------------------------------
def calc_rh_thompson(TEMPERATURE, PRESSURE, QVAPOR):
    """
    Calculate relative humidity following the Thompson scheme for supersaturation

    Args:
        TEMPERATURE: array-like
            Dry air temp [K]
        PRESSURE: array-like
            Air pressure [Pa]
        QVAPOR: array-like
            Water vapor mixing ratio [kg/kg]
   
    Returns:
        RH: array-like
            Relative humidity [%]
    """
    # RH (formula is used in Thompson scheme for supersaturation)
    C0 = 0.611583699e3
    C1 = 0.444606896e2
    C2 = 0.143177157e1
    C3 = 0.264224321e-1
    C4 = 0.299291081e-3
    C5 = 0.203154182e-5
    C6 = 0.702620698e-8
    C7 = 0.379534310e-11
    C8 = -0.321582393e-13
    X = TEMPERATURE - 273.16
    X[X < -80] = -80  #setting values less than -80C to -80C
    # X = X.where(X > -80, -80)
    # X = X.where(X > -80)
    # X = X.fillna(-80) #setting values less than -80C to -80C 
    ESL = C0 + X*(C1 + X*(C2 + X*(C3 + X*(C4 + X*(C5 + X*(C6 + X*(C7 + X*C8))))))) #saturation vapor pressure
    QVS = 0.622 * ESL / (PRESSURE - ESL) #saturation vapor mixing ratio
    RH = 1e2 * QVAPOR / QVS # %
    #SS = RH - 100 #supersaturation in %
    return RH

#-----------------------------------------------------------------------
def calc_theta_e(TEMPERATURE, PRESSURE, QVAPOR, Qliq, RH):
    """
    Calculate equivalent potential temperature following the Emanuel formula

    Args:
        TEMPERATURE: array-like
            Dry air temp [K]
        PRESSURE: array-like
            Air pressure [Pa]
        QVAPOR: array-like
            Water vapor mixing ratio [kg/kg]
        Qliq: array-like
            Total liquid condensate mixing ratio [kg/kg]
        RH: array-like
            Relative humidity [%]
    
    Returns:
        THETAE: array-like
            Equivalent potential temperature.
    """
    # constants:
    cpd = 1006 # J/kg K
    lv0 = 2501000
    Rv = 461.5
    Rd = 287.04
    cl = 4200

    teA = TEMPERATURE * (100000. / PRESSURE)**(Rd / (cpd + cl * Qliq))
    teB = np.exp((lv0 * QVAPOR) / ((cpd + Qliq * cl) * TEMPERATURE))
    teC = (RH/100)**((-QVAPOR * Rv) / (cpd + cl * Qliq))
    THETAE = teA * teB * teC
    return THETAE

#-----------------------------------------------------------------------
def theta_e_bolton(TEMPERATURE, QVAPOR, PRESSURE):
    """
    Calculate pseudoadiabatic equivalent potential temperature following the Bolton (1980) formula
    
    Error of < 0.3 K between -35 and 35C; from Thompson scheme

    Args:
        TEMPERATURE: array-like
            Dry air temp [K]
        PRESSURE: array-like
            Air pressure [Pa]
        QVAPOR: array-like
            Water vapor mixing ratio [kg/kg]

    Returns:
        THETAE_Bolton: array-like
            Equivalent potential temperature.
    """

    es = PRESSURE*QVAPOR/(0.622*QVAPOR)
    TDEW = (35.86*np.log(es) - 4947.2325)/(np.log(es) - 23.6837)
    TLCL = 1/(1/(TDEW - 56) + np.log(TEMPERATURE/TDEW)/800) + 56
    p1 = 3.376/TLCL - 0.00254
    p2 = 1e3*QVAPOR*(1 + 0.81*QVAPOR)
    THETAE_Bolton = (TEMPERATURE*(100000./PRESSURE)**(0.2854*(1 - 0.28*QVAPOR)))*np.exp(p1*p2) #K
    return THETAE_Bolton

#-----------------------------------------------------------------------
def calc_cellstats_singlefile(
    pixel_filename, 
    met_filename, 
    cld_filename,
    idx_track, 
    config,
):
    """
    Calculate statistics for cells in a single pixel file

    Args:
        pixel_filename: string
            Cell tracking pixel filename
        met_filename: string
            MET filename
        cld_filename: string
            CLD filename
        idx_track: np.array
            Tracknumber indices in the pixel file
        config: dictionary
            Dictionary containing config parameters

    Returns:
        out_dict: dictionary
            Dictionary containing the track statistics data
        out_dict_attrs: dictionary
            Dictionary containing the attributes of track statistics data
    """
    # Constants
    R_dry = 287.058   # J kg−1 K−1
    Cp_dry = 1005   # J kg-1 K−1

    print(met_filename)

    # Get thresholds from config
    W_up_thresh = config.get('W_up_thresh')
    W_down_thresh = config.get('W_down_thresh')
    Q_up_thresh = config.get('Q_up_thresh')
    Q_down_thresh = config.get('Q_down_thresh')
    min_core_npix = config.get('min_core_npix')
    ncores_min = config.get('ncores_min')
    core_expand_dist = config.get('core_expand_dist')
    core_shell_buffer_radius = config.get('core_shell_buffer_radius')
    geolimits = config.get('geolimits', None)

    # Read MET file
    dsm = xr.open_dataset(met_filename)
    # Rename dimenensions
    dsm = dsm.rename_dims({'south_north':'lat', 'west_east':'lon'})
    nz = dsm.sizes['HAMSL']
    ny = dsm.sizes['lat']
    nx = dsm.sizes['lon']
    height = dsm['HAMSL'].data
    DX = dsm.attrs['DX']
    DY = dsm.attrs['DY']
    grid_area = DX * DY / 1e6

    # Read CLD file
    dsc = xr.open_dataset(cld_filename)
    # Rename dimenensions
    dsc = dsc.rename_dims({'south_north':'lat', 'west_east':'lon'})

    # Read pixel-level track file
    ds = xr.open_dataset(pixel_filename, decode_times=False)
    time_pixel = ds['time']
    ny_p = ds.sizes['lat']
    nx_p = ds.sizes['lon']

    # Check dimensions between MET and pixel files
    if (ny_p < ny) | (nx_p < nx):
        # Get lat/lon limits
        buffer = 0
        latmin, latmax = geolimits[0]-buffer, geolimits[2]+buffer
        lonmin, lonmax = geolimits[1]-buffer, geolimits[3]+buffer
        # lonmin, lonmax = geolimits[0]-buffer, geolimits[1]+buffer
        # latmin, latmax = geolimits[2]-buffer, geolimits[3]+buffer
        # Make a 2D mask
        mask = ((dsm['XLONG'] >= lonmin) & (dsm['XLONG'] <= lonmax) & \
                (dsm['XLAT'] >= latmin) & (dsm['XLAT'] <= latmax)).squeeze()
        # Get y/x indices limits from the mask
        y_idx, x_idx = np.where(mask == True)
        xmin, xmax = np.min(x_idx), np.max(x_idx)
        ymin, ymax = np.min(y_idx), np.max(y_idx)
        # Subset 
        XLONG = dsm['XLONG'][ymin:ymax+1, xmin:xmax+1]
        XLAT = dsm['XLAT'][ymin:ymax+1, xmin:xmax+1]
        PRESSURE = dsm['PRESSURE'][:, :, ymin:ymax+1, xmin:xmax+1]*100  # [Pa]
        TEMPERATURE = dsm['TEMPERATURE'][:, :, ymin:ymax+1, xmin:xmax+1]
        QVAPOR = dsm['QVAPOR'][:, :, ymin:ymax+1, xmin:xmax+1]
        # TV = dsm['TV'][:, :, ymin:ymax+1, xmin:xmax+1]
        THETA = dsm['THETA'][:, :, ymin:ymax+1, xmin:xmax+1]
        WA = dsm['WA'][:, :, ymin:ymax+1, xmin:xmax+1]
        # Get cld variables
        QCLOUD = dsc['QCLOUD'][:, :, ymin:ymax+1, xmin:xmax+1]
        QRAIN = dsc['QRAIN'][:, :, ymin:ymax+1, xmin:xmax+1]
        QICE = dsc['QICE'][:, :, ymin:ymax+1, xmin:xmax+1]
        QSNOW = dsc['QSNOW'][:, :, ymin:ymax+1, xmin:xmax+1]
        QGRAUP = dsc['QGRAUP'][:, :, ymin:ymax+1, xmin:xmax+1]
        # Update ny, nx with the subset
        ny = XLONG.sizes['lat']
        nx = XLONG.sizes['lon']
    else:
        XLONG = dsm['XLONG']
        XLAT = dsm['XLAT']
        PRESSURE = dsm['PRESSURE']*100  # [Pa]
        TEMPERATURE = dsm['TEMPERATURE']
        QVAPOR = dsm['QVAPOR']
        THETA = dsm['THETA']
        # TV = dsm['TV']
        WA = dsm['WA']
        # Get cld variables
        QCLOUD = dsc['QCLOUD']
        QRAIN = dsc['QRAIN']
        QICE = dsc['QICE']
        QSNOW = dsc['QSNOW']
        QGRAUP = dsc['QGRAUP']

    # Check dimensions again after subset
    if (ny_p != ny) | (nx_p != nx):
        print(f'ERROR: Inconsistent number of grids between pixel-level and MET files.')
        print(f'ny: {ny}, ny_pixel: {ny_p}, nx: {nx}, nx_pixel: {nx_p}')
        sys.exit()

    # Drop 1D lat/lon coordinates, and reasign 2D XLONG/XLAT coordinates from Met file
    # It does not seem like this is necessary in Xarray 0.21.1
    ds = ds.drop_vars(['lon', 'lat']).assign_coords({'XLONG':XLONG, 'XLAT':XLAT})
    # Cell mask
    tracknumbermap = ds['tracknumber'].squeeze()

    out_dict2d = None
    out_dict3d = None
    out_dict_attrs = None

    # Proceed if number of matched cell is > 0
    nmatchcell = len(idx_track)
    if (nmatchcell > 0):

        # Create arrays for output statistics
        dims2d = (nmatchcell, nz)
        dims3d = (nmatchcell, nz, ncores_min)
        cell_nCore_up = np.full(dims2d, np.NaN, dtype=np.float32)
        # cell_ovlap_up = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_MassFlux_up = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_CoreMassFlux_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreArea_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMaxW_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanW_up = np.full(dims3d, np.NaN, dtype=np.float32)
        # cell_CoreMaxQC_up = np.full(dims3d, np.NaN, dtype=np.float32)
        # cell_CoreMeanQC_up = np.full(dims3d, np.NaN, dtype=np.float32)
        # cell_CoreMeanQC_prm = np.full(dims3d, np.NaN, dtype=np.float32)
        # cell_CoreMaxQR_up = np.full(dims3d, np.NaN, dtype=np.float32)
        # cell_CoreMeanQR_up = np.full(dims3d, np.NaN, dtype=np.float32)
        # cell_CoreMeanQV_up = np.full(dims3d, np.NaN, dtype=np.float32)
        # cell_CoreMeanQV_prm = np.full(dims3d, np.NaN, dtype=np.float32)

        # cell_nCore_down = np.full(dims2d, np.NaN, dtype=np.float32)
        # cell_MassFlux_down = np.full(dims2d, np.NaN, dtype=np.float32)
        # cell_CoreMassFlux_down = np.full(dims3d, np.NaN, dtype=np.float32)
        # cell_CoreArea_down = np.full(dims3d, np.NaN, dtype=np.float32)
        # cell_CoreMinW_down = np.full(dims3d, np.NaN, dtype=np.float32)
        # cell_CoreMeanW_down = np.full(dims3d, np.NaN, dtype=np.float32)
        # cell_CoreMinQ_down = np.full(dims3d, np.NaN, dtype=np.float32)
        # cell_CoreMeanQ_down = np.full(dims3d, np.NaN, dtype=np.float32)
       
        # new arrays for other things
        cell_dBZ_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_ThteMax_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_ThteMean_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_ThteMean_prm = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_ThteMax_Bolton_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_ThteMean_Bolton_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_ThteMean_Bolton_prm = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_ThtvMax_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_ThtvMean_prm = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_BuoyThtv_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_RhMean_prm = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_TrhoMax_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_TrhoMean_prm = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_BuoyTrho_up = np.full(dims3d, np.NaN, dtype=np.float32)
        # cell_PGF_up = np.full(dims3d, np.NaN, dtype=np.float32)

        # Make a 2D cloudy updraft mask and combine with the cell mask
        # The combined mask will include any updrafts that overlap with the cell mask, even if they are outside of the mask
        # This is useful IF the cell mask does not include any expansion, hence updrafts can exist outside of the cell mask
        # Refer to Enoch Jo's version on the implementation

        # Make a dilation structure
        struc = make_dilation_structure(core_shell_buffer_radius, DX, DY)

        # Loop over each match tracked cell
        for icell in range(nmatchcell):
            # Track number needs to add 1
            itracknum = idx_track[icell] + 1

            # Get the current time
            current_time = datetime.now().time()
            # Format and print the current time
            formatted_time = current_time.strftime("%H:%M:%S")
            print(f'Tracknumber: {itracknum}, current time: {formatted_time}')

            # Count the number of pixels for the original cell mask
            icellmask = tracknumbermap == itracknum
            inpix_cloud = np.count_nonzero(icellmask)

            # Expand the cell mask so that the updraft perimeter expansion is contained within the subset domain
            icellmask_expand = binary_dilation(icellmask, struc)
            # Convert to DataArray
            icellmask_expand = xr.DataArray(icellmask_expand, coords=icellmask.coords, dims=icellmask.dims)

            # Proceed if the number matching cloud pixel > 0
            if inpix_cloud > 0:

                # Subset 3D variables to the current cell mask
                iW = WA.where(icellmask_expand, drop=True).squeeze().data
                # iMassFlux = MassFlux.where(icellmask_expand, drop=True).squeeze().data
                # iRho = RHO_DRY.where(icellmask_expand, drop=True).squeeze().data
                iP = PRESSURE.where(icellmask_expand, drop=True).squeeze().data
                iT = TEMPERATURE.where(icellmask_expand, drop=True).squeeze().data
                iQVAPOR = QVAPOR.where(icellmask_expand, drop=True).squeeze().data
                iTheta = THETA.where(icellmask_expand, drop=True).squeeze().data
                iQCLOUD = QCLOUD.where(icellmask_expand, drop=True).squeeze().data
                iQRAIN = QRAIN.where(icellmask_expand, drop=True).squeeze().data
                iQICE = QICE.where(icellmask_expand, drop=True).squeeze().data
                iQSNOW = QSNOW.where(icellmask_expand, drop=True).squeeze().data
                iQGRAUP = QGRAUP.where(icellmask_expand, drop=True).squeeze().data

                # Total cloud condensates
                iQcld = iQCLOUD + iQICE + iQSNOW
                # Total liquid condensates
                iQliq = iQVAPOR + iQCLOUD + iQRAIN
                # Total condensates + vapor
                iQtotal = iQcld + iQRAIN + iQGRAUP + iQVAPOR

                # Calculate virtual temperature
                # TV = TEMPERATURE * (1 + QVAPOR / 0.622) / (1 + QVAPOR)
                iTV = iT * (1 + iQVAPOR / 0.622) / (1 + iQVAPOR)
                # Calculate moist air density using virtual temperature
                # RHO_MOIST = PRESSURE / (R_dry * TV)  # kg m-3
                iRHO_MOIST = iP / (R_dry * iTV)  # kg m-3
                # Calculate mass flux (kg m-2 s-1)
                # MassFlux = (RHO_MOIST * WA).squeeze()
                iMassFlux = (iRHO_MOIST * iW)

                # Virtual Potential Temperature
                iThtv = iTheta * (iQVAPOR + 0.622)/(0.622 * (1 + iQVAPOR))
                # Density Potential Temperature
                iTrho = iT * (1 + iQVAPOR/0.622) / (1 + iQtotal) * (100000/iP)**(R_dry/Cp_dry)
                # Thopmson RH
                iRH = calc_rh_thompson(iT, iP, iQVAPOR)
                # Emanuel ThetaE
                iThte = calc_theta_e(iT, iP, iQVAPOR, iQliq, iRH)
                # Bolton ThetaE
                iThte_Bolton = theta_e_bolton(iT, iQVAPOR, iP)
                # import pdb; pdb.set_trace()

                # Calculate new statistics of the cell
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)

                    # Loop over vertical level
                    for z in range(0, nz):
                        # print(height[z])
                        zW = iW[z,:,:]
                        zMassFlux = iMassFlux[z,:,:]
                        zQcld = iQcld[z,:,:]
                        zThtv = iThtv[z,:,:]
                        zThte = iThte[z,:,:]
                        zThte_Bolton = iThte_Bolton[z,:,:]
                        # zTv = iTv[z,:,:]
                        zTrho = iTrho[z,:,:]
                        zRH = iRH[z,:,:]

                        # Proceed if max(W) > threshold
                        if np.nanmax(zW) > W_up_thresh:

                            # Label updraft cores
                            dict_up = label_cores(zW, W_up_thresh, zQcld, Q_up_thresh, zMassFlux, ncores_min, min_core_npix, method='>')
                            ncores_all_up = dict_up['ncores_all']
                            ncores_up = dict_up['ncores_save']
                            core_npix_up = dict_up['core_npix']
                            core_numbers_up = dict_up['core_numbers']
                            core_label_up = dict_up['core_label']

                            # Get the min number of cores to save
                            ncores_save_up = min([ncores_up, ncores_min])

                            # Total number of cores
                            cell_nCore_up[icell, z] = ncores_all_up
                                       
                            # Calculate core statistics
                            MaFlx_core_up = np.full(ncores_save_up, np.NaN, dtype=np.float32)
                            W_max_up = np.full(ncores_save_up, np.NaN, dtype=np.float32)
                            W_mean_up = np.full(ncores_save_up, np.NaN, dtype=np.float32)

                            Thte_max_up = np.full(ncores_save_up, np.NaN, dtype=np.float32)
                            Thte_mean_up = np.full(ncores_save_up, np.NaN, dtype=np.float32)
                            Thte_mean_prm = np.full(ncores_save_up, np.NaN, dtype=np.float32)
                            Thte_Bolton_max_up = np.full(ncores_save_up, np.NaN, dtype=np.float32)
                            Thte_Bolton_mean_up = np.full(ncores_save_up, np.NaN, dtype=np.float32)
                            Thte_Bolton_mean_prm = np.full(ncores_save_up, np.NaN, dtype=np.float32)

                            Thtv_max_up = np.full(ncores_save_up, np.NaN, dtype=np.float32)
                            Thtv_mean_prm = np.full(ncores_save_up, np.NaN, dtype=np.float32)
                            Buoy_Thtv_up = np.full(ncores_save_up, np.NaN, dtype=np.float32)

                            Trho_max_up = np.full(ncores_save_up, np.NaN, dtype=np.float32)
                            Trho_mean_prm = np.full(ncores_save_up, np.NaN, dtype=np.float32)
                            Buoy_Trho_up = np.full(ncores_save_up, np.NaN, dtype=np.float32)
                            
                            Rh_mean_prm = np.full(ncores_save_up, np.NaN, dtype=np.float32)

                            # Dilate cores to get perimeter
                            # This dilates all cores by the same distance, thus is much faster to run
                            core_label_up_prm = expand_labels(core_label_up, distance=core_expand_dist) - core_label_up

                            # # Dilate core labels by the equivalent radius of each core
                            # # In theory more physical b/c larger cores would mix air in a wider region than narrower cores
                            # # But the for loop will make it much slower to run
                            # core_label_up_prm = np.zeros_like(core_label_up)
                            # for ii in range(ncores_save_up):
                            #     # Isolate current core
                            #     cell = np.zeros_like(core_label_up)
                            #     cell[core_label_up == core_numbers_up[ii]] = 1
                            #     # Expand core by its equivalent radius (in grid unit)
                            #     expand = round(np.sqrt(core_npix_up[ii]/np.pi))
                            #     dil = expand_labels(cell, distance = expand)
                            #     # Get core perimeter mask
                            #     core_label_up_prm[(dil - cell) == 1] = core_numbers_up[ii]
                            
                            # if (ncores_save_up > 1):
                            #     if (np.nanmax(core_npix_up) > 10):                                    
                            #         import matplotlib.pyplot as plt
                            #         import pdb; pdb.set_trace()

                            # Loop over each core
                            for ii in range(ncores_save_up):
                                # Get the masks for core and perimeter
                                icoremask = core_label_up == core_numbers_up[ii]
                                iperimask = core_label_up_prm == core_numbers_up[ii]

                                MaFlx_core_up[ii] = np.nansum(zMassFlux[icoremask])
                                W_max_up[ii] = np.nanmax(zW[icoremask])
                                W_mean_up[ii] = np.nanmean(zW[icoremask])

                                # ThetaE with Emanuel formula
                                Thte_max_up[ii] = np.nanmax(zThte[icoremask])
                                Thte_mean_up[ii] = np.nanmean(zThte[icoremask])
                                Thte_mean_prm[ii] = np.nanmean(zThte[iperimask])
                                # Theta with Bolton formula
                                Thte_Bolton_max_up[ii] = np.nanmax(zThte_Bolton[icoremask])
                                Thte_Bolton_mean_up[ii] = np.nanmean(zThte_Bolton[icoremask])
                                Thte_Bolton_mean_prm[ii] = np.nanmean(zThte_Bolton[iperimask])
                                # Buoyancy from ThetaV
                                Thtv_max_up[ii] = np.nanmax(zThtv[icoremask])
                                Thtv_mean_prm[ii] = np.nanmean(zThtv[iperimask])
                                Buoy_Thtv_up[ii] = 9.81*(Thtv_max_up[ii] - Thtv_mean_prm[ii]) / Thtv_mean_prm[ii]
                                Rh_mean_prm[ii] = np.nanmean(zRH[iperimask])
                                # Buoyancy from ThetaRho
                                Trho_max_up[ii] = np.nanmax(zTrho[icoremask])
                                Trho_mean_prm[ii] = np.nanmean(zTrho[iperimask])
                                Buoy_Trho_up[ii] = 9.81*(Trho_max_up[ii] - Trho_mean_prm[ii]) / Trho_mean_prm[ii]

                            # Calculate total mass flux for all labeled cores
                            if ncores_all_up > 0:
                                MaFlx_sum_up = np.nansum(zMassFlux[core_label_up > 0])
                            else:
                                MaFlx_sum_up = np.NaN

                            # Save data to output arrays
                            cell_MassFlux_up[icell, z] = MaFlx_sum_up * DX * DY
                            cell_CoreMassFlux_up[icell, z, 0:ncores_save_up] = MaFlx_core_up[0:ncores_save_up] * DX * DY
                            cell_CoreArea_up[icell, z, 0:ncores_save_up] = core_npix_up[0:ncores_save_up] * grid_area
                            cell_CoreMaxW_up[icell, z, 0:ncores_save_up] = W_max_up[0:ncores_save_up]
                            cell_CoreMeanW_up[icell, z, 0:ncores_save_up] = W_mean_up[0:ncores_save_up]

                            cell_ThteMax_up[icell , z, 0:ncores_save_up] = Thte_max_up[0:ncores_save_up]
                            cell_ThteMean_up[icell , z, 0:ncores_save_up] = Thte_mean_up[0:ncores_save_up]
                            cell_ThteMean_prm[icell , z, 0:ncores_save_up] = Thte_mean_prm[0:ncores_save_up]

                            cell_ThteMax_Bolton_up[icell , z, 0:ncores_save_up] = Thte_Bolton_max_up[0:ncores_save_up]
                            cell_ThteMean_Bolton_up[icell , z, 0:ncores_save_up] = Thte_Bolton_mean_up[0:ncores_save_up]
                            cell_ThteMean_Bolton_prm[icell , z, 0:ncores_save_up] = Thte_Bolton_mean_prm[0:ncores_save_up]

                            cell_ThtvMax_up[icell , z, 0:ncores_save_up] = Thtv_max_up[0:ncores_save_up]
                            cell_ThtvMean_prm[icell , z, 0:ncores_save_up] = Thtv_mean_prm[0:ncores_save_up]
                            cell_BuoyThtv_up[icell , z, 0:ncores_save_up] = Buoy_Thtv_up[0:ncores_save_up]

                            cell_TrhoMax_up[icell , z, 0:ncores_save_up] = Trho_max_up[0:ncores_save_up]
                            cell_TrhoMean_prm[icell , z, 0:ncores_save_up] = Trho_mean_prm[0:ncores_save_up]
                            cell_BuoyTrho_up[icell , z, 0:ncores_save_up] = Buoy_Trho_up[0:ncores_save_up]

                            cell_RhMean_prm[icell , z, 0:ncores_save_up] = Rh_mean_prm[0:ncores_save_up]
                        # if np.nanmax(zW) > W_up_thresh:
                        

                        # # Proceed if min(W) < threshold
                        # if np.nanmin(zW) < W_down_thresh:
                        #     # Label downdraft cores
                        #     dict_down = label_cores(zW, W_down_thresh, zQcld, Q_down_thresh, zMassFlux, ncores_min, min_core_npix, method='<')
                        #     ncores_all_down = dict_down['ncores_all']
                        #     ncores_down = dict_down['ncores_save']
                        #     core_npix_down = dict_down['core_npix']
                        #     core_numbers_down = dict_down['core_numbers']
                        #     core_label_down = dict_down['core_label']

                        #     # Get the min number of cores to save
                        #     ncores_save_down = min([ncores_down, ncores_min])

                        #     # Total number of cores
                        #     cell_nCore_down[icell, z] = ncores_all_down
                            
                        #     MaFlx_core_down = np.full(ncores_save_down, np.NaN, dtype=np.float32)
                        #     W_min_down = np.full(ncores_save_down, np.NaN, dtype=np.float32)
                        #     W_mean_down = np.full(ncores_save_down, np.NaN, dtype=np.float32)
                        #     for ii in range(ncores_save_down):
                        #         # Get the masks for core and perimeter
                        #         icoremask = core_label_down == core_numbers_down[ii]
                        #         # iperimask = core_label_down_prm == core_numbers_down[ii]

                        #         MaFlx_core_down[ii] = np.nansum(zMassFlux[icoremask])
                        #         W_min_down[ii] = np.nanmin(zW[icoremask])
                        #         W_mean_down[ii] = np.nanmean(zW[icoremask])
                        #     # Calculate total mass flux for all labeled cores
                        #     if ncores_all_down > 0:
                        #         MaFlx_sum_down = np.nansum(zMassFlux[core_label_down > 0])
                        #     else:
                        #         MaFlx_sum_down = np.NaN
                            
                        #     # Save data to output arrays
                        #     cell_MassFlux_down[icell, z] = MaFlx_sum_down * DX * DY
                        #     cell_CoreMassFlux_down[icell, z, 0:ncores_save_down] = MaFlx_core_down[0:ncores_save_down] * DX * DY
                        #     cell_CoreArea_down[icell, z, 0:ncores_save_down] = core_npix_down[0:ncores_save_down] * grid_area
                        #     cell_CoreMinW_down[icell, z, 0:ncores_save_down] = W_min_down[0:ncores_save_down]
                        #     cell_CoreMeanW_down[icell, z, 0:ncores_save_down] = W_mean_down[0:ncores_save_down]
                        # if np.nanmin(zW) < W_down_thresh:

                    # for z in range(0, nz):
                    # if ncores_save_up > 0:
                    #     import pdb; pdb.set_trace()

                    
            else:
                print(f'No cell matching track # {itracknum}')

        
        # Group outputs in dictionaries
        out_dict3d = {
            'CoreArea_up': cell_CoreArea_up,
            'CoreMaxW_up': cell_CoreMaxW_up,
            'CoreMeanW_up': cell_CoreMeanW_up,
            'CoreMassFlux_up': cell_CoreMassFlux_up,
            'CoreThteMax_up': cell_ThteMax_up,
            'CoreThteMean_up': cell_ThteMean_up,
            'CoreThteMean_prm': cell_ThteMean_prm,
            'CoreThteMax_Bolton_up': cell_ThteMax_Bolton_up,
            'CoreThteMean_Bolton_up': cell_ThteMean_Bolton_up,
            'CoreThteMean_Bolton_prm': cell_ThteMean_Bolton_prm,
            'CoreThtvMax_up': cell_ThtvMax_up,
            'CoreThtvMean_prm': cell_ThtvMean_prm,
            'CoreBuoyThtv_up': cell_BuoyThtv_up,
            'CoreRhMean_prm': cell_RhMean_prm,
            'CoreTrhoMax_up': cell_TrhoMax_up,
            'CoreTrhoMean_prm': cell_TrhoMean_prm,
            'CoreBuoyTrho_up': cell_BuoyTrho_up,
            # 'CorePGF_up': cell_PGF_up,
            # 'CoreArea_down': cell_CoreArea_down,
            # 'CoreMinW_down': cell_CoreMinW_down,
            # 'CoreMeanW_down': cell_CoreMeanW_down,
            # 'CoreMassFlux_down': cell_CoreMassFlux_down,
        }
        out_dict2d = {
            'nCore_up': cell_nCore_up,
            'MassFlux_up': cell_MassFlux_up,
            # 'nCore_down': cell_nCore_down,
            # 'MassFlux_down': cell_MassFlux_down,
        }
        out_dict_attrs = {
            # Updraft
            'nCore_up': {
                'long_name': 'Number of updraft cores',
                'units': 'count',
            },
            'CoreArea_up': {
                'long_name': 'Updraft core area',
                'units': 'km^2',
            },
            'CoreMaxW_up': {
                'long_name': 'Updraft core maximum W',
                'units': 'm/s',
            },
            'CoreMeanW_up': {
                'long_name': 'Updraft core mean W',
                'units': 'm/s',
            },
            'CoreMassFlux_up': {
                'long_name': 'Updraft core mass flux',
                'units': 'kg s^-1',
            },
            'MassFlux_up': {
                'long_name': 'Total updraft mass flux',
                'units': 'kg s^-1',
            },
            # ThetaE Emanuel
            'CoreThteMax_up': {
                'long_name': 'Updraft core max Theta e (Emanuel reversible formula)',
                'units': 'K',
            },
            'CoreThteMean_up': {
                'long_name': 'Updraft core mean Theta e (Emanuel reversible formula)',
                'units': 'K',
            },
            'CoreThteMean_prm': {
                'long_name': 'Updraft perim mean Theta e (Emanuel reversible formula)',
                'units': 'K',
            },
            # ThetaE Bolton
            'CoreThteMax_Bolton_up': {
                'long_name': 'Updraft core max Theta e (Bolton pseudo-adiabatic formula)',
                'units': 'K',
            },
            'CoreThteMean_Bolton_up': {
                'long_name': 'Updraft core mean Theta e (Bolton pseudo-adiabatic formula)',
                'units': 'K',
            },
            'CoreThteMean_Bolton_prm': {
                'long_name': 'Updraft perim mean Theta e (Bolton pseudo-adiabatic formula)',
                'units': 'K',
            },
            # ThetaV
            'CoreThtvMax_up': {
                'long_name': 'Updraft core max Theta v',
                'units': 'K',
            },
            'CoreThtvMean_prm': {
                'long_name': 'Updraft core boundary mean Theta v',
                'units': 'K',
            },
            'CoreBuoyThtv_up': {
                'long_name': 'Updraft core Buoyancy based on Theta v',
                'units': 'm s^-2',
            },
            # Theta rho
            'CoreTrhoMax_up': {
                'long_name': 'Updraft core max density Theta',
                'units': 'K',
            },
            'CoreTrhoMean_prm': {
                'long_name': 'Updraft core boundary mean density Theta',
                'units': 'K',
            },
            'CoreBuoyTrho_up': {
                'long_name': 'Updraft core Buoyancy based on density Theta (density potential temperature)',
                'units': 'm s^-2',
            },
            'CoreRhMean_prm':{
                'long_name': 'Updraft perim mean RH',
                'units': '%',
            },
            # # Downdraft
            # 'nCore_down': {
            #     'long_name': 'Number of downdraft cores',
            #     'units': 'count',
            # },
            # 'CoreArea_down': {
            #     'long_name': 'Downdraft core area',
            #     'units': 'km^2',
            # },
            # 'CoreMinW_down': {
            #     'long_name': 'Downdraft core minimum W',
            #     'units': 'm/s',
            # },
            # 'CoreMeanW_down': {
            #     'long_name': 'Downdraft core mean W',
            #     'units': 'm/s',
            # },
            # 'CoreMassFlux_down': {
            #     'long_name': 'Downdraft core mass flux',
            #     'units': 'kg s^-1',
            # },
            # 'MassFlux_down': {
            #     'long_name': 'Total downdraft mass flux',
            #     'units': 'kg s^-1',
            # },
        }
    # Clean up local variables to free memory
    del dsm, dsc, ds
    if 'PRESSURE' in locals(): del PRESSURE, TEMPERATURE, QVAPOR, THETA, WA
    if 'QCLOUD' in locals(): del QCLOUD, QRAIN, QICE, QSNOW, QGRAUP
    
    return out_dict3d, out_dict2d, out_dict_attrs


#-----------------------------------------------------------------------
if __name__ == '__main__':

    # Get configuration file name from input
    config_file = sys.argv[1]
    # Read configuration from yaml file
    stream = open(config_file, 'r')
    config = yaml.full_load(stream)

    run_parallel = config['run_parallel']
    n_workers = config['n_workers']
    threads_per_worker = config['threads_per_worker']
    startdate = config['startdate']
    enddate = config['enddate']
    time_window = config['time_window']
    stats_path = config['stats_path']
    pixelfile_path = config['pixelfile_path']
    metfile_path = config['metfile_path']
    metfile_path_2 = config['metfile_path_2']
    output_path = config['output_path']
    methamsl_filebase = config['methamsl_filebase']
    cldhamsl_filebase = config['cldhamsl_filebase']
    pixel_filebase = config['pixel_filebase']
    ncores_min = config['ncores_min']

    # Replace directory (some LASSO data are staged in a different directory)
    if os.path.isdir(metfile_path_2):
        metfile_path = metfile_path_2

    # Add start/end date to pixel file path
    pixelfile_path = f'{pixelfile_path}{startdate}_{enddate}/'

    # Track stats file basename
    stats_filebase = 'trackstats_'

    # Output statistics filename
    # output_basename = 'stats_3d_w_'
    output_basename = 'stats_3d_w_fixshell_'
    output_filename = f'{output_path}{output_basename}{startdate}_{enddate}.nc'
    os.makedirs(output_path, exist_ok=True)

    # Track statistics file dimension names
    tracks_dimname = 'tracks'
    times_dimname = 'times'
    z_dimname = 'z'
    core_dimname = 'core'

    # Track statistics file
    trackstats_file = f'{stats_path}{stats_filebase}{startdate}_{enddate}.nc'
    # Find all pixel-level files
    pixelfilelist = sorted(glob.glob(f'{pixelfile_path}{pixel_filebase}*.nc'))
    nfiles = len(pixelfilelist)
    # Find all Met files
    metfilelist = sorted(glob.glob(f'{metfile_path}{methamsl_filebase}*.nc'))
    cldfilelist = sorted(glob.glob(f'{metfile_path}{cldhamsl_filebase}*.nc'))
    nmetfiles = len(metfilelist)
    ncldfiles = len(cldfilelist)
    print(f'Number of MET files: {nmetfiles}')
    print(f'Number of CLD files: {ncldfiles}')

    # Get basetime from pixel files
    pixel_basetime, pixelfile_dict = calc_basetime(pixelfilelist, pixel_filebase)
    # Get basetime from met & cld files
    met_basetime, regfile_dict = calc_basetime(metfilelist, methamsl_filebase)
    cld_basetime, cldfile_dict = calc_basetime(cldfilelist, cldhamsl_filebase)

    # Find matching MET files for each pixel file
    match_metfilelist = [''] * nfiles
    match_cldfilelist = [''] * nfiles
    for ifile in range(nfiles):
        # Find MET time closest to the pixel file time and get the index
        # Save the filename if time difference is < time_window
        idx = np.argmin(np.abs(met_basetime - pixel_basetime[ifile]))
        if np.abs(met_basetime[idx] - pixel_basetime[ifile]) < time_window:
            match_metfilelist[ifile] = regfile_dict[met_basetime[idx]]
        else:
            print(f'No match met file found for: {pixelfilelist[ifile]}')

        idx = np.argmin(np.abs(cld_basetime - pixel_basetime[ifile]))
        if np.abs(cld_basetime[idx] - pixel_basetime[ifile]) < time_window:
            match_cldfilelist[ifile] = cldfile_dict[cld_basetime[idx]]
        else:
            print(f'No match cld file found for: {pixelfilelist[ifile]}')

    # Read track statistics file
    print(trackstats_file)
    dsstats = xr.open_dataset(trackstats_file, decode_times=False)
    ntracks = dsstats.sizes[tracks_dimname]
    ntimes = dsstats.sizes[times_dimname]
    coord_tracks = dsstats[tracks_dimname]
    coord_times = dsstats[times_dimname]
    stats_basetime = dsstats['base_time'].data
    stats_basetime_attrs = dsstats['base_time'].attrs
    # cell_area = dsstats['cell_area']
    pixel_radius = dsstats.attrs['pixel_radius_km']
    dsstats.close()

    print(f'Total Number of Tracks: {ntracks}')

    # Read a MET file to get vertical coordinates
    dsm = xr.open_dataset(match_metfilelist[0])
    nz = dsm.sizes['HAMSL']
    height = dsm['HAMSL']
    dsm.close()


    ##############################################################
    # Call function to calculate statistics
    trackindices_all = []
    timeindices_all = []
    final_results = []

    if run_parallel == 1:
        # Enhanced Dask configuration for large datasets
        dask_tmp_dir = config.get("dask_tmp_dir", "/tmp")
        dask.config.set({
            'temporary-directory': dask_tmp_dir,
            'distributed.worker.memory.target': 0.8,  # Target 80% memory usage
            'distributed.worker.memory.spill': 0.85,  # Spill to disk at 85%
            'distributed.worker.memory.pause': 0.9,   # Pause at 90%
            'distributed.worker.memory.terminate': 0.95,  # Terminate at 95%
            'distributed.comm.timeouts.connect': '60s',
            'distributed.comm.timeouts.tcp': '60s',
            'distributed.worker.daemon': False,
            'distributed.scheduler.idle-timeout': '1h',
            'distributed.worker.lifetime.duration': '4h',  # Restart workers every 4h
            'distributed.worker.lifetime.stagger': '1h',   # Stagger restarts
        })
        
        try:
            print(f"Initializing Dask cluster with {n_workers} workers...")
            
            # Reduce memory per worker for this large dataset
            memory_per_worker = min(48, int(256 / n_workers * 0.8))  # Use 80% of available memory
            
            cluster = LocalCluster(
                n_workers=n_workers, 
                threads_per_worker=threads_per_worker,
                memory_limit=f'{memory_per_worker}GB',
                processes=True,
                silence_logs=False,  # Keep logs for debugging
                dashboard_address=':8787',
                # Add worker resource limits
                worker_class='distributed.nanny.Nanny',  # Use nannies for better process management
                timeout='120s',
            )
            
            client = Client(cluster, timeout='120s')
            print(f"Dask cluster initialized successfully")
            print(f"Dashboard link: {client.dashboard_link}")
            print(f"Memory per worker: {memory_per_worker}GB")
            
        except Exception as e:
            print(f"Failed to initialize Dask cluster: {e}")
            print("Falling back to serial processing")
            run_parallel = 0

    # Enhanced processing loop with better monitoring
    print(f"Starting processing of {nfiles} files...")
    start_time = time.time()
    last_progress_time = start_time
    
    # Reduce chunk size for large datasets to prevent memory issues
    max_concurrent_tasks = min(n_workers * 2, 16)  # Limit concurrent tasks
    
    # Loop over each pixel-file and call function to calculate
    for ifile in range(nfiles):
        file_start_time = time.time()
        
        # Enhanced timeout detection
        current_time = time.time()
        if current_time - last_progress_time > 1800:  # 30 minutes
            print(f"WARNING: No progress for {(current_time - last_progress_time)/60:.1f} minutes")
            print(f"Current file: {ifile}, file: {pixelfilelist[ifile] if ifile < len(pixelfilelist) else 'N/A'}")
            # Try to recover by cleaning up memory and continuing
            cleanup_memory()
            if run_parallel == 1:
                try:
                    client.restart()
                    print("Dask client restarted")
                except:
                    print("Failed to restart client, continuing...")
            last_progress_time = current_time
        
        # Monitor memory every 20 files
        if ifile % 20 == 0:
            log_memory_usage(f"Processing file {ifile}/{nfiles}")
            if run_parallel == 1:
                try:
                    # Check cluster health
                    cluster_info = client.scheduler_info()
                    print(f"Active workers: {len(cluster_info['workers'])}")
                except:
                    print("Warning: Could not get cluster info")
        
        try:
            print(f"File {ifile}: {os.path.basename(pixelfilelist[ifile])}")
            
            # Find all matching time indices from track stats file to the current pixel file
            matchindices = np.array(
                np.where(np.abs(stats_basetime - pixel_basetime[ifile]) < time_window)
            )
            idx_track = matchindices[0]
            idx_time = matchindices[1]
            
            print(f"  Found {len(idx_track)} matching tracks")

            if len(idx_track) > 0:
                trackindices_all.append(idx_track)
                timeindices_all.append(idx_time)
                
                # Serial processing for large track counts to avoid memory issues
                if len(idx_track) > 50 or run_parallel == 0:
                    print(f"  Processing serially (large track count: {len(idx_track)})...")
                    iresult = calc_cellstats_singlefile(
                        pixelfilelist[ifile], 
                        match_metfilelist[ifile],
                        match_cldfilelist[ifile],
                        idx_track, 
                        config,
                    )
                    final_results.append(iresult)
                # Parallel processing for smaller track counts
                elif run_parallel == 1:
                    print(f"  Creating delayed task...")
                    iresult = dask.delayed(calc_cellstats_singlefile)(
                        pixelfilelist[ifile], 
                        match_metfilelist[ifile],
                        match_cldfilelist[ifile],
                        idx_track, 
                        config,
                    )
                    final_results.append(iresult)
                    
                    # Process in smaller batches to prevent memory buildup
                    if len(final_results) >= max_concurrent_tasks:
                        print(f"  Processing batch of {len(final_results)} tasks...")
                        try:
                            batch_results = dask.compute(*final_results[-max_concurrent_tasks:])
                            # Replace delayed objects with computed results
                            final_results[-max_concurrent_tasks:] = batch_results
                            cleanup_memory()
                            print(f"  Batch completed successfully")
                        except Exception as e:
                            print(f"  Batch processing failed: {e}, falling back to serial")
                            # Fall back to serial processing for remaining files
                            run_parallel = 0
            else:
                print(f"  No matching tracks found, skipping")
            
            last_progress_time = time.time()
            file_duration = last_progress_time - file_start_time
            print(f"  File {ifile} completed in {file_duration:.1f}s")
            
            # More frequent cleanup for large datasets
            if (ifile + 1) % 5 == 0:
                cleanup_memory()
                
        except Exception as e:
            print(f"ERROR processing file {ifile}: {e}")
            traceback.print_exc()
            # Try to recover
            cleanup_memory()
            continue

    print(f"File processing loop completed. Total files processed: {len(final_results)}")

    # Enhanced final computation with better error handling
    if run_parallel == 1 and len(final_results) > 0:
        print("Computing remaining statistics with Dask...")
        computation_start = time.time()
        
        try:
            # Filter out already computed results
            delayed_results = [r for r in final_results if hasattr(r, 'compute')]
            computed_results = [r for r in final_results if not hasattr(r, 'compute')]
            
            if delayed_results:
                # Process remaining delayed results in very small chunks
                chunk_size = min(5, len(delayed_results))  # Much smaller chunks
                
                for chunk_start in range(0, len(delayed_results), chunk_size):
                    chunk_end = min(chunk_start + chunk_size, len(delayed_results))
                    chunk = delayed_results[chunk_start:chunk_end]
                    
                    print(f"Computing final chunk {chunk_start//chunk_size + 1}/{(len(delayed_results)-1)//chunk_size + 1} "
                          f"({chunk_start}-{chunk_end-1})")
                    
                    # Add timeout for each chunk
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(dask.compute, *chunk)
                        try:
                            chunk_results = future.result(timeout=3600)  # 1 hour timeout per chunk
                            computed_results.extend(chunk_results)
                        except concurrent.futures.TimeoutError:
                            print(f"Chunk computation timed out, trying serial processing...")
                            # Fall back to serial for this chunk
                            for delayed_task in chunk:
                                try:
                                    result = delayed_task.compute()
                                    computed_results.append(result)
                                except Exception as e:
                                    print(f"Serial computation failed: {e}")
                                    computed_results.append((None, None, None))
                    
                    cleanup_memory()
                    print(f"Chunk completed in {time.time() - computation_start:.1f}s")
            
            final_results = computed_results
            print(f"All computations completed in {time.time() - computation_start:.1f}s")
            
        except Exception as e:
            print(f"ERROR during final Dask computation: {e}")
            traceback.print_exc()
            print("Falling back to serial processing for remaining tasks...")
            
            # Emergency serial fallback
            computed_results = []
            for i, result in enumerate(final_results):
                if hasattr(result, 'compute'):
                    try:
                        computed_results.append(result.compute())
                    except Exception as e:
                        print(f"Serial fallback failed for task {i}: {e}")
                        computed_results.append((None, None, None))
                else:
                    computed_results.append(result)
            final_results = computed_results
        
        # Close Dask client and cluster
        try:
            client.close()
            cluster.close()
        except:
            pass
            
    elif run_parallel == 0:
        print("Serial processing completed")

    # Make a variable list and get attributes from one of the returned dictionaries
    # Loop over each return results till one that is not None
    counter = len(final_results)-1
    while counter >= 0:
        if final_results[counter] is not None:
            var_names3d = list(final_results[counter][0].keys())
            var_names2d = list(final_results[counter][1].keys())
            var_attrs = final_results[counter][2]
            break
        counter -= 1

    # Loop over variable list to create the dictionary entry
    print(f'Creating output arrays ...')
    out_dict = {}
    out_dict_attrs = {}

    var_names = var_names3d + var_names2d
    # 3D variables 
    for ivar in var_names3d:
        out_dict[ivar] = np.full((ntracks, ntimes, nz, ncores_min), np.nan, dtype=np.float32)
        out_dict_attrs[ivar] = var_attrs[ivar]
    # 2D variables
    for ivar in var_names2d:
        out_dict[ivar] = np.full((ntracks, ntimes, nz), np.nan, dtype=np.float32)
        out_dict_attrs[ivar] = var_attrs[ivar]

    # Put the results to output track stats variables
    # Loop over each returned results
    for ifile in range(len(final_results)):
        # Check the return results
        if final_results[ifile] is not None:
            iVAR3d = final_results[ifile][0]
            iVAR2d = final_results[ifile][1]
            if iVAR3d is not None:
                trackindices = trackindices_all[ifile]
                timeindices = timeindices_all[ifile]
                # Loop over each variable and assign values to output dictionary
                for ivar in var_names3d:
                    if iVAR3d[ivar].ndim == 3:
                        # Check shape compatibility before assignment
                        expected_shape = out_dict[ivar][trackindices,timeindices,:,:].shape
                        actual_shape = iVAR3d[ivar].shape
                        if expected_shape == actual_shape:
                            out_dict[ivar][trackindices,timeindices,:,:] = iVAR3d[ivar]
                        else:
                            print(f'WARNING: Shape mismatch for {ivar} in file {ifile}:')
                            print(f'  Expected shape: {expected_shape}, got: {actual_shape}')
                            print(f'  Track indices: {len(trackindices)}, Time indices: {len(timeindices)}')
                            print(f'  Skipping assignment for this file.')
                    else:
                        print(f'Warning: {ivar} dimension is not 3.')
            if iVAR2d is not None:
                trackindices = trackindices_all[ifile]
                timeindices = timeindices_all[ifile]
                # Loop over each variable and assign values to output dictionary
                for ivar in var_names2d:
                    if iVAR2d[ivar].ndim == 2:
                        # Check shape compatibility before assignment
                        expected_shape = out_dict[ivar][trackindices,timeindices,:].shape
                        actual_shape = iVAR2d[ivar].shape
                        if expected_shape == actual_shape:
                            out_dict[ivar][trackindices,timeindices,:] = iVAR2d[ivar]
                        else:
                            print(f'WARNING: Shape mismatch for {ivar} in file {ifile}:')
                            print(f'  Expected shape: {expected_shape}, got: {actual_shape}')
                            print(f'  Track indices: {len(trackindices)}, Time indices: {len(timeindices)}')
                            print(f'  Skipping assignment for this file.')
                    else:
                        print(f'Warning: {ivar} dimension is not 2.')

    ##########################################################
    # Write to netcdf
    print('Writing output netcdf ... ')

    # Define variable list
    var_dict = {}
    # Define output variable dictionary
    for key, value in out_dict.items():
        if value.ndim == 2:
            var_dict[key] = ([tracks_dimname, times_dimname], value, out_dict_attrs[key])
        if value.ndim == 3:
            var_dict[key] = ([tracks_dimname, times_dimname, z_dimname], value, out_dict_attrs[key])
        if value.ndim == 4:
            var_dict[key] = ([tracks_dimname, times_dimname, z_dimname, core_dimname], value, out_dict_attrs[key])
    # Add base_time from track stats to the output dictionary
    var_dict['base_time'] = ([tracks_dimname, times_dimname], stats_basetime, stats_basetime_attrs)
    # Define coordinate list
    core_dim_attrs = {
        'long_name': 'Core number',
    }
    coord_dict = {
        tracks_dimname: ([tracks_dimname], coord_tracks.data, coord_tracks.attrs),
        times_dimname: ([times_dimname], coord_times.data, coord_times.attrs),
        z_dimname: ([z_dimname], height.data, height.attrs),
        core_dimname: ([core_dimname], np.arange(0, ncores_min), core_dim_attrs),
    }
    # Define global attributes
    gattr_dict = {
        'title':  'Tracked cell W statistics', \
        'Institution': 'Pacific Northwest National Laboratoy', \
        'Contact': 'Zhe Feng, zhe.feng@pnnl.gov', \
        'Created_on':  time.ctime(time.time()), \
        'source_trackfile': trackstats_file, \
        'startdate': startdate, \
        'enddate': enddate, \
    }
    # Define xarray dataset
    dsout = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)

    # Delete file if it already exists
    if os.path.isfile(output_filename):
        os.remove(output_filename)
        
    # Set encoding/compression for all variables
    comp = dict(zlib=True)
    encoding = {var: comp for var in dsout.data_vars}

    # Write to netcdf file
    dsout.to_netcdf(path=output_filename, mode="w",
                    format="NETCDF4", unlimited_dims=tracks_dimname, encoding=encoding)
    print(f'Output saved: {output_filename}')

    # Clean up Dask cluster if it was used
    if run_parallel == 1:
        print('Closing Dask cluster...')
        try:
            client.close()
            cluster.close()
        except:
            pass
        print('Dask cluster closed.')

    # Final cleanup
    cleanup_memory()
    log_memory_usage("Final cleanup")
    print('Processing completed successfully.')