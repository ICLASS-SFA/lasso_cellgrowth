"""
Calculates 3D W core statistics from gridded Met data and Draft data for tracked convective cells.
The 3D core statistics are written to netCDF file matching the cell track statistics file format.
"""
import numpy as np
import os, sys, glob
import time
from datetime import datetime
from pytz import utc
import yaml
import xarray as xr
from scipy import ndimage
from skimage.segmentation import expand_labels
from scipy.ndimage import generate_binary_structure, binary_dilation, iterate_structure
import warnings
import dask
from dask.distributed import Client, LocalCluster
import psutil
import gc
# import matplotlib.pyplot as plt

def log_memory_usage(stage_name):
    """Log current memory usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"{stage_name}: Memory usage: {memory_info.rss / 1024**3:.2f} GB")

def cleanup_memory():
    """Enhanced memory cleanup function"""
    gc.collect()
    # Clear Dask cache more aggressively
    try:
        import dask
        import dask.array as da
        # Clear all caches
        da.core.clear_cache()
        # Force garbage collection in Dask
        from dask.base import clear_cache
        clear_cache()
        # Clear any remaining delayed objects
        dask.base.clear_cache()
    except (ImportError, AttributeError):
        pass

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
    draft_filename,
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
        draft_filename: string
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

    # Read MET file with dask lazy evaluation to prevent I/O blocking
    # Large files (2775 × 2145 × 40) should not be loaded into memory immediately
    dsm = xr.open_dataset(met_filename, chunks={'HAMSL': 10, 'south_north': 500, 'west_east': 500})
    # Rename dimenensions
    dsm = dsm.rename_dims({'south_north':'lat', 'west_east':'lon'})
    nz = dsm.sizes['HAMSL']
    ny = dsm.sizes['lat']
    nx = dsm.sizes['lon']
    height = dsm['HAMSL'].data
    DX = dsm.attrs['DX']
    DY = dsm.attrs['DY']
    grid_area = DX * DY / 1e6

    # Read draft file with dask lazy evaluation to prevent I/O blocking
    dsc = xr.open_dataset(draft_filename, chunks={'HAMSL': 10, 'south_north': 500, 'west_east': 500})
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
        # PRESSURE = dsm['PRESSURE'][:, :, ymin:ymax+1, xmin:xmax+1]*100  # [Pa]
        # TEMPERATURE = dsm['TEMPERATURE'][:, :, ymin:ymax+1, xmin:xmax+1]
        # QVAPOR = dsm['QVAPOR'][:, :, ymin:ymax+1, xmin:xmax+1]
        # # TV = dsm['TV'][:, :, ymin:ymax+1, xmin:xmax+1]
        # THETA = dsm['THETA'][:, :, ymin:ymax+1, xmin:xmax+1]
        WA = dsm['WA'][:, :, ymin:ymax+1, xmin:xmax+1]
        # Get draft variables
        # QCLOUD = dsc['QCLOUD'][:, :, ymin:ymax+1, xmin:xmax+1]
        # QRAIN = dsc['QRAIN'][:, :, ymin:ymax+1, xmin:xmax+1]
        # QICE = dsc['QICE'][:, :, ymin:ymax+1, xmin:xmax+1]
        # QSNOW = dsc['QSNOW'][:, :, ymin:ymax+1, xmin:xmax+1]
        # QGRAUP = dsc['QGRAUP'][:, :, ymin:ymax+1, xmin:xmax+1]
        Qcld = dsc['Qcld'][:, :, ymin:ymax+1, xmin:xmax+1]
        # Qliq = dsc['Qliq'][:, :, ymin:ymax+1, xmin:xmax+1]
        # Qtotal = dsc['Qtotal'][:, :, ymin:ymax+1, xmin:xmax+1]
        MassFlux = dsc['MassFlux'][:, :, ymin:ymax+1, xmin:xmax+1]
        ThetaV = dsc['ThetaV'][:, :, ymin:ymax+1, xmin:xmax+1]
        ThetaRho = dsc['ThetaRho'][:, :, ymin:ymax+1, xmin:xmax+1]
        RH = dsc['RH'][:, :, ymin:ymax+1, xmin:xmax+1]
        ThetaE = dsc['ThetaE'][:, :, ymin:ymax+1, xmin:xmax+1]
        ThetaE_Bolton = dsc['ThetaE_Bolton'][:, :, ymin:ymax+1, xmin:xmax+1]
        # Update ny, nx with the subset
        ny = XLONG.sizes['lat']
        nx = XLONG.sizes['lon']
    else:
        XLONG = dsm['XLONG']
        XLAT = dsm['XLAT']
        # PRESSURE = dsm['PRESSURE']*100  # [Pa]
        # TEMPERATURE = dsm['TEMPERATURE']
        # QVAPOR = dsm['QVAPOR']
        # THETA = dsm['THETA']
        # TV = dsm['TV']
        WA = dsm['WA']
        # Get draft variables
        Qcld = dsc['Qcld']
        # Qliq = dsc['Qliq']
        # Qtotal = dsc['Qtotal']
        MassFlux = dsc['MassFlux']
        ThetaV = dsc['ThetaV']
        ThetaRho = dsc['ThetaRho']
        RH = dsc['RH']
        ThetaE = dsc['ThetaE']
        ThetaE_Bolton = dsc['ThetaE_Bolton']
        # QCLOUD = dsc['QCLOUD']
        # QRAIN = dsc['QRAIN']
        # QICE = dsc['QICE']
        # QSNOW = dsc['QSNOW']
        # QGRAUP = dsc['QGRAUP']

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
            # Start timing for this track
            track_start_time = time.time()
            
            # Track number needs to add 1
            itracknum = idx_track[icell] + 1

            # Memory management for small files (max 88 tracks)
            if icell % 25 == 0:  # Less frequent progress reporting
                print(f'Processing track {icell+1}/{nmatchcell} (track #{itracknum}) - starting...')
            
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
                # Note: These will be dask arrays due to chunked loading
                iW = WA.where(icellmask_expand, drop=True).squeeze()

                iQcld = Qcld.where(icellmask_expand, drop=True).squeeze()
                # iQliq = Qliq.where(icellmask_expand, drop=True).squeeze()
                # iQtotal = Qtotal.where(icellmask_expand, drop=True).squeeze()
                iMassFlux = MassFlux.where(icellmask_expand, drop=True).squeeze()
                iThtv = ThetaV.where(icellmask_expand, drop=True).squeeze()
                iTrho = ThetaRho.where(icellmask_expand, drop=True).squeeze()
                iRH = RH.where(icellmask_expand, drop=True).squeeze()
                iThte = ThetaE.where(icellmask_expand, drop=True).squeeze()
                iThte_Bolton = ThetaE_Bolton.where(icellmask_expand, drop=True).squeeze()

                # iMassFlux = MassFlux.where(icellmask_expand, drop=True).squeeze().data
                # iRho = RHO_DRY.where(icellmask_expand, drop=True).squeeze().data
                # iP = PRESSURE.where(icellmask_expand, drop=True).squeeze().data
                # iT = TEMPERATURE.where(icellmask_expand, drop=True).squeeze().data
                # iQVAPOR = QVAPOR.where(icellmask_expand, drop=True).squeeze().data
                # iTheta = THETA.where(icellmask_expand, drop=True).squeeze().data
                # iQCLOUD = QCLOUD.where(icellmask_expand, drop=True).squeeze().data
                # iQRAIN = QRAIN.where(icellmask_expand, drop=True).squeeze().data
                # iQICE = QICE.where(icellmask_expand, drop=True).squeeze().data
                # iQSNOW = QSNOW.where(icellmask_expand, drop=True).squeeze().data
                # iQGRAUP = QGRAUP.where(icellmask_expand, drop=True).squeeze().data

                # # Total cloud condensates
                # iQcld = iQCLOUD + iQICE + iQSNOW
                # # Total liquid condensates
                # iQliq = iQVAPOR + iQCLOUD + iQRAIN
                # # Total condensates + vapor
                # iQtotal = iQcld + iQRAIN + iQGRAUP + iQVAPOR

                # # Calculate virtual temperature
                # # TV = TEMPERATURE * (1 + QVAPOR / 0.622) / (1 + QVAPOR)
                # iTV = iT * (1 + iQVAPOR / 0.622) / (1 + iQVAPOR)
                # # Calculate moist air density using virtual temperature
                # # RHO_MOIST = PRESSURE / (R_dry * TV)  # kg m-3
                # iRHO_MOIST = iP / (R_dry * iTV)  # kg m-3
                # # Calculate mass flux (kg m-2 s-1)
                # # MassFlux = (RHO_MOIST * WA).squeeze()
                # iMassFlux = (iRHO_MOIST * iW)

                # # Virtual Potential Temperature
                # iThtv = iTheta * (iQVAPOR + 0.622)/(0.622 * (1 + iQVAPOR))
                # # Density Potential Temperature
                # iTrho = iT * (1 + iQVAPOR/0.622) / (1 + iQtotal) * (100000/iP)**(R_dry/Cp_dry)
                # # Thopmson RH
                # iRH = calc_rh_thompson(iT, iP, iQVAPOR)
                # # Emanuel ThetaE
                # iThte = calc_theta_e(iT, iP, iQVAPOR, iQliq, iRH)
                # # Bolton ThetaE
                # iThte_Bolton = theta_e_bolton(iT, iQVAPOR, iP)

                # iQcld = Qcld.where(icellmask_expand, drop=True).squeeze().data
                # import pdb; pdb.set_trace()

                # Calculate new statistics of the cell
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)

                    # For small track counts, process all levels (simpler and fast enough)
                    # Pre-compute max W across all levels for optimization
                    # Note: iW is now a dask array, so we need to compute to get actual values
                    max_W_all_levels = np.nanmax(iW, axis=(1,2))
                    if hasattr(max_W_all_levels, 'compute'):
                        max_W_all_levels = max_W_all_levels.compute()
                    
                    active_levels = np.where(max_W_all_levels > W_up_thresh)[0]
                    
                    if len(active_levels) == 0:
                        print(f'  Track {itracknum}: No levels with W > {W_up_thresh}, skipping')
                        continue
                    
                    # For small files, no need for detailed progress reporting
                    if nmatchcell > 50:
                        print(f'  Track {itracknum}: Processing {len(active_levels)} active levels out of {nz}')

                    # Start timing for vertical level processing
                    levels_start_time = time.time()

                    # Loop over vertical level (only process active levels)
                    for z_idx, z in enumerate(active_levels):
                        # Extract level data and compute to get numpy arrays for processing
                        zW = iW[z,:,:].values if hasattr(iW[z,:,:], 'values') else iW[z,:,:].compute()
                        zMassFlux = iMassFlux[z,:,:].values if hasattr(iMassFlux[z,:,:], 'values') else iMassFlux[z,:,:].compute()
                        zQcld = iQcld[z,:,:].values if hasattr(iQcld[z,:,:], 'values') else iQcld[z,:,:].compute()
                        zThtv = iThtv[z,:,:].values if hasattr(iThtv[z,:,:], 'values') else iThtv[z,:,:].compute()
                        zThte = iThte[z,:,:].values if hasattr(iThte[z,:,:], 'values') else iThte[z,:,:].compute()
                        zThte_Bolton = iThte_Bolton[z,:,:].values if hasattr(iThte_Bolton[z,:,:], 'values') else iThte_Bolton[z,:,:].compute()
                        zTrho = iTrho[z,:,:].values if hasattr(iTrho[z,:,:], 'values') else iTrho[z,:,:].compute()
                        zRH = iRH[z,:,:].values if hasattr(iRH[z,:,:], 'values') else iRH[z,:,:].compute()

                        # Process if max W at this level exceeds threshold
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

                            # Loop over each core - optimized for better performance
                            # Pre-compute all core and perimeter masks at once
                            core_masks = []
                            perim_masks = []
                            for ii in range(ncores_save_up):
                                core_masks.append(core_label_up == core_numbers_up[ii])
                                perim_masks.append(core_label_up_prm == core_numbers_up[ii])
                            
                            for ii in range(ncores_save_up):
                                icoremask = core_masks[ii]
                                iperimask = perim_masks[ii]

                                # Skip if core mask is empty (shouldn't happen but safety check)
                                if not np.any(icoremask):
                                    continue

                                MaFlx_core_up[ii] = np.nansum(zMassFlux[icoremask])
                                W_max_up[ii] = np.nanmax(zW[icoremask])
                                W_mean_up[ii] = np.nanmean(zW[icoremask])

                                # ThetaE with Emanuel formula
                                Thte_max_up[ii] = np.nanmax(zThte[icoremask])
                                Thte_mean_up[ii] = np.nanmean(zThte[icoremask])
                                if np.any(iperimask):  # Check if perimeter exists
                                    Thte_mean_prm[ii] = np.nanmean(zThte[iperimask])
                                else:
                                    Thte_mean_prm[ii] = np.nan
                                    
                                # Theta with Bolton formula
                                Thte_Bolton_max_up[ii] = np.nanmax(zThte_Bolton[icoremask])
                                Thte_Bolton_mean_up[ii] = np.nanmean(zThte_Bolton[icoremask])
                                if np.any(iperimask):
                                    Thte_Bolton_mean_prm[ii] = np.nanmean(zThte_Bolton[iperimask])
                                else:
                                    Thte_Bolton_mean_prm[ii] = np.nan
                                    
                                # Buoyancy from ThetaV
                                Thtv_max_up[ii] = np.nanmax(zThtv[icoremask])
                                if np.any(iperimask):
                                    Thtv_mean_prm[ii] = np.nanmean(zThtv[iperimask])
                                    Buoy_Thtv_up[ii] = 9.81*(Thtv_max_up[ii] - Thtv_mean_prm[ii]) / Thtv_mean_prm[ii]
                                    Rh_mean_prm[ii] = np.nanmean(zRH[iperimask])
                                else:
                                    Thtv_mean_prm[ii] = np.nan
                                    Buoy_Thtv_up[ii] = np.nan
                                    Rh_mean_prm[ii] = np.nan
                                    
                                # Buoyancy from ThetaRho
                                Trho_max_up[ii] = np.nanmax(zTrho[icoremask])
                                if np.any(iperimask):
                                    Trho_mean_prm[ii] = np.nanmean(zTrho[iperimask])
                                    Buoy_Trho_up[ii] = 9.81*(Trho_max_up[ii] - Trho_mean_prm[ii]) / Trho_mean_prm[ii]
                                else:
                                    Trho_mean_prm[ii] = np.nan
                                    Buoy_Trho_up[ii] = np.nan

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

                    # End timing for vertical level processing  
                    levels_end_time = time.time()
                    levels_duration = levels_end_time - levels_start_time
                    
                    # Report vertical level processing time for detailed diagnostics
                    if nmatchcell > 50 or icell % 25 == 0:
                        print(f'    Track {itracknum}: Vertical levels completed in {levels_duration:.2f}s ({len(active_levels)} levels)')

                    # for z in range(0, nz):
                    # if ncores_save_up > 0:
                    #     import pdb; pdb.set_trace()

                # Track processing completed successfully
                track_end_time = time.time()
                track_duration = track_end_time - track_start_time
                
                # Print timing info for every track or every 25th track  
                if icell % 25 == 0 or nmatchcell <= 50:
                    print(f'  Track {icell+1}/{nmatchcell} (#{itracknum}) completed in {track_duration:.2f}s')
                    
            else:
                track_end_time = time.time()
                track_duration = track_end_time - track_start_time
                print(f'No cell matching track # {itracknum} (processed in {track_duration:.2f}s)')

        
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
    draftfile_path = config['draft_path']
    output_path = config['output_path']
    methamsl_filebase = config['methamsl_filebase']
    cldhamsl_filebase = config['cldhamsl_filebase']
    drafthamsl_filebase = config['drafthamsl_filebase']
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
    # cldfilelist = sorted(glob.glob(f'{metfile_path}{cldhamsl_filebase}*.nc'))
    draftfilelist = sorted(glob.glob(f'{draftfile_path}{drafthamsl_filebase}*.nc'))
    nmetfiles = len(metfilelist)
    print(f'Number of MET files: {nmetfiles}')
    
    # Get basetime from pixel files
    pixel_basetime, pixelfile_dict = calc_basetime(pixelfilelist, pixel_filebase)
    # Get basetime from met & cld files
    met_basetime, regfile_dict = calc_basetime(metfilelist, methamsl_filebase)
    # cld_basetime, cldfile_dict = calc_basetime(cldfilelist, cldhamsl_filebase)
    draft_basetime, draftfile_dict = calc_basetime(draftfilelist, drafthamsl_filebase)

    # Find matching MET files for each pixel file
    match_metfilelist = [''] * nfiles
    match_draftfilelist = [''] * nfiles
    # match_cldfilelist = [''] * nfiles
    for ifile in range(nfiles):
        # Find MET time closest to the pixel file time and get the index
        # Save the filename if time difference is < time_window
        idx = np.argmin(np.abs(met_basetime - pixel_basetime[ifile]))
        if np.abs(met_basetime[idx] - pixel_basetime[ifile]) < time_window:
            match_metfilelist[ifile] = regfile_dict[met_basetime[idx]]
        else:
            print(f'No match met file found for: {pixelfilelist[ifile]}')

        idx = np.argmin(np.abs(draft_basetime - pixel_basetime[ifile]))
        if np.abs(draft_basetime[idx] - pixel_basetime[ifile]) < time_window:
            match_draftfilelist[ifile] = draftfile_dict[draft_basetime[idx]]
        else:
            print(f'No match draft file found for: {pixelfilelist[ifile]}')

        # idx = np.argmin(np.abs(cld_basetime - pixel_basetime[ifile]))
        # if np.abs(cld_basetime[idx] - pixel_basetime[ifile]) < time_window:
        #     match_cldfilelist[ifile] = cldfile_dict[cld_basetime[idx]]
        # else:
        #     print(f'No match cld file found for: {pixelfilelist[ifile]}')

    # Read track statistics file
    print(trackstats_file)
    dsstats = xr.open_dataset(trackstats_file, decode_times=False)
    ntracks = dsstats.sizes[tracks_dimname]
    ntimes = dsstats.sizes[times_dimname]
    stats_basetime = dsstats['base_time'].data
    stats_basetime_attrs = dsstats['base_time'].attrs
    # cell_area = dsstats['cell_area']
    pixel_radius = dsstats.attrs['pixel_radius_km']
    dsstats.close()

    print(f'Total Number of Tracks: {ntracks}')

    # Read a MET file to get vertical coordinates (chunked to prevent memory issues)
    dsm = xr.open_dataset(match_metfilelist[0], chunks={'HAMSL': 10, 'south_north': 500, 'west_east': 500})
    nz = dsm.sizes['HAMSL']
    height = dsm['HAMSL']
    dsm.close()


    ##############################################################
    # Call function to calculate statistics
    trackindices_all = []
    timeindices_all = []
    final_results = []

    if run_parallel == 1:
        # Dask configuration optimized for small files (max 88 tracks each)
        dask_tmp_dir = config.get("dask_tmp_dir", "/tmp")
        dask.config.set({
            'temporary-directory': dask_tmp_dir,
            'distributed.worker.memory.target': 0.70,  # Less conservative since files are small
            'distributed.worker.memory.spill': 0.80,   # Spill to disk at 80%
            'distributed.worker.memory.pause': 0.85,   # Pause at 85%
            'distributed.worker.memory.terminate': 0.90,  # Terminate at 90%
            'distributed.comm.timeouts.connect': '60s',   # Shorter timeouts for small files
            'distributed.comm.timeouts.tcp': '60s',
            'distributed.worker.daemon': False,
            'distributed.scheduler.idle-timeout': '30m',  # Shorter idle timeout
            'distributed.worker.lifetime.duration': '2h', # Shorter lifetime for small workloads
            'distributed.worker.lifetime.stagger': '20m',
            'array.slicing.split_large_chunks': False,    # Not needed for small files
        })
        
        # Configure xarray to use dask for large arrays (prevents I/O blocking)
        xr.set_options(
            keep_attrs=True,
            file_cache_maxsize=128,  # Limit file cache to prevent memory issues
        )
        
        try:
            print(f"Initializing Dask cluster with {n_workers} workers...")
            
            # Get actual system memory and optimize for small files (max 88 tracks)
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
            usable_memory_gb = total_memory_gb * 0.80  # Use 80% of total memory
            # For small files, we can use more workers with less memory each
            memory_per_worker = max(12, min(32, int(usable_memory_gb / n_workers)))
            print(f"Detected {total_memory_gb:.1f}GB total memory, allocating {memory_per_worker}GB per worker")
            
            cluster = LocalCluster(
                n_workers=n_workers, 
                threads_per_worker=threads_per_worker,
                memory_limit=f'{memory_per_worker}GB',
                processes=True,
                silence_logs=False,  # Keep logs for debugging
                dashboard_address=':8787',
                worker_class='distributed.nanny.Nanny',  # Use nannies for better process management
                timeout='60s',  # Shorter timeout for small files
            )
            
            client = Client(cluster, timeout='120s')
            print(f"Dask client created, waiting for cluster to be ready...")
            
            # Wait for cluster to be ready and test basic connectivity
            try:
                client.wait_for_workers(n_workers, timeout=60)
                print(f"Dask cluster initialized successfully with {n_workers} workers")
                print(f"Dashboard link: {client.dashboard_link}")
                print(f"Memory per worker: {memory_per_worker}GB")
            except Exception as e:
                print(f"Warning: Cluster readiness check failed: {e}")
                print("Proceeding anyway, cluster might still be initializing...")
            
            # Print cluster resource summary (with error handling)
            try:
                sched_info = client.scheduler_info()
                print(f"Cluster resource summary:")
                print(f"  Workers: {len(sched_info['workers'])}")
                print(f"  Total cores: {sum(w['nthreads'] for w in sched_info['workers'].values())}")
                print(f"  Total memory: {sum(w['memory_limit'] for w in sched_info['workers'].values()) / 1e9:.1f} GB")
            except Exception as e:
                print(f"Warning: Could not get cluster resource summary: {e}")
            
            print("Dask cluster initialization completed, proceeding to file processing...")
            
        except Exception as e:
            print(f"Failed to initialize Dask cluster: {e}")
            print("Falling back to serial processing")
            run_parallel = 0

    # Enhanced processing loop with better monitoring
    print(f"Starting processing of {nfiles} files...")
    start_time = time.time()
    last_progress_time = start_time
    
    # Diagnostic mode: set to True to analyze track distribution without processing
    diagnostic_mode = config.get('diagnostic_mode', False)
    
    # Configuration optimized for small files (5-88 tracks each)
    memory_cleanup_interval = config.get('memory_cleanup_interval', 25)  # Less frequent cleanup
    
    if diagnostic_mode:
        print("="*60)
        print("DIAGNOSTIC MODE: Analyzing track distribution across files")
        print("="*60)
        track_distribution = []
    
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
        
        # Monitor memory every 30 files (less frequent for small files)
        if ifile % 30 == 0:
            log_memory_usage(f"Processing file {ifile}/{nfiles}")
            if run_parallel == 1:
                try:
                    # Check cluster health with timeout
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(client.scheduler_info)
                        cluster_info = future.result(timeout=10)  # 10 second timeout
                    print(f"Active workers: {len(cluster_info['workers'])}")
                except concurrent.futures.TimeoutError:
                    print("Warning: Cluster health check timed out")
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

            if diagnostic_mode:
                # In diagnostic mode, just collect track counts and skip processing
                track_distribution.append({
                    'file_index': ifile,
                    'filename': os.path.basename(pixelfilelist[ifile]),
                    'num_tracks': len(idx_track),
                    'track_indices': idx_track.tolist() if len(idx_track) > 0 else []
                })
                print(f"  File {ifile}: {len(idx_track)} tracks")
                continue

            if len(idx_track) > 0:
                trackindices_all.append(idx_track)
                timeindices_all.append(idx_time)
                
                # All files are small (<100 tracks), so use parallel processing for all
                if run_parallel == 1:
                    print(f"  Creating delayed task (track count: {len(idx_track)})...")
                    iresult = dask.delayed(calc_cellstats_singlefile)(
                        pixelfilelist[ifile], 
                        match_metfilelist[ifile],
                        match_draftfilelist[ifile],
                        idx_track, 
                        config,
                    )
                    final_results.append(iresult)
                else:
                    # Serial processing
                    print(f"  Processing serially (track count: {len(idx_track)})...")
                    iresult = calc_cellstats_singlefile(
                        pixelfilelist[ifile], 
                        match_metfilelist[ifile],
                        match_draftfilelist[ifile],
                        idx_track, 
                        config,
                    )
                    final_results.append(iresult)
                    
            else:
                print(f"  No matching tracks found, skipping")
            
            last_progress_time = time.time()
            file_duration = last_progress_time - file_start_time
            print(f"  File {ifile} completed in {file_duration:.1f}s")
            
            # Regular cleanup based on memory_cleanup_interval
            if (ifile + 1) % memory_cleanup_interval == 0:
                cleanup_memory()
                
        except Exception as e:
            print(f"ERROR processing file {ifile}: {e}")
            import traceback
            traceback.print_exc()
            # Try to recover
            cleanup_memory()
            continue

    print(f"File processing loop completed. Total files processed: {len(final_results)}")
    
    # If in diagnostic mode, print summary and exit
    if diagnostic_mode:
        print("\n" + "="*60)
        print("TRACK DISTRIBUTION ANALYSIS COMPLETE")
        print("="*60)
        
        # Calculate statistics
        track_counts = [item['num_tracks'] for item in track_distribution]
        total_tracks = sum(track_counts)
        max_tracks = max(track_counts) if track_counts else 0
        min_tracks = min(track_counts) if track_counts else 0
        avg_tracks = total_tracks / len(track_counts) if track_counts else 0
        
        # Distribution analysis
        small_files = len([c for c in track_counts if c < 100])
        medium_files = len([c for c in track_counts if 100 <= c <= 300])
        large_files = len([c for c in track_counts if c > 300])
        
        print(f"Total files analyzed: {len(track_distribution)}")
        print(f"Total tracks across all files: {total_tracks}")
        print(f"Track count statistics:")
        print(f"  Min tracks per file: {min_tracks}")
        print(f"  Max tracks per file: {max_tracks}")
        print(f"  Average tracks per file: {avg_tracks:.1f}")
        print(f"\nFile distribution:")
        print(f"  Files with <100 tracks: {small_files} ({small_files/len(track_counts)*100:.1f}%)")
        print(f"  Files with 100-300 tracks: {medium_files} ({medium_files/len(track_counts)*100:.1f}%)")
        print(f"  Files with >300 tracks: {large_files} ({large_files/len(track_counts)*100:.1f}%)")
        
        # Show files with most tracks
        sorted_files = sorted(track_distribution, key=lambda x: x['num_tracks'], reverse=True)
        print(f"\nTop 10 files with most tracks:")
        for i, file_info in enumerate(sorted_files[:10]):
            print(f"  {i+1:2d}. File {file_info['file_index']:3d}: {file_info['num_tracks']:4d} tracks - {file_info['filename']}")
        
        # Show files with no tracks
        empty_files = [item for item in track_distribution if item['num_tracks'] == 0]
        if empty_files:
            print(f"\nFiles with no tracks: {len(empty_files)}")
            for file_info in empty_files[:5]:  # Show first 5
                print(f"  File {file_info['file_index']:3d}: {file_info['filename']}")
                
        print(f"\nDiagnostic mode complete. Set 'diagnostic_mode: false' in config to run actual processing.")
        print("="*60)
        sys.exit(0)  # Exit early in diagnostic mode

    # Final computation with improved error handling for 145 files
    if run_parallel == 1 and len(final_results) > 0:
        print("Computing remaining statistics with Dask...")
        computation_start = time.time()
        
        try:
            # Filter out already computed results
            delayed_results = [r for r in final_results if hasattr(r, 'compute')]
            computed_results = [r for r in final_results if not hasattr(r, 'compute')]
            
            if delayed_results:
                print(f"Computing {len(delayed_results)} delayed results...")
                
                # For the fixed 145 files, compute delayed results with timeout protection
                batch_results = dask.compute(*delayed_results)
                computed_results.extend(batch_results)
                print(f"All delayed computations completed successfully")
            
            final_results = computed_results
            print(f"All computations completed in {time.time() - computation_start:.1f}s")
            
        except Exception as e:
            print(f"ERROR during final Dask computation: {e}")
            import traceback
            traceback.print_exc()
            print("Falling back to serial processing for remaining tasks...")
            
            # Emergency serial fallback
            computed_results = []
            for i, result in enumerate(final_results):
                if hasattr(result, 'compute'):
                    try:
                        print(f"Computing task {i+1}/{len(final_results)} serially...")
                        computed_results.append(result.compute())
                    except Exception as e2:
                        print(f"Serial fallback failed for task {i+1}: {e2}")
                        computed_results.append((None, None, None))
                else:
                    computed_results.append(result)
                    
                # Clean memory periodically during serial fallback
                if (i + 1) % 10 == 0:
                    cleanup_memory()
                    
            final_results = computed_results
        
        # Close Dask client and cluster
        try:
            client.close()
            cluster.close()
        except:
            pass
            
    elif run_parallel == 0:
        print("Serial processing completed")

    # Rest of the code for saving results...