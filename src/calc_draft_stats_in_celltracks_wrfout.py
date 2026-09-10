"""
Calculates 3D W core statistics from gridded WRF data for tracked convective cells.
The 3D core statistics are written to netCDF file matching the cell track statistics file format.
"""
import numpy as np
import os, sys, glob, fnmatch
import time
from datetime import datetime, timedelta
from pytz import utc
import yaml
import xarray as xr
from scipy import ndimage
import warnings
import dask
from dask.distributed import Client, LocalCluster
from wrf import interplevel

#-----------------------------------------------------------------------
def calc_basetime(filelist, filebase, fmt='yyyymmdd.hhmm'):
    """
    Calculates basetime (Epoch Time) and a filename dictionary from a list of filenames.
    The basetime uses year, month, day, hour, minute but set all seconds to 0.

    Args:
        filelist: list
            A list of input file names
        filebase: string
            Basename of the files
        fmt: string
            Format of time in filename

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
    if fmt == 'yyyymmdd.hhmm':
        for ifile in range(nfiles):
            fname = os.path.basename(filelist[ifile])
            # File name format: basename_yyyymmdd.hhmm
            TEMP_filetime = datetime(
                int(fname[prelength:(prelength+4)]), 
                int(fname[prelength+4:(prelength+6)]), 
                int(fname[prelength+6:(prelength+8)]),
                int(fname[prelength+9:(prelength+11)]), 
                int(fname[prelength+11:(prelength+13)]), 0, tzinfo=utc
            )
            # file_basetime[ifile] = calendar.timegm(TEMP_filetime.timetuple())
            file_basetime[ifile] = TEMP_filetime.timestamp()
            file_dict[file_basetime[ifile]] = filelist[ifile]
    if fmt == 'yyyy.mm.dd.hh.mm':
        for ifile in range(nfiles):
            fname = os.path.basename(filelist[ifile])
            # File name format: basename_yyyy.mm.dd.hh.mm
            TEMP_filetime = datetime(
                int(fname[prelength:(prelength+4)]), 
                int(fname[prelength+5:(prelength+7)]), 
                int(fname[prelength+8:(prelength+10)]),
                int(fname[prelength+11:(prelength+13)]), 
                int(fname[prelength+14:(prelength+16)]), 0, tzinfo=utc
            )
            # file_basetime[ifile] = calendar.timegm(TEMP_filetime.timetuple())
            file_basetime[ifile] = TEMP_filetime.timestamp()
            file_dict[file_basetime[ifile]] = filelist[ifile]
    return file_basetime, file_dict


#-----------------------------------------------------------------------
def label_cores(W, W_thresh, ncores_min, min_core_npix, method='>'):
    """
    Label up/down draft cores using threshold and connectivity.

    Args:
        W: np.array
            Vertical velocity array
        W_thresh: float
            Vertical velocity threshold
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
    if method == '>':
        core_label, ncores = ndimage.label((W > W_thresh))
    elif method == '<':
        core_label, ncores = ndimage.label((W < W_thresh))
    else:
        print(f'Error: Undefined method to label cores: {method}!')
    
    # Get core sizes
    core_numbers, npix_core = np.unique(core_label, return_counts=True)
    
    # Remove 0 label result since that is the background
    npix_core = npix_core[core_numbers > 0]
    core_numbers = core_numbers[core_numbers > 0]
    
    # Remove small cores
    mask = npix_core > min_core_npix
    npix_core = npix_core[mask]
    core_numbers = core_numbers[mask]

    # Sort the core size by descending order
    sort_idx = npix_core.argsort()[::-1]
    npix_core_sorted = npix_core[sort_idx]
    core_numbers_sorted = core_numbers[sort_idx]
    
    # Save the largest X cores
    ncores_all = len(npix_core)
    ncores_save = np.nanmin([ncores_all, ncores_min])
    
    # Put output variables in a dictionary
    out_dict = {
        'ncores_all': ncores_all,
        'ncores_save': ncores_save,
        'core_npix': npix_core_sorted[:ncores_save+1],
        'core_numbers': core_numbers_sorted[:ncores_save+1],
        'core_label': core_label,
    }
    return out_dict

#-----------------------------------------------------------------------
def calc_cellstats_singlefile(
    pixel_filename, 
    wrf_filename, 
    idx_track,
    config,
):
    """
    Calculate statistics for cells in a single pixel file

    Args:
        pixel_filename: string
            Cell tracking pixel filename
        wrf_filename: string
            WRF filename
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
    print(wrf_filename)

    # Get thresholds from config
    W_up_thresh = config['W_up_thresh']
    W_down_thresh = config['W_down_thresh']
    min_core_npix = config['min_core_npix']
    ncores_min = config['ncores_min']
    geolimits = config.get('geolimits', None)

    # Read WRF file
    dsm = xr.open_dataset(wrf_filename)
    # Rename dimenensions
    dsm = dsm.rename_dims({'bottom_top':'level', 'south_north':'lat', 'west_east':'lon'})
    # nz = dsm.dims['level']
    zlevs_lo = (np.arange(40)+1)*125 #125 m steps between 0 and 5000 m
    zlevs_hi = (np.arange(40)+1)*250 + 5000 #200 m steps between 5000 and 15000 m
    zlevs = np.concatenate((zlevs_lo, zlevs_hi))
    height = xr.DataArray(data=zlevs, dims=["level"], coords=dict(), attrs=dict(description="Geopotential height", units="m")) #convert to xarray DataArray
    nz = len(height)
    ny = dsm.dims['lat']
    nx = dsm.dims['lon']
    DX = dsm.attrs['DX'] / 1e3 #km
    DY = dsm.attrs['DY'] / 1e3 #km
    grid_area = DX * DY #km^2
    
    # Read pixel-level track file
    ds = xr.open_dataset(pixel_filename, decode_times=False)
    time_pixel = ds['time']
    ny_p = ds.dims['lat']
    nx_p = ds.dims['lon']

    # Check dimensions between WRF and pixel files
    # could add checks for variable units
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
        #Don't yet understand why I need to shift these but if I don't, then there are too many points and the mask definition is violated
        xmin, xmax = np.min(x_idx)+2, np.max(x_idx)
        ymin, ymax = np.min(y_idx)+1, np.max(y_idx)-1
        
        # Subset 
        XLONG = dsm['CAC_LONG'].squeeze()[ymin:ymax+1, xmin:xmax+1]
        XLAT = dsm['CAC_LAT'].squeeze()[ymin:ymax+1, xmin:xmax+1]
        GPH = dsm['CAC_GPH'].squeeze()[:, ymin:ymax+1, xmin:xmax+1]/9.8 #m
        PRESSURE = dsm['CAC_P'].squeeze()[:, ymin:ymax+1, xmin:xmax+1] #Pa
        TEMPERATURE = dsm['CAC_T'].squeeze()[:, ymin:ymax+1, xmin:xmax+1] #K
        QVAPOR = 1e-3*dsm['CAC_QV'].squeeze()[:, ymin:ymax+1, xmin:xmax+1] #kg/kg
        QCLOUD = 1e-3*dsm['CAC_QC'].squeeze()[:, ymin:ymax+1, xmin:xmax+1] #kg/kg
        QRAIN = 1e-3*dsm['CAC_QR'].squeeze()[:, ymin:ymax+1, xmin:xmax+1] #kg/kg
        QICE = 1e-3*dsm['CAC_QI'].squeeze()[:, ymin:ymax+1, xmin:xmax+1] #kg/kg
        QSNOW = 1e-3*dsm['CAC_QS'].squeeze()[:, ymin:ymax+1, xmin:xmax+1] #kg/kg
        QGRAUPEL = 1e-3*dsm['CAC_QG'].squeeze()[:, ymin:ymax+1, xmin:xmax+1] #kg/kg
        QTOTAL = QCLOUD + QRAIN + QICE + QSNOW + QGRAUPEL #kg/kg
        NCLOUD = dsm['CAC_NC'].squeeze()[:, ymin:ymax+1, xmin:xmax+1] #1/kg
        NRAIN = dsm['CAC_NR'].squeeze()[:, ymin:ymax+1, xmin:xmax+1] #1/kg
        NICE = dsm['CAC_NI'].squeeze()[:, ymin:ymax+1, xmin:xmax+1] #1/kg
        NWA = dsm['CAC_WA'].squeeze()[:, ymin:ymax+1, xmin:xmax+1] #1/kg
        NIA = dsm['CAC_IA'].squeeze()[:, ymin:ymax+1, xmin:xmax+1] #1/kg
        HD = dsm['CAC_HD'].squeeze()[:, ymin:ymax+1, xmin:xmax+1] #K/s
        W = dsm['CAC_W'].squeeze()[:, ymin:ymax+1, xmin:xmax+1] #m/s
        
        # Update ny, nx with the subset
        ny = XLAT.sizes['lat']
        nx = XLONG.sizes['lon']
    else:
        XLONG = dsm['CAC_LONG'].squeeze()
        XLAT = dsm['CAC_LAT'].squeeze()
        GPH = dsm['CAC_GPH'].squeeze()/9.8
        PRESSURE = dsm['CAC_P'].squeeze()
        TEMPERATURE = dsm['CAC_T'].squeeze()
        QVAPOR = 1e-3*dsm['CAC_QV'].squeeze()
        QCLOUD = 1e-3*dsm['CAC_QC'].squeeze()
        QRAIN = 1e-3*dsm['CAC_QR'].squeeze()
        QICE = 1e-3*dsm['CAC_QI'].squeeze()
        QSNOW = 1e-3*dsm['CAC_QS'].squeeze()
        QGRAUPEL = 1e-3*dsm['CAC_QG'].squeeze()
        QTOTAL = QCLOUD + QRAIN + QICE + QSNOW + QGRAUPEL
        NCLOUD = dsm['CAC_NC'].squeeze()
        NRAIN = dsm['CAC_NR'].squeeze()
        NICE = dsm['CAC_NI'].squeeze()
        NWA = dsm['CAC_WA'].squeeze()
        NIA = dsm['CAC_IA'].squeeze()
        HD = dsm['CAC_HD'].squeeze()
        W = dsm['CAC_W'].squeeze()
        
    # Check dimensions again after subset
    if (ny_p != ny) | (nx_p != nx):
        print(f'ERROR: Inconsistent number of grids between pixel-level and WRF files.')
        print(f'ny: {ny}, ny_pixel: {ny_p}, nx: {nx}, nx_pixel: {nx_p}')
        sys.exit()

    # Drop 1D lat/lon coordinates, and reasign 2D XLONG/XLAT coordinates from WRF file
    # It does not seem like this is necessary in Xarray 0.21.1
    ds = ds.drop_vars(['lon', 'lat']).assign_coords({'XLONG':XLONG, 'XLAT':XLAT})
    tracknumbermap = ds['tracknumber'].squeeze()
    
    # Calculate additional variables
    R_dry = 287.04   # J kg−1 K−1
    Cp = 1004.
    
    # virtual temperature
    TV = TEMPERATURE*(1 + QVAPOR/0.622)/(1 + QVAPOR) #K
    
    # theta-e following Bolton (1980); error of < 0.3 K between -35 and 35C; from Thompson scheme
    # more accurate formula from Emanuel could be implemented; ice effects also excluded
    es = PRESSURE*QVAPOR/(0.622*QVAPOR)
    TDEW = (35.86*np.log(es) - 4947.2325)/(np.log(es) - 23.6837)
    TLCL = 1/(1/(TDEW - 56) + np.log(TEMPERATURE/TDEW)/800) + 56
    p1 = 3.376/TLCL - 0.00254
    p2 = 1e3*QVAPOR*(1 + 0.81*QVAPOR)
    THETAE = (TEMPERATURE*(100000./PRESSURE)**(0.2854*(1 - 0.28*QVAPOR)))*np.exp(p1*p2) #K
    
    # #moist static energy (another option apart from theta-e)
    # LV = 2.5e6 - 2112.*(TEMPERATURE - 273.15)
    # MSE = Cp*TEMPERATURE + 9.8*GHT + LV*QVAPOR #J/kg
    
    #density temp
    TRHO = TEMPERATURE*((1 + QVAPOR/0.622)/(1 + QTOTAL+QVAPOR))
    
    #RH (formula is used in Thompson scheme for supersaturation)
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
    X = X.where(X > -80)
    X = X.fillna(-80) #setting values less than -80C to -80C 
    ESL = C0 + X*(C1 + X*(C2 + X*(C3 + X*(C4 + X*(C5 + X*(C6 + X*(C7 + X*C8))))))) #saturation vapor pressure
    QVS = 0.622*ESL/(PRESSURE - ESL) #saturation vapor mixing ratio
    RH = 1e2*QVAPOR/QVS # %
    #SS = RH - 100 #supersaturation in %
    
    # Calculate moist air density using virtual temperature
    RHO_DRY = PRESSURE / (R_dry * TEMPERATURE)  # kg m-3
    RHO_MOIST = PRESSURE / (R_dry * TV)  # kg m-3

    # Calculate mass flux (kg m-2 s-1)
    MASSFLUX = (RHO_MOIST*W).squeeze()
    
    # vertical condensate flux? requires mass weighted fall speed calculations
    
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
        cell_nCore_down = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_nonCoreArea = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_nonCoreMassFlux = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_nonCoreMeanW = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_nonCoreMeanT = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_nonCoreMeanQv = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_nonCoreMeanThetae = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_nonCoreMeanTv = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_nonCoreMeanTrho = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_nonCoreMeanRH = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_nonCoreMeanQc = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_nonCoreMeanQr = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_nonCoreMeanQi = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_nonCoreMeanQs = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_nonCoreMeanQg = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_nonCoreMeanQt = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_nonCoreMeanNc = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_nonCoreMeanNr = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_nonCoreMeanNi = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_nonCoreMeanNwa = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_nonCoreMeanNia = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_nonCoreMeanHd = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_nonCoreMeanP = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_nonCoreMeanRho = np.full(dims2d, np.NaN, dtype=np.float32)
        
        cell_MassFlux_up = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_Area_up = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_MeanW_up = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_MeanT_up = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_MeanQv_up = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_MeanThetae_up = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_MeanTv_up = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_MeanTrho_up = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_MeanRH_up = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_MeanQc_up = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_MeanQr_up = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_MeanQi_up = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_MeanQs_up = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_MeanQg_up = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_MeanQt_up = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_MeanNc_up = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_MeanNr_up = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_MeanNi_up = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_MeanNwa_up = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_MeanNia_up = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_MeanHd_up = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_MeanP_up = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_MeanRho_up = np.full(dims2d, np.NaN, dtype=np.float32)
        
        cell_CoreMassFlux_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreArea_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanW_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanT_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanQv_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanThetae_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanTv_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanTrho_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanRH_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanQc_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanQr_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanQi_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanQs_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanQg_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanQt_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanNc_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanNr_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanNi_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanNwa_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanNia_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanHd_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanP_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanRho_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMaxW_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMaxT_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMaxQv_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMaxThetae_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMaxTv_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMaxTrho_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMaxRH_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMaxQc_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMaxQr_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMaxQi_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMaxQs_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMaxQg_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMaxQt_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMaxNc_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMaxNr_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMaxNi_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMaxNwa_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMaxNia_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMaxHd_up = np.full(dims3d, np.NaN, dtype=np.float32)
        
        cell_MassFlux_down = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_Area_down = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_MeanW_down = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_MeanT_down = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_MeanQv_down = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_MeanThetae_down = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_MeanTv_down = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_MeanTrho_down = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_MeanRH_down = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_MeanQc_down = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_MeanQr_down = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_MeanQi_down = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_MeanQs_down = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_MeanQg_down = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_MeanQt_down = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_MeanNc_down = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_MeanNr_down = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_MeanNi_down = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_MeanNwa_down = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_MeanNia_down = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_MeanHd_down = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_MeanP_down = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_MeanRho_down = np.full(dims2d, np.NaN, dtype=np.float32)
        
        cell_CoreMassFlux_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreArea_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanW_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanT_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanQv_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanThetae_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanTv_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanTrho_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanRH_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanQc_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanQr_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanQi_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanQs_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanQg_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanQt_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanNc_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanNr_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanNi_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanNwa_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanNia_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanHd_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanP_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanRho_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMinW_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMinT_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMinQv_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMinThetae_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMinTv_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMinTrho_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMinRH_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMaxQc_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMaxQr_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMaxQi_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMaxQs_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMaxQg_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMaxQt_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMaxNc_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMaxNr_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMaxNi_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMaxNwa_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMaxNia_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMinHd_down = np.full(dims3d, np.NaN, dtype=np.float32)

        # Loop over each match tracked cell
        for icell in range(nmatchcell):
            # Track number needs to add 1
            itracknum = idx_track[icell] + 1

            # Count the number of pixels for the original cell mask
            inpix_cloud = np.count_nonzero(tracknumbermap == itracknum)

            # Proceed if the number matching cloud pixel > 0
            if inpix_cloud > 0:

                # Subset 3D variables to the current cell mask
                iGPH = GPH.where(tracknumbermap == itracknum, drop=True).squeeze().data
                iW = W.where(tracknumbermap == itracknum, drop=True).squeeze().data
                iMassFlux = MASSFLUX.where(tracknumbermap == itracknum, drop=True).squeeze().data
                iRho = RHO_DRY.where(tracknumbermap == itracknum, drop=True).squeeze().data
                iP = PRESSURE.where(tracknumbermap == itracknum, drop=True).squeeze().data
                iT = TEMPERATURE.where(tracknumbermap == itracknum, drop=True).squeeze().data
                iQv = QVAPOR.where(tracknumbermap == itracknum, drop=True).squeeze().data
                iThetae = THETAE.where(tracknumbermap == itracknum, drop=True).squeeze().data
                iTv = TV.where(tracknumbermap == itracknum, drop=True).squeeze().data
                iTrho = TRHO.where(tracknumbermap == itracknum, drop=True).squeeze().data
                iRH = RH.where(tracknumbermap == itracknum, drop=True).squeeze().data
                iQc = QCLOUD.where(tracknumbermap == itracknum, drop=True).squeeze().data
                iQr = QRAIN.where(tracknumbermap == itracknum, drop=True).squeeze().data
                iQi = QICE.where(tracknumbermap == itracknum, drop=True).squeeze().data
                iQs = QSNOW.where(tracknumbermap == itracknum, drop=True).squeeze().data
                iQg = QGRAUPEL.where(tracknumbermap == itracknum, drop=True).squeeze().data
                iQt = QTOTAL.where(tracknumbermap == itracknum, drop=True).squeeze().data
                iNc = NCLOUD.where(tracknumbermap == itracknum, drop=True).squeeze().data
                iNr = NRAIN.where(tracknumbermap == itracknum, drop=True).squeeze().data
                iNi = NICE.where(tracknumbermap == itracknum, drop=True).squeeze().data
                iNwa = NWA.where(tracknumbermap == itracknum, drop=True).squeeze().data
                iNia = NIA.where(tracknumbermap == itracknum, drop=True).squeeze().data
                iHd = HD.where(tracknumbermap == itracknum, drop=True).squeeze().data
                
                # Interpolate variables to constant height grid (may not work well with dask)
                MassFlux_Interp = interplevel(iMassFlux, iGPH, height)
                Rho_Interp = interplevel(iRho, iGPH, height)
                P_Interp = interplevel(iP, iGPH, height)
                W_Interp = interplevel(iW, iGPH, height)
                T_Interp = interplevel(iT, iGPH, height)
                Qv_Interp = interplevel(iQv, iGPH, height)
                Thetae_Interp = interplevel(iThetae, iGPH, height)
                Tv_Interp = interplevel(iTv, iGPH, height)
                Trho_Interp = interplevel(iTrho, iGPH, height)
                RH_Interp = interplevel(iRH, iGPH, height)
                Qc_Interp = interplevel(iQc, iGPH, height)
                Qr_Interp = interplevel(iQr, iGPH, height)
                Qi_Interp = interplevel(iQi, iGPH, height)
                Qs_Interp = interplevel(iQs, iGPH, height)
                Qg_Interp = interplevel(iQg, iGPH, height)
                Qt_Interp = interplevel(iQt, iGPH, height)
                Nc_Interp = interplevel(iNc, iGPH, height)
                Nr_Interp = interplevel(iNr, iGPH, height)
                Ni_Interp = interplevel(iNi, iGPH, height)
                Nwa_Interp = interplevel(iNwa, iGPH, height)
                Nia_Interp = interplevel(iNia, iGPH, height)
                Hd_Interp = interplevel(iHd, iGPH, height)    
                                
                # Calculate new statistics of the cell
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)

                    # Loop over vertical level
                    for z in range(0, nz):
                        # print(height[z])
                        zW = W_Interp[z,:,:]
                        zMassFlux = MassFlux_Interp[z,:,:]
                        zRho = Rho_Interp[z,:,:]
                        zP = P_Interp[z,:,:]
                        zT = T_Interp[z,:,:]
                        zQv = Qv_Interp[z,:,:]
                        zThetae = Thetae_Interp[z,:,:]
                        zTv = Tv_Interp[z,:,:]
                        zTrho = Trho_Interp[z,:,:]
                        zRH = RH_Interp[z,:,:]
                        zQc = Qc_Interp[z,:,:]
                        zQr = Qr_Interp[z,:,:]
                        zQi = Qi_Interp[z,:,:]
                        zQs = Qs_Interp[z,:,:]
                        zQg = Qg_Interp[z,:,:]
                        zQt = Qt_Interp[z,:,:]
                        zNc = Nc_Interp[z,:,:]
                        zNr = Nr_Interp[z,:,:]
                        zNi = Ni_Interp[z,:,:]
                        zNwa = Nwa_Interp[z,:,:]
                        zNia = Nia_Interp[z,:,:]
                        zHd = Hd_Interp[z,:,:]
                        
                        # Label updraft cores
                        dict_up = label_cores(zW, W_up_thresh, ncores_min, min_core_npix, method='>')
                        ncores_all_up = dict_up['ncores_all']
                        ncores_up = dict_up['ncores_save']
                        core_npix_up = dict_up['core_npix']
                        core_numbers_up = dict_up['core_numbers']
                        core_label_up = dict_up['core_label']
                        
                        # Label downdraft cores
                        dict_down = label_cores(zW, W_down_thresh, ncores_min, min_core_npix, method='<')
                        ncores_all_down = dict_down['ncores_all']
                        ncores_down = dict_down['ncores_save']
                        core_npix_down = dict_down['core_npix']
                        core_numbers_down = dict_down['core_numbers']
                        core_label_down = dict_down['core_label']
                                                
                        # Calculate core statistics
                        # Updrafts
                        MaFlx_core_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        W_mean_core_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        T_mean_core_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        Qv_mean_core_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        Thetae_mean_core_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        Tv_mean_core_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        Trho_mean_core_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        RH_mean_core_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        Qc_mean_core_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        Qr_mean_core_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        Qi_mean_core_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        Qs_mean_core_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        Qg_mean_core_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        Qt_mean_core_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        Nc_mean_core_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        Nr_mean_core_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        Ni_mean_core_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        Nwa_mean_core_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        Nia_mean_core_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        Hd_mean_core_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        P_mean_core_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        Rho_mean_core_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        W_max_core_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        T_max_core_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        Qv_max_core_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        Thetae_max_core_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        Tv_max_core_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        Trho_max_core_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        RH_max_core_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        Qc_max_core_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        Qr_max_core_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        Qi_max_core_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        Qs_max_core_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        Qg_max_core_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        Qt_max_core_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        Nc_max_core_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        Nr_max_core_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        Ni_max_core_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        Nwa_max_core_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        Nia_max_core_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        Hd_max_core_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        for ii in range(ncores_up):
                            MaFlx_core_up[ii] = np.nansum(np.array(zMassFlux)[core_label_up == core_numbers_up[ii]])
                            W_mean_core_up[ii] = np.nanmean(np.array(zW)[core_label_up == core_numbers_up[ii]])
                            T_mean_core_up[ii] = np.nanmean(np.array(zT)[core_label_up == core_numbers_up[ii]])
                            Qv_mean_core_up[ii] = np.nanmean(np.array(zQv)[core_label_up == core_numbers_up[ii]])
                            Thetae_mean_core_up[ii] = np.nanmean(np.array(zThetae)[core_label_up == core_numbers_up[ii]])
                            Tv_mean_core_up[ii] = np.nanmean(np.array(zTv)[core_label_up == core_numbers_up[ii]])
                            Trho_mean_core_up[ii] = np.nanmean(np.array(zTrho)[core_label_up == core_numbers_up[ii]])
                            RH_mean_core_up[ii] = np.nanmean(np.array(zRH)[core_label_up == core_numbers_up[ii]])
                            Qc_mean_core_up[ii] = np.nanmean(np.array(zQc)[core_label_up == core_numbers_up[ii]])
                            Qr_mean_core_up[ii] = np.nanmean(np.array(zQr)[core_label_up == core_numbers_up[ii]])
                            Qi_mean_core_up[ii] = np.nanmean(np.array(zQi)[core_label_up == core_numbers_up[ii]])
                            Qs_mean_core_up[ii] = np.nanmean(np.array(zQs)[core_label_up == core_numbers_up[ii]])
                            Qg_mean_core_up[ii] = np.nanmean(np.array(zQg)[core_label_up == core_numbers_up[ii]])
                            Qt_mean_core_up[ii] = np.nanmean(np.array(zQt)[core_label_up == core_numbers_up[ii]])
                            Nc_mean_core_up[ii] = np.nanmean(np.array(zNc)[core_label_up == core_numbers_up[ii]])
                            Nr_mean_core_up[ii] = np.nanmean(np.array(zNr)[core_label_up == core_numbers_up[ii]])
                            Ni_mean_core_up[ii] = np.nanmean(np.array(zNi)[core_label_up == core_numbers_up[ii]])
                            Nwa_mean_core_up[ii] = np.nanmean(np.array(zNwa)[core_label_up == core_numbers_up[ii]])
                            Nia_mean_core_up[ii] = np.nanmean(np.array(zNia)[core_label_up == core_numbers_up[ii]])
                            Hd_mean_core_up[ii] = np.nanmean(np.array(zHd)[core_label_up == core_numbers_up[ii]])
                            P_mean_core_up[ii] = np.nanmean(np.array(zP)[core_label_up == core_numbers_up[ii]])
                            Rho_mean_core_up[ii] = np.nanmean(np.array(zRho)[core_label_up == core_numbers_up[ii]])
                            W_max_core_up[ii] = np.nanmax(np.array(zW)[core_label_up == core_numbers_up[ii]])
                            T_max_core_up[ii] = np.nanmax(np.array(zT)[core_label_up == core_numbers_up[ii]])
                            Qv_max_core_up[ii] = np.nanmax(np.array(zQv)[core_label_up == core_numbers_up[ii]])
                            Thetae_max_core_up[ii] = np.nanmax(np.array(zThetae)[core_label_up == core_numbers_up[ii]])
                            Tv_max_core_up[ii] = np.nanmax(np.array(zTv)[core_label_up == core_numbers_up[ii]])
                            Trho_max_core_up[ii] = np.nanmax(np.array(zTrho)[core_label_up == core_numbers_up[ii]])
                            RH_max_core_up[ii] = np.nanmax(np.array(zRH)[core_label_up == core_numbers_up[ii]])
                            Qc_max_core_up[ii] = np.nanmax(np.array(zQc)[core_label_up == core_numbers_up[ii]])
                            Qr_max_core_up[ii] = np.nanmax(np.array(zQr)[core_label_up == core_numbers_up[ii]])
                            Qi_max_core_up[ii] = np.nanmax(np.array(zQi)[core_label_up == core_numbers_up[ii]])
                            Qs_max_core_up[ii] = np.nanmax(np.array(zQs)[core_label_up == core_numbers_up[ii]])
                            Qg_max_core_up[ii] = np.nanmax(np.array(zQg)[core_label_up == core_numbers_up[ii]])
                            Qt_max_core_up[ii] = np.nanmax(np.array(zQt)[core_label_up == core_numbers_up[ii]])
                            Nc_max_core_up[ii] = np.nanmax(np.array(zNc)[core_label_up == core_numbers_up[ii]])
                            Nr_max_core_up[ii] = np.nanmax(np.array(zNr)[core_label_up == core_numbers_up[ii]])
                            Ni_max_core_up[ii] = np.nanmax(np.array(zNi)[core_label_up == core_numbers_up[ii]])
                            Nwa_max_core_up[ii] = np.nanmax(np.array(zNwa)[core_label_up == core_numbers_up[ii]])
                            Nia_max_core_up[ii] = np.nanmax(np.array(zNia)[core_label_up == core_numbers_up[ii]])
                            Hd_max_core_up[ii] = np.nanmax(np.array(zHd)[core_label_up == core_numbers_up[ii]])
                            
                        # Calculate total mass flux for all labeled cores
                        if ncores_all_up > 0:
                            MaFlx_sum_up = np.nansum(np.array(zMassFlux)[core_label_up > 0])
                            W_mean_up = np.nanmean(np.array(zW)[core_label_up > 0])
                            T_mean_up = np.nanmean(np.array(zT)[core_label_up > 0])
                            Qv_mean_up = np.nanmean(np.array(zQv)[core_label_up > 0])
                            Thetae_mean_up = np.nanmean(np.array(zThetae)[core_label_up > 0])
                            Tv_mean_up = np.nanmean(np.array(zTv)[core_label_up > 0])
                            Trho_mean_up = np.nanmean(np.array(zTrho)[core_label_up > 0])
                            RH_mean_up = np.nanmean(np.array(zRH)[core_label_up > 0])
                            Qc_mean_up = np.nanmean(np.array(zQc)[core_label_up > 0])
                            Qr_mean_up = np.nanmean(np.array(zQr)[core_label_up > 0])
                            Qi_mean_up = np.nanmean(np.array(zQi)[core_label_up > 0])
                            Qs_mean_up = np.nanmean(np.array(zQs)[core_label_up > 0])
                            Qg_mean_up = np.nanmean(np.array(zQg)[core_label_up > 0])
                            Qt_mean_up = np.nanmean(np.array(zQt)[core_label_up > 0])
                            Nc_mean_up = np.nanmean(np.array(zNc)[core_label_up > 0])
                            Nr_mean_up = np.nanmean(np.array(zNr)[core_label_up > 0])
                            Ni_mean_up = np.nanmean(np.array(zNi)[core_label_up > 0])
                            Nwa_mean_up = np.nanmean(np.array(zNwa)[core_label_up > 0])
                            Nia_mean_up = np.nanmean(np.array(zNia)[core_label_up > 0])
                            Hd_mean_up = np.nanmean(np.array(zHd)[core_label_up > 0])
                            P_mean_up = np.nanmean(np.array(zP)[core_label_up > 0])
                            Rho_mean_up = np.nanmean(np.array(zRho)[core_label_up > 0])
                        else:
                            MaFlx_sum_up = np.NaN
                            W_mean_up = np.NaN
                            T_mean_up = np.NaN
                            Qv_mean_up = np.NaN
                            Thetae_mean_up = np.NaN
                            Tv_mean_up = np.NaN
                            Trho_mean_up = np.NaN
                            RH_mean_up = np.NaN
                            Qc_mean_up = np.NaN
                            Qr_mean_up = np.NaN
                            Qi_mean_up = np.NaN
                            Qs_mean_up = np.NaN
                            Qg_mean_up = np.NaN
                            Qt_mean_up = np.NaN
                            Nc_mean_up = np.NaN
                            Nr_mean_up = np.NaN
                            Ni_mean_up = np.NaN
                            Nwa_mean_up = np.NaN
                            Nia_mean_up = np.NaN
                            Hd_mean_up = np.NaN
                            Hd_mean_up = np.NaN
                            P_mean_up = np.NaN
                            Rho_mean_up = np.NaN
                        # if (ncores_all_up > 2):                         
                        #     import pdb; pdb.set_trace()
                        
                        # Downdrafts
                        MaFlx_core_down = np.full(ncores_down, np.NaN, dtype=np.float32)
                        W_mean_core_down = np.full(ncores_down, np.NaN, dtype=np.float32)
                        T_mean_core_down = np.full(ncores_down, np.NaN, dtype=np.float32)
                        Qv_mean_core_down = np.full(ncores_down, np.NaN, dtype=np.float32)
                        Thetae_mean_core_down = np.full(ncores_down, np.NaN, dtype=np.float32)
                        Tv_mean_core_down = np.full(ncores_down, np.NaN, dtype=np.float32)
                        Trho_mean_core_down = np.full(ncores_down, np.NaN, dtype=np.float32)
                        RH_mean_core_down = np.full(ncores_down, np.NaN, dtype=np.float32)
                        Qc_mean_core_down = np.full(ncores_down, np.NaN, dtype=np.float32)
                        Qr_mean_core_down = np.full(ncores_down, np.NaN, dtype=np.float32)
                        Qi_mean_core_down = np.full(ncores_down, np.NaN, dtype=np.float32)
                        Qs_mean_core_down = np.full(ncores_down, np.NaN, dtype=np.float32)
                        Qg_mean_core_down = np.full(ncores_down, np.NaN, dtype=np.float32)
                        Qt_mean_core_down = np.full(ncores_down, np.NaN, dtype=np.float32)
                        Nc_mean_core_down = np.full(ncores_down, np.NaN, dtype=np.float32)
                        Nr_mean_core_down = np.full(ncores_down, np.NaN, dtype=np.float32)
                        Ni_mean_core_down = np.full(ncores_down, np.NaN, dtype=np.float32)
                        Nwa_mean_core_down = np.full(ncores_down, np.NaN, dtype=np.float32)
                        Nia_mean_core_down = np.full(ncores_down, np.NaN, dtype=np.float32)
                        Hd_mean_core_down = np.full(ncores_down, np.NaN, dtype=np.float32)
                        P_mean_core_down = np.full(ncores_down, np.NaN, dtype=np.float32)
                        Rho_mean_core_down = np.full(ncores_down, np.NaN, dtype=np.float32)
                        W_min_core_down = np.full(ncores_down, np.NaN, dtype=np.float32)
                        T_min_core_down = np.full(ncores_down, np.NaN, dtype=np.float32)
                        Qv_min_core_down = np.full(ncores_down, np.NaN, dtype=np.float32)
                        Thetae_min_core_down = np.full(ncores_down, np.NaN, dtype=np.float32)
                        Tv_min_core_down = np.full(ncores_down, np.NaN, dtype=np.float32)
                        Trho_min_core_down = np.full(ncores_down, np.NaN, dtype=np.float32)
                        RH_min_core_down = np.full(ncores_down, np.NaN, dtype=np.float32)
                        Qc_max_core_down = np.full(ncores_down, np.NaN, dtype=np.float32)
                        Qr_max_core_down = np.full(ncores_down, np.NaN, dtype=np.float32)
                        Qi_max_core_down = np.full(ncores_down, np.NaN, dtype=np.float32)
                        Qs_max_core_down = np.full(ncores_down, np.NaN, dtype=np.float32)
                        Qg_max_core_down = np.full(ncores_down, np.NaN, dtype=np.float32)
                        Qt_max_core_down = np.full(ncores_down, np.NaN, dtype=np.float32)
                        Nc_max_core_down = np.full(ncores_down, np.NaN, dtype=np.float32)
                        Nr_max_core_down = np.full(ncores_down, np.NaN, dtype=np.float32)
                        Ni_max_core_down = np.full(ncores_down, np.NaN, dtype=np.float32)
                        Nwa_max_core_down = np.full(ncores_down, np.NaN, dtype=np.float32)
                        Nia_max_core_down = np.full(ncores_down, np.NaN, dtype=np.float32)
                        Hd_min_core_down = np.full(ncores_down, np.NaN, dtype=np.float32)
                        for ii in range(ncores_down):
                            MaFlx_core_down[ii] = np.nansum(np.array(zMassFlux)[core_label_down == core_numbers_down[ii]])
                            W_mean_core_down[ii] = np.nanmean(np.array(zW)[core_label_down == core_numbers_down[ii]])
                            T_mean_core_down[ii] = np.nanmean(np.array(zT)[core_label_down == core_numbers_down[ii]])
                            Qv_mean_core_down[ii] = np.nanmean(np.array(zQv)[core_label_down == core_numbers_down[ii]])
                            Thetae_mean_core_down[ii] = np.nanmean(np.array(zThetae)[core_label_down == core_numbers_down[ii]])
                            Tv_mean_core_down[ii] = np.nanmean(np.array(zTv)[core_label_down == core_numbers_down[ii]])
                            Trho_mean_core_down[ii] = np.nanmean(np.array(zTrho)[core_label_down == core_numbers_down[ii]])
                            RH_mean_core_down[ii] = np.nanmean(np.array(zRH)[core_label_down == core_numbers_down[ii]])
                            Qc_mean_core_down[ii] = np.nanmean(np.array(zQc)[core_label_down == core_numbers_down[ii]])
                            Qr_mean_core_down[ii] = np.nanmean(np.array(zQr)[core_label_down == core_numbers_down[ii]])
                            Qi_mean_core_down[ii] = np.nanmean(np.array(zQi)[core_label_down == core_numbers_down[ii]])
                            Qs_mean_core_down[ii] = np.nanmean(np.array(zQs)[core_label_down == core_numbers_down[ii]])
                            Qg_mean_core_down[ii] = np.nanmean(np.array(zQg)[core_label_down == core_numbers_down[ii]])
                            Qt_mean_core_down[ii] = np.nanmean(np.array(zQt)[core_label_down == core_numbers_down[ii]])
                            Nc_mean_core_down[ii] = np.nanmean(np.array(zNc)[core_label_down == core_numbers_down[ii]])
                            Nr_mean_core_down[ii] = np.nanmean(np.array(zNr)[core_label_down == core_numbers_down[ii]])
                            Ni_mean_core_down[ii] = np.nanmean(np.array(zNi)[core_label_down == core_numbers_down[ii]])
                            Nwa_mean_core_down[ii] = np.nanmean(np.array(zNwa)[core_label_down == core_numbers_down[ii]])
                            Nia_mean_core_down[ii] = np.nanmean(np.array(zNia)[core_label_down == core_numbers_down[ii]])
                            Hd_mean_core_down[ii] = np.nanmean(np.array(zHd)[core_label_down == core_numbers_down[ii]])
                            P_mean_core_down[ii] = np.nanmean(np.array(zP)[core_label_down == core_numbers_down[ii]])
                            Rho_mean_core_down[ii] = np.nanmean(np.array(zRho)[core_label_down == core_numbers_down[ii]])
                            W_min_core_down[ii] = np.nanmin(np.array(zW)[core_label_down == core_numbers_down[ii]])
                            T_min_core_down[ii] = np.nanmin(np.array(zT)[core_label_down == core_numbers_down[ii]])
                            Qv_min_core_down[ii] = np.nanmin(np.array(zQv)[core_label_down == core_numbers_down[ii]])
                            Thetae_min_core_down[ii] = np.nanmin(np.array(zThetae)[core_label_down == core_numbers_down[ii]])
                            Tv_min_core_down[ii] = np.nanmin(np.array(zTv)[core_label_down == core_numbers_down[ii]])
                            Trho_min_core_down[ii] = np.nanmin(np.array(zTrho)[core_label_down == core_numbers_down[ii]])
                            RH_min_core_down[ii] = np.nanmin(np.array(zRH)[core_label_down == core_numbers_down[ii]])
                            Qc_max_core_down[ii] = np.nanmax(np.array(zQc)[core_label_down == core_numbers_down[ii]])
                            Qr_max_core_down[ii] = np.nanmax(np.array(zQr)[core_label_down == core_numbers_down[ii]])
                            Qi_max_core_down[ii] = np.nanmax(np.array(zQi)[core_label_down == core_numbers_down[ii]])
                            Qs_max_core_down[ii] = np.nanmax(np.array(zQs)[core_label_down == core_numbers_down[ii]])
                            Qg_max_core_down[ii] = np.nanmax(np.array(zQg)[core_label_down == core_numbers_down[ii]])
                            Qt_max_core_down[ii] = np.nanmax(np.array(zQt)[core_label_down == core_numbers_down[ii]])
                            Nc_max_core_down[ii] = np.nanmax(np.array(zNc)[core_label_down == core_numbers_down[ii]])
                            Nr_max_core_down[ii] = np.nanmax(np.array(zNr)[core_label_down == core_numbers_down[ii]])
                            Ni_max_core_down[ii] = np.nanmax(np.array(zNi)[core_label_down == core_numbers_down[ii]])
                            Nwa_max_core_down[ii] = np.nanmax(np.array(zNwa)[core_label_down == core_numbers_down[ii]])
                            Nia_max_core_down[ii] = np.nanmax(np.array(zNia)[core_label_down == core_numbers_down[ii]])
                            Hd_min_core_down[ii] = np.nanmin(np.array(zHd)[core_label_down == core_numbers_down[ii]])
                            
                        # Calculate total mass flux for all labeled cores
                        if ncores_all_down > 0:
                            MaFlx_sum_down = np.nansum(np.array(zMassFlux)[core_label_down > 0])
                            W_mean_down = np.nanmean(np.array(zW)[core_label_down > 0])
                            T_mean_down = np.nanmean(np.array(zT)[core_label_down > 0])
                            Qv_mean_down = np.nanmean(np.array(zQv)[core_label_down > 0])
                            Thetae_mean_down = np.nanmean(np.array(zThetae)[core_label_down > 0])
                            Tv_mean_down = np.nanmean(np.array(zTv)[core_label_down > 0])
                            Trho_mean_down = np.nanmean(np.array(zTrho)[core_label_down > 0])
                            RH_mean_down = np.nanmean(np.array(zRH)[core_label_down > 0])
                            Qc_mean_down = np.nanmean(np.array(zQc)[core_label_down > 0])
                            Qr_mean_down = np.nanmean(np.array(zQr)[core_label_down > 0])
                            Qi_mean_down = np.nanmean(np.array(zQi)[core_label_down > 0])
                            Qs_mean_down = np.nanmean(np.array(zQs)[core_label_down > 0])
                            Qg_mean_down = np.nanmean(np.array(zQg)[core_label_down > 0])
                            Qt_mean_down = np.nanmean(np.array(zQt)[core_label_down > 0])
                            Nc_mean_down = np.nanmean(np.array(zNc)[core_label_down > 0])
                            Nr_mean_down = np.nanmean(np.array(zNr)[core_label_down > 0])
                            Ni_mean_down = np.nanmean(np.array(zNi)[core_label_down > 0])
                            Nwa_mean_down = np.nanmean(np.array(zNwa)[core_label_down > 0])
                            Nia_mean_down = np.nanmean(np.array(zNia)[core_label_down > 0])
                            Hd_mean_down = np.nanmean(np.array(zHd)[core_label_down > 0])
                            P_mean_down = np.nanmean(np.array(zP)[core_label_down > 0])
                            Rho_mean_down = np.nanmean(np.array(zRho)[core_label_down > 0])
                        else:
                            MaFlx_sum_down = np.NaN
                            W_mean_down = np.NaN
                            T_mean_down = np.NaN
                            Qv_mean_down = np.NaN
                            Thetae_mean_down = np.NaN
                            Tv_mean_down = np.NaN
                            Trho_mean_down = np.NaN
                            RH_mean_down = np.NaN
                            Qc_mean_down = np.NaN
                            Qr_mean_down = np.NaN
                            Qi_mean_down = np.NaN
                            Qs_mean_down = np.NaN
                            Qg_mean_down = np.NaN
                            Qt_mean_down = np.NaN
                            Nc_mean_down = np.NaN
                            Nr_mean_down = np.NaN
                            Ni_mean_down = np.NaN
                            Nwa_mean_down = np.NaN
                            Nia_mean_down = np.NaN
                            Hd_mean_down = np.NaN
                            Hd_mean_down = np.NaN
                            P_mean_down = np.NaN
                            Rho_mean_down = np.NaN
                        
                        # Save data to output arrays
                        #2D arrays
                        #Non-core areas
                        cell_nCore_up[icell, z] = ncores_all_up
                        cell_nCore_down[icell, z] = ncores_all_down
                        cell_nonCoreMassFlux[icell, z] = np.nanmean(zMassFlux.where(np.logical_and(zW >= W_down_thresh, zW <= W_up_thresh))) * DX * DY * 1e6
                        cell_nonCoreMeanW[icell, z] = np.nanmean(zW.where(np.logical_and(zW >= W_down_thresh, zW <= W_up_thresh)))
                        cell_nonCoreArea[icell, z] = (inpix_cloud - np.nansum(core_npix_up) - np.nansum(core_npix_down)) * grid_area
                        cell_nonCoreMeanT[icell, z] = np.nanmean(zT.where(np.logical_and(zW >= W_down_thresh, zW <= W_up_thresh)))
                        cell_nonCoreMeanQv[icell, z] = np.nanmean(zQv.where(np.logical_and(zW >= W_down_thresh, zW <= W_up_thresh)))
                        cell_nonCoreMeanThetae[icell, z] = np.nanmean(zThetae.where(np.logical_and(zW >= W_down_thresh, zW <= W_up_thresh)))
                        cell_nonCoreMeanTv[icell, z] = np.nanmean(zTv.where(np.logical_and(zW >= W_down_thresh, zW <= W_up_thresh)))
                        cell_nonCoreMeanTrho[icell, z] = np.nanmean(zTrho.where(np.logical_and(zW >= W_down_thresh, zW <= W_up_thresh)))
                        cell_nonCoreMeanRH[icell, z] = np.nanmean(zRH.where(np.logical_and(zW >= W_down_thresh, zW <= W_up_thresh)))
                        cell_nonCoreMeanQc[icell, z] = np.nanmean(zQc.where(np.logical_and(zW >= W_down_thresh, zW <= W_up_thresh)))
                        cell_nonCoreMeanQr[icell, z] = np.nanmean(zQr.where(np.logical_and(zW >= W_down_thresh, zW <= W_up_thresh)))
                        cell_nonCoreMeanQi[icell, z] = np.nanmean(zQi.where(np.logical_and(zW >= W_down_thresh, zW <= W_up_thresh)))
                        cell_nonCoreMeanQs[icell, z] = np.nanmean(zQs.where(np.logical_and(zW >= W_down_thresh, zW <= W_up_thresh)))
                        cell_nonCoreMeanQg[icell, z] = np.nanmean(zQg.where(np.logical_and(zW >= W_down_thresh, zW <= W_up_thresh)))
                        cell_nonCoreMeanQt[icell, z] = np.nanmean(zQt.where(np.logical_and(zW >= W_down_thresh, zW <= W_up_thresh)))
                        cell_nonCoreMeanNc[icell, z] = np.nanmean(zNc.where(np.logical_and(zW >= W_down_thresh, zW <= W_up_thresh)))
                        cell_nonCoreMeanNr[icell, z] = np.nanmean(zNr.where(np.logical_and(zW >= W_down_thresh, zW <= W_up_thresh)))
                        cell_nonCoreMeanNi[icell, z] = np.nanmean(zNi.where(np.logical_and(zW >= W_down_thresh, zW <= W_up_thresh)))
                        cell_nonCoreMeanNwa[icell, z] = np.nanmean(zNwa.where(np.logical_and(zW >= W_down_thresh, zW <= W_up_thresh)))
                        cell_nonCoreMeanNia[icell, z] = np.nanmean(zNia.where(np.logical_and(zW >= W_down_thresh, zW <= W_up_thresh)))
                        cell_nonCoreMeanHd[icell, z] = np.nanmean(zHd.where(np.logical_and(zW >= W_down_thresh, zW <= W_up_thresh)))
                        cell_nonCoreMeanP[icell, z] = np.nanmean(zP.where(np.logical_and(zW >= W_down_thresh, zW <= W_up_thresh)))
                        cell_nonCoreMeanRho[icell, z] = np.nanmean(zRho.where(np.logical_and(zW >= W_down_thresh, zW <= W_up_thresh)))
                        
                        #3D arrays
                        #Updrafts
                        ncores_save_up = min([ncores_up, ncores_min])
                        cell_MassFlux_up[icell, z] = MaFlx_sum_up * DX * DY * 1e6
                        cell_Area_up[icell, z] = np.nansum(core_npix_up) * grid_area
                        cell_MeanW_up[icell, z] = W_mean_up
                        cell_MeanT_up[icell, z] = T_mean_up
                        cell_MeanQv_up[icell, z] = Qv_mean_up
                        cell_MeanThetae_up[icell, z] = Thetae_mean_up
                        cell_MeanTv_up[icell, z] = Tv_mean_up
                        cell_MeanTrho_up[icell, z] = Trho_mean_up
                        cell_MeanRH_up[icell, z] = RH_mean_up
                        cell_MeanQc_up[icell, z] = Qc_mean_up
                        cell_MeanQr_up[icell, z] = Qr_mean_up
                        cell_MeanQi_up[icell, z] = Qi_mean_up
                        cell_MeanQs_up[icell, z] = Qs_mean_up
                        cell_MeanQg_up[icell, z] = Qg_mean_up
                        cell_MeanQt_up[icell, z] = Qt_mean_up
                        cell_MeanNc_up[icell, z] = Nc_mean_up
                        cell_MeanNr_up[icell, z] = Nr_mean_up
                        cell_MeanNi_up[icell, z] = Ni_mean_up
                        cell_MeanNwa_up[icell, z] = Nwa_mean_up
                        cell_MeanNia_up[icell, z] = Nia_mean_up
                        cell_MeanHd_up[icell, z] = Hd_mean_up
                        cell_MeanP_up[icell, z] = P_mean_up
                        cell_MeanRho_up[icell, z] = Rho_mean_up
                        cell_CoreMassFlux_up[icell, z, 0:ncores_save_up] = MaFlx_core_up[0:ncores_save_up] * DX * DY * 1e6
                        cell_CoreArea_up[icell, z, 0:ncores_save_up] = core_npix_up[0:ncores_save_up] * grid_area
                        cell_CoreMeanW_up[icell, z, 0:ncores_save_up] = W_mean_core_up[0:ncores_save_up]
                        cell_CoreMeanT_up[icell, z, 0:ncores_save_up] = T_mean_core_up[0:ncores_save_up]
                        cell_CoreMeanQv_up[icell, z, 0:ncores_save_up] = Qv_mean_core_up[0:ncores_save_up]
                        cell_CoreMeanThetae_up[icell, z, 0:ncores_save_up] = Thetae_mean_core_up[0:ncores_save_up]
                        cell_CoreMeanTv_up[icell, z, 0:ncores_save_up] = Tv_mean_core_up[0:ncores_save_up]
                        cell_CoreMeanTrho_up[icell, z, 0:ncores_save_up] = Trho_mean_core_up[0:ncores_save_up]
                        cell_CoreMeanRH_up[icell, z, 0:ncores_save_up] = RH_mean_core_up[0:ncores_save_up]
                        cell_CoreMeanQc_up[icell, z, 0:ncores_save_up] = Qc_mean_core_up[0:ncores_save_up]
                        cell_CoreMeanQr_up[icell, z, 0:ncores_save_up] = Qr_mean_core_up[0:ncores_save_up]
                        cell_CoreMeanQi_up[icell, z, 0:ncores_save_up] = Qi_mean_core_up[0:ncores_save_up]
                        cell_CoreMeanQs_up[icell, z, 0:ncores_save_up] = Qs_mean_core_up[0:ncores_save_up]
                        cell_CoreMeanQg_up[icell, z, 0:ncores_save_up] = Qg_mean_core_up[0:ncores_save_up]
                        cell_CoreMeanQt_up[icell, z, 0:ncores_save_up] = Qt_mean_core_up[0:ncores_save_up]
                        cell_CoreMeanNc_up[icell, z, 0:ncores_save_up] = Nc_mean_core_up[0:ncores_save_up]
                        cell_CoreMeanNr_up[icell, z, 0:ncores_save_up] = Nr_mean_core_up[0:ncores_save_up]
                        cell_CoreMeanNi_up[icell, z, 0:ncores_save_up] = Ni_mean_core_up[0:ncores_save_up]
                        cell_CoreMeanNwa_up[icell, z, 0:ncores_save_up] = Nwa_mean_core_up[0:ncores_save_up]
                        cell_CoreMeanNia_up[icell, z, 0:ncores_save_up] = Nia_mean_core_up[0:ncores_save_up]
                        cell_CoreMeanHd_up[icell, z, 0:ncores_save_up] = Hd_mean_core_up[0:ncores_save_up]
                        cell_CoreMeanP_up[icell, z, 0:ncores_save_up] = P_mean_core_up[0:ncores_save_up]
                        cell_CoreMeanRho_up[icell, z, 0:ncores_save_up] = Rho_mean_core_up[0:ncores_save_up]
                        cell_CoreMaxW_up[icell, z, 0:ncores_save_up] = W_max_core_up[0:ncores_save_up]
                        cell_CoreMaxT_up[icell, z, 0:ncores_save_up] = T_max_core_up[0:ncores_save_up]
                        cell_CoreMaxQv_up[icell, z, 0:ncores_save_up] = Qv_max_core_up[0:ncores_save_up]
                        cell_CoreMaxThetae_up[icell, z, 0:ncores_save_up] = Thetae_max_core_up[0:ncores_save_up]
                        cell_CoreMaxTv_up[icell, z, 0:ncores_save_up] = Tv_max_core_up[0:ncores_save_up]
                        cell_CoreMaxTrho_up[icell, z, 0:ncores_save_up] = Trho_max_core_up[0:ncores_save_up]
                        cell_CoreMaxRH_up[icell, z, 0:ncores_save_up] = RH_max_core_up[0:ncores_save_up]
                        cell_CoreMaxQc_up[icell, z, 0:ncores_save_up] = Qc_max_core_up[0:ncores_save_up]
                        cell_CoreMaxQr_up[icell, z, 0:ncores_save_up] = Qr_max_core_up[0:ncores_save_up]
                        cell_CoreMaxQi_up[icell, z, 0:ncores_save_up] = Qi_max_core_up[0:ncores_save_up]
                        cell_CoreMaxQs_up[icell, z, 0:ncores_save_up] = Qs_max_core_up[0:ncores_save_up]
                        cell_CoreMaxQg_up[icell, z, 0:ncores_save_up] = Qg_max_core_up[0:ncores_save_up]
                        cell_CoreMaxQt_up[icell, z, 0:ncores_save_up] = Qt_max_core_up[0:ncores_save_up]
                        cell_CoreMaxNc_up[icell, z, 0:ncores_save_up] = Nc_max_core_up[0:ncores_save_up]
                        cell_CoreMaxNr_up[icell, z, 0:ncores_save_up] = Nr_max_core_up[0:ncores_save_up]
                        cell_CoreMaxNi_up[icell, z, 0:ncores_save_up] = Ni_max_core_up[0:ncores_save_up]
                        cell_CoreMaxNwa_up[icell, z, 0:ncores_save_up] = Nwa_max_core_up[0:ncores_save_up]
                        cell_CoreMaxNia_up[icell, z, 0:ncores_save_up] = Nia_max_core_up[0:ncores_save_up]
                        cell_CoreMaxHd_up[icell, z, 0:ncores_save_up] = Hd_max_core_up[0:ncores_save_up]
                        
                        #Downdrafts
                        ncores_save_down = min([ncores_down, ncores_min])
                        cell_MassFlux_down[icell, z] = MaFlx_sum_down * DX * DY * 1e6
                        cell_Area_down[icell, z] = np.nansum(core_npix_down) * grid_area
                        cell_MeanW_down[icell, z] = W_mean_down
                        cell_MeanT_down[icell, z] = T_mean_down
                        cell_MeanQv_down[icell, z] = Qv_mean_down
                        cell_MeanThetae_down[icell, z] = Thetae_mean_down
                        cell_MeanTv_down[icell, z] = Tv_mean_down
                        cell_MeanTrho_down[icell, z] = Trho_mean_down
                        cell_MeanRH_down[icell, z] = RH_mean_down
                        cell_MeanQc_down[icell, z] = Qc_mean_down
                        cell_MeanQr_down[icell, z] = Qr_mean_down
                        cell_MeanQi_down[icell, z] = Qi_mean_down
                        cell_MeanQs_down[icell, z] = Qs_mean_down
                        cell_MeanQg_down[icell, z] = Qg_mean_down
                        cell_MeanQt_down[icell, z] = Qt_mean_down
                        cell_MeanNc_down[icell, z] = Nc_mean_down
                        cell_MeanNr_down[icell, z] = Nr_mean_down
                        cell_MeanNi_down[icell, z] = Ni_mean_down
                        cell_MeanNwa_down[icell, z] = Nwa_mean_down
                        cell_MeanNia_down[icell, z] = Nia_mean_down
                        cell_MeanHd_down[icell, z] = Hd_mean_down
                        cell_MeanP_down[icell, z] = P_mean_down
                        cell_MeanRho_down[icell, z] = Rho_mean_down
                        cell_CoreMassFlux_down[icell, z, 0:ncores_save_down] = MaFlx_core_down[0:ncores_save_down] * DX * DY * 1e6
                        cell_CoreArea_down[icell, z, 0:ncores_save_down] = core_npix_down[0:ncores_save_down] * grid_area
                        cell_CoreMeanW_down[icell, z, 0:ncores_save_down] = W_mean_core_down[0:ncores_save_down]
                        cell_CoreMeanT_down[icell, z, 0:ncores_save_down] = T_mean_core_down[0:ncores_save_down]
                        cell_CoreMeanQv_down[icell, z, 0:ncores_save_down] = Qv_mean_core_down[0:ncores_save_down]
                        cell_CoreMeanThetae_down[icell, z, 0:ncores_save_down] = Thetae_mean_core_down[0:ncores_save_down]
                        cell_CoreMeanTv_down[icell, z, 0:ncores_save_down] = Tv_mean_core_down[0:ncores_save_down]
                        cell_CoreMeanTrho_down[icell, z, 0:ncores_save_down] = Trho_mean_core_down[0:ncores_save_down]
                        cell_CoreMeanRH_down[icell, z, 0:ncores_save_down] = RH_mean_core_down[0:ncores_save_down]
                        cell_CoreMeanQc_down[icell, z, 0:ncores_save_down] = Qc_mean_core_down[0:ncores_save_down]
                        cell_CoreMeanQr_down[icell, z, 0:ncores_save_down] = Qr_mean_core_down[0:ncores_save_down]
                        cell_CoreMeanQi_down[icell, z, 0:ncores_save_down] = Qi_mean_core_down[0:ncores_save_down]
                        cell_CoreMeanQs_down[icell, z, 0:ncores_save_down] = Qs_mean_core_down[0:ncores_save_down]
                        cell_CoreMeanQg_down[icell, z, 0:ncores_save_down] = Qg_mean_core_down[0:ncores_save_down]
                        cell_CoreMeanQt_down[icell, z, 0:ncores_save_down] = Qt_mean_core_down[0:ncores_save_down]
                        cell_CoreMeanNc_down[icell, z, 0:ncores_save_down] = Nc_mean_core_down[0:ncores_save_down]
                        cell_CoreMeanNr_down[icell, z, 0:ncores_save_down] = Nr_mean_core_down[0:ncores_save_down]
                        cell_CoreMeanNi_down[icell, z, 0:ncores_save_down] = Ni_mean_core_down[0:ncores_save_down]
                        cell_CoreMeanNwa_down[icell, z, 0:ncores_save_down] = Nwa_mean_core_down[0:ncores_save_down]
                        cell_CoreMeanNia_down[icell, z, 0:ncores_save_down] = Nia_mean_core_down[0:ncores_save_down]
                        cell_CoreMeanHd_down[icell, z, 0:ncores_save_down] = Hd_mean_core_down[0:ncores_save_down]
                        cell_CoreMeanP_down[icell, z, 0:ncores_save_down] = P_mean_core_down[0:ncores_save_down]
                        cell_CoreMeanRho_down[icell, z, 0:ncores_save_down] = Rho_mean_core_down[0:ncores_save_down]
                        cell_CoreMinW_down[icell, z, 0:ncores_save_down] = W_min_core_down[0:ncores_save_down]
                        cell_CoreMinT_down[icell, z, 0:ncores_save_down] = T_min_core_down[0:ncores_save_down]
                        cell_CoreMinQv_down[icell, z, 0:ncores_save_down] = Qv_min_core_down[0:ncores_save_down]
                        cell_CoreMinThetae_down[icell, z, 0:ncores_save_down] = Thetae_min_core_down[0:ncores_save_down]
                        cell_CoreMinTv_down[icell, z, 0:ncores_save_down] = Tv_min_core_down[0:ncores_save_down]
                        cell_CoreMinTrho_down[icell, z, 0:ncores_save_down] = Trho_min_core_down[0:ncores_save_down]
                        cell_CoreMinRH_down[icell, z, 0:ncores_save_down] = RH_min_core_down[0:ncores_save_down]
                        cell_CoreMaxQc_down[icell, z, 0:ncores_save_down] = Qc_max_core_down[0:ncores_save_down]
                        cell_CoreMaxQr_down[icell, z, 0:ncores_save_down] = Qr_max_core_down[0:ncores_save_down]
                        cell_CoreMaxQi_down[icell, z, 0:ncores_save_down] = Qi_max_core_down[0:ncores_save_down]
                        cell_CoreMaxQs_down[icell, z, 0:ncores_save_down] = Qs_max_core_down[0:ncores_save_down]
                        cell_CoreMaxQg_down[icell, z, 0:ncores_save_down] = Qg_max_core_down[0:ncores_save_down]
                        cell_CoreMaxQt_down[icell, z, 0:ncores_save_down] = Qt_max_core_down[0:ncores_save_down]
                        cell_CoreMaxNc_down[icell, z, 0:ncores_save_down] = Nc_max_core_down[0:ncores_save_down]
                        cell_CoreMaxNr_down[icell, z, 0:ncores_save_down] = Nr_max_core_down[0:ncores_save_down]
                        cell_CoreMaxNi_down[icell, z, 0:ncores_save_down] = Ni_max_core_down[0:ncores_save_down]
                        cell_CoreMaxNwa_down[icell, z, 0:ncores_save_down] = Nwa_max_core_down[0:ncores_save_down]
                        cell_CoreMaxNia_down[icell, z, 0:ncores_save_down] = Nia_max_core_down[0:ncores_save_down]
                        cell_CoreMinHd_down[icell, z, 0:ncores_save_down] = Hd_min_core_down[0:ncores_save_down]
                        
                        # if np.nansum(cell_CoreMassFlux_up[icell, z, :]) > 0:
                        #     import pdb; pdb.set_trace()
                    # if np.max(cell_MassFlux_up[icell, :]) > 0:
                    #     import pdb; pdb.set_trace()
                    
            else:
                print(f'No cell matching track # {itracknum}')

        
        # Group outputs in dictionaries
        out_up_dict3d = {
            'CoreArea_up': cell_CoreArea_up,
            'CoreMassFlux_up': cell_CoreMassFlux_up,
            'CoreMeanW_up': cell_CoreMeanW_up,
            'CoreMeanT_up': cell_CoreMeanT_up,
            'CoreMeanQv_up': cell_CoreMeanQv_up,
            'CoreMeanThetae_up': cell_CoreMeanThetae_up,
            'CoreMeanTv_up': cell_CoreMeanTv_up,
            'CoreMeanTrho_up': cell_CoreMeanTrho_up,
            'CoreMeanRH_up': cell_CoreMeanRH_up,
            'CoreMeanQc_up': cell_CoreMeanQc_up,
            'CoreMeanQr_up': cell_CoreMeanQr_up,
            'CoreMeanQi_up': cell_CoreMeanQi_up,
            'CoreMeanQs_up': cell_CoreMeanQs_up,
            'CoreMeanQg_up': cell_CoreMeanQg_up,
            'CoreMeanQt_up': cell_CoreMeanQt_up,
            'CoreMeanNc_up': cell_CoreMeanNc_up,
            'CoreMeanNr_up': cell_CoreMeanNr_up,
            'CoreMeanNi_up': cell_CoreMeanNi_up,
            'CoreMeanNwa_up': cell_CoreMeanNwa_up,
            'CoreMeanNia_up': cell_CoreMeanNia_up,
            'CoreMeanHd_up': cell_CoreMeanHd_up,
            'CoreMeanP_up': cell_CoreMeanP_up,
            'CoreMeanRho_up': cell_CoreMeanRho_up,
            'CoreMaxW_up': cell_CoreMaxW_up,
            'CoreMaxT_up': cell_CoreMaxT_up,
            'CoreMaxQv_up': cell_CoreMaxQv_up,
            'CoreMaxThetae_up': cell_CoreMaxThetae_up,
            'CoreMaxTv_up': cell_CoreMaxTv_up,
            'CoreMaxTrho_up': cell_CoreMaxTrho_up,
            'CoreMaxRH_up': cell_CoreMaxRH_up,
            'CoreMaxQc_up': cell_CoreMaxQc_up,
            'CoreMaxQr_up': cell_CoreMaxQr_up,
            'CoreMaxQi_up': cell_CoreMaxQi_up,
            'CoreMaxQs_up': cell_CoreMaxQs_up,
            'CoreMaxQg_up': cell_CoreMaxQg_up,
            'CoreMaxQt_up': cell_CoreMaxQt_up,
            'CoreMaxNc_up': cell_CoreMaxNc_up,
            'CoreMaxNr_up': cell_CoreMaxNr_up,
            'CoreMaxNi_up': cell_CoreMaxNi_up,
            'CoreMaxNwa_up': cell_CoreMaxNwa_up,
            'CoreMaxNia_up': cell_CoreMaxNia_up,
            'CoreMaxHd_up': cell_CoreMaxHd_up,
        }
        out_down_dict3d = {
            'CoreArea_down': cell_CoreArea_down,
            'CoreMassFlux_down': cell_CoreMassFlux_down,
            'CoreMeanW_down': cell_CoreMeanW_down,
            'CoreMeanT_down': cell_CoreMeanT_down,
            'CoreMeanQv_down': cell_CoreMeanQv_down,
            'CoreMeanThetae_down': cell_CoreMeanThetae_down,
            'CoreMeanTv_down': cell_CoreMeanTv_down,
            'CoreMeanTrho_down': cell_CoreMeanTrho_down,
            'CoreMeanRH_down': cell_CoreMeanRH_down,
            'CoreMeanQc_down': cell_CoreMeanQc_down,
            'CoreMeanQr_down': cell_CoreMeanQr_down,
            'CoreMeanQi_down': cell_CoreMeanQi_down,
            'CoreMeanQs_down': cell_CoreMeanQs_down,
            'CoreMeanQg_down': cell_CoreMeanQg_down,
            'CoreMeanQt_down': cell_CoreMeanQt_down,
            'CoreMeanNc_down': cell_CoreMeanNc_down,
            'CoreMeanNr_down': cell_CoreMeanNr_down,
            'CoreMeanNi_down': cell_CoreMeanNi_down,
            'CoreMeanNwa_down': cell_CoreMeanNwa_down,
            'CoreMeanNia_down': cell_CoreMeanNia_down,
            'CoreMeanHd_down': cell_CoreMeanHd_down,
            'CoreMeanP_down': cell_CoreMeanP_down,
            'CoreMeanRho_down': cell_CoreMeanRho_down,
            'CoreMinW_down': cell_CoreMinW_down,
            'CoreMinT_down': cell_CoreMinT_down,
            'CoreMinQv_down': cell_CoreMinQv_down,
            'CoreMinThetae_down': cell_CoreMinThetae_down,
            'CoreMinTv_down': cell_CoreMinTv_down,
            'CoreMinTrho_down': cell_CoreMinTrho_down,
            'CoreMinRH_down': cell_CoreMinRH_down,
            'CoreMaxQc_down': cell_CoreMaxQc_down,
            'CoreMaxQr_down': cell_CoreMaxQr_down,
            'CoreMaxQi_down': cell_CoreMaxQi_down,
            'CoreMaxQs_down': cell_CoreMaxQs_down,
            'CoreMaxQg_down': cell_CoreMaxQg_down,
            'CoreMaxQt_down': cell_CoreMaxQt_down,
            'CoreMaxNc_down': cell_CoreMaxNc_down,
            'CoreMaxNr_down': cell_CoreMaxNr_down,
            'CoreMaxNi_down': cell_CoreMaxNi_down,
            'CoreMaxNwa_down': cell_CoreMaxNwa_down,
            'CoreMaxNia_down': cell_CoreMaxNia_down,
            'CoreMinHd_down': cell_CoreMinHd_down,
        }
        out_up_dict2d = {
            'nCore_up': cell_nCore_up,
            'MassFlux_up': cell_MassFlux_up,
            'Area_up': cell_Area_up,
            'MeanW_up': cell_MeanW_up,
            'MeanT_up': cell_MeanT_up,
            'MeanQv_up': cell_MeanQv_up,
            'MeanThetae_up': cell_MeanThetae_up,
            'MeanTv_up': cell_MeanTv_up,
            'MeanTrho_up': cell_MeanTrho_up,
            'MeanRH_up': cell_MeanRH_up,
            'MeanQc_up': cell_MeanQc_up,
            'MeanQr_up': cell_MeanQr_up,
            'MeanQi_up': cell_MeanQi_up,
            'MeanQs_up': cell_MeanQs_up,
            'MeanQg_up': cell_MeanQg_up,
            'MeanQt_up': cell_MeanQt_up,
            'MeanNc_up': cell_MeanNc_up,
            'MeanNr_up': cell_MeanNr_up,
            'MeanNi_up': cell_MeanNi_up,
            'MeanNwa_up': cell_MeanNwa_up,
            'MeanNia_up': cell_MeanNia_up,
            'MeanHd_up': cell_MeanHd_up,
            'MeanP_up': cell_MeanP_up,
            'MeanRho_up': cell_MeanRho_up
        }
        out_down_dict2d = {
            'nCore_down': cell_nCore_down,
            'MassFlux_down': cell_MassFlux_down,
            'Area_down': cell_Area_down,
            'MeanW_down': cell_MeanW_down,
            'MeanT_down': cell_MeanT_down,
            'MeanQv_down': cell_MeanQv_down,
            'MeanThetae_down': cell_MeanThetae_down,
            'MeanTv_down': cell_MeanTv_down,
            'MeanTrho_down': cell_MeanTrho_down,
            'MeanRH_down': cell_MeanRH_down,
            'MeanQc_down': cell_MeanQc_down,
            'MeanQr_down': cell_MeanQr_down,
            'MeanQi_down': cell_MeanQi_down,
            'MeanQs_down': cell_MeanQs_down,
            'MeanQg_down': cell_MeanQg_down,
            'MeanQt_down': cell_MeanQt_down,
            'MeanNc_down': cell_MeanNc_down,
            'MeanNr_down': cell_MeanNr_down,
            'MeanNi_down': cell_MeanNi_down,
            'MeanNwa_down': cell_MeanNwa_down,
            'MeanNia_down': cell_MeanNia_down,
            'MeanHd_down': cell_MeanHd_down,
            'MeanP_down': cell_MeanP_down,
            'MeanRho_down': cell_MeanRho_down,
        }
        out_non_dict2d = {
            'nonCoreMassFlux': cell_nonCoreMassFlux,
            'nonCoreMeanW': cell_nonCoreMeanW,
            'nonCoreArea': cell_nonCoreArea,
            'nonCoreMeanT': cell_nonCoreMeanT,
            'nonCoreMeanQv': cell_nonCoreMeanQv,
            'nonCoreMeanThetae': cell_nonCoreMeanThetae,
            'nonCoreMeanTv': cell_nonCoreMeanTv,
            'nonCoreMeanTrho': cell_nonCoreMeanTrho,
            'nonCoreMeanRH': cell_nonCoreMeanRH,
            'nonCoreMeanQc': cell_nonCoreMeanQc,
            'nonCoreMeanQr': cell_nonCoreMeanQr,
            'nonCoreMeanQi': cell_nonCoreMeanQi,
            'nonCoreMeanQs': cell_nonCoreMeanQs,
            'nonCoreMeanQg': cell_nonCoreMeanQg,
            'nonCoreMeanQt': cell_nonCoreMeanQt,
            'nonCoreMeanNc': cell_nonCoreMeanNc,
            'nonCoreMeanNr': cell_nonCoreMeanNr,
            'nonCoreMeanNi': cell_nonCoreMeanNi,
            'nonCoreMeanNwa': cell_nonCoreMeanNwa,
            'nonCoreMeanNia': cell_nonCoreMeanNia,
            'nonCoreMeanHd': cell_nonCoreMeanHd,
            'nonCoreMeanP': cell_nonCoreMeanP,
            'nonCoreMeanRho': cell_nonCoreMeanRho,
        }
        out_up_dict_attrs = {
            'nCore_up': {
                'long_name': 'Number of updraft cores',
                'units': 'count',
            },
            'Area_up': {
                'long_name': 'Total updraft area',
                'units': 'km^2',
            },
            'MassFlux_up': {
                'long_name': 'Total updraft mass flux',
                'units': 'kg/s',
            },
            'MeanW_up': {
                'long_name': 'Total updraft mean vertical wind speed',
                'units': 'm/s',         
            },
            'MeanT_up': {
                'long_name': 'Total updraft mean temperature',
                'units': 'K',
            },
            'MeanQv_up': {
                'long_name': 'Total updraft mean water vapor mixing ratio',
                'units': 'kg/kg',
            },
            'MeanThetae_up': {
                'long_name': 'Total updraft mean equivalent potential temperature',
                'units': 'K',
            },
            'MeanTv_up': {
                'long_name': 'Total updraft mean virtual temperature',
                'units': 'K',
            },
            'MeanTrho_up': {
                'long_name': 'Total updraft mean density temperature',
                'units': 'K',
            },
            'MeanRH_up': {
                'long_name': 'Total updraft mean relative humidity',
                'units': '%',
            },
            'MeanQc_up': {
                'long_name': 'Total updraft mean cloud water mixing ratio',
                'units': 'kg/kg',
            },
            'MeanQr_up': {
                'long_name': 'Total updraft mean rain mixing ratio',
                'units': 'kg/kg',
            },
            'MeanQi_up': {
                'long_name': 'Total updraft mean cloud ice mixing ratio',
                'units': 'kg/kg',
            },
            'MeanQs_up': {
                'long_name': 'Total updraft mean snow mixing ratio',
                'units': 'kg/kg',
            },
            'MeanQg_up': {
                'long_name': 'Total updraft mean graupel mixing ratio',
                'units': 'kg/kg',
            },
            'MeanQt_up': {
                'long_name': 'Total updraft mean total condensate mixing ratio',
                'units': 'kg/kg',
            },
            'MeanNc_up': {
                'long_name': 'Total updraft mean cloud droplet number concentration',
                'units': '1/kg',
            },
            'MeanNr_up': {
                'long_name': 'Total updraft mean raindrop number concentration',
                'units': '1/kg',
            },
            'MeanNi_up': {
                'long_name': 'Total updraft mean cloud ice number concentration',
                'units': '1/kg',
            },
            'MeanNwa_up': {
                'long_name': 'Total updraft mean water-friendly aerosol concentration',
                'units': '1/kg',
            },
            'MeanNia_up': {
                'long_name': 'Total updraft mean ice-friendly aerosol concentration',
                'units': '1/kg',
            },
            'MeanHd_up': {
                'long_name': 'Total updraft mean latent heating',
                'units': 'K/s',
            },
            'MeanP_up': {
                'long_name': 'Total updraft mean pressure',
                'units': 'Pa',
            },
            'MeanRho_up': {
                'long_name': 'Total updraft mean dry air density',
                'units': 'kg/m^3',
            },
            'CoreArea_up': {
                'long_name': 'Updraft core area',
                'units': 'km^2',
            },
            'CoreMassFlux_up': {
                'long_name': 'Updraft core vertical mass flux',
                'units': 'kg/s',
            },
            'CoreMeanW_up': {
                'long_name': 'Updraft core mean vertical wind speed',
                'units': 'm/s',         
            },
            'CoreMeanT_up': {
                'long_name': 'Updraft core mean temperature',
                'units': 'K',
            },
            'CoreMeanQv_up': {
                'long_name': 'Updraft core mean water vapor mixing ratio',
                'units': 'kg/kg',
            },
            'CoreMeanThetae_up': {
                'long_name': 'Updraft core mean equivalent potential temperature',
                'units': 'K',
            },
            'CoreMeanTv_up': {
                'long_name': 'Updraft core mean virtual temperature',
                'units': 'K',
            },
            'CoreMeanTrho_up': {
                'long_name': 'Updraft core mean density temperature',
                'units': 'K',
            },
            'CoreMeanRH_up': {
                'long_name': 'Updraft core mean relative humidity',
                'units': '%',
            },
            'CoreMeanQc_up': {
                'long_name': 'Updraft core mean cloud water mixing ratio',
                'units': 'kg/kg',
            },
            'CoreMeanQr_up': {
                'long_name': 'Updraft core mean rain mixing ratio',
                'units': 'kg/kg',
            },
            'CoreMeanQi_up': {
                'long_name': 'Updraft core mean cloud ice mixing ratio',
                'units': 'kg/kg',
            },
            'CoreMeanQs_up': {
                'long_name': 'Updraft core mean snow mixing ratio',
                'units': 'kg/kg',
            },
            'CoreMeanQg_up': {
                'long_name': 'Updraft core mean graupel mixing ratio',
                'units': 'kg/kg',
            },
            'CoreMeanQt_up': {
                'long_name': 'Updraft core mean total condensate mixing ratio',
                'units': 'kg/kg',
            },
            'CoreMeanNc_up': {
                'long_name': 'Updraft core mean cloud droplet number concentration',
                'units': '1/kg',
            },
            'CoreMeanNr_up': {
                'long_name': 'Updraft core mean raindrop number concentration',
                'units': '1/kg',
            },
            'CoreMeanNi_up': {
                'long_name': 'Updraft core mean cloud ice number concentration',
                'units': '1/kg',
            },
            'CoreMeanNwa_up': {
                'long_name': 'Updraft core mean water-friendly aerosol concentration',
                'units': '1/kg',
            },
            'CoreMeanNia_up': {
                'long_name': 'Updraft core mean ice-friendly aerosol concentration',
                'units': '1/kg',
            },
            'CoreMeanHd_up': {
                'long_name': 'Updraft core mean latent heating',
                'units': 'K/s',
            },
            'CoreMeanP_up': {
                'long_name': 'Updraft core mean pressure',
                'units': 'Pa',
            },
            'CoreMeanRho_up': {
                'long_name': 'Updraft core mean dry air density',
                'units': 'kg/m^3',
            },
            'CoreMaxW_up': {
                'long_name': 'Updraft core maximum vertical wind speed',
                'units': 'm/s',                
            },
            'CoreMaxT_up': {
                'long_name': 'Updraft core maximum temperature',
                'units': 'K',
            },
            'CoreMaxQv_up': {
                'long_name': 'Updraft core maximum water vapor mixing ratio',
                'units': 'kg/kg',
            },
            'CoreMaxThetae_up': {
                'long_name': 'Updraft core maximum equivalent potential temperature',
                'units': 'K',
            },
            'CoreMaxTv_up': {
                'long_name': 'Updraft core maximum virtual temperature',
                'units': 'K',
            },
            'CoreMaxTrho_up': {
                'long_name': 'Updraft core maximum density temperature',
                'units': 'K',
            },
            'CoreMaxRH_up': {
                'long_name': 'Updraft core maximum relative humidity',
                'units': '%',
            },
            'CoreMaxQc_up': {
                'long_name': 'Updraft core maximum cloud water mixing ratio',
                'units': 'kg/kg',
            },
            'CoreMaxQr_up': {
                'long_name': 'Updraft core maximum rain mixing ratio',
                'units': 'kg/kg',
            },
            'CoreMaxQi_up': {
                'long_name': 'Updraft core maximum cloud ice mixing ratio',
                'units': 'kg/kg',
            },
            'CoreMaxQs_up': {
                'long_name': 'Updraft core maximum snow mixing ratio',
                'units': 'kg/kg',
            },
            'CoreMaxQg_up': {
                'long_name': 'Updraft core maximum graupel mixing ratio',
                'units': 'kg/kg',
            },
            'CoreMaxQt_up': {
                'long_name': 'Updraft core maximum total condensate mixing ratio',
                'units': 'kg/kg',
            },
            'CoreMaxNc_up': {
                'long_name': 'Updraft core maximum cloud droplet number concentration',
                'units': '1/kg',
            },
            'CoreMaxNr_up': {
                'long_name': 'Updraft core maximum raindrop number concentration',
                'units': '1/kg',
            },
            'CoreMaxNi_up': {
                'long_name': 'Updraft core maximum cloud ice number concentration',
                'units': '1/kg',
            },
            'CoreMaxNwa_up': {
                'long_name': 'Updraft core maximum water-friendly aerosol concentration',
                'units': '1/kg',
            },
            'CoreMaxNia_up': {
                'long_name': 'Updraft core maximum ice-friendly aerosol concentration',
                'units': '1/kg',
            },
            'CoreMaxHd_up': {
                'long_name': 'Updraft core maximum latent heating',
                'units': 'K/s',
            }
        }
        out_down_dict_attrs = {            
            'nCore_down': {
                'long_name': 'Number of downdraft cores',
                'units': 'count',
            },
            'Area_down': {
                'long_name': 'Total downdraft area',
                'units': 'km^2',
            },
            'MassFlux_down': {
                'long_name': 'Total downdraft mass flux',
                'units': 'kg/s',
            },
            'MeanW_down': {
                'long_name': 'Total downdraft mean vertical wind speed',
                'units': 'm/s',         
            },
            'MeanT_down': {
                'long_name': 'Total downdraft mean temperature',
                'units': 'K',
            },
            'MeanQv_down': {
                'long_name': 'Total downdraft mean water vapor mixing ratio',
                'units': 'kg/kg',
            },
            'MeanThetae_down': {
                'long_name': 'Total downdraft mean equivalent potential temperature',
                'units': 'K',
            },
            'MeanTv_down': {
                'long_name': 'Total downdraft mean virtual temperature',
                'units': 'K',
            },
            'MeanTrho_down': {
                'long_name': 'Total downdraft mean density temperature',
                'units': 'K',
            },
            'MeanRH_down': {
                'long_name': 'Total downdraft mean relative humidity',
                'units': '%',
            },
            'MeanQc_down': {
                'long_name': 'Total downdraft mean cloud water mixing ratio',
                'units': 'kg/kg',
            },
            'MeanQr_down': {
                'long_name': 'Total downdraft mean rain mixing ratio',
                'units': 'kg/kg',
            },
            'MeanQi_down': {
                'long_name': 'Total downdraft mean cloud ice mixing ratio',
                'units': 'kg/kg',
            },
            'MeanQs_down': {
                'long_name': 'Total downdraft mean snow mixing ratio',
                'units': 'kg/kg',
            },
            'MeanQg_down': {
                'long_name': 'Total downdraft mean graupel mixing ratio',
                'units': 'kg/kg',
            },
            'MeanQt_down': {
                'long_name': 'Total downdraft mean total condensate mixing ratio',
                'units': 'kg/kg',
            },
            'MeanNc_down': {
                'long_name': 'Total downdraft mean cloud droplet number concentration',
                'units': '1/kg',
            },
            'MeanNr_down': {
                'long_name': 'Total downdraft mean raindrop number concentration',
                'units': '1/kg',
            },
            'MeanNi_down': {
                'long_name': 'Total downdraft mean cloud ice number concentration',
                'units': '1/kg',
            },
            'MeanNwa_down': {
                'long_name': 'Total downdraft mean water-friendly aerosol concentration',
                'units': '1/kg',
            },
            'MeanNia_down': {
                'long_name': 'Total downdraft mean ice-friendly aerosol concentration',
                'units': '1/kg',
            },
            'MeanHd_down': {
                'long_name': 'Total downdraft mean latent heating',
                'units': 'K/s',
            },
            'MeanP_down': {
                'long_name': 'Total downdraft mean pressure',
                'units': 'Pa',
            },
            'MeanRho_down': {
                'long_name': 'Total downdraft mean dry air density',
                'units': 'kg/m^3',
            },
            'CoreArea_down': {
                'long_name': 'Downdraft core area',
                'units': 'km^2',
            },
            'CoreMassFlux_down': {
                'long_name': 'Downdraft core vertical mass flux',
                'units': 'kg/s',
            },
            'CoreMeanW_down': {
                'long_name': 'Downdraft core mean vertical wind speed',
                'units': 'm/s',
            },
            'CoreMeanT_down': {
                'long_name': 'Downdraft core mean temperature',
                'units': 'K',
            },
            'CoreMeanQv_down': {
                'long_name': 'Downdraft core mean water vapor mixing ratio',
                'units': 'kg/kg',
            },
            'CoreMeanThetae_down': {
                'long_name': 'Downdraft core mean equivalent potential temperature',
                'units': 'K',
            },
            'CoreMeanTv_down': {
                'long_name': 'Downdraft core mean virtual temperature',
                'units': 'K',
            },
            'CoreMeanTrho_down': {
                'long_name': 'Downdraft core mean density temperature',
                'units': 'K',
            },
            'CoreMeanRH_down': {
                'long_name': 'Downdraft core mean relative humidity',
                'units': '%',
            },
            'CoreMeanQc_down': {
                'long_name': 'Downdraft core mean cloud water mixing ratio',
                'units': 'kg/kg',
            },
            'CoreMeanQr_down': {
                'long_name': 'Downdraft core mean rain mixing ratio',
                'units': 'kg/kg',
            },
            'CoreMeanQi_down': {
                'long_name': 'Downdraft core mean cloud ice mixing ratio',
                'units': 'kg/kg',
            },
            'CoreMeanQs_down': {
                'long_name': 'Downdraft core mean snow mixing ratio',
                'units': 'kg/kg',
            },
            'CoreMeanQg_down': {
                'long_name': 'Downdraft core mean graupel mixing ratio',
                'units': 'kg/kg',
            },
            'CoreMeanQt_down': {
                'long_name': 'Downdraft core mean total condensate mixing ratio',
                'units': 'kg/kg',
            },
            'CoreMeanNc_down': {
                'long_name': 'Downdraft core mean cloud droplet number concentration',
                'units': '1/kg',
            },
            'CoreMeanNr_down': {
                'long_name': 'Downdraft core mean raindrop number concentration',
                'units': '1/kg',
            },
            'CoreMeanNi_down': {
                'long_name': 'Downdraft core mean cloud ice number concentration',
                'units': '1/kg',
            },
            'CoreMeanNwa_down': {
                'long_name': 'Downdraft core mean water-friendly aerosol concentration',
                'units': '1/kg',
            },
            'CoreMeanNia_down': {
                'long_name': 'Downdraft core mean ice-friendly aerosol concentration',
                'units': '1/kg',
            },
            'CoreMeanHd_down': {
                'long_name': 'Downdraft core mean latent heating',
                'units': 'K/s',
            },
            'CoreMeanP_down': {
                'long_name': 'Downdraft core mean pressure',
                'units': 'Pa',
            },
            'CoreMeanRho_down': {
                'long_name': 'Downdraft core mean dry air density',
                'units': 'kg/m^3',
            },
            'CoreMinW_down': {
                'long_name': 'Downdraft core minimum vertical wind speed',
                'units': 'm/s',                
            },
            'CoreMinT_down': {
                'long_name': 'Downdraft core minimum temperature',
                'units': 'K',
            },
            'CoreMinQv_down': {
                'long_name': 'Downdraft core minimum water vapor mixing ratio',
                'units': 'kg/kg',
            },
            'CoreMinThetae_down': {
                'long_name': 'Downdraft core minimum equivalent potential temperature',
                'units': 'K',
            },
            'CoreMinTv_down': {
                'long_name': 'Downdraft core minimum virtual temperature',
                'units': 'K',
            },
            'CoreMinTrho_down': {
                'long_name': 'Downdraft core minimum density temperature',
                'units': 'K',
            },
            'CoreMinRH_down': {
                'long_name': 'Downdraft core minimum relative humidity',
                'units': '%',
            },
            'CoreMaxQc_down': {
                'long_name': 'Downdraft core maximum cloud water mixing ratio',
                'units': 'kg/kg',
            },
            'CoreMaxQr_down': {
                'long_name': 'Downdraft core maximum rain mixing ratio',
                'units': 'kg/kg',
            },
            'CoreMaxQi_down': {
                'long_name': 'Downdraft core maximum cloud ice mixing ratio',
                'units': 'kg/kg',
            },
            'CoreMaxQs_down': {
                'long_name': 'Downdraft core maximum snow mixing ratio',
                'units': 'kg/kg',
            },
            'CoreMaxQg_down': {
                'long_name': 'Downdraft core maximum graupel mixing ratio',
                'units': 'kg/kg',
            },
            'CoreMaxQt_down': {
                'long_name': 'Downdraft core maximum total condensate mixing ratio',
                'units': 'kg/kg',
            },
            'CoreMaxNc_down': {
                'long_name': 'Downdraft core maximum cloud droplet number concentration',
                'units': '1/kg',
            },
            'CoreMaxNr_down': {
                'long_name': 'Downdraft core maximum raindrop number concentration',
                'units': '1/kg',
            },
            'CoreMaxNi_down': {
                'long_name': 'Downdraft core maximum cloud ice number concentration',
                'units': '1/kg',
            },
            'CoreMaxNwa_down': {
                'long_name': 'Downdraft core maximum water-friendly aerosol concentration',
                'units': '1/kg',
            },
            'CoreMaxNia_down': {
                'long_name': 'Downdraft core maximum ice-friendly aerosol concentration',
                'units': '1/kg',
            },
            'CoreMinHd_down': {
                'long_name': 'Downdraft core minimum latent heating',
                'units': 'K/s',
            }
        }
        out_non_dict_attrs = {
            'nonCoreArea': {
                'long_name': 'Non-draft area',
                'units': 'km^2',
            },
            'nonCoreMassFlux': {
                'long_name': 'Non-draft vertical mass flux',
                'units': 'kg/s',
            },
            'nonCoreMeanW': {
                'long_name': 'Non-draft mean vertical wind speed',
                'units': 'm/s',
            },
            'nonCoreMeanT': {
                'long_name': 'Non-draft mean temperature',
                'units': 'K',
            },
            'nonCoreMeanQv': {
                'long_name': 'Non-draft mean water vapor mixing ratio',
                'units': 'kg/kg',
            },
            'nonCoreMeanThetae': {
                'long_name': 'Non-draft mean equivalent potential temperature',
                'units': 'K',
            },
            'nonCoreMeanTv': {
                'long_name': 'Non-draft mean virtual temperature',
                'units': 'K',
            },
            'nonCoreMeanTrho': {
                'long_name': 'Non-draft mean density temperature',
                'units': 'K',
            },
            'nonCoreMeanRH': {
                'long_name': 'Non-draft mean relative humidity',
                'units': '%',
            },
            'nonCoreMeanQc': {
                'long_name': 'Non-draft mean cloud water mixing ratio',
                'units': 'kg/kg',
            },
            'nonCoreMeanQr': {
                'long_name': 'Non-draft mean rain mixing ratio',
                'units': 'kg/kg',
            },
            'nonCoreMeanQi': {
                'long_name': 'Non-draft mean cloud ice mixing ratio',
                'units': 'kg/kg',
            },
            'nonCoreMeanQs': {
                'long_name': 'Non-draft mean snow mixing ratio',
                'units': 'kg/kg',
            },
            'nonCoreMeanQg': {
                'long_name': 'Non-draft mean graupel mixing ratio',
                'units': 'kg/kg',
            },
            'nonCoreMeanQt': {
                'long_name': 'Non-draft mean total condensate mixing ratio',
                'units': 'kg/kg',
            },
            'nonCoreMeanNc': {
                'long_name': 'Non-draft mean cloud droplet number concentration',
                'units': '1/kg',
            },
            'nonCoreMeanNr': {
                'long_name': 'Non-draft mean raindrop number concentration',
                'units': '1/kg',
            },
            'nonCoreMeanNi': {
                'long_name': 'Non-draft mean cloud ice number concentration',
                'units': '1/kg',
            },
            'nonCoreMeanNwa': {
                'long_name': 'Non-draft mean water-friendly aerosol concentration',
                'units': '1/kg',
            },
            'nonCoreMeanNia': {
                'long_name': 'Non-draft mean ice-friendly aerosol concentration',
                'units': '1/kg',
            },
            'nonCoreMeanHd': {
                'long_name': 'Non-draft mean latent heating',
                'units': 'K/s',
            },
            'nonCoreMeanP': {
                'long_name': 'Non-draft mean pressure',
                'units': 'Pa',
            },
            'nonCoreMeanRho': {
                'long_name': 'Non-draft mean dry air density',
                'units': 'kg/m^3',
            },
        }
        # import pdb; pdb.set_trace()
    return out_up_dict3d, out_up_dict2d, out_up_dict_attrs, out_down_dict3d, out_down_dict2d, out_down_dict_attrs, out_non_dict2d, out_non_dict_attrs


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
    wrffile_path = config['wrffile_path']
    output_path = config['output_path']
    wrf_filebase = config['wrf_filebase']
    pixel_filebase = config['pixel_filebase']
    ncores_min = config['ncores_min']

    # Add start/end date to pixel file path
    pixelfile_path = f'{pixelfile_path}{startdate}_{enddate}/'

    # Track stats file basename
    stats_filebase = 'stats_tracknumbersv1.0_'

    # Output statistics filename
    updraft_output_filename = f'{output_path}cell_track_updraft_stats_{startdate}_{enddate}.nc'
    downdraft_output_filename = f'{output_path}cell_track_downdraft_stats_{startdate}_{enddate}.nc'
    nondraft_output_filename = f'{output_path}cell_track_nondraft_stats_{startdate}_{enddate}.nc'
    os.makedirs(output_path, exist_ok=True)

    # Track statistics file dimension names
    tracks_dimname = 'tracks'
    times_dimname = 'times'
    z_dimname = 'levels'
    core_dimname = 'core'

    # Track statistics file
    trackstats_file = f'{stats_path}{stats_filebase}{startdate}_{enddate}.nc'
    # Find all pixel-level files
    pixelfilelist = sorted(glob.glob(f'{pixelfile_path}{pixel_filebase}*.nc'))
    nfiles = len(pixelfilelist)
    # Find all WRF files
    wrffilelist = sorted(glob.glob(f'{wrffile_path}*/*_3d'))
    nwrffiles = len(wrffilelist)
    print(f'Number of WRF files: {nwrffiles}')
    
    # Get basetime from pixel files
    pixel_basetime, pixelfile_dict = calc_basetime(pixelfilelist, pixel_filebase, fmt='yyyymmdd.hhmm')
    # Get basetime from WRF files
    wrf_basetime, wrffile_dict = calc_basetime(wrffilelist, wrf_filebase, fmt='yyyy.mm.dd.hh.mm')

    # Find matching WRF files for each pixel file
    match_wrffilelist = [''] * nfiles
    for ifile in range(nfiles):
        # Find WRF time closest to the pixel file time and get the index
        # Save the filename if time difference is < time_window
        idx = np.argmin(np.abs(wrf_basetime - pixel_basetime[ifile]))        
        if np.abs(wrf_basetime[idx] - pixel_basetime[ifile]) < time_window:
            match_wrffilelist[ifile] = wrffile_dict[wrf_basetime[idx]]
        else:
            print(f'No match file found for: {pixelfilelist[ifile]}')

    # Read track statistics file
    print(trackstats_file)
    dsstats = xr.open_dataset(trackstats_file, decode_times=False)
    ntracks = dsstats.dims[tracks_dimname]
    ntimes = dsstats.dims[times_dimname]
    ntimes_short = 30
    stats_basetime = dsstats['basetime'][:,:ntimes_short].data
    stats_basetime_attrs = dsstats['basetime'].attrs
    # cell_area = dsstats['cell_area']
    pixel_radius = dsstats.attrs['pixel_radius_km']
    dsstats.close()

    print(f'Total Number of Tracks: {ntracks}')

    # # Read a WRF file to get number of vertical levels
    # dsm = xr.open_dataset(match_wrffilelist[0])
    # nz = dsm.dims['bottom_top']
    # height = dsm['CAC_GPH'][0,:,:,:]
    # dsm.close()
    # Define constant height levels (WRF values are variable)
    zlevs_lo = (np.arange(40)+1)*125 #125 m steps between 0 and 5000 m
    zlevs_hi = (np.arange(40)+1)*250 + 5000 #200 m steps between 5000 and 15000 m
    zlevs = np.concatenate((zlevs_lo, zlevs_hi))
    height = xr.DataArray(data=zlevs, dims=["level"], coords=dict(), attrs=dict(description="Geopotential height", units="m")) #convert to xarray DataArray
    nz = len(height)

    ##############################################################
    # Call function to calculate statistics
    trackindices_all = []
    timeindices_all = []
    final_results = []

    if run_parallel == 1:
        # Initialize dask
        dask_tmp_dir = config.get("dask_tmp_dir", "/tmp")
        dask.config.set({'temporary-directory': dask_tmp_dir})
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker)
        client = Client(cluster)

    # Loop over each pixel-file and call function to calculate
    for ifile in range(nfiles):
    #for ifile in range(300):
        # Find all matching time indices from track stats file to the current pixel file
        matchindices = np.array(
            np.where(np.abs(stats_basetime - pixel_basetime[ifile]) < time_window)
        )
        # The returned match indices are for [tracks, times] dimensions respectively
        idx_track = matchindices[0]
        idx_time = matchindices[1]

        if len(idx_track) > 0:
            # Save matchindices for the current pixel file to the overall list
            trackindices_all.append(idx_track)
            timeindices_all.append(idx_time)
            # Serial
            if run_parallel == 0:
                iresult = calc_cellstats_singlefile(
                    pixelfilelist[ifile], 
                    match_wrffilelist[ifile],
                    idx_track,
                    config,
                )
            # Parallel
            elif run_parallel == 1:
                iresult = dask.delayed(calc_cellstats_singlefile)(
                    pixelfilelist[ifile], 
                    match_wrffilelist[ifile],
                    idx_track, 
                    config,
                )
            final_results.append(iresult)
    
    if run_parallel == 1:
        # Trigger Dask computation
        print("Computing statistics ...")
        final_results = dask.compute(*final_results)


    # Make a variable list and get attributes from one of the returned dictionaries
    # Loop over each return results till one that is not None
    counter = len(final_results)-1
    while counter >= 0:
        if final_results[counter] is not None:
            upvar_names3d = list(final_results[counter][0].keys())
            upvar_names2d = list(final_results[counter][1].keys())
            upvar_attrs = final_results[counter][2]
            downvar_names3d = list(final_results[counter][3].keys())
            downvar_names2d = list(final_results[counter][4].keys())
            downvar_attrs = final_results[counter][5]
            nonvar_names2d = list(final_results[counter][6].keys())
            nonvar_attrs = final_results[counter][7]
            break
        counter -= 1

    # Updrafts
    # Loop over variable list to create the dictionary entry
    print(f'Creating updraft output arrays ...')
    out_dict = {}
    out_dict_attrs = {}

    var_names = upvar_names3d + upvar_names2d
    # 3D variables 
    for ivar in upvar_names3d:
        out_dict[ivar] = np.full((ntracks, ntimes_short, nz, ncores_min), np.nan, dtype=np.float32)
        out_dict_attrs[ivar] = upvar_attrs[ivar]
    # 2D variables
    for ivar in upvar_names2d:
        out_dict[ivar] = np.full((ntracks, ntimes_short, nz), np.nan, dtype=np.float32)
        out_dict_attrs[ivar] = upvar_attrs[ivar]

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
                for ivar in upvar_names3d:
                    if iVAR3d[ivar].ndim == 3:
                        out_dict[ivar][trackindices,timeindices,:,:] = iVAR3d[ivar]
                    else:
                        print(f'Warning: {ivar} dimension is not 3.')
            if iVAR2d is not None:
                trackindices = trackindices_all[ifile]
                timeindices = timeindices_all[ifile]
                # Loop over each variable and assign values to output dictionary
                for ivar in upvar_names2d:
                    if iVAR2d[ivar].ndim == 2:
                        out_dict[ivar][trackindices,timeindices,:] = iVAR2d[ivar]
                    else:
                        print(f'Warning: {ivar} dimension is not 2.')

    ##########################################################
    # Write to netcdf
    print('Writing updraft output netcdf ... ')

    # Define variable list
    var_dict = {}
    # Define output variable dictionary
    for key, value in out_dict.items():
        if np.ndim(value) == 2:
            var_dict[key] = ([tracks_dimname, times_dimname], value, out_dict_attrs[key])
        if np.ndim(value) == 3:
            var_dict[key] = ([tracks_dimname, times_dimname, z_dimname], value, out_dict_attrs[key])
        if np.ndim(value) == 4:
            var_dict[key] = ([tracks_dimname, times_dimname, z_dimname, core_dimname], value, out_dict_attrs[key])
    # Add base_time from track stats to the output dictionary
    out_dict['basetime'] = ([tracks_dimname, times_dimname], stats_basetime, dsstats['basetime'].attrs)
    # Define coordinate list
    core_dim_attrs = {
        'long_name': 'Core number',
    }
    coord_dict = {
        tracks_dimname: ([tracks_dimname], np.arange(0, ntracks)),
        times_dimname: ([times_dimname], np.arange(0, ntimes_short)),
        z_dimname: ([z_dimname], height.data, height.attrs),
        core_dimname: ([core_dimname], np.arange(0, ncores_min), core_dim_attrs),
    }
    # Define global attributes
    gattr_dict = {
        'title':  'Tracked cell updraft statistics', \
        'Institution': 'Pacific Northwest National Laboratoy', \
        'Contact': 'Zhe Feng, zhe.feng@pnnl.gov; Adam Varble, adam.varble@pnnl.gov', \
        'Created_on':  time.ctime(time.time()), \
        'source_trackfile': trackstats_file, \
        'startdate': startdate, \
        'enddate': enddate, \
    }
    # Define xarray dataset
    dsout = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)

    # Delete file if it already exists
    if os.path.isfile(updraft_output_filename):
        os.remove(updraft_output_filename)
        
    # Set encoding/compression for all variables
    comp = dict(zlib=True)
    encoding = {var: comp for var in dsout.data_vars}

    # Write to netcdf file
    dsout.to_netcdf(path=updraft_output_filename, mode="w",
                    format="NETCDF4", unlimited_dims=tracks_dimname, encoding=encoding)
    print(f'Output saved: {updraft_output_filename}')
    
    
    # Downdrafts
    # Loop over variable list to create the dictionary entry
    print(f'Creating downdraft output arrays ...')
    out_dict = {}
    out_dict_attrs = {}

    var_names = downvar_names3d + downvar_names2d
    # 3D variables 
    for ivar in downvar_names3d:
        out_dict[ivar] = np.full((ntracks, ntimes_short, nz, ncores_min), np.nan, dtype=np.float32)
        out_dict_attrs[ivar] = downvar_attrs[ivar]
    # 2D variables
    for ivar in downvar_names2d:
        out_dict[ivar] = np.full((ntracks, ntimes_short, nz), np.nan, dtype=np.float32)
        out_dict_attrs[ivar] = downvar_attrs[ivar]
        
    # Put the results to output track stats variables
    # Loop over each returned results
    for ifile in range(len(final_results)):
        # Check the return results
        if final_results[ifile] is not None:
            iVAR3d = final_results[ifile][3]
            iVAR2d = final_results[ifile][4]
            if iVAR3d is not None:
                trackindices = trackindices_all[ifile]
                timeindices = timeindices_all[ifile]
                # Loop over each variable and assign values to output dictionary
                for ivar in downvar_names3d:
                    if iVAR3d[ivar].ndim == 3:
                        out_dict[ivar][trackindices,timeindices,:,:] = iVAR3d[ivar]
                    else:
                        print(f'Warning: {ivar} dimension is not 3.')
            if iVAR2d is not None:
                trackindices = trackindices_all[ifile]
                timeindices = timeindices_all[ifile]
                # Loop over each variable and assign values to output dictionary
                for ivar in downvar_names2d:
                    if iVAR2d[ivar].ndim == 2:
                        out_dict[ivar][trackindices,timeindices,:] = iVAR2d[ivar]
                    else:
                        print(f'Warning: {ivar} dimension is not 2.')
        
    ##########################################################
    # Write to netcdf
    print('Writing downdraft output netcdf ... ')

    # Define variable list
    var_dict = {}
    # Define output variable dictionary
    for key, value in out_dict.items():
        if np.ndim(value) == 2:
            var_dict[key] = ([tracks_dimname, times_dimname], value, out_dict_attrs[key])
        if np.ndim(value) == 3:
            var_dict[key] = ([tracks_dimname, times_dimname, z_dimname], value, out_dict_attrs[key])
        if np.ndim(value) == 4:
            var_dict[key] = ([tracks_dimname, times_dimname, z_dimname, core_dimname], value, out_dict_attrs[key])
    # Add base_time from track stats to the output dictionary
    out_dict['basetime'] = ([tracks_dimname, times_dimname], stats_basetime, dsstats['basetime'].attrs)
    # Define coordinate list
    core_dim_attrs = {
        'long_name': 'Core number',
    }
    coord_dict = {
        tracks_dimname: ([tracks_dimname], np.arange(0, ntracks)),
        times_dimname: ([times_dimname], np.arange(0, ntimes_short)),
        z_dimname: ([z_dimname], height.data, height.attrs),
        core_dimname: ([core_dimname], np.arange(0, ncores_min), core_dim_attrs),
    }
    # Define global attributes
    gattr_dict = {
        'title':  'Tracked cell downdraft statistics', \
        'Institution': 'Pacific Northwest National Laboratoy', \
        'Contact': 'Zhe Feng, zhe.feng@pnnl.gov; Adam Varble, adam.varble@pnnl.gov', \
        'Created_on':  time.ctime(time.time()), \
        'source_trackfile': trackstats_file, \
        'startdate': startdate, \
        'enddate': enddate, \
    }
    # Define xarray dataset
    dsout = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)

    # Delete file if it already exists
    if os.path.isfile(downdraft_output_filename):
        os.remove(downdraft_output_filename)
        
    # Set encoding/compression for all variables
    comp = dict(zlib=True)
    encoding = {var: comp for var in dsout.data_vars}

    # Write to netcdf file
    dsout.to_netcdf(path=downdraft_output_filename, mode="w",
                    format="NETCDF4", unlimited_dims=tracks_dimname, encoding=encoding)
    print(f'Output saved: {downdraft_output_filename}')
    
    
    # Nondraft Regions
    # Loop over variable list to create the dictionary entry
    print(f'Creating nondraft output arrays ...')
    out_dict = {}
    out_dict_attrs = {}

    var_names = nonvar_names2d
    # 2D variables
    for ivar in nonvar_names2d:
        out_dict[ivar] = np.full((ntracks, ntimes_short, nz), np.nan, dtype=np.float32)
        out_dict_attrs[ivar] = nonvar_attrs[ivar]

    # Put the results to output track stats variables
    # Loop over each returned results
    for ifile in range(len(final_results)):
        # Check the return results
        if final_results[ifile] is not None:
            iVAR2d = final_results[ifile][6]
            if iVAR2d is not None:
                trackindices = trackindices_all[ifile]
                timeindices = timeindices_all[ifile]
                # Loop over each variable and assign values to output dictionary
                for ivar in nonvar_names2d:
                    if iVAR2d[ivar].ndim == 2:
                        out_dict[ivar][trackindices,timeindices,:] = iVAR2d[ivar]
                    else:
                        print(f'Warning: {ivar} dimension is not 2.')

    ##########################################################
    # Write to netcdf
    print('Writing nondraft output netcdf ... ')

    # Define variable list
    var_dict = {}
    # Define output variable dictionary
    for key, value in out_dict.items():
        if np.ndim(value) == 2:
            var_dict[key] = ([tracks_dimname, times_dimname], value, out_dict_attrs[key])
        if np.ndim(value) == 3:
            var_dict[key] = ([tracks_dimname, times_dimname, z_dimname], value, out_dict_attrs[key])
    # Add base_time from track stats to the output dictionary
    out_dict['basetime'] = ([tracks_dimname, times_dimname], stats_basetime, dsstats['basetime'].attrs)
    # Define coordinate list
    coord_dict = {
        tracks_dimname: ([tracks_dimname], np.arange(0, ntracks)),
        times_dimname: ([times_dimname], np.arange(0, ntimes_short)),
        z_dimname: ([z_dimname], height.data, height.attrs),
    }
    # Define global attributes
    gattr_dict = {
        'title':  'Tracked cell non draft region statistics', \
        'Institution': 'Pacific Northwest National Laboratoy', \
        'Contact': 'Zhe Feng, zhe.feng@pnnl.gov; Adam Varble, adam.varble@pnnl.gov', \
        'Created_on':  time.ctime(time.time()), \
        'source_trackfile': trackstats_file, \
        'startdate': startdate, \
        'enddate': enddate, \
    }
    # Define xarray dataset
    dsout = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)

    # Delete file if it already exists
    if os.path.isfile(nondraft_output_filename):
        os.remove(nondraft_output_filename)
        
    # Set encoding/compression for all variables
    comp = dict(zlib=True)
    encoding = {var: comp for var in dsout.data_vars}

    # Write to netcdf file
    dsout.to_netcdf(path=nondraft_output_filename, mode="w",
                    format="NETCDF4", unlimited_dims=tracks_dimname, encoding=encoding)
    print(f'Output saved: {nondraft_output_filename}')