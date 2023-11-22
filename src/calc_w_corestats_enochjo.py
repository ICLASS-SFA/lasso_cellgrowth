"""
Calculates 3D W core statistics from gridded Met data for tracked convective cells.
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
from scipy.ndimage import generate_binary_structure, binary_dilation,iterate_structure
from skimage.measure import label
from skimage.segmentation import expand_labels

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
        # File name format: basename_yyyymmdd.hhmmss # EJ
        TEMP_filetime = datetime(
            int(fname[prelength:(prelength+4)]), 
            int(fname[prelength+4:(prelength+6)]), 
            int(fname[prelength+6:(prelength+8)]),
            int(fname[prelength+9:(prelength+11)]), 
            int(fname[prelength+11:(prelength+13)]),
            int(fname[prelength+13:(prelength+15)]), tzinfo=utc
        )
        # file_basetime[ifile] = calendar.timegm(TEMP_filetime.timetuple())
        file_basetime[ifile] = TEMP_filetime.timestamp()
        file_dict[file_basetime[ifile]] = filelist[ifile]
    return file_basetime, file_dict

# #-----------------------------------------------------------------------
# def convert_lasso_times(data_path, data_basename):
#     """
#     Convert LASSO regridded file times to Epoch time.

#     Args:
#         data_path: string
#             Input data path.
#         data_basename: string
#             Input data basename.
    
#     Returns:
#         file_basetime: np.array
#             Epoch time corresponding to the input files.
#         file_dict: dictionary
#             Direction key by basetime and value is the file names
#     """
#     # Isolate all possible files
#     filenames = sorted(fnmatch.filter(os.listdir(data_path), data_basename + '*'))
#     nfiles = len(filenames)
#     # Make array to store basetime
#     file_basetime = np.zeros(nfiles, dtype=int)
#     file_dict = {}

#     # Get start time from the file name
#     # e.g., corlasso_sub_metOnHamsl.M1.m1.gefs18_2018120400_f143000_d3.nc
#     # start time: 2018120400
#     # forecast time: 143000
#     nleadingchar = len(data_basename)
#     start_datetime = filenames[0][nleadingchar:nleadingchar+10]
#     syear = start_datetime[0:4]
#     smonth = start_datetime[4:6]
#     sday = start_datetime[6:8]
#     shour = start_datetime[8:10]
#     start_time = datetime(
#         int(syear), int(smonth), int(sday), int(shour), tzinfo=utc,
#     )
#     # Get forecast times from each file name
#     nleadingchar_fxtime = nleadingchar + len(start_datetime) + 2
#     for ii in range(0, nfiles):
#         fx_time = filenames[ii][nleadingchar_fxtime:nleadingchar_fxtime+6]
#         fx_hour = int(fx_time[0:2])
#         fx_min = int(fx_time[2:4])
#         fx_sec = int(fx_time[4:6])
#         # Add forecast time to start time to get the real time
#         rtime = start_time + timedelta(hours=fx_hour, minutes=fx_min, seconds=fx_sec)
#         # Convert to Epoch time (base time)
#         file_basetime[ii] = rtime.timestamp()
#         file_dict[file_basetime[ii]] = data_path + filenames[ii]
    
#     return file_basetime, file_dict

#-----------------------------------------------------------------------
def label_cores(W, W_thresh, Q, Q_thresh, VMF, ncores_min, min_core_npix, method='>'):
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
    struct = generate_binary_structure(2, 1).astype(int)
    if method == '>':
        core_label, ncores = ndimage.label((W > W_thresh) & (Q > Q_thresh),structure = struct) # EJ
    elif method == '<':
        core_label, ncores = ndimage.label((W < W_thresh) & (Q < Q_thresh),structure = struct) # EJ
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

#     if ( len(core_numbers) > 0 ) & ( method == '>' ):
#         import pdb;pdb.set_trace()
    
    zVMF = np.zeros_like(npix_core)
    for i in range(0,len(core_numbers)):
        zVMF[i] = np.nansum(VMF[core_label == core_numbers[i]]) # Find VMF of cores
    
    # Sort the core size by descending order
    # sort_idx = npix_core.argsort()[::-1] # Original Method of sorting by area
    sort_idx = zVMF.argsort()[::-1] # New Method of sorting by VMF
    npix_core_sorted = npix_core[sort_idx]
    core_numbers_sorted = core_numbers[sort_idx]
    
    
    
    # Save the largest X cores
    ncores_all = len(npix_core)
    ncores_save = np.nanmin([ncores_all, ncores_min])
    
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

#-----------------------------------------------------------------------
def calc_cellstats_singlefile(
    pixel_filename, 
    met_filename,
    ent_filename, 
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
        ent_filename: string (EJ)
            ENT filename
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
    print(pixel_filename)
    print(met_filename)
    print(ent_filename)

    # Get thresholds from config
    W_up_thresh = config['W_up_thresh']
    W_down_thresh = config['W_down_thresh']
    Q_up_thresh = config['Q_up_thresh'] # EJ
    Q_down_thresh = config['Q_down_thresh'] # EJ
    min_core_npix = config['min_core_npix']
    ncores_min = config['ncores_min']
    geolimits = config.get('geolimits', None)

    # Read MET file
    dsm = xr.open_dataset(met_filename)
    # Rename dimenensions
    dsm = dsm.rename_dims({'south_north':'lat', 'west_east':'lon'})
    nz = dsm.dims['HAMSL']
    ny = dsm.dims['lat']
    nx = dsm.dims['lon']
    height = dsm['HAMSL'].data
    DX = dsm.attrs['DX']
    DY = dsm.attrs['DY']
    grid_area = DX * DY / 1e6
        
    # Read ENT file (EJ)
    # No need to rename dimensions as they are already 'lat' and 'lon'
    dse = xr.open_dataset(ent_filename)
    dse = dse.rename_dims({'time':'Time','hgt':'HAMSL'})    
    dse = dse.drop('time')
    dse = dse.assign_coords(Time=dsm.coords['Time'].data) 
    EntrDetr = dse['entr_detr']

    # Separate the net entrainment file to entrainment and detrainment (EJ)
    Entr = EntrDetr.where(EntrDetr > 0)
    Detr = EntrDetr.where(EntrDetr < 0)

    # Read pixel-level track file
    ds = xr.open_dataset(pixel_filename, decode_times=False, mask_and_scale=False)
    time_pixel = ds['time']
    ny_p = ds.dims['lat']
    nx_p = ds.dims['lon']
    
#     import pdb
#     pdb.set_trace()

    # Check dimensions between MET and pixel files
    # (EJ) will skip this step for now, as 'XLAT','XLONG' does not exist in ENT files.
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
        PRESSURE = dsm['PRESSURE'][:, :, ymin:ymax+1, xmin:xmax+1]
        TV = dsm['tv'][:, :, ymin:ymax+1, xmin:xmax+1] # EJ
        WA = dsm['WA'][:, :, ymin:ymax+1, xmin:xmax+1]
        #qc = dsm['QCLOUD'][:, :, ymin:ymax+1, xmin:xmax+1] # EJ
        #qi = dsm['QICE'][:, :, ymin:ymax+1, xmin:xmax+1] # EJ
        #qs = dsm['QSNOW'][:, :, ymin:ymax+1, xmin:xmax+1] # EJ
        qv = dsm['QVAPOR'][:, :, ymin:ymax+1, xmin:xmax+1] # EJ
        #qr = dsm['QRAIN'][:, :, ymin:ymax+1, xmin:xmax+1] # EJ
        #qg = dsm['QGRAUP'][:, :, ymin:ymax+1, xmin:xmax+1] # EJ
        Thte = dsm['THETA_E'][:, :, ymin:ymax+1, xmin:xmax+1] # EJ
        TH = dsm['THETA'][:, :, ymin:ymax+1, xmin:xmax+1] # EJ
        QR = dsm['QRAIN'][:, :, ymin:ymax+1, xmin:xmax+1] # EJ
        QC = dsm['QCLOUD'][:, :, ymin:ymax+1, xmin:xmax+1] # EJ
        QT = dsm['QT'][:, :, ymin:ymax+1, xmin:xmax+1] # EJ
        dBZ = dsm['REFL_10CM'][:, :, ymin:ymax+1, xmin:xmax+1] # EJ
        # Update ny, nx with the subset
        ny = XLONG.sizes['lat']
        nx = XLONG.sizes['lon']
    else:
        XLONG = dsm['XLONG']
        XLAT = dsm['XLAT']
        PRESSURE = dsm['PRESSURE'] 
        TV = dsm['tv'] # EJ
        WA = dsm['WA']
        dBZ = dsm['REFL_10CM'] # EJ
        #qc = dsm['QCLOUD'] # EJ
        #qi = dsm['QICE'] # EJ
        #qs = dsm['QSNOW'] # EJ
        qv = dsm['QVAPOR'] # EJ
        #qr = dsm['QRAIN'] # EJ
        #qg = dsm['QGRAUP'] # EJ
        Thte = dsm['THETA_E'] # EJ
        TH = dsm['THETA'] # EJ
        QR = dsm['QRAIN'] # EJ
        QC = dsm['QCLOUD'] # EJ
        QT = dsm['QT'] # EJ

    # Check dimensions again after subset
    if (ny_p != ny) | (nx_p != nx):
        print(f'ERROR: Inconsistent number of grids between pixel-level and MET files.')
        print(f'ny: {ny}, ny_pixel: {ny_p}, nx: {nx}, nx_pixel: {nx_p}')
        sys.exit()

    # Drop 1D lat/lon coordinates, and reasign 2D XLONG/XLAT coordinates from Met file
    # It does not seem like this is necessary in Xarray 0.21.1
    ds = ds.drop_vars(['lon', 'lat']).assign_coords({'XLONG':XLONG, 'XLAT':XLAT})
#     tracknumbermap = ds['tracknumber_expand'].squeeze() # EJ (Before)
    tracknumbermap = ds['conv_core_label'].squeeze() # EJ now using stringent criteria as we are including updrafts that overlap with boundary

    # Calculate moist air density using virtual temperature
    R_dry = 287.058   # J kg−1 K−1
    Mrho = 100 * PRESSURE / (R_dry * TV)  # kg m-3
    
    QA = QC + QR
    
    # import pdb; pdb.set_trace()
    
    # Calculate vertical pressure gradient
    # dpdz = np.zeros_like(PRESSURE)
    # dpdz[0,2:-2,:,:] = ( PRESSURE[0,2:,:,:]-PRESSURE[0,:-2,:,:] ) / 200 # EJ hard-coded 2*dz as attribute does not exist
    # vpgf = - 1/Mrho * dpdz
    
    # Calculate Inflow of qv
    Vapr = EntrDetr.where( (EntrDetr > 0) ) * qv
    
    # Calculate Virtual Potential Temperature # EJ
    Thtv = TH * (qv + 0.622)/(0.622 * (1 + qv))
    
    # Calculate Temperature (AMS)
    Temp = TH*(PRESSURE/1000)**(2/7) # Remember that PRESSURE is in hPa
    
    # Calculate Saturated Vapor Pressure (NWS)
    # es = 6.11*10**((7.5*Temp)/(237.3+Temp))
    
    # Calculate Saturated Mixing Ratio (NWS)
    # ws = 621.97*(es/(PRESSURE-es))
    
    # Calculate Density Temperature (Eqn. 4.3.6 of some Emanuel textbook)
    # "Note that Tv is a special case of Trho, since when condensed water is absent rT = r."
    Trho = Temp*(1 + qv/0.622)/(1 + QT)

    # Calculate mass flux (kg m-2 s-1)
    MassFlux = (Mrho * WA).squeeze()

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
        cell_ovlap_up = np.full(dims2d, np.NaN, dtype=np.float32) # EJ
        cell_MassFlux_up = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_CoreMassFlux_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreArea_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMaxW_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanW_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMaxQC_up = np.full(dims3d, np.NaN, dtype=np.float32) # EJ
        cell_CoreMeanQC_up = np.full(dims3d, np.NaN, dtype=np.float32) # EJ
        cell_CoreMeanQC_prm = np.full(dims3d, np.NaN, dtype=np.float32) # EJ
        cell_CoreMaxQR_up = np.full(dims3d, np.NaN, dtype=np.float32) # EJ
        cell_CoreMeanQR_up = np.full(dims3d, np.NaN, dtype=np.float32) # EJ
        cell_CoreMeanQV_up = np.full(dims3d, np.NaN, dtype=np.float32) # EJ
        cell_CoreMeanQV_prm = np.full(dims3d, np.NaN, dtype=np.float32) # EJ

        cell_nCore_down = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_MassFlux_down = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_CoreMassFlux_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreArea_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMinW_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanW_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMinQ_down = np.full(dims3d, np.NaN, dtype=np.float32) # EJ
        cell_CoreMeanQ_down = np.full(dims3d, np.NaN, dtype=np.float32) # EJ
        
        # (EJ) new 3d variables for entrainment
        cell_Entr_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_Detr_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_Vapr_up = np.full(dims3d, np.NaN, dtype=np.float32)
        
        # (EJ) new arrays for other things
        cell_dBZ_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_ThteMax_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_ThteMean_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_ThteMean_prm = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_ThtvMax_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_ThtvMean_prm = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_BuoyThtv_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_TrhoMax_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_TrhoMean_prm = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_BuoyTrho_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_PGF_up = np.full(dims3d, np.NaN, dtype=np.float32) # EJ
        
        # This section of code used to determine how large the sub-domain needs to be
        cell_cloudy = np.full(QA.shape, 0, dtype=np.float32)     # Create zero array
        icloud = (WA > W_up_thresh) & (QA > Q_up_thresh)         # Find cloudy region
        cell_cloudy[icloud] = 1                                  # Set cloudy region to 1
        zcloud = np.nanmax(cell_cloudy,axis=1).squeeze()         # Reduce to 2D
        tracknumbermap_mod = (zcloud + tracknumbermap).data # Merge the cloudy regions with tmap
        tracknumbermap_mod[tracknumbermap_mod > 0] = 1           # binarize and re-label           
        tracknumbermap_label = xr.DataArray(label(tracknumbermap_mod),coords=tracknumbermap.coords, dims=tracknumbermap.dims )
        
        # Loop over each match tracked cell
        for icell in range(nmatchcell):
            # Track number needs to add 1
            itracknum = idx_track[icell] + 1

            # Count the number of pixels for the original cell mask
            inpix_cloud = np.count_nonzero(tracknumbermap == itracknum)

            # Proceed if the number matching cloud pixel > 0
            if inpix_cloud > 0:
                
                # Need to find a (any) cell index corresponding to the current itracknum
                ind_tmap = np.where(tracknumbermap == itracknum)
                
                # Find the corresponding cell index in the re-labelled tracknumbermap
                correct = tracknumbermap_label.data[ind_tmap[0][0],ind_tmap[1][0]]
                
                # Then, carve out the appropriate region in the modified tracknumbermap
                # e.g., where(tracknumbermap_label == correct)
                
                # Setting the largest-possible sub-domain as 1.
                ind = np.where(tracknumbermap_label == correct)
                tracknumbermap_evo = np.zeros_like(tracknumbermap)
                tracknumbermap_evo[ind[0].min():ind[0].max(),ind[1].min():ind[1].max()] = 1
                tracknumbermap_final = xr.DataArray(tracknumbermap_evo,coords=tracknumbermap.coords, dims=tracknumbermap.dims )
                
#                 import pdb; pdb.set_trace()
#                 from matplotlib import pyplot as plt
#         
#                 tmp = np.zeros_like(tracknumbermap_label.data)
#                 ind = np.where(tracknumbermap_final == 1)
#                 tmp[ind] = 1
#         
#                 plt.clf
#                 f1 = plt.figure(figsize=(5, 5))
#                 pm = plt.pcolormesh(tmp[1500:1800,400:600])
#                 plt.colorbar(pm)
#                 plt.savefig('/ccsopen/home/enochjo/test.png')
                                
                # Subset 3D variables to the current cell mask
                iW = WA.where(tracknumbermap_final == 1, drop=True).squeeze().data
                iQ = QA.where(tracknumbermap_final == 1, drop=True).squeeze().data
                iQC = QC.where(tracknumbermap_final == 1, drop=True).squeeze().data # EJ
                iQR = QR.where(tracknumbermap_final == 1, drop=True).squeeze().data # EJ
                iQV = qv.where(tracknumbermap_final == 1, drop=True).squeeze().data # EJ
                iMassFlux = MassFlux.where(tracknumbermap_final == 1, drop=True).squeeze().data  
                idBZ = dBZ.where(tracknumbermap_final == 1, drop=True).squeeze().data # EJ 
                iThte = Thte.where(tracknumbermap_final == 1, drop=True).squeeze().data # EJ 
                iThtv = Thtv.where(tracknumbermap_final == 1, drop=True).squeeze().data # EJ 
                iVapr = Vapr.where(tracknumbermap_final == 1, drop=True).squeeze().data # EJ 
                iEntr = Entr.where(tracknumbermap_final == 1, drop=True).squeeze().data # EJ
                iDetr = Detr.where(tracknumbermap_final == 1, drop=True).squeeze().data # EJ
                iTrho = Trho.where(tracknumbermap_final == 1, drop=True).squeeze().data # EJ
                iPres = PRESSURE.where(tracknumbermap_final == 1, drop=True).squeeze().data # EJ
                iMrho = Mrho.where(tracknumbermap_final == 1, drop=True).squeeze().data # EJ
                zTnum = tracknumbermap.where(tracknumbermap_final == 1, drop=True).squeeze().data

                # Calculate new statistics of the cell
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                                        
                    # Loop over vertical levels
                    for z in range(0, nz):

                        zW = iW[z,:,:]
                        zQ = iQ[z,:,:] #EJ
                        zQC = iQC[z,:,:] #EJ
                        zQR = iQR[z,:,:] #EJ
                        zQV = iQV[z,:,:] #EJ
                        zMassFlux = iMassFlux[z,:,:]
                        zdBZ = idBZ[z,:,:] #EJ
                        zThte = iThte[z,:,:] #EJ
                        zThtv = iThtv[z,:,:] #EJ
                        zEntr = iEntr[z,:,:] #EJ
                        zDetr = iDetr[z,:,:] #EJ
                        zVapr = iVapr[z,:,:] #EJ
                        zTrho = iTrho[z,:,:] #EJ
                        zMrho = iMrho[z,:,:] #EJ
                        
                        zz = z
                        if (z < 1): zz = 1
                        if (z > 98): zz = 98
                        
                        zPrs1 = iPres[zz-1,:,:] *100 #EJ
                        zPrs2 = iPres[zz+1,:,:] *100 #EJ
                        
#                         from matplotlib import pyplot as plt
#                         import pdb; pdb.set_trace()
#                         plt.clf
#                         f1 = plt.figure(figsize=(5, 5))
#                         pm = plt.pcolormesh(zdBZ)
#                         plt.colorbar(pm)
#                         plt.savefig('/ccsopen/home/enochjo/test.png')
                        
                        cell_cloudy = np.full(zW.shape, 0, dtype=np.float32)     # Create zero array
                        icloud = (zW > W_up_thresh) & (zQ > Q_up_thresh)         # Find cloudy region
                        cell_cloudy[icloud] = 1                                  # Set cloudy region to 1
                        tracknumbermap_mod = (cell_cloudy + zTnum)               # Merge the cloudy regions with tmap
                        tracknumbermap_mod[tracknumbermap_mod > 0] = 1           # binarize and re-label        
                        tmap_label = label(tracknumbermap_mod)
                        
                        zW_mask = np.zeros_like(zW)
                        ind = np.where(zTnum == itracknum)
                        ind_conv = tmap_label[ind[0][0],ind[1][0]]
                        ind = tmap_label == ind_conv
                        zW_mask[ind] = 1
                        
                        # Label updraft cores
                        dict_up = label_cores(zW*zW_mask, W_up_thresh, zQ, Q_up_thresh, zMassFlux, ncores_min, min_core_npix, method='>')
                        ncores_all_up = dict_up['ncores_all']
                        ncores_up = dict_up['ncores_save']
                        core_npix_up = dict_up['core_npix']
                        core_numbers_up = dict_up['core_numbers']
                        core_label_up = dict_up['core_label']
                        # Label downdraft cores
                        dict_down = label_cores(zW, W_down_thresh, zQ, Q_down_thresh, zMassFlux, ncores_min, min_core_npix, method='<')
                        ncores_all_down = dict_down['ncores_all']
                        ncores_down = dict_down['ncores_save']
                        core_npix_down = dict_down['core_npix']
                        core_numbers_down = dict_down['core_numbers']
                        core_label_down = dict_down['core_label']

                        # Dilate core labels by a certain number of pixels
                        core_label_up_prm = np.zeros_like(core_label_up)
                        for ii in range(ncores_up):
                            cell = np.zeros_like(core_label_up)
                            cell[core_label_up == core_numbers_up[ii]] = 1
                            expand = round(np.sqrt(core_npix_up[ii]/np.pi))
                            dil = expand_labels(cell, distance = expand)
                            core_label_up_prm[(dil - cell) == 1] = core_numbers_up[ii]
                            
                        # Getting rid of all the perimeter labels that exist within adjacent cores.
                        # core_label_up_prm[core_label_up > 0] = 0 # Don't do this
                        
#                         # The old way of doing things
#                         core_label_up_dil = np.zeros_like(core_label_up)
#                         core_label_up_prm = np.zeros_like(core_label_up)
#                         for ii in range(ncores_up):
#                             cell = np.zeros_like(core_label_up)
#                             cell[core_label_up == core_numbers_up[ii]] = 1
#                             dil = binary_dilation(cell, structure = struct, iterations = 2)
#                             core_label_up_dil[dil == 1] = core_numbers_up[ii] 
#                             core_label_up_prm[(dil - cell) == 1] = core_numbers_up[ii]

                        # Dilation needed just in case entrainment/detrainment has extra values adjacent to cores
                        core_label_up_dil = expand_labels(core_label_up, distance = 2) # Includes core
                        # core_label_up_prm = expand_labels(core_label_up, distance = 2) - core_label_up # Just perimeter
                        # ^^ Not needed for now as we are dynamically adjusting the perimeter.
                        
#                             from matplotlib import pyplot as plt
#                             plt.clf
#                             f1 = plt.figure(figsize=(5, 5))
#                             pm = plt.pcolormesh(core_label_up_prm)
#                             plt.colorbar(pm)
#                             plt.savefig('/ccsopen/home/enochjo/test.png')
                        
                        # Total number of cores
                        cell_nCore_up[icell, z] = ncores_all_up
                        cell_nCore_down[icell, z] = ncores_all_down
                        cell_ovlap_up[icell, z] = 0 # EJ
                        
                        # Calculate core statistics
                        # MaFlx_sum_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        MaFlx_core_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        W_max_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        W_mean_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        QC_max_up = np.full(ncores_up, np.NaN, dtype=np.float32) # EJ
                        QC_mean_up = np.full(ncores_up, np.NaN, dtype=np.float32) # EJ
                        QC_mean_prm = np.full(ncores_up, np.NaN, dtype=np.float32) # EJ
                        QR_max_up = np.full(ncores_up, np.NaN, dtype=np.float32) # EJ
                        QR_mean_up = np.full(ncores_up, np.NaN, dtype=np.float32) # EJ
                        QV_mean_up = np.full(ncores_up, np.NaN, dtype=np.float32) # EJ
                        QV_mean_prm = np.full(ncores_up, np.NaN, dtype=np.float32) # EJ
                        Entr_up = np.full(ncores_up, np.NaN, dtype=np.float32) # EJ
                        Detr_up = np.full(ncores_up, np.NaN, dtype=np.float32) # EJ
                        Vapr_up = np.full(ncores_up, np.NaN, dtype=np.float32) # EJ
                        dBZ_up = np.full(ncores_up, np.NaN, dtype=np.float32) # EJ
                        Thte_max_up = np.full(ncores_up, np.NaN, dtype=np.float32) # EJ
                        Thte_mean_up = np.full(ncores_up, np.NaN, dtype=np.float32) # EJ
                        Thte_mean_prm = np.full(ncores_up, np.NaN, dtype=np.float32) # EJ
                        Thtv_max_up = np.full(ncores_up, np.NaN, dtype=np.float32) # EJ
                        Thtv_mean_prm = np.full(ncores_up, np.NaN, dtype=np.float32) # EJ
                        Buoy_Thtv_up = np.full(ncores_up, np.NaN, dtype=np.float32) # EJ
                        Trho_max_up = np.full(ncores_up, np.NaN, dtype=np.float32) # EJ
                        Trho_mean_prm = np.full(ncores_up, np.NaN, dtype=np.float32) # EJ
                        Buoy_Trho_up = np.full(ncores_up, np.NaN, dtype=np.float32) # EJ
                        Pres_pert_top = np.full(ncores_up, np.NaN, dtype=np.float32) # EJ
                        Pres_pert_bot = np.full(ncores_up, np.NaN, dtype=np.float32) # EJ
                        Mrho_mean_up = np.full(ncores_up, np.NaN, dtype=np.float32) # EJ
                        PGF_up = np.full(ncores_up, np.NaN, dtype=np.float32) # EJ
                        
                        
                        for ii in range(ncores_up):                            
                            MaFlx_core_up[ii] = np.nansum(zMassFlux[core_label_up == core_numbers_up[ii]])
                            W_max_up[ii] = np.nanmax(zW[core_label_up == core_numbers_up[ii]])
                            W_mean_up[ii] = np.nanmean(zW[core_label_up == core_numbers_up[ii]])
                            QC_max_up[ii] = np.nanmax(zQC[core_label_up == core_numbers_up[ii]]) # EJ
                            QC_mean_up[ii] = np.nanmean(zQC[core_label_up == core_numbers_up[ii]]) # EJ
                            QC_mean_prm[ii] = np.nanmean(zQC[core_label_up_prm == core_numbers_up[ii]]) # EJ
                            QR_max_up[ii] = np.nanmax(zQR[core_label_up == core_numbers_up[ii]]) # EJ
                            QR_mean_up[ii] = np.nanmean(zQR[core_label_up == core_numbers_up[ii]]) # EJ
                            QV_mean_up[ii] = np.nanmean(zQV[core_label_up == core_numbers_up[ii]]) # EJ
                            QV_mean_prm[ii] = np.nanmean(zQV[core_label_up_prm == core_numbers_up[ii]]) # EJ
                            Entr_up[ii] = np.nansum(zEntr[core_label_up_dil == core_numbers_up[ii]]) # EJ
                            Detr_up[ii] = np.nansum(zDetr[core_label_up_dil == core_numbers_up[ii]]) # EJ
                            Vapr_up[ii] = np.nansum(zVapr[core_label_up == core_numbers_up[ii]]) # EJ
                            dBZ_up[ii] = np.nanmax(zdBZ[core_label_up == core_numbers_up[ii]]) # EJ
                            Thte_max_up[ii] = np.nanmax(zThte[core_label_up == core_numbers_up[ii]]) # EJ
                            Thte_mean_up[ii] = np.nanmean(zThte[core_label_up == core_numbers_up[ii]]) # EJ
                            Thte_mean_prm[ii] = np.nanmean(zThte[core_label_up_prm == core_numbers_up[ii]]) # EJ
                            Thtv_max_up[ii] = np.nanmax(zThtv[core_label_up == core_numbers_up[ii]]) # EJ
                            Thtv_mean_prm[ii] = np.nanmean(zThtv[core_label_up_prm == core_numbers_up[ii]]) # EJ
                            Buoy_Thtv_up[ii] = 9.81*(Thtv_max_up[ii]-Thtv_mean_prm[ii])/Thtv_mean_prm[ii] # EJ
                            Trho_max_up[ii] = np.nanmax(zTrho[core_label_up == core_numbers_up[ii]]) # EJ
                            Trho_mean_prm[ii] = np.nanmean(zTrho[core_label_up_prm == core_numbers_up[ii]]) # EJ
                            Buoy_Trho_up[ii] = 9.81*(Trho_max_up[ii]-Trho_mean_prm[ii])/Trho_mean_prm[ii] # EJ
                            
                            Pres_pert_top[ii] = np.nanmean(zPrs2[core_label_up_prm == core_numbers_up[ii]])\
                                - np.nanmax(zPrs2[core_label_up == core_numbers_up[ii]])
                            Pres_pert_bot[ii] = np.nanmean(zPrs1[core_label_up_prm == core_numbers_up[ii]])\
                                - np.nanmax(zPrs1[core_label_up == core_numbers_up[ii]])
                            Mrho_mean_up[ii] = np.nanmean(zMrho[core_label_up == core_numbers_up[ii]])
                            PGF_up[ii] = - 1/Mrho_mean_up[ii]\
                                * ( Pres_pert_top[ii] - Pres_pert_bot[ii] )/200
                            
                        # Calculate total mass flux for all labeled cores
                        MaFlx_sum_up = np.nansum(zMassFlux[core_label_up > 0])

                        # MaFlx_sum_down = np.full(ncores_down, np.NaN, dtype=np.float32)
                        MaFlx_core_down = np.full(ncores_down, np.NaN, dtype=np.float32)
                        W_min_down = np.full(ncores_down, np.NaN, dtype=np.float32)
                        W_mean_down = np.full(ncores_down, np.NaN, dtype=np.float32) 
                        Q_min_down = np.full(ncores_down, np.NaN, dtype=np.float32) # EJ
                        Q_mean_down = np.full(ncores_down, np.NaN, dtype=np.float32) # EJ
                        for ii in range(ncores_down):
                            MaFlx_core_down[ii] = np.nansum(zMassFlux[core_label_down == core_numbers_down[ii]])
                            W_min_down[ii] = np.nanmin(zW[core_label_down == core_numbers_down[ii]])
                            W_mean_down[ii] = np.nanmean(zW[core_label_down == core_numbers_down[ii]])
                            Q_min_down[ii] = np.nanmin(zQC[core_label_down == core_numbers_down[ii]]) # EJ
                            Q_mean_down[ii] = np.nanmean(zQC[core_label_down == core_numbers_down[ii]]) # EJ
                        # Calculate total mass flux for all labeled cores
                        MaFlx_sum_down = np.nansum(zMassFlux[core_label_down > 0])
                        
                        # Save data to output arrays
                        ncores_save_up = min([ncores_up, ncores_min])
                        cell_MassFlux_up[icell, z] = MaFlx_sum_up * DX * DY
                        cell_CoreMassFlux_up[icell, z, 0:ncores_save_up] = MaFlx_core_up[0:ncores_save_up] * DX * DY
                        cell_CoreArea_up[icell, z, 0:ncores_save_up] = core_npix_up[0:ncores_save_up] * grid_area
                        cell_CoreMaxW_up[icell, z, 0:ncores_save_up] = W_max_up[0:ncores_save_up]
                        cell_CoreMeanW_up[icell, z, 0:ncores_save_up] = W_mean_up[0:ncores_save_up]
                        
                        cell_CoreMaxQC_up[icell, z, 0:ncores_save_up] = QC_max_up[0:ncores_save_up] # EJ
                        cell_CoreMeanQC_up[icell, z, 0:ncores_save_up] = QC_mean_up[0:ncores_save_up] # EJ
                        cell_CoreMeanQC_prm[icell, z, 0:ncores_save_up] = QC_mean_prm[0:ncores_save_up] # EJ
                        cell_CoreMaxQR_up[icell, z, 0:ncores_save_up] = QR_max_up[0:ncores_save_up] # EJ
                        cell_CoreMeanQR_up[icell, z, 0:ncores_save_up] = QR_mean_up[0:ncores_save_up] # EJ
                        cell_CoreMeanQV_up[icell, z, 0:ncores_save_up] = QV_mean_up[0:ncores_save_up] # EJ
                        cell_CoreMeanQV_prm[icell, z, 0:ncores_save_up] = QV_mean_prm[0:ncores_save_up] # EJ
                        cell_Entr_up[icell, z, 0:ncores_save_up] = Entr_up[0:ncores_save_up] # EJ
                        cell_Detr_up[icell, z, 0:ncores_save_up] = Detr_up[0:ncores_save_up] # EJ
                        cell_Vapr_up[icell, z, 0:ncores_save_up] = Vapr_up[0:ncores_save_up] # EJ
                        cell_dBZ_up[icell, z, 0:ncores_save_up] = dBZ_up[0:ncores_save_up] # EJ
                        cell_ThteMax_up[icell , z, 0:ncores_save_up] = Thte_max_up[0:ncores_save_up] # EJ
                        cell_ThteMean_up[icell , z, 0:ncores_save_up] = Thte_mean_up[0:ncores_save_up] # EJ
                        cell_ThteMean_prm[icell , z, 0:ncores_save_up] = Thte_mean_prm[0:ncores_save_up] # EJ
                        cell_ThtvMax_up[icell , z, 0:ncores_save_up] = Thtv_max_up[0:ncores_save_up] # EJ
                        cell_ThtvMean_prm[icell , z, 0:ncores_save_up] = Thtv_mean_prm[0:ncores_save_up] # EJ
                        cell_BuoyThtv_up[icell , z, 0:ncores_save_up] = Buoy_Thtv_up[0:ncores_save_up] # EJ
                        cell_TrhoMax_up[icell , z, 0:ncores_save_up] = Trho_max_up[0:ncores_save_up] # EJ
                        cell_TrhoMean_prm[icell , z, 0:ncores_save_up] = Trho_mean_prm[0:ncores_save_up] # EJ
                        cell_BuoyTrho_up[icell , z, 0:ncores_save_up] = Buoy_Trho_up[0:ncores_save_up] # EJ
                        cell_PGF_up[icell , z, 0:ncores_save_up] = PGF_up[0:ncores_save_up] # EJ

                        ncores_save_down = min([ncores_down, ncores_min])
                        cell_MassFlux_down[icell, z] = MaFlx_sum_down * DX * DY
                        cell_CoreMassFlux_down[icell, z, 0:ncores_save_down] = MaFlx_core_down[0:ncores_save_down] * DX * DY
                        cell_CoreArea_down[icell, z, 0:ncores_save_down] = core_npix_down[0:ncores_save_down] * grid_area
                        cell_CoreMinW_down[icell, z, 0:ncores_save_down] = W_min_down[0:ncores_save_down]
                        cell_CoreMeanW_down[icell, z, 0:ncores_save_down] = W_mean_down[0:ncores_save_down]
                        cell_CoreMinQ_down[icell, z, 0:ncores_save_down] = Q_min_down[0:ncores_save_down] # EJ
                        cell_CoreMeanQ_down[icell, z, 0:ncores_save_down] = Q_mean_down[0:ncores_save_down] # EJ
                    
            else:
                print(f'No cell matching track # {itracknum}')

                
        # Group outputs in dictionaries
        # (EJ: added Entrainment variables and dBZ)
        out_dict3d = {
            'CoreArea_up': cell_CoreArea_up,
            'CoreMaxW_up': cell_CoreMaxW_up,
            'CoreMeanW_up': cell_CoreMeanW_up,
            'CoreMaxQC_up': cell_CoreMaxQC_up,
            'CoreMeanQC_up': cell_CoreMeanQC_up,
            'CoreMeanQC_prm': cell_CoreMeanQC_prm,
            'CoreMaxQR_up': cell_CoreMaxQR_up,
            'CoreMeanQR_up': cell_CoreMeanQR_up,
            'CoreMeanQV_up': cell_CoreMeanQV_up,
            'CoreMeanQV_prm': cell_CoreMeanQV_prm,
            'CoreMassFlux_up': cell_CoreMassFlux_up,
            'CoreReflMax_up': cell_dBZ_up,
            'CoreThteMax_up': cell_ThteMax_up,
            'CoreThteMean_up': cell_ThteMean_up,
            'CoreThteMean_prm': cell_ThteMean_prm,
            'CoreThtvMax_up': cell_ThtvMax_up,
            'CoreThtvMean_prm': cell_ThtvMean_prm,
            'CoreBuoyThtv_up': cell_BuoyThtv_up,
            'CoreTrhoMax_up': cell_TrhoMax_up,
            'CoreTrhoMean_prm': cell_TrhoMean_prm,
            'CoreBuoyTrho_up': cell_BuoyTrho_up,
            'CorePGF_up': cell_PGF_up,
            
            'Entr_up': cell_Entr_up,
            'Detr_up': cell_Detr_up,
            'Vapor_up': cell_Vapr_up,

            'CoreArea_down': cell_CoreArea_down,
            'CoreMinW_down': cell_CoreMinW_down,
            'CoreMeanW_down': cell_CoreMeanW_down,
            'CoreMinQ_down': cell_CoreMinQ_down,
            'CoreMeanQ_down': cell_CoreMeanQ_down,
            'CoreMassFlux_down': cell_CoreMassFlux_down,
        }
        out_dict2d = {
            'nCore_up': cell_nCore_up,
            'MassFlux_up': cell_MassFlux_up,
            'Ovlap_up': cell_ovlap_up,

            'nCore_down': cell_nCore_down,
            'MassFlux_down': cell_MassFlux_down,
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
            'CoreMaxQC_up': {
                'long_name': 'Updraft core mean QC',
                'units': 'kg/kg',
            },
            'CoreMeanQC_up': {
                'long_name': 'Updraft core mean QC',
                'units': 'kg/kg',
            },
            'CoreMeanQC_prm': {
                'long_name': 'Updraft perim mean QC',
                'units': 'kg/kg',
            },
            'CoreMaxQR_up': {
                'long_name': 'Updraft core mean QR',
                'units': 'kg/kg',
            },
            'CoreMeanQR_up': {
                'long_name': 'Updraft core mean QR',
                'units': 'kg/kg',
            },
            'CoreMeanQV_up': {
                'long_name': 'Updraft core mean QV',
                'units': 'kg/kg',
            },
            'CoreMeanQV_prm': {
                'long_name': 'Updraft perim mean QV',
                'units': 'kg/kg',
            },
            'CoreMassFlux_up': {
                'long_name': 'Updraft core mass flux',
                'units': 'kg s^-1',
            },
            'CoreReflMax_up': {
                'long_name': 'Updraft core reflectivity',
                'units': 'dBZ',
            },
            'CoreThteMax_up': {
                'long_name': 'Updraft core max Theta e',
                'units': 'K',
            },
            'CoreThteMean_up': {
                'long_name': 'Updraft core mean Theta e',
                'units': 'K',
            },
            'CoreThteMean_prm': {
                'long_name': 'Updraft perim mean Theta e',
                'units': 'K',
            },
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
            'CoreTrhoMax_up': {
                'long_name': 'Updraft core max density T',
                'units': 'K',
            },
            'CoreTrhoMean_prm': {
                'long_name': 'Updraft core boundary mean density T',
                'units': 'K',
            },
            'CoreBuoyTrho_up': {
                'long_name': 'Updraft core Buoyancy based on Density T',
                'units': 'm s^-2',
            },
            'CorePGF_up': {
                'long_name': 'Pressure Gradient Force opposing Buoyancy',
                'units': 'm s^-2',
            },
            'MassFlux_up': {
                'long_name': 'Total updraft mass flux',
                'units': 'kg s^-1',
            },
            'Ovlap_up': {
                'long_name': 'Overlap Percentage',
                'units': '%',
            },
            'Entr_up': {
                'long_name': 'Total entrainment',
                'units': 'kg s^-1',
            },
            'Detr_up': {
                'long_name': 'Total detrainment',
                'units': 'kg s^-1',
            },
            'Vapor_up': {
                'long_name': 'Total flux of water vapor into core',
                'units': 'kg s^-1',
            },
            # Downdraft
            'nCore_down': {
                'long_name': 'Number of downdraft cores',
                'units': 'count',
            },
            'CoreArea_down': {
                'long_name': 'Downdraft core area',
                'units': 'km^2',
            },
            'CoreMinW_down': {
                'long_name': 'Downdraft core minimum W',
                'units': 'm/s',
            },
            'CoreMeanW_down': {
                'long_name': 'Downdraft core mean W',
                'units': 'm/s',
            },
            'CoreMinQ_down': {
                'long_name': 'Downdraft core mean Q',
                'units': 'kg/kg',
            },
            'CoreMeanQ_down': {
                'long_name': 'Downdraft core mean Q',
                'units': 'kg/kg',
            },
            'CoreMassFlux_down': {
                'long_name': 'Downdraft core mass flux',
                'units': 'kg s^-1',
            },
            'MassFlux_down': {
                'long_name': 'Total downdraft mass flux',
                'units': 'kg s^-1',
            },
        }
        # import pdb; pdb.set_trace()
    return out_dict3d, out_dict2d, out_dict_attrs


#-----------------------------------------------------------------------
if __name__ == '__main__':

    # Get configuration file name from input
    config_file = sys.argv[1]
    # track_start = int(sys.argv[2]) # EJ
    # track_end = int(sys.argv[3]) # EJ
    # digits = int(sys.argv[4]) # EJ
    
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
    # pixelfile_path_tmp = config['pixelfile_path_tmp']
    metfile_path = config['metfile_path']
    output_path = config['output_path']
    met_filebase = config['met_filebase']
    pixel_filebase = config['pixel_filebase']
    ncores_min = config['ncores_min']
    entfile_path = config['entfile_path'] # EJ
    ent_filebase = config['ent_filebase'] # EJ

    # Add start/end date to pixel file path
    pixelfile_path = f'{pixelfile_path}{startdate}_{enddate}/'
    #pixelfile_path_tmp = f'{pixelfile_path_tmp}{startdate}_{enddate}/'

    # Track stats file basename
    stats_filebase = 'trackstats_'

    # Output statistics filename
    # track_start_str = str(track_start).zfill(digits) # EJ
    # output_filename = f'{output_path}stats_3d_w_{startdate}_{enddate}_t{track_start_str}.nc' # EJ
    output_filename = f'{output_path}stats_3d_w_{startdate}_{enddate}.nc'
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
    
    # Time Travel
    
        
    nfiles = len(pixelfilelist) 
#     pixelfilelist_tmp = sorted(glob.glob(f'{pixelfile_path_tmp}{pixel_filebase}*.nc')) # EJ
#     nfiles = len(pixelfilelist_tmp) # EJ
    # Find all Met files
    metfilelist = sorted(glob.glob(f'{metfile_path}{met_filebase}*.nc'))
    nmetfiles = len(metfilelist)
    # Find all Ent files (EJ)
    entfilelist = sorted(glob.glob(f'{entfile_path}{ent_filebase}*.nc'))
    nentfiles = len(entfilelist)

    # Get basetime from pixel files
    pixel_basetime, pixelfile_dict = calc_basetime(pixelfilelist, pixel_filebase)
#     pixel_basetime, pixelfile_dict = calc_basetime(pixelfilelist_tmp, pixel_filebase)
    # Get basetime from MET files
    met_basetime, metfile_dict = calc_basetime(metfilelist, met_filebase)
    # met_basetime, metfile_dict = convert_lasso_times(metfile_path, met_filebase)
    # Get basetime from ENT files (EJ)
    ent_basetime, entfile_dict = calc_basetime(entfilelist, ent_filebase)
    
    

    # Find matching MET files for each pixel file
    match_metfilelist = [''] * nfiles
    for ifile in range(nfiles):
        # Find MET time closest to the pixel file time and get the index
        # Save the filename if time difference is < time_window
        idx = np.argmin(np.abs(met_basetime - pixel_basetime[ifile]))        
        if np.abs(met_basetime[idx] - pixel_basetime[ifile]) < time_window:
            match_metfilelist[ifile] = metfile_dict[met_basetime[idx]]
        else:
            print(f'No match file found for: {pixelfilelist[ifile]}')
    
    # Find matching ENT files for each pixel file (EJ)
    match_entfilelist = [''] * nfiles
    for ifile in range(nfiles):
        # Find MET time closest to the pixel file time and get the index
        # Save the filename if time difference is < time_window
        idx = np.argmin(np.abs(ent_basetime - pixel_basetime[ifile]))        
        if np.abs(ent_basetime[idx] - pixel_basetime[ifile]) < time_window:
            match_entfilelist[ifile] = entfile_dict[ent_basetime[idx]]
        else:
            print(f'No match file found for: {pixelfilelist[ifile]}')
    
    # For one minute intervals
#     pixelfilelist = np.repeat(pixelfilelist, 4)[:-3] # Duplicate for 1 min intervals (EJ)

    # Read track statistics file
    print(trackstats_file)
    dsstats = xr.open_dataset(trackstats_file, decode_times=False)
    ntracks = dsstats.dims[tracks_dimname]
    ntimes = dsstats.dims[times_dimname]
    # coord_tracks = dsstats['tracks'].sel(tracks=slice(track_start, track_end)) # EJ (added)
    # stats_basetime = dsstats['base_time'].sel(tracks=slice(track_start, track_end)) # EJ (replaced)
    stats_basetime = dsstats['base_time'].data
    stats_basetime_attrs = dsstats['base_time'].attrs
    stats_cloudnumber = dsstats['cloudnumber'].data
    # ntracks = len(coord_tracks) # EJ (replaced)
    
    # cell_area = dsstats['cell_area']
    pixel_radius = dsstats.attrs['pixel_radius_km']
    dsstats.close()

    print(f'Total Number of Tracks: {ntracks}')

    # Read a MET file to get vertical coordinates
    dsm = xr.open_dataset(match_metfilelist[0])
    nz = dsm.dims['HAMSL']
    height = dsm['HAMSL']
    dsm.close()


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
    # EJ change back to range(nfiles) for 15s.
    for ifile in range(nfiles):
#     for ifile in range(200,300):
        # print(ifile)
        # Find all matching time indices from track stats file to the current pixel file
        matchindices = np.array(
            np.where(np.abs(stats_basetime - pixel_basetime[ifile]) < time_window)
        ) # (EJ) somehow replace stats_basetime
        #import pdb
        #pdb.set_trace()
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
                    match_metfilelist[ifile],
                    match_entfilelist[ifile],
                    idx_track, 
                    config,
                )
            # Parallel
            elif run_parallel == 1:
                iresult = dask.delayed(calc_cellstats_singlefile)(
                    pixelfilelist[ifile], 
                    match_metfilelist[ifile],
                    match_entfilelist[ifile],
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
                        out_dict[ivar][trackindices,timeindices,:,:] = iVAR3d[ivar]
                    else:
                        print(f'Warning: {ivar} dimension is not 3.')
            if iVAR2d is not None:
                trackindices = trackindices_all[ifile]
                timeindices = timeindices_all[ifile]
                # Loop over each variable and assign values to output dictionary
                for ivar in var_names2d:
                    if iVAR2d[ivar].ndim == 2:
                        out_dict[ivar][trackindices,timeindices,:] = iVAR2d[ivar]
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
    out_dict['base_time'] = ([tracks_dimname, times_dimname], stats_basetime, dsstats['base_time'].attrs)
    # Define coordinate list
    core_dim_attrs = {
        'long_name': 'Core number',
    }
    coord_dict = {
        tracks_dimname: ([tracks_dimname], np.arange(0, ntracks)),
        times_dimname: ([times_dimname], np.arange(0, ntimes)),
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