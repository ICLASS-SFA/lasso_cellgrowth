"""
Calculates 3D W core statistics from gridded Met data for tracked convective cells.
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
import warnings
import dask
from dask.distributed import Client, LocalCluster
# import matplotlib.pyplot as plt

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
    met_filename, 
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
    print(met_filename)

    # Get thresholds from config
    W_up_thresh = config['W_up_thresh']
    W_down_thresh = config['W_down_thresh']
    min_core_npix = config['min_core_npix']
    ncores_min = config['ncores_min']
    core_expand_dist = config.get('core_expand_dist', 1)
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

    # Read pixel-level track file
    ds = xr.open_dataset(pixel_filename, decode_times=False)
    time_pixel = ds['time']
    ny_p = ds.dims['lat']
    nx_p = ds.dims['lon']

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
        PRESSURE = dsm['PRESSURE'][:, :, ymin:ymax+1, xmin:xmax+1]
        TEMPERATURE = dsm['TEMPERATURE'][:, :, ymin:ymax+1, xmin:xmax+1]
        QVAPOR = dsm['QVAPOR'][:, :, ymin:ymax+1, xmin:xmax+1]
        # TV = dsm['TV'][:, :, ymin:ymax+1, xmin:xmax+1]
        THETA = dsm['THETA'][:, :, ymin:ymax+1, xmin:xmax+1]
        WA = dsm['WA'][:, :, ymin:ymax+1, xmin:xmax+1]
        # Update ny, nx with the subset
        ny = XLONG.sizes['lat']
        nx = XLONG.sizes['lon']
    else:
        XLONG = dsm['XLONG']
        XLAT = dsm['XLAT']
        PRESSURE = dsm['PRESSURE']
        TEMPERATURE = dsm['TEMPERATURE']
        QVAPOR = dsm['QVAPOR']
        THETA = dsm['THETA']
        # TV = dsm['TV']
        WA = dsm['WA']

    # Check dimensions again after subset
    if (ny_p != ny) | (nx_p != nx):
        print(f'ERROR: Inconsistent number of grids between pixel-level and MET files.')
        print(f'ny: {ny}, ny_pixel: {ny_p}, nx: {nx}, nx_pixel: {nx_p}')
        sys.exit()

    # Drop 1D lat/lon coordinates, and reasign 2D XLONG/XLAT coordinates from Met file
    # It does not seem like this is necessary in Xarray 0.21.1
    ds = ds.drop_vars(['lon', 'lat']).assign_coords({'XLONG':XLONG, 'XLAT':XLAT})
    tracknumbermap = ds['tracknumber'].squeeze()

    # Calculate virtual temperature
    TV = TEMPERATURE * (1 + QVAPOR / 0.622) / (1 + QVAPOR)

    # theta-e following Bolton (1980); error of < 0.3 K between -35 and 35C; from Thompson scheme
    # more accurate formula from Emanuel could be implemented; ice effects also excluded
    es = PRESSURE*QVAPOR/(0.622*QVAPOR)
    TDEW = (35.86*np.log(es) - 4947.2325)/(np.log(es) - 23.6837)
    TLCL = 1/(1/(TDEW - 56) + np.log(TEMPERATURE/TDEW)/800) + 56
    p1 = 3.376/TLCL - 0.00254
    p2 = 1e3*QVAPOR*(1 + 0.81*QVAPOR)
    THETAE = (TEMPERATURE*(100000./PRESSURE)**(0.2854*(1 - 0.28*QVAPOR)))*np.exp(p1*p2) #K

    # Density temp
    # TRHO = TEMPERATURE*((1 + QVAPOR/0.622)/(1 + QTOTAL+QVAPOR))
    
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
    # X = TEMPERATURE - 273.16
    # X = X.where(X > -80)
    # X = X.fillna(-80) #setting values less than -80C to -80C 
    # ESL = C0 + X*(C1 + X*(C2 + X*(C3 + X*(C4 + X*(C5 + X*(C6 + X*(C7 + X*C8))))))) #saturation vapor pressure
    # QVS = 0.622*ESL/(PRESSURE - ESL) #saturation vapor mixing ratio
    # RH = 1e2*QVAPOR/QVS # %
    #SS = RH - 100 #supersaturation in %

    # Calculate moist air density using virtual temperature
    R_dry = 287.058   # J kg−1 K−1
    # RHO_DRY = PRESSURE / (R_dry * TEMPERATURE)  # kg m-3
    RHO_MOIST = 100 * PRESSURE / (R_dry * TV)  # kg m-3

    # Calculate mass flux (kg m-2 s-1)
    MassFlux = (RHO_MOIST * WA).squeeze()

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

        cell_nCore_down = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_MassFlux_down = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_CoreMassFlux_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreArea_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMinW_down = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanW_down = np.full(dims3d, np.NaN, dtype=np.float32)
        # cell_CoreMinQ_down = np.full(dims3d, np.NaN, dtype=np.float32)
        # cell_CoreMeanQ_down = np.full(dims3d, np.NaN, dtype=np.float32)
       
        # new arrays for other things
        cell_dBZ_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_ThteMax_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_ThteMean_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_ThteMean_prm = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_ThtvMax_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_ThtvMean_prm = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_BuoyThtv_up = np.full(dims3d, np.NaN, dtype=np.float32)
        # cell_TrhoMax_up = np.full(dims3d, np.NaN, dtype=np.float32)
        # cell_TrhoMean_prm = np.full(dims3d, np.NaN, dtype=np.float32)
        # cell_BuoyTrho_up = np.full(dims3d, np.NaN, dtype=np.float32)
        # cell_PGF_up = np.full(dims3d, np.NaN, dtype=np.float32)
    

        # Loop over each match tracked cell
        for icell in range(nmatchcell):
            # Track number needs to add 1
            itracknum = idx_track[icell] + 1

            # Count the number of pixels for the original cell mask
            icellmask = tracknumbermap == itracknum
            inpix_cloud = np.count_nonzero(icellmask)

            # Proceed if the number matching cloud pixel > 0
            if inpix_cloud > 0:

                # Subset 3D variables to the current cell mask
                # icellmask = tracknumbermap == itracknum
                iW = WA.where(icellmask, drop=True).squeeze().data
                iMassFlux = MassFlux.where(icellmask, drop=True).squeeze().data
                # iRho = RHO_DRY.where(icellmask, drop=True).squeeze().data
                # iP = PRESSURE.where(icellmask, drop=True).squeeze().data
                # iT = TEMPERATURE.where(icellmask, drop=True).squeeze().data
                iQv = QVAPOR.where(icellmask, drop=True).squeeze().data
                iTheta = THETA.where(icellmask, drop=True).squeeze().data
                # iThetae = THETAE.where(icellmask, drop=True).squeeze().data
                iThte = THETAE.where(icellmask, drop=True).squeeze().data
                # iTv = TV.where(icellmask, drop=True).squeeze().data
                # iTrho = TRHO.where(icellmask, drop=True).squeeze().data
                # iRH = RH.where(icellmask, drop=True).squeeze().data
                # iQc = QCLOUD.where(icellmask, drop=True).squeeze().data
                # iQr = QRAIN.where(icellmask, drop=True).squeeze().data
                # iQi = QICE.where(icellmask, drop=True).squeeze().data
                # iQs = QSNOW.where(icellmask, drop=True).squeeze().data
                # iQg = QGRAUPEL.where(icellmask, drop=True).squeeze().data
                # iQt = QTOTAL.where(icellmask, drop=True).squeeze().data
                # iNc = NCLOUD.where(icellmask, drop=True).squeeze().data
                # iNr = NRAIN.where(icellmask, drop=True).squeeze().data
                # iNi = NICE.where(icellmask, drop=True).squeeze().data
                # iNwa = NWA.where(icellmask, drop=True).squeeze().data
                # iNia = NIA.where(icellmask, drop=True).squeeze().data
                # iHd = HD.where(icellmask, drop=True).squeeze().data

                # Virtual Potential Temperature
                iThtv = iTheta * (iQv + 0.622)/(0.622 * (1 + iQv))

                # Calculate new statistics of the cell
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)

                    # Loop over vertical level
                    for z in range(0, nz):
                        # print(height[z])
                        zW = iW[z,:,:]
                        zMassFlux = iMassFlux[z,:,:]
                        # zRho = iRho[z,:,:]
                        # zP = iP[z,:,:]
                        # zT = iT[z,:,:]
                        # zQv = iQv[z,:,:]
                        zThtv = iThtv[z,:,:]
                        zThte = iThte[z,:,:]
                        # zTv = iTv[z,:,:]
                        # zTrho = iTrho[z,:,:]
                        # zRH = iRH[z,:,:]
                        # zQc = iQc[z,:,:]
                        # zQr = iQr[z,:,:]
                        # zQi = iQi[z,:,:]
                        # zQs = iQs[z,:,:]
                        # zQg = iQg[z,:,:]
                        # zQt = iQt[z,:,:]
                        # zNc = iNc[z,:,:]
                        # zNr = iNr[z,:,:]
                        # zNi = iNi[z,:,:]
                        # zNwa = iNwa[z,:,:]
                        # zNia = iNia[z,:,:]
                        # zHd = iHd[z,:,:]

                        # Proceed if max(W) > threshold
                        if np.nanmax(zW) > W_up_thresh:

                            # Label updraft cores
                            dict_up = label_cores(zW, W_up_thresh, ncores_min, min_core_npix, method='>')
                            ncores_all_up = dict_up['ncores_all']
                            ncores_up = dict_up['ncores_save']
                            core_npix_up = dict_up['core_npix']
                            core_numbers_up = dict_up['core_numbers']
                            core_label_up = dict_up['core_label']

                            # # Dilate core labels by the equivalent diameter of each core
                            # # This is more physical b/c larger cores would mix air in a wider region than narrower cores
                            # # But the for loop will make it much slower to run
                            # core_label_up_prm = np.zeros_like(core_label_up)
                            # for ii in range(ncores_up):
                            #     # Isolate current core
                            #     cell = np.zeros_like(core_label_up)
                            #     cell[core_label_up == core_numbers_up[ii]] = 1
                            #     # Expand core by its equivalent diameter (in grid unit)
                            #     expand = round(np.sqrt(core_npix_up[ii]/np.pi))
                            #     dil = expand_labels(cell, distance = expand)
                            #     # Get core perimeter mask
                            #     core_label_up_prm[(dil - cell) == 1] = core_numbers_up[ii]

                            # Get the min number of cores to save
                            ncores_save_up = min([ncores_up, ncores_min])

                            # Total number of cores
                            cell_nCore_up[icell, z] = ncores_all_up
            
                            # Dilate cores to get perimeter
                            # This dilates all cores by the same distance, thus is much faster to run but less physical
                            core_label_up_prm = expand_labels(core_label_up, distance=core_expand_dist) - core_label_up
                            
                            # Calculate core statistics
                            MaFlx_core_up = np.full(ncores_save_up, np.NaN, dtype=np.float32)
                            W_max_up = np.full(ncores_save_up, np.NaN, dtype=np.float32)
                            W_mean_up = np.full(ncores_save_up, np.NaN, dtype=np.float32)

                            Thte_max_up = np.full(ncores_save_up, np.NaN, dtype=np.float32)
                            Thte_mean_up = np.full(ncores_save_up, np.NaN, dtype=np.float32)
                            Thte_mean_prm = np.full(ncores_save_up, np.NaN, dtype=np.float32)
                            Thtv_max_up = np.full(ncores_save_up, np.NaN, dtype=np.float32)
                            Thtv_mean_prm = np.full(ncores_save_up, np.NaN, dtype=np.float32)
                            Buoy_Thtv_up = np.full(ncores_save_up, np.NaN, dtype=np.float32)

                            # Loop over each core
                            for ii in range(ncores_save_up):
                                # Get the masks for core and perimeter
                                icoremask = core_label_up == core_numbers_up[ii]
                                iperimask = core_label_up_prm == core_numbers_up[ii]

                                MaFlx_core_up[ii] = np.nansum(zMassFlux[icoremask])
                                W_max_up[ii] = np.nanmax(zW[icoremask])
                                W_mean_up[ii] = np.nanmean(zW[icoremask])

                                # if (ncores_save_up > 0): import pdb; pdb.set_trace()
                                Thte_max_up[ii] = np.nanmax(zThte[icoremask])
                                Thte_mean_up[ii] = np.nanmean(zThte[icoremask])
                                Thte_mean_prm[ii] = np.nanmean(zThte[iperimask])
                                Thtv_max_up[ii] = np.nanmax(zThtv[icoremask])
                                Thtv_mean_prm[ii] = np.nanmean(zThtv[iperimask])
                                Buoy_Thtv_up[ii] = 9.81*(Thtv_max_up[ii] - Thtv_mean_prm[ii]) / Thtv_mean_prm[ii]

                            # Calculate total mass flux for all labeled cores
                            if ncores_all_up > 0:
                                MaFlx_sum_up = np.nansum(zMassFlux[core_label_up > 0])
                            else:
                                MaFlx_sum_up = np.NaN
                            # if (ncores_all_up > 2):                         
                            #     import pdb; pdb.set_trace()

                            # Save data to output arrays
                            cell_MassFlux_up[icell, z] = MaFlx_sum_up * DX * DY
                            cell_CoreMassFlux_up[icell, z, 0:ncores_save_up] = MaFlx_core_up[0:ncores_save_up] * DX * DY
                            cell_CoreArea_up[icell, z, 0:ncores_save_up] = core_npix_up[0:ncores_save_up] * grid_area
                            cell_CoreMaxW_up[icell, z, 0:ncores_save_up] = W_max_up[0:ncores_save_up]
                            cell_CoreMeanW_up[icell, z, 0:ncores_save_up] = W_mean_up[0:ncores_save_up]

                            cell_ThteMax_up[icell , z, 0:ncores_save_up] = Thte_max_up[0:ncores_save_up]
                            cell_ThteMean_up[icell , z, 0:ncores_save_up] = Thte_mean_up[0:ncores_save_up]
                            cell_ThteMean_prm[icell , z, 0:ncores_save_up] = Thte_mean_prm[0:ncores_save_up]
                            cell_ThtvMax_up[icell , z, 0:ncores_save_up] = Thtv_max_up[0:ncores_save_up]
                            cell_ThtvMean_prm[icell , z, 0:ncores_save_up] = Thtv_mean_prm[0:ncores_save_up]
                            cell_BuoyThtv_up[icell , z, 0:ncores_save_up] = Buoy_Thtv_up[0:ncores_save_up]
                        # if np.nanmax(zW) > W_up_thresh:
                        

                        # Proceed if min(W) < threshold
                        if np.nanmin(zW) < W_down_thresh:
                            # Label downdraft cores
                            dict_down = label_cores(zW, W_down_thresh, ncores_min, min_core_npix, method='<')
                            ncores_all_down = dict_down['ncores_all']
                            ncores_down = dict_down['ncores_save']
                            core_npix_down = dict_down['core_npix']
                            core_numbers_down = dict_down['core_numbers']
                            core_label_down = dict_down['core_label']

                            # Get the min number of cores to save
                            ncores_save_down = min([ncores_down, ncores_min])

                            # Total number of cores
                            cell_nCore_down[icell, z] = ncores_all_down
                            
                            MaFlx_core_down = np.full(ncores_save_down, np.NaN, dtype=np.float32)
                            W_min_down = np.full(ncores_save_down, np.NaN, dtype=np.float32)
                            W_mean_down = np.full(ncores_save_down, np.NaN, dtype=np.float32)
                            for ii in range(ncores_save_down):
                                # Get the masks for core and perimeter
                                icoremask = core_label_down == core_numbers_down[ii]
                                # iperimask = core_label_down_prm == core_numbers_down[ii]

                                MaFlx_core_down[ii] = np.nansum(zMassFlux[icoremask])
                                W_min_down[ii] = np.nanmin(zW[icoremask])
                                W_mean_down[ii] = np.nanmean(zW[icoremask])
                            # Calculate total mass flux for all labeled cores
                            if ncores_all_down > 0:
                                MaFlx_sum_down = np.nansum(zMassFlux[core_label_down > 0])
                            else:
                                MaFlx_sum_down = np.NaN
                            
                            # Save data to output arrays
                            cell_MassFlux_down[icell, z] = MaFlx_sum_down * DX * DY
                            cell_CoreMassFlux_down[icell, z, 0:ncores_save_down] = MaFlx_core_down[0:ncores_save_down] * DX * DY
                            cell_CoreArea_down[icell, z, 0:ncores_save_down] = core_npix_down[0:ncores_save_down] * grid_area
                            cell_CoreMinW_down[icell, z, 0:ncores_save_down] = W_min_down[0:ncores_save_down]
                            cell_CoreMeanW_down[icell, z, 0:ncores_save_down] = W_mean_down[0:ncores_save_down]
                        # if np.nanmin(zW) < W_down_thresh:

                    # for z in range(0, nz):

                    
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
            'CoreThtvMax_up': cell_ThtvMax_up,
            'CoreThtvMean_prm': cell_ThtvMean_prm,
            'CoreBuoyThtv_up': cell_BuoyThtv_up,
            # 'CoreTrhoMax_up': cell_TrhoMax_up,
            # 'CoreTrhoMean_prm': cell_TrhoMean_prm,
            # 'CoreBuoyTrho_up': cell_BuoyTrho_up,
            # 'CorePGF_up': cell_PGF_up,

            'CoreArea_down': cell_CoreArea_down,
            'CoreMinW_down': cell_CoreMinW_down,
            'CoreMeanW_down': cell_CoreMeanW_down,
            'CoreMassFlux_down': cell_CoreMassFlux_down,
        }
        out_dict2d = {
            'nCore_up': cell_nCore_up,
            'MassFlux_up': cell_MassFlux_up,

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
            'CoreMassFlux_up': {
                'long_name': 'Updraft core mass flux',
                'units': 'kg s^-1',
            },
            'MassFlux_up': {
                'long_name': 'Total updraft mass flux',
                'units': 'kg s^-1',
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
    output_path = config['output_path']
    reg_filebase = config['reg_filebase']
    pixel_filebase = config['pixel_filebase']
    ncores_min = config['ncores_min']

    # Add start/end date to pixel file path
    pixelfile_path = f'{pixelfile_path}{startdate}_{enddate}/'

    # Track stats file basename
    stats_filebase = 'trackstats_'

    # Output statistics filename
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
    nfiles = len(pixelfilelist)
    # Find all Met files
    # regfilelist = sorted(glob.glob(f'{regfile_path}{reg_filebase}*.nc'))
    metfilelist = sorted(glob.glob(f'{metfile_path}{reg_filebase}*.nc'))
    nmetfiles = len(metfilelist)
    print(f'Number of MET files: {nmetfiles}')
    
    # Get basetime from pixel files
    pixel_basetime, pixelfile_dict = calc_basetime(pixelfilelist, pixel_filebase)
    # Get basetime from MET files
    met_basetime, regfile_dict = calc_basetime(metfilelist, reg_filebase)

    # Find matching MET files for each pixel file
    match_regfilelist = [''] * nfiles
    for ifile in range(nfiles):
        # Find MET time closest to the pixel file time and get the index
        # Save the filename if time difference is < time_window
        idx = np.argmin(np.abs(met_basetime - pixel_basetime[ifile]))
        if np.abs(met_basetime[idx] - pixel_basetime[ifile]) < time_window:
            match_regfilelist[ifile] = regfile_dict[met_basetime[idx]]
        else:
            print(f'No match file found for: {pixelfilelist[ifile]}')

    # Read track statistics file
    print(trackstats_file)
    dsstats = xr.open_dataset(trackstats_file, decode_times=False)
    ntracks = dsstats.dims[tracks_dimname]
    ntimes = dsstats.dims[times_dimname]
    stats_basetime = dsstats['base_time'].data
    stats_basetime_attrs = dsstats['base_time'].attrs
    # cell_area = dsstats['cell_area']
    pixel_radius = dsstats.attrs['pixel_radius_km']
    dsstats.close()

    print(f'Total Number of Tracks: {ntracks}')

    # Read a MET file to get vertical coordinates
    dsm = xr.open_dataset(match_regfilelist[0])
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
    for ifile in range(nfiles):
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
                    match_regfilelist[ifile],
                    idx_track, 
                    config,
                )
            # Parallel
            elif run_parallel == 1:
                iresult = dask.delayed(calc_cellstats_singlefile)(
                    pixelfilelist[ifile], 
                    match_regfilelist[ifile],
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