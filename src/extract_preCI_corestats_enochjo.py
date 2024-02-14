"""
Extracts 3D data from raw WRF files for tracked convective cells.
This version separates tracks by chunks.
"""
from genericpath import isfile
import numpy as np
import os, sys
import time
import yaml
import xarray as xr
import pandas as pd
import warnings
import dask
from dask.distributed import Client, LocalCluster, wait
from scipy import ndimage
from scipy.ndimage import generate_binary_structure, binary_dilation,iterate_structure
from skimage.measure import label
from skimage.segmentation import expand_labels
# import cc3d
from skimage.measure import centroid

#-----------------------------------------------------------------------
def ensmemb_short_to_long(ensmemb):
    """
    Convert the more modern short naming convention for ensemble members
    to the longer way used when starting LASSO-CACTI, e.g., eda05->eda_en05
    """
    if ensmemb[0:3] == "fnl":
        ensmemb_long = "fnl"
    elif ensmemb[0:4] == "era5":
        ensmemb_long = "era5"
    elif ensmemb[0:3] == "eda":
        ensmemb_long = f"eda_en{ensmemb[-2:]}"
    elif ensmemb[0:4] == "gefs":
        ensmemb_long = f"gefs_en{ensmemb[-2:]}"
    #end if
    return ensmemb_long

#-----------------------------------------------------------------------
def indentify_domain_number(dx):
    '''
    Use the grid spacing to determine which domain was run
    within the LASSO-CACTI nest setup.

    :param dx: Grid spacing (m)
    :return: domain number
    '''
    if dx == 7500:
        dom = 1
    elif dx == 2500:
        dom = 2
    elif dx == 500:
        dom = 3
    elif dx == 100:
        dom = 4
    # end if
    return dom

#-----------------------------------------------------------------------
def location_to_idx(lat, lon, center):
    """ 
    Convert a latitude and longitude into an index
    
    Args:
        lat: np.array
            Latitude array
        lon: np.array
            Longitufde array
        center: tuple(float)
            location tuple to find (lat, lon)
    
    Returns:
        lat_idx: int
            Index for latitude
        lon_idx: int
            Index for longitude
    """
    # This is for 2D lat/lon
    diff = abs(lat - center[0]) + abs(lon - center[1])
    lat_idx, lon_idx = np.unravel_index(diff.argmin(), diff.shape)
    return lat_idx, lon_idx

#--------------------------------------------------------------------------
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

#--------------------------------------------------------------------------
def pad_array(in_array, lat_idx, lon_idx, ny, nx, ny_d, nx_d, crad, sub_y=1, sub_x=1, fillval=np.NaN):
    """
    Pad 2D or 3D array to ny, nx dimensions center at lat_idx, lon_idx.
    
    Args:
        in_array: np.array
            Input 2D (y, x) or 3D array (z, y, x)
        lat_idx: int
            Center index on latitude (y) dimension
        lon_idx: int
            Center index on longitude (x) dimension
        ny: int
            Number of 1/2 grids to extract data in y dimension (In other words, radius)
        nx: int
            Number of 1/2 grids to extract data in x dimension
        ny_d: int
            Number of grids in the domain in y dimension (The OG size)
        nx_d: int
            Number of grids in the domain in x dimension
        sub_y: int, optional, default=1
            Number of grids to sub-sample in y dimension
        sub_x: int, optional, default=1
            Number of grids to sub-sample in x dimension
        fillval: optional, default=np.NaN
            Default fill value to pad the array.

    Returns:
        out_array: np.array
            Padded output array (z, y, x)
    """
    # Constrain lat/lon indices within domain boundary
    iy_min = 0 if (lat_idx-ny < 0) else lat_idx-ny
    iy_max = ny_d if (lat_idx+ny+1 > ny_d) else lat_idx+ny+1
    ix_min = 0 if (lon_idx-nx < 0) else lon_idx-nx
    ix_max = nx_d if (lon_idx+nx+1 > nx_d) else lon_idx+nx+1
    # Number of grids on left, right, bottom, top
    nx_l = lon_idx - ix_min
    nx_r = ix_max - lon_idx - 1
    ny_b = lat_idx - iy_min
    ny_t = iy_max - lat_idx - 1
    # Number of grids to pad on each side
    pnx_l = nx - nx_l
    pnx_r = nx - nx_r
    pny_b = ny - ny_b
    pny_t = ny - ny_t
    
#     circle = np.ones((crad*2+1,crad*2+1))
#     for ij in range(0,crad*2+1):
#         for ii in range(0,crad*2+1):
#             radi = np.sqrt( ( ij-(ny+1) )**2+( ii-(nx+1) )**2)
#             if radi > crad:
#                 circle[ij,ii] = np.nan
#     
#     circle = circle.astype(int)
    
    # Check array dimensions
    ndim = in_array.ndim
    if ndim == 3:
        # Subset array within the domain
        in_array = in_array[:, iy_min:iy_max, ix_min:ix_max] 
        # Pad array on y & x dimensions
        out_array = np.pad(in_array, ((0,0), (pny_b,pny_t), (pnx_l,pnx_r)), 'constant', constant_values=fillval)
        # Sub-sample array
        out_array = out_array[:, ::sub_y, ::sub_x] #* circle[::sub_y, ::sub_x] # EJ multiply by ones/NaN array.
    if ndim == 2:
        # Subset array within the domain
        in_array = in_array[iy_min:iy_max, ix_min:ix_max]
        # Pad array on y & x dimensions
        out_array = np.pad(in_array, ((pny_b,pny_t), (pnx_l,pnx_r)), 'constant', constant_values=fillval)
        # Sub-sample array
        out_array = out_array[::sub_y, ::sub_x] #* circle[::sub_y, ::sub_x]
    return out_array

#-----------------------------------------------------------------------
def extract_env_prof(
    fname_pixel, 
    # fname_wrfout,
    fname_met,
    fname_cld,
    fname_ent,
    idx_track, 
    _lat,
    _lon,
    config,
    cradii,
):
    """
    Extract data center at a track.
    
    Args:
        fname_pixel: string
            Tracking pixel file name
        fname_wrfout: string
            WRF out file name
        fname_met: string
            Met file name
        fname_cld: string
            Cloud file name
        idx_track: np.array
            Track indices in the pixel file
        _lat: np.array
            Center latitude of a track
        _lon: np.array
            Center longitude of a track
        config: dictionary
            Dictionary containing config parameters

    Returns:
        out_dict3d: dictionary
            Dictionary containing the 3d track statistics data
        out_dict2d: dictionary
            Dictionary containing the 2d track statistics data
        out_dict_attrs: dictionary
            Dictionary containing the attributes of track statistics data
        out_coords: dictionary
            Dictionary containing the coordinates of track statistics data
    """
    # Get values from config
    nx = config['nx']
    ny = config['ny']
    nz = config.get('nz', 100) # EJ
#     nz = config['HAMSL']
    sub_x = config.get('sub_x', 1)
    sub_y = config.get('sub_y', 1)
    DX = config.get('DX',100)
    DY = config.get('DY',100)
#     DX = 100
#     DY = 100
    
    W_up_thresh = config['W_up_thresh']
    W_down_thresh = config['W_down_thresh']
    Q_up_thresh = config['Q_up_thresh'] # EJ
    Q_down_thresh = config['Q_down_thresh'] # EJ
    min_core_npix = config['min_core_npix']
    ncores_min = config['ncores_min']
    geolimits = config.get('geolimits', None)

    # Check file existance
    # wrfout_exist = os.path.isfile(fname_wrfout)
    met_exist = os.path.isfile(fname_met)
    cld_exist = os.path.isfile(fname_cld)
    ent_exist = os.path.isfile(fname_ent)
    pixel_exist = os.path.isfile(fname_pixel)

    

    if met_exist & ent_exist:
        # Read Met file
        print(fname_met)
        dsm = xr.open_dataset(fname_met)
        # Rename dimenensions
        dsm = dsm.rename_dims({'south_north':'lat', 'west_east':'lon'})
        nz = dsm.dims['HAMSL']
        ny = dsm.dims['lat']
        nx = dsm.dims['lon']
        height = dsm['HAMSL'].data
        DX = dsm.attrs['DX']
        DY = dsm.attrs['DY']
        grid_area = DX * DY / 1e6
        
        XLONG = dsm['XLONG']
        XLAT = dsm['XLAT']
        PRESSURE = dsm['PRESSURE'] 
        TV = dsm['tv'] # EJ
        WA = dsm['WA']
        dBZ = dsm['REFL_10CM'] # EJ
        #QC = dsm['QCLOUD'] # EJ
        #qi = dsm['QICE'] # EJ
        #qs = dsm['QSNOW'] # EJ
        qv = dsm['QVAPOR'] # EJ
        #qr = dsm['QRAIN'] # EJ
        #qg = dsm['QGRAUP'] # EJ
        Thte = dsm['THETA_E'] # EJ
        TH = dsm['THETA'] # EJ
        QA = dsm['QA'] # EJ
        QT = dsm['QT'] # EJ
        #QR = dsm['QRAIN'] # EJ
        
        # QA = QC + QR
        
        # Calculate moist air density using virtual temperature
        R_dry = 287.058   # J kg−1 K−1
        Mrho = 100 * PRESSURE / (R_dry * TV)  # kg m-3

        # Calculate Virtual Potential Temperature # EJ
        # Thtv = TH * (qv + 0.622)/(0.622 * (1 + qv))

        # Calculate Temperature (AMS)
        Temp = TH*(PRESSURE/1000)**(2/7) # Remember that PRESSURE is in hPa
        tc = Temp - 273.15
        
        # Calculate RH (Thompson Scheme)
#         C0 = 0.611583699e3
#         C1 = 0.444606896e2
#         C2 = 0.143177157e1
#         C3 = 0.264224321e-1
#         C4 = 0.299291081e-3
#         C5 = 0.203154182e-5
#         C6 = 0.702620698e-8
#         C7 = 0.379534310e-11
#         C8 = -0.321582393e-13
#         X = tc.where(tc > -80)
#         X = X.fillna(-80) #setting values less than -80C to -80C 
#         ESL = C0 + X*(C1 + X*(C2 + X*(C3 + X*(C4 + X*(C5 + X*(C6 + X*(C7 + X*C8))))))) #saturation vapor pressure
#         QVS = 0.622*ESL/(PRESSURE - ESL) #saturation vapor mixing ratio
#         RH = 1e2*qv/QVS # %
        
        # Calculate Saturated Vapor Pressure (NWS)
        es = 6.11*10**((7.5*tc)/(237.3+tc))

        # Calculate Saturated Mixing Ratio (NWS)
        ws = 0.62197*(es/(PRESSURE-es))
        
        RH = qv/ws*100

        # Calculate Density Temperature (Eqn. 4.3.6 of some Emanuel textbook)
        # "Note that Tv is a special case of Trho, since when condensed water is absent rT = r."
        Trho = Temp*(1 + qv/0.622)/(1 + QT)
        
        # Calculate Theta E using the Emmanuel Textbook
        cpd = 1006
        g = 9.81
        cpv = 1870
        E = 0.622
        lv0 = 2501000
        Rv = 461.5
        Rd = 287.04
        cw = 4190
        cc = 2320
        ccl = 4200
        cvv = 1410
        cvd = 719
    
        alv = lv0 - cc*tc
    
        Tv = Temp*(1+0.608*QT)
        Thtv = Tv*(1000/PRESSURE)**0.286
    
        teA = Temp*(1000/PRESSURE)**(Rd/(cpd + ccl*QT))
        teB = np.exp( (lv0*QT)/((cpd + QT*ccl)*Temp))
        teC = (RH/100)**( (-1*QT*Rv) / (cpd + ccl*QT) )
        Thte = teA * teB * teC

        # Calculate mass flux (kg m-2 s-1)
        MassFlux = (Mrho * WA).squeeze()
        
    if ent_exist:
        # print(fname_ent)
        # Read ENT file (EJ)
        # No need to rename dimensions as they are already 'lat' and 'lon'
        dse = xr.open_dataset(fname_ent)
        dse = dse.rename_dims({'time':'Time','hgt':'HAMSL'})    
        dse = dse.drop('time')
        dse = dse.assign_coords(Time=dsm.coords['Time'].data) 
        EntrDetr = dse['entr_detr']

        # Separate the net entrainment file to entrainment and detrainment (EJ)
        Entr = EntrDetr.where(EntrDetr > 0)
        Detr = EntrDetr.where(EntrDetr < 0)
        
        # Calculate Inflow of qv
        Vapr = EntrDetr.where( (EntrDetr > 0) ) * qv
        
    if pixel_exist & met_exist:
        # print(fname_pixel)
        dsp = xr.open_dataset(fname_pixel)
        nx_p = dsp.sizes['lon']
        ny_p = dsp.sizes['lat']
        pixel_attrs = {
            'conv_core': dsp['conv_core'].attrs,
            'conv_mask': dsp['conv_mask'].attrs,
            # 'tracknumber_cmask': dsp['tracknumber_cmask'].attrs,
            'tracknumber': dsp['tracknumber'].attrs,
            # 'comp_ref': dsp['comp_ref'].attrs,
            'dbz_comp': dsp['dbz_comp'].attrs,
            'echotop10': dsp['echotop10'].attrs,
        }
        
#         dsp = dsp.drop_vars(['lon', 'lat']).assign_coords({'XLONG':XLONG, 'XLAT':XLAT})
        tracknumbermap = dsp['tracknumber'].squeeze()
#         tracknumbermap = dsp['conv_core_label'].squeeze() # EJ
        
    else:
        pixel_attrs = {
            'conv_core': '',
            'conv_mask': '',
            # 'tracknumber_cmask': '',
            'tracknumber': '',
            'dbz_comp': '',
            'echotop10': '',
        }

    out_dict1d = None
    out_dict2d = None
    out_dict3d = None
    out_dict_attrs = None


    # Proceed if number of matched cell is > 0
    nmatchcell = len(idx_track)
    if (nmatchcell > 0):
        # Create arrays for output statistics (EJ)
        dims1d = (nmatchcell)
        dims2d = (nmatchcell, nz)
        dims3d = (nmatchcell, nz, ncores_min)
        
        # Adding some 1d variables that are needed
        cell_maxETH_10dbz = np.full(dims1d, np.NaN, dtype=np.float32)
        cell_max_dbz = np.full(dims1d, np.NaN, dtype=np.float32)
        
        cell_nCore_up = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_MassFlux_up = np.full(dims2d, np.NaN, dtype=np.float32)
        cell_ovlap_up = np.full(dims2d, np.NaN, dtype=np.float32) # EJ
        cell_CoreMassFlux_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreArea_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMaxW_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanW_up = np.full(dims3d, np.NaN, dtype=np.float32)
#         cell_CoreMaxQC_up = np.full(dims3d, np.NaN, dtype=np.float32)
#         cell_CoreMeanQC_up = np.full(dims3d, np.NaN, dtype=np.float32)
#         cell_CoreMeanQC_prm = np.full(dims3d, np.NaN, dtype=np.float32)
#         cell_CoreMaxQR_up = np.full(dims3d, np.NaN, dtype=np.float32) # EJ
#         cell_CoreMeanQR_up = np.full(dims3d, np.NaN, dtype=np.float32) # EJ
        cell_CoreMeanQV_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanQV_prm = np.full(dims3d, np.NaN, dtype=np.float32)
        
        cell_CoreMeanTHe_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_CoreMeanTHv_up = np.full(dims3d, np.NaN, dtype=np.float32)
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
        cell_NS_up = np.full(dims3d, np.NaN, dtype=np.float32) # EJ
        cell_WE_up = np.full(dims3d, np.NaN, dtype=np.float32) # EJ
        cell_rhMean_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_rhMean_prm = np.full(dims3d, np.NaN, dtype=np.float32)
        
        # (EJ) new 3d variables for entrainment
        cell_Entr_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_Detr_up = np.full(dims3d, np.NaN, dtype=np.float32)
        cell_Vapr_up = np.full(dims3d, np.NaN, dtype=np.float32)
        
        if (pixel_exist == True) & (met_exist == True) & (ent_exist == True):
            cell_cloudy = np.full(QA.shape, 0, dtype=np.float32)     # Create zero array
            icloud = (WA > W_up_thresh) & (QA > Q_up_thresh)         # Find cloudy region
            cell_cloudy[icloud] = 1                                  # Set cloudy region to 1
            zcloud = np.nanmax(cell_cloudy,axis=1).squeeze()         # Reduce to 2D
        
        
        
        
        wcount = 0
        # Loop over each track
        for icell in range(nmatchcell):
        
            itracknum = idx_track[icell] + 1 # EJ
        
            # track center location (lat, lon)
            center = (_lat[icell], _lon[icell])

            # Tracking pixel file
            if (pixel_exist == True) & (met_exist == True) & (ent_exist == True):
                # print(icell)
            
                lat_idx, lon_idx = location_to_idx(XLAT.data, XLONG.data, center)
                
                # Calculate Equivalent diameter of cell
                # The issue with manually setting Area is that some cells do not exist
                # at this current time.
                # Ara = len(np.where(tracknumbermap == itracknum)[0])*100**2
                # crad = int(round(np.sqrt(Ara/np.pi)/100))
                
                # Use "area" in the file trackstats_20190129.1500_20190129.1800.nc
                
                #crad = 80
                
                # crad = cradii[icell] + 50 # Manually adding dilation
                crad = cradii[icell] # No more dilation now that we are including overlapping updrafts
                
                # create a ny x nx nan array with just the cell location with ones  
                if (lon_idx - crad < 0):
                    xbound1 = 0
                else:
                    xbound1 = lon_idx - crad
                
                if (lon_idx + crad > nx_p):
                    xbound2 = nx_p
                else:
                    xbound2 = lon_idx + crad
                
                if (lat_idx - crad < 0):
                    ybound1 = 0
                else:
                    ybound1 = lat_idx - crad
                
                if (lat_idx + crad > ny_p):
                    ybound2 = ny_p
                else:
                    ybound2 = lat_idx + crad
                
                cookiecutter = np.zeros((ny_p,nx_p))*np.nan
                for ij in range(ybound1,ybound2):
                    for ii in range(xbound1,xbound2):
                        radi = np.sqrt( ( ij-lat_idx )**2+( ii-lon_idx )**2)
                        if radi < crad:
                            cookiecutter[ij,ii] = 1
                            
                da_cc = xr.DataArray(cookiecutter, coords=tracknumbermap.coords, dims=tracknumbermap.dims) # Convert to dask format
                
                # Figure out how much overlap there is between current cell mask and other neighboring cells.
                # Remove current cell from tracknumbermap and binarize it
                tmap_new = np.copy(tracknumbermap)
                
                # Removing current cell from tmap
                numbah = tracknumbermap.data[lat_idx,lon_idx]
                
                # This if statement is for the case where the cell exists at t = 0
                # It removes the current cell before calculating percentages
                if np.isnan(numbah) == 0:
                    ind = tracknumbermap == numbah
                    # tmap_new[ind] = np.nan
                    tmap_new[ind] = 0
                
                # Binarizing tmap_new
                ind = tmap_new > 0 # not necessary maybe
                tmap_new[ind] = 1
            
                # Find where cell exists in cookiecutter
                # Carve those points out in tmap_new
                # Find out percentage of tmap_new occupied by other cells
                ind = cookiecutter > 0
                ovlap = np.zeros_like(tmap_new)
             
                ovlap[ind] = tmap_new[ind]
            
                ind1 = len(np.where(ovlap > 0)[0]) # np.count_nonzero(ovlap)
                ind2 = len(np.where(cookiecutter > 0)[0]) # np.count_nonzero(cookiecutter)
                opct = ind1/ind2*100
                
                # Modifying da_cc
                cc_mod = da_cc.data
                cc_mod[np.isnan(cc_mod)] = 0
                tracknumbermap_mod = (zcloud + cc_mod) # Merge the cloudy regions with tmap
                tracknumbermap_mod[tracknumbermap_mod > 0] = 1           # binarize and re-label
                tracknumbermap_label = xr.DataArray(label(tracknumbermap_mod),coords=tracknumbermap.coords, dims=tracknumbermap.dims )
                                
                # Need to find a (any) cell index corresponding to the current itracknum
                ind_tmap = np.where(da_cc == 1)
                # Find the corresponding cell index in the re-labelled tracknumbermap
                correct = tracknumbermap_label.data[ind_tmap[0][0],ind_tmap[1][0]]
                
                ipos = np.where(tracknumbermap_label == correct)
                tracknumbermap_evo = np.zeros_like(tracknumbermap)
                tracknumbermap_evo[ipos[0].min():ipos[0].max(),ipos[1].min():ipos[1].max()] = 1
                tracknumbermap_final = xr.DataArray(tracknumbermap_evo,coords=tracknumbermap.coords, dims=tracknumbermap.dims )
                
#                 import pdb; pdb.set_trace()
#                 # For testing the overlapping updrafts 
#                 tmp = np.zeros_like(tracknumbermap_label.data)
#                 ind = np.where(tracknumbermap_label == correct)
#                 tmp[ind] = 1
#                 
#                 plt.clf
#                 f1 = plt.figure(figsize=(5, 5))
#                 pm = plt.pcolormesh(tmp[900:1100,400:600])
#                 pm = plt.pcolormesh(cookiecutter[900:1100,400:600])
#                 pm = plt.pcolormesh(tracknumbermap[900:1100,400:600])
#                 plt.colorbar(pm)
#                 plt.savefig('/ccsopen/home/enochjo/test1.png')

                
#                 if opct > 0: # For testing opct
#                     import pdb; pdb.set_trace()
#                     
#                     from matplotlib import pyplot as plt
#                     f1 = plt.figure(figsize=(5, 5))
#                     pm = plt.pcolormesh(tmap_new)
#                     pm = plt.pcolormesh(tracknumbermap)
#                     pm = plt.pcolormesh(cookiecutter)
#                     plt.colorbar(pm)
#                     plt.savefig('/ccsopen/home/enochjo/test1.png')
                                
                
                
                iW = WA.where(tracknumbermap_final == 1, drop=True).squeeze().data
                iQ = QA.where(tracknumbermap_final == 1, drop=True).squeeze().data
#                 iQC = QC.where(tracknumbermap_final == 1, drop=True).squeeze().data
#                 iQR = QR.where(tracknumbermap_final == 1, drop=True).squeeze().data
                iQV = qv.where(tracknumbermap_final == 1, drop=True).squeeze().data
                iMassFlux = MassFlux.where(tracknumbermap_final == 1, drop=True).squeeze().data
                idBZ = dBZ.where(tracknumbermap_final == 1, drop=True).squeeze().data
                iThte = Thte.where(tracknumbermap_final == 1, drop=True).squeeze().data
                iThtv = Thtv.where(tracknumbermap_final == 1, drop=True).squeeze().data
                iVapr = Vapr.where(tracknumbermap_final == 1, drop=True).squeeze().data
                iEntr = Entr.where(tracknumbermap_final == 1, drop=True).squeeze().data
                iDetr = Detr.where(tracknumbermap_final == 1, drop=True).squeeze().data
                iTrho = Trho.where(tracknumbermap_final == 1, drop=True).squeeze().data
                iPres = PRESSURE.where(tracknumbermap_final == 1, drop=True).squeeze().data
                iMrho = Mrho.where(tracknumbermap_final == 1, drop=True).squeeze().data
                iRH = RH.where(tracknumbermap_final == 1, drop=True).squeeze().data # EJ
                zTnum = da_cc.where(tracknumbermap_final == 1, drop=True).squeeze().data
#                 iTnum = np.repeat(zTnum[np.newaxis,:,:],100,axis=0)
                # This array is going to be populated with the "z" loop
#                 iCor1 = np.zeros_like(iW)
                                
                # Calculate new statistics of the cell
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    
#                     ind = np.where(np.isnan(iTnum) == 1)
#                     iTnum[ind] = 0
#                     
#                     cell_cloudy = np.full(iW.shape, 0, dtype=np.float32)     # Create zero array
#                     icloud = (iW > W_up_thresh) & (iQ > Q_up_thresh)         # Find cloudy region
#                     cell_cloudy[icloud] = 1
#                     tracknumbermap_mod = (cell_cloudy + iTnum)               # Merge the cloudy regions with tmap
#                     tracknumbermap_mod[tracknumbermap_mod > 0] = 1           # binarize and re-label        
#                     tmap_label = label(tracknumbermap_mod)
#                     
#                     # This section ensures that we are only analyzing regions that correspond to our current reflectivity track
#                     # ... and not some neighboring track that has encroached on our territory.
#                     iW_mask = np.zeros_like(iW)
#                     ind = np.where(iTnum == 1)
#                     ind_conv = tmap_label[ind[0][0],ind[1][0]]
#                     ind = tmap_label == ind_conv
#                     iW_mask[ind] = 1
#                     
#                     # Z Loop to obtain vertically-aligned updrafts.
#                     for z in range(0, nz):
#                         dict_up = label_cores(iW[z,:,:]*iW_mask[z,:,:], W_up_thresh, iQ[z,:,:], Q_up_thresh, iMassFlux[z,:,:], ncores_min, min_core_npix, method='>')
#                         iCor1[z,:,:] = dict_up['core_label'] # EJ
#                         
#                     # You should have a fully populated 3D (x,y,z) iCor1 variable here
#                     # Then do the 3d labelling here.
#                     
#                     ibiry = np.zeros_like(iCor1).astype(int)
#                     ibiry[iCor1>0] = 1 # Converting the sort-of-3D core array to binary
#                     labels_out = cc3d.connected_components(ibiry,connectivity=6) # Find core array in 3D
#                     core_idx,core_sze = np.unique(labels_out, return_counts=True) # unique labels
#                     core_idx = np.delete(core_idx,0) # Remove the values corresponding to 0
#                     core_sze = np.delete(core_sze,0)                    
#                     sort_idx = core_sze.argsort()[::-1] # Ordering based on size of updraft (could change to VMF later if necessary)
#                     
#                     # This loop re-labels the 3D cores according to size of updraft
#                     iCore = np.zeros_like(iCor1)
#                     cc = 1
#                     for i in range(0,len(core_idx)):
#                         ind = np.where(labels_out == core_idx[sort_idx][i])
#                         iCore[ind] = cc
#                         cc = cc+1

                    for z in range(0, nz):                        
                        zW = iW[z,:,:]
                        zQ = iQ[z,:,:] #EJ
#                         zQC = iQC[z,:,:] #EJ
#                         zQR = iQR[z,:,:] #EJ
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
                        zRH = iRH[z,:,:] #EJ
#                         zCore = iCore[z,:,:]
                        
                        zz = z
                        if (z < 1): zz = 1
                        if (z > 98): zz = 98
                        
                        zPrs1 = iPres[zz-1,:,:] *100 #EJ
                        zPrs2 = iPres[zz+1,:,:] *100 #EJ
                        
#                         from matplotlib import pyplot as plt
#                         import pdb; pdb.set_trace()
#                         plt.clf
#                         f1 = plt.figure(figsize=(5, 5))
# #                         pm = plt.pcolormesh(tmap_label)
#                         pm = plt.pcolormesh(zW_mask)
#                         plt.colorbar(pm)
#                         plt.savefig('/ccsopen/home/enochjo/test1.png')
                        
                        ind = np.where(np.isnan(zTnum) == 1)
                        zTnum[ind] = 0
                        
                        cell_cloudy = np.full(zW.shape, 0, dtype=np.float32)     # Create zero array
                        icloud = (zW > W_up_thresh) & (zQ > Q_up_thresh)         # Find cloudy region
                        cell_cloudy[icloud] = 1                                  # Set cloudy region to 1
                        tracknumbermap_mod = (cell_cloudy + zTnum)               # Merge the cloudy regions with tmap
                        tracknumbermap_mod[tracknumbermap_mod > 0] = 1           # binarize and re-label        
                        tmap_label = label(tracknumbermap_mod)
                        
                        zW_mask = np.zeros_like(zW)
                        ind = np.where(zTnum == 1)
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
                        
#                         # Label updraft cores (EJ)
#                         core_numbers_up,core_npix_up = np.unique(zCore, return_counts=True)
#                         core_numbers_up = np.delete(core_numbers_up,0) # Excluding 0
#                         core_npix_up = np.delete(core_npix_up,0)
#                         ncores_all_up = len(core_numbers_up)
#                         ncores_up = np.nanmin([ncores_all_up, ncores_min])
#                         core_label_up = zCore
                        
                        # Find Centroids here (EJ)
                        cpoints = np.zeros((len(core_numbers_up),2))
                        for i in range(0,len(core_numbers_up)):
                            tmp_image = np.zeros_like(core_label_up)
                            ind = np.where(core_label_up==core_numbers_up[i])
                            tmp_image[ind] = 1
                            cpoints[i,:] = centroid(tmp_image)
                        
                        # Label downdraft cores
                        # dict_down = label_cores(zW, W_down_thresh, zQ, Q_down_thresh, zMassFlux, ncores_min, min_core_npix, method='<')
                        # ncores_all_down = dict_down['ncores_all']
                        # ncores_down = dict_down['ncores_save']
                        # core_npix_down = dict_down['core_npix']
                        # core_numbers_down = dict_down['core_numbers']
                        # core_label_down = dict_down['core_label']
                        
                        # Dilate core labels by a certain number of pixels
                        core_label_up_prm = np.zeros_like(core_label_up)
                        for ii in range(ncores_up):
                            cell = np.zeros_like(core_label_up)
                            cell[core_label_up == core_numbers_up[ii]] = 1
                            expand = round(np.sqrt(core_npix_up[ii]/np.pi))
                            dil = expand_labels(cell, distance = expand)
                            core_label_up_prm[(dil - cell) == 1] = core_numbers_up[ii]
                            
                        # Getting rid of all the perimeter labels that exist within adjacent cores.
                        # core_label_up_prm[core_label_up > 0] = 0
                        
#                         # The old way of doing things
#                         struct = generate_binary_structure(2, 6)                        
#                         core_label_up_dil = np.zeros_like(core_label_up)
#                         core_label_up_prm = np.zeros_like(core_label_up)
#                         for ii in range(ncores_up):
#                             cell = np.zeros_like(core_label_up)
#                             cell[core_label_up == core_numbers_up[ii]] = 1
#                             dil = binary_dilation(cell, structure = struct, iterations = 2)
#                             core_label_up_dil[dil == 1] = core_numbers_up[ii]
#                             core_label_up_prm[(dil - cell) == 1] = core_numbers_up[ii]

                        core_label_up_dil = expand_labels(core_label_up, distance=2) # Includes core
                        # core_label_up_prm = expand_labels(core_label_up, distance=2) - core_label_up # Just perimeter
                        
                        # Total number of cores
                        cell_nCore_up[icell, z] = ncores_all_up
                        cell_ovlap_up[icell, z] = opct # EJ
                        
                        # Calculate core statistics
                        # MaFlx_sum_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        MaFlx_core_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        W_max_up = np.full(ncores_up, np.NaN, dtype=np.float32)
                        W_mean_up = np.full(ncores_up, np.NaN, dtype=np.float32)
#                         QC_max_up = np.full(ncores_up, np.NaN, dtype=np.float32) # EJ
#                         QC_mean_up = np.full(ncores_up, np.NaN, dtype=np.float32) # EJ
#                         QC_mean_prm = np.full(ncores_up, np.NaN, dtype=np.float32) # EJ
                        QV_mean_up = np.full(ncores_up, np.NaN, dtype=np.float32) # EJ
                        QV_mean_prm = np.full(ncores_up, np.NaN, dtype=np.float32) # EJ
#                         QR_max_up = np.full(ncores_up, np.NaN, dtype=np.float32) # EJ
#                         QR_mean_up = np.full(ncores_up, np.NaN, dtype=np.float32) # EJ
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
                        NS_up = np.full(ncores_up, np.NaN, dtype=np.float32) # EJ
                        WE_up = np.full(ncores_up, np.NaN, dtype=np.float32) # EJ
                        RH_mean_up = np.full(ncores_up, np.NaN, dtype=np.float32) # EJ
                        RH_mean_prm = np.full(ncores_up, np.NaN, dtype=np.float32) # EJ
                        
                        
                        for ii in range(ncores_up):                            
                            MaFlx_core_up[ii] = np.nansum(zMassFlux[core_label_up == core_numbers_up[ii]])
                            W_max_up[ii] = np.nanmax(zW[core_label_up == core_numbers_up[ii]])
                            W_mean_up[ii] = np.nanmean(zW[core_label_up == core_numbers_up[ii]])
#                             QC_max_up[ii] = np.nanmax(zQC[core_label_up == core_numbers_up[ii]]) # EJ
#                             QC_mean_up[ii] = np.nanmean(zQC[core_label_up == core_numbers_up[ii]]) # EJ
#                             QC_mean_prm[ii] = np.nanmean(zQC[core_label_up_prm == core_numbers_up[ii]]) # EJ
#                             QR_max_up[ii] = np.nanmax(zQR[core_label_up == core_numbers_up[ii]]) # EJ
#                             QR_mean_up[ii] = np.nanmean(zQR[core_label_up == core_numbers_up[ii]]) # EJ
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
                            NS_up[ii] = ipos[0].min()+cpoints[ii,0] # EJ
                            WE_up[ii] = ipos[1].min()+cpoints[ii,1]
                            RH_mean_up[ii] = np.nanmean(zRH[core_label_up == core_numbers_up[ii]]) # EJ
                            RH_mean_prm[ii] = np.nanmean(zRH[core_label_up_prm == core_numbers_up[ii]]) # EJ
                            
                            Pres_pert_top[ii] = np.nanmean(zPrs2[core_label_up_prm == core_numbers_up[ii]])\
                                - np.nanmax(zPrs2[core_label_up == core_numbers_up[ii]])
                            Pres_pert_bot[ii] = np.nanmean(zPrs1[core_label_up_prm == core_numbers_up[ii]])\
                                - np.nanmax(zPrs1[core_label_up == core_numbers_up[ii]])
                            Mrho_mean_up[ii] = np.nanmean(zMrho[core_label_up == core_numbers_up[ii]])
                            PGF_up[ii] = - 1/Mrho_mean_up[ii]\
                                * ( Pres_pert_top[ii] - Pres_pert_bot[ii] )/200
                        
                        
                        # Calculate total mass flux for all labeled cores
                        MaFlx_sum_up = np.nansum(zMassFlux[core_label_up > 0])
                        
                        # Save data to output arrays
                        ncores_save_up = min([ncores_up, ncores_min])
                        cell_MassFlux_up[icell, z] = MaFlx_sum_up * DX * DY
                        cell_CoreMassFlux_up[icell, z, 0:ncores_save_up] = MaFlx_core_up[0:ncores_save_up] * DX * DY
                        cell_CoreArea_up[icell, z, 0:ncores_save_up] = core_npix_up[0:ncores_save_up] * grid_area
                        cell_CoreMaxW_up[icell, z, 0:ncores_save_up] = W_max_up[0:ncores_save_up]
                        cell_CoreMeanW_up[icell, z, 0:ncores_save_up] = W_mean_up[0:ncores_save_up]
                        
#                         cell_CoreMaxQC_up[icell, z, 0:ncores_save_up] = QC_max_up[0:ncores_save_up] # EJ
#                         cell_CoreMeanQC_up[icell, z, 0:ncores_save_up] = QC_mean_up[0:ncores_save_up] # EJ
#                         cell_CoreMeanQC_prm[icell, z, 0:ncores_save_up] = QC_mean_prm[0:ncores_save_up] # EJ
#                         cell_CoreMaxQR_up[icell, z, 0:ncores_save_up] = QR_max_up[0:ncores_save_up] # EJ
#                         cell_CoreMeanQR_up[icell, z, 0:ncores_save_up] = QR_mean_up[0:ncores_save_up] # EJ
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
                        cell_NS_up[icell , z, 0:ncores_save_up] = NS_up[0:ncores_save_up] # EJ
                        cell_WE_up[icell , z, 0:ncores_save_up] = WE_up[0:ncores_save_up] # EJ
                        cell_rhMean_up[icell , z, 0:ncores_save_up] = RH_mean_up[0:ncores_save_up] # EJ
                        cell_rhMean_prm[icell , z, 0:ncores_save_up] = RH_mean_prm[0:ncores_save_up] # EJ

                    ind = np.where(idBZ > 10)[0]
                    if ind.size > 0: 
                        cell_maxETH_10dbz[icell] = np.max(ind)/10 # to convert to km.
                    cell_max_dbz[icell] = np.nanmax(idBZ,axis=(0,1,2))
                    
            else:
                # print(f'No cell matching track # {itracknum}')
                # Note that this error message will be mentioned if the relevant files do not exist
                # It does not necessarily mean that the tracks do not exist. 
                # Therefore changed the error/warning message:
                if wcount == 0:
                    print('Met file does not exist')
                    wcount = wcount + 1
                
        out_dict3d = {
            'CoreArea_up': cell_CoreArea_up,
            'CoreMaxW_up': cell_CoreMaxW_up,
            'CoreMeanW_up': cell_CoreMeanW_up,
#             'CoreMaxQC_up': cell_CoreMaxQC_up,
#             'CoreMeanQC_up': cell_CoreMeanQC_up,
#             'CoreMeanQC_prm': cell_CoreMeanQC_prm,
#             'CoreMaxQR_up': cell_CoreMaxQR_up,
#             'CoreMeanQR_up': cell_CoreMeanQR_up,
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
            'CoreNS_up': cell_NS_up,
            'CoreWE_up': cell_WE_up,
            'CoreRHMean_up': cell_rhMean_up,
            'CoreRHMean_prm': cell_rhMean_prm,
            'Entr_up': cell_Entr_up,
            'Detr_up': cell_Detr_up,
            'Vapor_up': cell_Vapr_up,
        }
        out_dict2d = {
            'nCore_up': cell_nCore_up,
            'MassFlux_up': cell_MassFlux_up,
            'Ovlap_up': cell_ovlap_up,
        }
        out_dict1d = {
            'maxETH_10dbz': cell_maxETH_10dbz,
            'max_dbz': cell_max_dbz,
        }
        out_dict_attrs = {
            # Updraft
            'nCore_up': {
                'long_name': 'Number of updraft cores',
                'units': 'count',
            },
            'maxETH_10dbz': {
                'long_name': 'Maximum 10dBZ echo-top height in a convective cell',
                'units': 'km',
            },
            'max_dbz': {
                'long_name': 'Maximum reflectivity in a convective cell',
                'units': 'dBZ',
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
#             'CoreMaxQC_up': {
#                 'long_name': 'Updraft core mean QC',
#                 'units': 'kg/kg',
#             },
#             'CoreMeanQC_up': {
#                 'long_name': 'Updraft core mean QC',
#                 'units': 'kg/kg',
#             },
#             'CoreMeanQC_prm': {
#                 'long_name': 'Updraft perim mean QC',
#                 'units': 'kg/kg',
#             },
#             'CoreMaxQR_up': {
#                 'long_name': 'Updraft core mean QR',
#                 'units': 'kg/kg',
#             },
#             'CoreMeanQR_up': {
#                 'long_name': 'Updraft core mean QR',
#                 'units': 'kg/kg',
#             },
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
            'CoreNS_up': {
                'long_name': 'North-South Location of Updraft Centroid',
                'units': 'Grid points',
            },
            'CoreWE_up': {
                'long_name': 'West-East Location of Updraft Centroid',
                'units': 'Grid points',
            },
            'CoreRHMean_up': {
                'long_name': 'Mean RH within updraft core',
                'units': '%',
            },
            'CoreRHMean_prm': {
                'long_name': 'Mean RH within updraft perimeter',
                'units': '%',
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
        }
    return out_dict3d, out_dict2d, out_dict1d, out_dict_attrs


#-----------------------------------------------------------------------
if __name__ == '__main__':

    # Get configuration file name from input
    config_file = sys.argv[1]
    # track_start = int(sys.argv[2])
    # track_end = int(sys.argv[3])
    # digits = int(sys.argv[4])
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
    met_filebase = config['met_filebase']
    cld_filebase = config['cld_filebase']
    pixel_filebase = config['pixel_filebase']
    entfile_path = config['entfile_path'] # EJ
    ent_filebase = config['ent_filebase'] # EJ
    ncores_min = config['ncores_min']
    wrfout_path1 = config['wrfout_path1']
    wrfout_path2 = config['wrfout_path2']
    nhours = config['nhours']
    nminutes = config['nminutes']
    # ntimes_max = config['ntimes_max']
    ensmember = config['ensmember']
    domain = config['domain']
    nx = config['nx']
    ny = config['ny']

    # # Add ensemble member to WRF path
    # _startdate = startdate[0:8]
    # base_date = f"{_startdate[0:4]}-{_startdate[4:6]}-{_startdate[6:8]}T00"
    # ensmember_long = ensmemb_short_to_long(ensmember)
    # wrfout_path1 = f'{wrfout_path1}{_startdate}_{ensmember_long}/run/merged/'
    # wrfout_path2 = f'{wrfout_path2}{_startdate}/{ensmember_long}/run/merged/'
    # # Check which directory exists
    # if os.path.isdir(wrfout_path1):
    #     wrfout_path = wrfout_path1
    # elif os.path.isdir(wrfout_path2):
    #     wrfout_path = wrfout_path2
    # else:
    #     print(f'WRF path does not exist: {wrfout_path1}')
    #     print(f'WRF path does not exist: {wrfout_path2}')
    #     print(f'Code will exit now.')
    #     sys.exit()

    # Add start/end date to pixel file path
    pixelfile_path = f'{pixelfile_path}{startdate}_{enddate}/'

    # Output statistics filename
    # track_start_str = str(track_start).zfill(digits)
    # output_filename = f'{output_path}stats_3d_env_{startdate}_{enddate}_t{track_start_str}.nc'
    output_filename = f'{output_path}stats_3d_env_{startdate}_{enddate}.nc'
    os.makedirs(output_path, exist_ok=True)

    # Track statistics file dimension names
    tracks_dimname = 'tracks'
    times_dimname = 'times'
    z_dimname = 'z'
    core_dimname = 'core'

    # Track statistics file
    stats_filebase = 'trackstats_'
    trackstats_file = f'{stats_path}{stats_filebase}{startdate}_{enddate}.nc'

    # Read track statistics file
    # import pdb; pdb.set_trace()
    print(trackstats_file)
    dsstats = xr.open_dataset(trackstats_file, decode_times=True)
    # Subset stats times to reduce array size
    # dsstats = dsstats.isel(times=slice(0, ntimes_max))
    # ntracks = dsstats.dims[tracks_dimname]
    ntracks = dsstats.dims[tracks_dimname]
    ntimes = dsstats.dims[times_dimname] # EJ change this
    coord_tracks = dsstats['tracks']
    stats_basetime = dsstats['base_time']
    stats_lon = dsstats['meanlon']
    stats_lat = dsstats['meanlat']
    # ntimes = dsstats.dims[times_dimname]
    # coord_tracks = dsstats['tracks'].sel(tracks=slice(track_start, track_end))
    # stats_basetime = dsstats['base_time'].sel(tracks=slice(track_start, track_end))
    # stats_lon = dsstats['meanlon'].sel(tracks=slice(track_start, track_end))
    # stats_lat = dsstats['meanlat'].sel(tracks=slice(track_start, track_end))
    # ntracks = len(coord_tracks)
    time_res_hour = dsstats.attrs['time_resolution_hour']
    dsstats.close()

    print(f'Total Number of Tracks: {ntracks}')

    # Convert time resolution of data to minutes
    # time_res_min = np.round(time_res_hour * 60).astype(int)
    time_res_sec = np.round(time_res_hour * 60 * 60).astype(int) # EJ
    # ntimes_per_hour = np.round(60. / time_res_min).astype(int)

    # Select initiation time, and round to the nearest minute
    time0 = stats_basetime.isel(times=0).dt.round('S')
    # Get initiation lat/lon    
    stats_lon0 = stats_lon.isel(times=0).data
    stats_lat0 = stats_lat.isel(times=0).data
    
    # Make an array to store the full time series
#     ntimes_prior = np.ceil(nhours / time_res_hour).astype(int)
    ntimes_prior =  np.ceil(nminutes * 60 / time_res_sec).astype(int) # EJ
    # ntimes_full = np.ceil(ntimes_prior + ntimes_max).astype(int)
    ntimes_full = np.ceil(ntimes_prior + 1).astype(int)
    # 
    full_times = np.ndarray((ntracks, ntimes_full), dtype='datetime64[ns]')
    full_basetimes = np.full((ntracks, ntimes_full), np.NaN, dtype=float)
    full_lons = np.full((ntracks, ntimes_full), np.NaN, dtype=np.float32)
    full_lats = np.full((ntracks, ntimes_full), np.NaN, dtype=np.float32)

    # Get track data numpy arrays for better performance
    stats_min0 = time0.data
    # stats_mins = stats_basetime.dt.round(f'{time_res_min:.0f}min').data
    stats_mins = stats_basetime.data
    stats_lons = stats_lon.data
    stats_lats = stats_lat.data
    
    # ntimes = ntimes_prior # Doesn't work for some reason
    
    #import pdb; pdb.set_trace()

    # Loop over each track
    for itrack in range(0, ntracks):
        # Calculate start/end times prior to initiation
        # time0_start = stats_hour0[itrack] - pd.offsets.Hour(nhours-1)
        # time0_end = stats_hour0[itrack] - pd.offsets.Hour(1)
        # time0_start = stats_min0[itrack] - pd.offsets.Hour(nhours)
        time0_start = stats_min0[itrack] - pd.offsets.Minute(nminutes) # EJ
        time0_end = stats_min0[itrack] - pd.offsets.Second(time_res_sec) # EJ
        # Generate hourly time series leading up to initiation
        prior_times = np.array(pd.date_range(time0_start, time0_end, freq=f'{time_res_sec:.0f}S'))

        # Save full history of times
        full_times[itrack,0:ntimes_prior] = prior_times
        full_times[itrack,ntimes_prior] = stats_mins[itrack,0]
        # full_times[itrack,ntimes_prior:] = stats_mins[itrack,:]

        # Convert full times to Epoch time
        itimes = full_times[itrack,:]
        # Find indices that is a time
        idx = ~np.isnat(itimes)
        full_basetimes[itrack, idx] = np.array([tt.tolist()/1e9 for tt in itimes[idx]])

        # Repeat initiation lat/lon by X hours (i.e., stay at the initiation location)
        ilon0 = np.repeat(stats_lon0[itrack], ntimes_prior)
        ilat0 = np.repeat(stats_lat0[itrack], ntimes_prior)
        # Save full history of lat/lon
        full_lons[itrack,0:ntimes_prior] = ilon0
        full_lats[itrack,0:ntimes_prior] = ilat0
        full_lats[itrack,ntimes_prior] = stats_lats[itrack,0]
        full_lons[itrack,ntimes_prior] = stats_lons[itrack,0]
        # full_lats[itrack,ntimes_prior:] = stats_lats[itrack,:]
        # full_lons[itrack,ntimes_prior:] = stats_lons[itrack,:]

    # # Convert to Xarray DataArray
    # coord_relativetimes = np.arange(-ntimes_prior, ntimes_max, 1)
    coord_relativetimes = np.arange(-ntimes_prior, 1, 1)
    coord_relativetimes_attrs = {
        'description': 'Relative times for track lifecycle',
        'units': 'unitless',
        'comment': f'Multiply by {time_res_sec:.0f}S to get physical time',
    }
    full_basetimes_attrs = {
        'long_name': stats_basetime.attrs['long_name'],
        'units': 'Seconds since 1970-1-1',
    }
    out_ntimes = len(coord_relativetimes)
    # Check number of times for output
    if out_ntimes != ntimes_full:
        print(f'out_ntimes ({out_ntimes}) != ntimes_full ({ntimes_full})')
        sys.exit()

    # Find unique valid times
    uniq_times = np.unique(full_times)
    uniq_times = uniq_times[~np.isnat(uniq_times)]
    nfiles = len(uniq_times)
    # Convert unique times to basetime
    uniq_basetimes = np.array([tt.tolist()/1e9 for tt in uniq_times])
    # import pdb; pdb.set_trace()


    ##############################################################
    # Call function to calculate statistics
    trackindices_all = []
    timeindices_all = []
    results = []

    if run_parallel == 1:
        # Initialize dask
        dask_tmp_dir = config.get("dask_tmp_dir", "/tmp")
        dask.config.set({'temporary-directory': dask_tmp_dir})
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker)
        client = Client(cluster)
    
    # import pdb; pdb.set_trace()
    # Loop over each pixel-file and call function to calculate
    for ifile in range(nfiles):
#     for ifile in range(300,350):
    # for ifile in range(0, 12):
        # Convert time string to match different files
        itime = uniq_times[ifile]
        itime_pixel = pd.to_datetime(str(itime)).strftime('%Y%m%d_%H%M%S')
        itime_wrfout = pd.to_datetime(str(itime)).strftime('%Y-%m-%d_%H_%M_%S')
        itime_met = pd.to_datetime(str(itime)).strftime('%Y%m%d.%H%M%S')
        itime_cld = pd.to_datetime(str(itime)).strftime('%Y%m%d.%H%M%S')

        # File names
        fname_pixel = f'{pixelfile_path}{pixel_filebase}{itime_pixel}.nc'
        # fname_wrfout = f'{wrfout_path}wrfout_{domain}_{itime_wrfout}'
        # New MET file time format: yyyymmdd.hhmmss
        fname_met = f'{metfile_path}{met_filebase}{itime_met}.nc'
        fname_cld = f'{metfile_path}{cld_filebase}{itime_met}.nc'
        fname_ent = f'{entfile_path}{ent_filebase}{itime_met}.nc'

        # Get all MCS tracks/times indices in the same time (file)
        idx_track, idx_time = np.where(full_basetimes == uniq_basetimes[ifile])
        
        # Realized that 'cell_area' is the same as 'area'
        # careas = dsstats['cell_area'].isel(times = 0).isel(tracks = idx_track) # This is the expanded area
        careas = dsstats['core_area'].isel(times = 0).isel(tracks = idx_track) # We now want the core area
        # careas2 = dsstats['area'].isel(times = 0).isel(tracks = idx_track)
        # cradii = np.ceil( (np.sqrt(careas/np.pi).data + 5 )*1000/100).astype(int) # inflation
        cradii = np.ceil(np.sqrt(careas/np.pi).data*1000/100+1).astype(int) # round up
#         import pdb;pdb.set_trace()

        if len(idx_track) > 0:
            # Save matchindices for the current pixel file to the overall list
            trackindices_all.append(idx_track)
            timeindices_all.append(idx_time)

            # Get the track lat/lon/time values
            _lat = full_lats[idx_track, idx_time]
            _lon = full_lons[idx_track, idx_time]
            
            # Serial
            if run_parallel == 0:
                result = extract_env_prof(
                    fname_pixel, 
                    # fname_wrfout,
                    fname_met,
                    fname_cld,
                    fname_ent,
                    idx_track, 
                    _lat,
                    _lon,
                    config,
                    cradii,
                )
                results.append(result)
            # Parallel
            elif run_parallel >= 1:
                result = dask.delayed(extract_env_prof)(
                    fname_pixel, 
                    # fname_wrfout,
                    fname_met,
                    fname_cld,
                    fname_ent,
                    idx_track, 
                    _lat,
                    _lon,
                    config,
                    cradii,
                )
                results.append(result)
            else:
                print(f'Invalid parallization option run_parallel: {run_parallel}')

    # import pdb; pdb.set_trace()
    
    final_results = results

    # Trigger dask computation
    if run_parallel == 0:
        final_results = results
    elif run_parallel >= 1:
        final_results = dask.compute(*results)
        wait(final_results)
    else:
        print(f'Invalid parallization option run_parallel: {run_parallel}')

    
    # Make a variable list and get attributes from one of the returned dictionaries
    # Loop over each return results till one that is not None
    counter = len(final_results)-1
    while counter >= 0:
        if final_results[counter] is not None:
            var_names3d = list(final_results[counter][0].keys())
            var_names2d = list(final_results[counter][1].keys())
            var_names1d = list(final_results[counter][2].keys())
            var_attrs = final_results[counter][3]
            break
        counter -= 1
        
#     import pdb; pdb.set_trace()

    # Read a MET file to get vertical coordinates
    dsm = xr.open_dataset(fname_met)
    nz = dsm.dims['HAMSL']
    height = dsm['HAMSL']
    dsm.close()
    
    # Loop over variable list to create the dictionary entry
    print(f'Creating output arrays ...')
    out_dict = {}
    out_dict_attrs = {}

    var_names = var_names3d + var_names2d + var_names1d
    # 3D variables 
    for ivar in var_names3d:
        out_dict[ivar] = np.full((ntracks, ntimes, nz, ncores_min), np.nan, dtype=np.float32)
        out_dict_attrs[ivar] = var_attrs[ivar]
    # 2D variables
    for ivar in var_names2d:
        out_dict[ivar] = np.full((ntracks, ntimes, nz), np.nan, dtype=np.float32)
        out_dict_attrs[ivar] = var_attrs[ivar]
    # 1D variables
    for ivar in var_names1d:
        out_dict[ivar] = np.full((ntracks, ntimes), np.nan, dtype=np.float32)
        out_dict_attrs[ivar] = var_attrs[ivar]

    # Put the results to output track stats variables
    # Loop over each returned results
    for ifile in range(len(final_results)):
        # Check the return results
        if final_results[ifile] is not None:
            iVAR3d = final_results[ifile][0]
            iVAR2d = final_results[ifile][1]
            iVAR1d = final_results[ifile][2]
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
            if iVAR1d is not None:
                trackindices = trackindices_all[ifile]
                timeindices = timeindices_all[ifile]
                # Loop over each variable and assign values to output dictionary
                for ivar in var_names1d:
                    if iVAR1d[ivar].ndim == 1:
                        out_dict[ivar][trackindices,timeindices] = iVAR1d[ivar]
                    else:
                        print(f'Warning: {ivar} dimension is not 1.')
    
#     # import pdb; pdb.set_trace()
#     from matplotlib import pyplot as plt
#     tmp1 = np.nanmax(out_dict['CoreArea_up'][:,:,:,0],axis=(0))
#     # Shape:(685, 720, 100, 35)
#     f1 = plt.figure(figsize=(9, 3))
#     plt.pcolormesh(tmp1)
#     plt.savefig('/ccsopen/home/enochjo/test.png')
#     # Confirmed that things are being generated.
    
    
    ##########################################################
    # Write to netcdf
    print('Writing output netcdf ... ')

    # Define variable list
    var_dict = {}
    # Define output variable dictionary
    for key, value in out_dict.items():
        if value.ndim == 1:
            var_dict[key] = ([tracks_dimname], value, out_dict_attrs[key])
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
    
#     import pdb; pdb.set_trace()
#     from matplotlib import pyplot as plt
#     tmp1 = dsout['CoreArea_up'].max(dim=('tracks','core'))
#     tmp1 = np.nanmax(out_dict['CoreArea_up'][:,:,:,0],axis=(0))
#     # Shape:(685, 720, 100, 35)
#     f1 = plt.figure(figsize=(9, 3))
#     plt.pcolormesh(tmp1)
#     plt.savefig('/ccsopen/home/enochjo/test.png')
    
#     import pdb;pdb.set_trace()
    # Set encoding/compression for all variables
    comp = dict(zlib=True)
    # comp = dict(zlib=True, complevel=0)
    encoding = {var: comp for var in dsout.data_vars }

    # Write to netcdf file
    dsout.to_netcdf(path=output_filename, mode="w",
                    format="NETCDF4", unlimited_dims=tracks_dimname, encoding=encoding)
    print(f'Output saved: {output_filename}')