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
def pad_array(in_array, lat_idx, lon_idx, ny, nx, ny_d, nx_d, sub_y=1, sub_x=1, fillval=np.NaN):
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
            Number of 1/2 grids to extract data in y dimension
        nx: int
            Number of 1/2 grids to extract data in x dimension
        ny_d: int
            Number of grids in the domain in y dimension
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
    # Check array dimensions
    ndim = in_array.ndim
    if ndim == 3:
        # Subset array within the domain
        in_array = in_array[:, iy_min:iy_max, ix_min:ix_max]
        # Pad array on y & x dimensions
        out_array = np.pad(in_array, ((0,0), (pny_b,pny_t), (pnx_l,pnx_r)), 'constant', constant_values=fillval)
        # Sub-sample array
        out_array = out_array[:, ::sub_y, ::sub_x]
    if ndim == 2:
        # Subset array within the domain
        in_array = in_array[iy_min:iy_max, ix_min:ix_max]
        # Pad array on y & x dimensions
        out_array = np.pad(in_array, ((pny_b,pny_t), (pnx_l,pnx_r)), 'constant', constant_values=fillval)
        # Sub-sample array
        out_array = out_array[::sub_y, ::sub_x]
    return out_array

#-----------------------------------------------------------------------
def extract_env_prof(
    fname_pixel, 
    # fname_wrfout,
    fname_met,
    fname_cld,
    idx_track, 
    _lat,
    _lon,
    config,
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
    nz = config.get('nz', 149)
    sub_x = config.get('sub_x', 1)
    sub_y = config.get('sub_y', 1)
    DX = config.get('DX')
    DY = config.get('DY')

    # Max difference in closest lat/lon point to the cell center [degree]
    dlat_max, dlon_max = 0.04, 0.04

    # Check file existance
    # wrfout_exist = os.path.isfile(fname_wrfout)
    met_exist = os.path.isfile(fname_met)
    cld_exist = os.path.isfile(fname_cld)
    pixel_exist = os.path.isfile(fname_pixel)
    # import pdb; pdb.set_trace()

    # Skip processing if any required file does not exist
    if not (pixel_exist and met_exist and cld_exist):
        print(f'WARNING: Skipping track - missing required file(s):')
        if not pixel_exist:
            print(f'  Missing pixel file: {fname_pixel}')
        if not met_exist:
            print(f'  Missing met file: {fname_met}')
        if not cld_exist:
            print(f'  Missing cld file: {fname_cld}')
        return None, None, None, None

    if pixel_exist:
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
    else:
        pixel_attrs = {
            'conv_core': '',
            'conv_mask': '',
            # 'tracknumber_cmask': '',
            'tracknumber': '',
            'dbz_comp': '',
            'echotop10': '',
        }

    # if wrfout_exist:
    #     # Read WRF out file
    #     print(fname_wrfout)
    #     nc = Dataset(fname_wrfout)
    #     # Domain dimension
    #     nx_d = nc.dimensions['west_east'].size
    #     ny_d = nc.dimensions['south_north'].size
    #     nz = nc.dimensions['bottom_top'].size
    #     # 3D Variables
    #     height = wrf.getvar(nc, 'z', units='m')
    #     tk = wrf.getvar(nc, 'tk')
    #     pressure = wrf.getvar(nc, 'pressure')
    #     qv = wrf.getvar(nc, 'QVAPOR')
    #     rh = wrf.getvar(nc, 'rh')
    #     umet, vmet = wrf.getvar(nc, 'uvmet')
    #     wa = wrf.getvar(nc, 'wa')
    #     # 2D variables
    #     XLAT = wrf.getvar(nc, 'XLAT')
    #     XLONG = wrf.getvar(nc, 'XLONG')
    #     pwv = wrf.getvar(nc, 'pw')
    #     T2 = wrf.getvar(nc, 'T2')
    #     Q2 = wrf.getvar(nc, 'Q2')
    #     PSFC = wrf.getvar(nc, 'PSFC')
    #     U10, V10 = wrf.getvar(nc, 'uvmet10')
    #     PBLH = wrf.getvar(nc, 'PBLH')
    #     RAINNC = wrf.getvar(nc, 'RAINNC')
    #     HGT = wrf.getvar(nc, 'HGT')
    #     # Attributes
    #     DX = nc.getncattr('DX')
    #     DY = nc.getncattr('DY')
    #     # Close WRF file
    #     nc.close()

    #     # Remove attributes ('projection' in particular conflicts with Xarray)
    #     attrs_to_remove = ['FieldType', 'projection', 'MemoryOrder', 'stagger', 'coordinates', 'missing_value']
    #     for key in attrs_to_remove:
    #         pressure.attrs.pop(key, None)
    #         height.attrs.pop(key, None)
    #         tk.attrs.pop(key, None)
    #         qv.attrs.pop(key, None)
    #         rh.attrs.pop(key, None)
    #         umet.attrs.pop(key, None)
    #         vmet.attrs.pop(key, None)
    #         wa.attrs.pop(key, None)
    #         pwv.attrs.pop(key, None)
    #         T2.attrs.pop(key, None)
    #         Q2.attrs.pop(key, None)
    #         PSFC.attrs.pop(key, None)
    #         U10.attrs.pop(key, None)
    #         V10.attrs.pop(key, None)
    #         PBLH.attrs.pop(key, None)
    #         RAINNC.attrs.pop(key, None)
    #         HGT.attrs.pop(key, None)


    if met_exist:
        # Read Met file
        print(fname_met)
        dsm = xr.open_dataset(fname_met)
        nx_d = dsm.sizes['west_east']
        ny_d = dsm.sizes['south_north']
        nz = dsm.sizes['bottom_top']
        # 3D Variables
        height = dsm['HAMSL'].squeeze()
        pressure = dsm['PRESSURE'].squeeze()
        tk = dsm['TEMPERATURE'].squeeze()
        qv = dsm['QVAPOR'].squeeze()
        rh = dsm['RH'].squeeze()
        umet = dsm['UMET'].squeeze()
        vmet = dsm['VMET'].squeeze()
        wa = dsm['WA'].squeeze()
        # 2D variables
        XLAT = dsm['XLAT'].squeeze()
        XLONG = dsm['XLONG'].squeeze()
        T2 = dsm['T2'].squeeze()
        Q2 = dsm['Q2'].squeeze()
        PSFC = dsm['PSFC'].squeeze()
        U10 = dsm['UMET10'].squeeze()
        V10 = dsm['VMET10'].squeeze()
        RAINNC = dsm['RAINNC'].squeeze()
        HGT = dsm['HGT'].squeeze()
        # # Attributes
        # DX = dsm.attrs['DX']
        # DY = dsm.attrs['DY']

        # Remove attributes ('projection' in particular conflicts with Xarray)
        attrs_to_remove = ['FieldType', 'projection', 'MemoryOrder', 'stagger', 'coordinates', 'missing_value']
        for key in attrs_to_remove:
            height.attrs.pop(key, None)
            pressure.attrs.pop(key, None)
            tk.attrs.pop(key, None)
            qv.attrs.pop(key, None)
            rh.attrs.pop(key, None)
            umet.attrs.pop(key, None)
            vmet.attrs.pop(key, None)
            wa.attrs.pop(key, None)
            # pwv.attrs.pop(key, None)
            T2.attrs.pop(key, None)
            Q2.attrs.pop(key, None)
            PSFC.attrs.pop(key, None)
            U10.attrs.pop(key, None)
            V10.attrs.pop(key, None)
            # PBLH.attrs.pop(key, None)
            RAINNC.attrs.pop(key, None)
            HGT.attrs.pop(key, None)

        # Save variable attributes
        pressure_attrs = pressure.attrs
        height_attrs = height.attrs
        temperature_attrs = tk.attrs
        qv_attrs = qv.attrs
        rh_attrs = rh.attrs
        u_attrs = umet.attrs
        v_attrs = vmet.attrs
        w_attrs = wa.attrs
        T2_attrs = T2.attrs
        Q2_attrs = Q2.attrs
        PSFC_attrs = PSFC.attrs
        U10_attrs = U10.attrs
        V10_attrs = V10.attrs
        RAINNC_attrs = RAINNC.attrs
        HGT_attrs = HGT.attrs

    else:
        # Create empty attributes
        pressure_attrs = ''
        height_attrs = ''
        temperature_attrs = ''
        qv_attrs = ''
        rh_attrs = ''
        u_attrs = ''
        v_attrs = ''
        w_attrs = ''
        # PWV_attrs = ''
        T2_attrs = ''
        Q2_attrs = ''
        PSFC_attrs = ''
        U10_attrs = ''
        V10_attrs = ''
        RAINNC_attrs = ''
        HGT_attrs = ''


    if cld_exist:
        # Read Cloud file
        print(fname_cld)
        dsc = xr.open_dataset(fname_cld)
        nx_c = dsc.sizes['west_east']
        ny_c = dsc.sizes['south_north']
        LWP = dsc['LWP'].squeeze()
        IWP = dsc['IWP'].squeeze()
        PWV = dsc['PRECIPWATER'].squeeze()
        # Total liquid condensates (for calculating Emanuel's ThetaE)
        Qliq = (qv + dsc['QCLOUD'] + dsc['QRAIN']).squeeze()
        # import pdb; pdb.set_trace()

        # Remove attributes ('projection' in particular conflicts with Xarray)
        attrs_to_remove = ['FieldType', 'projection', 'MemoryOrder', 'stagger', 'coordinates', 'missing_value']
        for key in attrs_to_remove:
            LWP.attrs.pop(key, None)
            IWP.attrs.pop(key, None)
            PWV.attrs.pop(key, None)
            Qliq.attrs.pop(key, None)

        # Save variable attributes
        LWP_attrs = LWP.attrs
        IWP_attrs = IWP.attrs
        PWV_attrs = PWV.attrs
        Qliq_attrs = {
            'description': 'Total liquid mixing ratio (QVAPOR + QCLOUD + QRAIN)', 
            'units': 'kg kg-1',
        }
        cld_attrs = {
            'LWP': LWP_attrs,
            'IWP': IWP_attrs,
            'PWV': PWV_attrs,
            'Qliq': Qliq_attrs,
        }

    else:
        # Create empty attributes
        cld_attrs = {
            'LWP': '',
            'IWP': '',
            'PWV': '',
            'Qliq': '',
        }
    # import pdb; pdb.set_trace()

    # Make array to store output
    # Number of tracks in the file
    out_ntracks = len(idx_track)
    # out_ny = 2*ny+1
    # out_nx = 2*nx+1
    out_ny = np.round((2 * ny) / sub_y + 1).astype(int)
    out_nx = np.round((2 * nx) / sub_x + 1).astype(int)
    out_dims = (nz, out_ny, out_nx)
    DX_sub = DX * sub_x
    DY_sub = DY * sub_y
    # Spatial coordinates
    xcoords = np.arange(-((out_nx-1)/2).astype(int), ((out_nx+1)/2).astype(int), 1)
    ycoords = np.arange(-((out_ny-1)/2).astype(int), ((out_ny+1)/2).astype(int), 1)
    zcoords = np.arange(0, nz)
    xcoords_attrs = {'long_name':'X-distance from cell center', 'units':f'{DX_sub:.0f}m'}
    ycoords_attrs = {'long_name':'Y-distance from cell center', 'units':f'{DY_sub:.0f}m'}
    zcoords_attrs = {'long_name':'Vertical level'}
    out_coords = {
        'dims': out_dims,
        'xcoords': xcoords,
        'ycoords': ycoords,
        'zcoords': zcoords,
        'xcoords_attrs': xcoords_attrs,
        'ycoords_attrs': ycoords_attrs,
        'zcoords_attrs': zcoords_attrs,
        'DX': DX_sub,
        'DY': DY_sub,
    }
    # 3D variables
    out_Z = np.full((out_ntracks, nz, out_ny, out_nx), np.NaN, dtype=float)
    out_P = np.full((out_ntracks, nz, out_ny, out_nx), np.NaN, dtype=float)
    out_T = np.full((out_ntracks, nz, out_ny, out_nx), np.NaN, dtype=float)
    out_QV = np.full((out_ntracks, nz, out_ny, out_nx), np.NaN, dtype=float)
    out_RH = np.full((out_ntracks, nz, out_ny, out_nx), np.NaN, dtype=float)
    out_U = np.full((out_ntracks, nz, out_ny, out_nx), np.NaN, dtype=float)
    out_V = np.full((out_ntracks, nz, out_ny, out_nx), np.NaN, dtype=float)
    out_W = np.full((out_ntracks, nz, out_ny, out_nx), np.NaN, dtype=float)
    out_Qliq = np.full((out_ntracks, nz, out_ny, out_nx), np.NaN, dtype=float)

    # 2D variables
    # out_LCL = np.full((out_ntracks, out_ny, out_nx), np.NaN, dtype=float)
    # out_LFC = np.full((out_ntracks, out_ny, out_nx), np.NaN, dtype=float)
    # out_LPL = np.full((out_ntracks, out_ny, out_nx), np.NaN, dtype=float)
    # out_LNB = np.full((out_ntracks, out_ny, out_nx), np.NaN, dtype=float)
    # out_MUCAPE = np.full((out_ntracks, out_ny, out_nx), np.NaN, dtype=float)
    # out_MUCIN = np.full((out_ntracks, out_ny, out_nx), np.NaN, dtype=float)
    out_LWP = np.full((out_ntracks, out_ny, out_nx), np.NaN, dtype=float)
    out_IWP = np.full((out_ntracks, out_ny, out_nx), np.NaN, dtype=float)
    out_PWV = np.full((out_ntracks, out_ny, out_nx), np.NaN, dtype=float)
    out_T2 = np.full((out_ntracks, out_ny, out_nx), np.NaN, dtype=float)
    out_Q2 = np.full((out_ntracks, out_ny, out_nx), np.NaN, dtype=float)
    out_PSFC = np.full((out_ntracks, out_ny, out_nx), np.NaN, dtype=float)
    out_U10 = np.full((out_ntracks, out_ny, out_nx), np.NaN, dtype=float)
    out_V10 = np.full((out_ntracks, out_ny, out_nx), np.NaN, dtype=float)
    # out_PBLH = np.full((out_ntracks, out_ny, out_nx), np.NaN, dtype=float)
    out_RAINNC = np.full((out_ntracks, out_ny, out_nx), np.NaN, dtype=float)
    out_HGT = np.full((out_ntracks, out_ny, out_nx), np.NaN, dtype=float)
    # 2D cell variables
    out_convcore = np.full((out_ntracks, out_ny, out_nx), np.NaN, dtype=float)
    out_convmask = np.full((out_ntracks, out_ny, out_nx), np.NaN, dtype=float)
    out_tnmask = np.full((out_ntracks, out_ny, out_nx), np.NaN, dtype=float)
    out_refl = np.full((out_ntracks, out_ny, out_nx), np.NaN, dtype=float)
    out_eth10 = np.full((out_ntracks, out_ny, out_nx), np.NaN, dtype=float)

    out_dict3d = None
    out_dict2d = None
    out_dict_attrs = None

    # Proceed if number of matched cell is > 0
    nmatchcell = len(idx_track)
    if (nmatchcell > 0):

        # Loop over each track
        for itrack in range(0, out_ntracks):
            # track center location (lat, lon)
            center = (_lat[itrack], _lon[itrack])

            # Tracking pixel file
            if (pixel_exist == True) & (met_exist == True):
                lat_idx, lon_idx = location_to_idx(dsp['latitude'].squeeze().data, dsp['longitude'].squeeze().data, center)
                # lat_idx, lon_idx = location_to_idx(XLAT.data, XLONG.data, center)
                _convcore = pad_array(dsp['conv_core'].astype('float32').squeeze().data, lat_idx, lon_idx, ny, nx, ny_p, nx_p, sub_y, sub_x)
                _convmask = pad_array(dsp['conv_mask'].astype('float32').squeeze().data, lat_idx, lon_idx, ny, nx, ny_p, nx_p, sub_y, sub_x)
                _tnmask = pad_array(dsp['tracknumber'].astype('float32').squeeze().data, lat_idx, lon_idx, ny, nx, ny_p, nx_p, sub_y, sub_x)
                _refl = pad_array(dsp['dbz_comp'].squeeze().data, lat_idx, lon_idx, ny, nx, ny_p, nx_p, sub_y, sub_x)
                _eth10 = pad_array(dsp['echotop10'].squeeze().data, lat_idx, lon_idx, ny, nx, ny_p, nx_p, sub_y, sub_x)

                iny, inx = _refl.shape
                if (iny == out_ny) & (inx == out_nx):
                    out_convcore[itrack, :, :] = _convcore
                    out_convmask[itrack, :, :] = _convmask
                    out_tnmask[itrack, :, :] = _tnmask
                    out_refl[itrack, :, :] = _refl
                    out_eth10[itrack, :, :] = _eth10

            # # Met file
            # if met_exist:
            #     lat_idx, lon_idx = location_to_idx(dsm['XLAT'], dsm['XLONG'], center)
            #     _LCL = pad_array(dsm['LCL'].squeeze().data, lat_idx, lon_idx, ny, nx, ny_m, nx_m, sub_y, sub_x)
            #     _LFC = pad_array(dsm['LFC'].squeeze().data, lat_idx, lon_idx, ny, nx, ny_m, nx_m, sub_y, sub_x)
            #     _LPL = pad_array(dsm['LPL'].squeeze().data, lat_idx, lon_idx, ny, nx, ny_m, nx_m, sub_y, sub_x)
            #     _LNB = pad_array(dsm['LNB'].squeeze().data, lat_idx, lon_idx, ny, nx, ny_m, nx_m, sub_y, sub_x)
            #     _MUCAPE = pad_array(dsm['MUCAPE'].squeeze().data, lat_idx, lon_idx, ny, nx, ny_m, nx_m, sub_y, sub_x)
            #     _MUCIN = pad_array(dsm['MUCIN'].squeeze().data, lat_idx, lon_idx, ny, nx, ny_m, nx_m, sub_y, sub_x)

            #     iny, inx = _LCL.shape
            #     if (iny == out_ny) & (inx == out_nx):
            #         out_LCL[itrack, :, :] = _LCL
            #         out_LFC[itrack, :, :] = _LFC
            #         out_LPL[itrack, :, :] = _LPL
            #         out_LNB[itrack, :, :] = _LNB
            #         out_MUCAPE[itrack, :, :] = _MUCAPE
            #         out_MUCIN[itrack, :, :] = _MUCIN

            # Cloud file
            if cld_exist:
                lat_idx, lon_idx = location_to_idx(dsc['XLAT'].squeeze().data, dsc['XLONG'].squeeze().data, center)
                # Double check closest lat/lon distance
                closest_lat = dsc['XLAT'].squeeze().data[lat_idx, lon_idx]
                closest_lon = dsc['XLONG'].squeeze().data[lat_idx, lon_idx]
                if (np.abs(closest_lat - center[0]) > dlat_max) | (np.abs(closest_lon - center[1]) > dlon_max):
                    print(f'ERROR (cloud file): closest lat ({closest_lat:.2f} to {center[0]:.2f}) > {dlat_max} or '+\
                          f'closest lon ({closest_lon:.2f} to {center[1]:.2f}) > {dlon_max}')
                    sys.exit(f'Code exits now.')
                _LWP = pad_array(LWP.data, lat_idx, lon_idx, ny, nx, ny_c, nx_c, sub_y, sub_x)
                _IWP = pad_array(IWP.data, lat_idx, lon_idx, ny, nx, ny_c, nx_c, sub_y, sub_x)
                _PWV = pad_array(PWV.data, lat_idx, lon_idx, ny, nx, ny_d, nx_d, sub_y, sub_x)

                iny, inx = _LWP.shape
                if (iny == out_ny) & (inx == out_nx):
                    out_LWP[itrack, :, :] = _LWP
                    out_IWP[itrack, :, :] = _IWP
                    out_PWV[itrack, :, :] = _PWV

            # WRF out file
            # if wrfout_exist:
            if met_exist:
                # Find closet lat/lon index to track center location                
                lat_idx, lon_idx = location_to_idx(XLAT.data, XLONG.data, center)
                # Double check closest lat/lon distance
                closest_lat = XLAT.squeeze().data[lat_idx, lon_idx]
                closest_lon = XLONG.squeeze().data[lat_idx, lon_idx]
                if (np.abs(closest_lat - center[0]) > dlat_max) | (np.abs(closest_lon - center[1]) > dlon_max):
                    print(f'ERROR (met file): closest lat ({closest_lat:.2f} to {center[0]:.2f}) > {dlat_max} or '+\
                          f'closest lon ({closest_lon:.2f} to {center[1]:.2f}) > {dlon_max}')
                    sys.exit(f'Code exits now.')

                # Extract and pad 3D variables
                _tk = pad_array(tk.data, lat_idx, lon_idx, ny, nx, ny_d, nx_d, sub_y, sub_x)
                _Z = pad_array(height.data, lat_idx, lon_idx, ny, nx, ny_d, nx_d, sub_y, sub_x)
                _pressure = pad_array(pressure.data, lat_idx, lon_idx, ny, nx, ny_d, nx_d, sub_y, sub_x)
                _qv = pad_array(qv.data, lat_idx, lon_idx, ny, nx, ny_d, nx_d, sub_y, sub_x)
                _rh = pad_array(rh.data, lat_idx, lon_idx, ny, nx, ny_d, nx_d, sub_y, sub_x)
                _u = pad_array(umet.data, lat_idx, lon_idx, ny, nx, ny_d, nx_d, sub_y, sub_x)
                _v = pad_array(vmet.data, lat_idx, lon_idx, ny, nx, ny_d, nx_d, sub_y, sub_x)
                _w = pad_array(wa.data, lat_idx, lon_idx, ny, nx, ny_d, nx_d, sub_y, sub_x)
                _Qliq = pad_array(Qliq.data, lat_idx, lon_idx, ny, nx, ny_d, nx_d, sub_y, sub_x)
                # Extract and pad 2D variables
                # _PWV = pad_array(pwv.data, lat_idx, lon_idx, ny, nx, ny_d, nx_d, sub_y, sub_x)
                _T2 = pad_array(T2.data, lat_idx, lon_idx, ny, nx, ny_d, nx_d, sub_y, sub_x)
                _Q2 = pad_array(Q2.data, lat_idx, lon_idx, ny, nx, ny_d, nx_d, sub_y, sub_x)
                _PSFC = pad_array(PSFC.data, lat_idx, lon_idx, ny, nx, ny_d, nx_d, sub_y, sub_x)
                _U10 = pad_array(U10.data, lat_idx, lon_idx, ny, nx, ny_d, nx_d, sub_y, sub_x)
                _V10 = pad_array(V10.data, lat_idx, lon_idx, ny, nx, ny_d, nx_d, sub_y, sub_x)
                # _PBLH = pad_array(PBLH.data, lat_idx, lon_idx, ny, nx, ny_d, nx_d, sub_y, sub_x)
                _RAINNC = pad_array(RAINNC.data, lat_idx, lon_idx, ny, nx, ny_d, nx_d, sub_y, sub_x)
                _HGT = pad_array(HGT.data, lat_idx, lon_idx, ny, nx, ny_d, nx_d, sub_y, sub_x)

                inz, iny, inx = _Z.shape
                if (iny == out_ny) & (inx == out_nx):
                    # 3D
                    out_T[itrack, :, :, :] = _tk
                    out_Z[itrack, :, :, :] = _Z
                    out_P[itrack, :, :, :] = _pressure
                    out_QV[itrack, :, :, :] = _qv
                    out_RH[itrack, :, :, :] = _rh
                    out_U[itrack, :, :, :] = _u
                    out_V[itrack, :, :, :] = _v
                    out_W[itrack, :, :, :] = _w
                    out_Qliq[itrack, :, :, :] = _Qliq
                    # 2D
                    # out_PWV[itrack, :, :] = _pwv
                    out_T2[itrack, :, :] = _T2
                    out_Q2[itrack, :, :] = _Q2
                    out_PSFC[itrack, :, :] = _PSFC
                    out_U10[itrack, :, :] = _U10
                    out_V10[itrack, :, :] = _V10
                    # out_PBLH[itrack, :, :] = _PBLH
                    out_RAINNC[itrack, :, :] = _RAINNC
                    out_HGT[itrack, :, :] = _HGT

        # Put output variables to a dictionary for easier acceess
        out_dict3d = {
            'pressure': out_P,
            'height': out_Z,
            'temperature': out_T,
            'qv': out_QV,
            'rh': out_RH,
            'u': out_U,
            'v': out_V,
            'w': out_W,
            'Qliq': out_Qliq,
        }
        out_dict2d = {
            # 'LCL': out_LCL,
            # 'LFC': out_LFC,
            # 'LPL': out_LPL,
            # 'LNB': out_LNB,
            # 'MUCAPE': out_MUCAPE,
            # 'MUCIN': out_MUCIN,
            'conv_core': out_convcore,
            'conv_mask': out_convmask,
            # 'tracknumber_cmask': out_tnmask,
            'tracknumber': out_tnmask,
            # 'comp_ref': out_refl,
            'dbz_comp': out_refl,
            'echotop10': out_eth10,
            'LWP': out_LWP,
            'IWP': out_IWP,
            'PWV': out_PWV,
            'T2': out_T2,
            'Q2': out_Q2,
            'PSFC': out_PSFC,
            'U10': out_U10,
            'V10': out_V10,
            # 'PBLH': out_PBLH,
            'RAINNC': out_RAINNC,
            'HGT': out_HGT,
        }
        out_dict_attrs = {
            'pressure': pressure_attrs,
            'height': height_attrs,
            'temperature': temperature_attrs,
            'qv': qv_attrs,
            'rh': rh_attrs,
            'u': u_attrs,
            'v': v_attrs,
            'w': w_attrs,
            'Qliq': Qliq_attrs,
            # 'PWV': PWV_attrs,
            'T2': T2_attrs,
            'Q2': Q2_attrs,
            'PSFC': PSFC_attrs,
            'U10': U10_attrs,
            'V10': V10_attrs,
            # 'PBLH': PBLH_attrs,
            'RAINNC': RAINNC_attrs,
            'HGT': HGT_attrs,
            # 'DX': DX_sub,
            # 'DY': DY_sub,
        }
        # Merge attribute dictionaries
        # new_attrs = {**pixel_attrs, **met_attrs}
        new_attrs = {**pixel_attrs, **cld_attrs}
        out_dict_attrs = {**out_dict_attrs, **new_attrs}
        # import pdb; pdb.set_trace()

    return out_dict3d, out_dict2d, out_dict_attrs, out_coords


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
    wrfout_path = config.get('wrfout_path', None)
    metfile_path = config['metfile_path']
    cldfile_path = config['cldfile_path']
    metfile_path_2 = config['metfile_path_2']
    cldfile_path_2 = config['cldfile_path_2']
    output_path = config['output_path']
    # wrfout_filebase = config.get('wrfout_filebase', None)
    met_filebase = config.get('met_filebase', None)
    cld_filebase = config.get('cld_filebase', None)
    pixel_filebase = config['pixel_filebase']
    # wrfout_path1 = config['wrfout_path1']
    # wrfout_path2 = config['wrfout_path2']
    # nhours = config['nhours']
    minutes_prior = config['minutes_prior']
    freq_prior = config['freq_prior']
    # ntimes_max = config['ntimes_max']
    ensmember = config['ensmember']
    domain = config['domain']
    nx = config['nx']
    ny = config['ny']

    # Replace directory (some LASSO data are staged in a different directory)
    if os.path.isdir(metfile_path_2):
        metfile_path = metfile_path_2
    if os.path.isdir(cldfile_path_2):
        cldfile_path = cldfile_path_2

    # Time threshold to match generated times with data times
    dt_thresh = 1.0     # [second]

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
    y_dimname = 'y'
    x_dimname = 'x'

    # Track statistics file
    stats_filebase = 'trackstats_'
    trackstats_file = f'{stats_path}{stats_filebase}{startdate}_{enddate}.nc'

    # Read track statistics file
    print(trackstats_file)
    dsstats = xr.open_dataset(trackstats_file, decode_times=True)
    # Subset stats times to reduce array size
    # dsstats = dsstats.isel(times=slice(0, ntimes_max))
    # ntracks = dsstats.sizes[tracks_dimname]
    ntracks = dsstats.sizes[tracks_dimname]
    ntimes = dsstats.sizes[times_dimname]
    coord_tracks = dsstats['tracks']
    stats_basetime = dsstats['base_time']
    stats_lon = dsstats['meanlon']
    stats_lat = dsstats['meanlat']
    # ntimes = dsstats.sizes[times_dimname]
    # coord_tracks = dsstats['tracks'].sel(tracks=slice(track_start, track_end))
    # stats_basetime = dsstats['base_time'].sel(tracks=slice(track_start, track_end))
    # stats_lon = dsstats['meanlon'].sel(tracks=slice(track_start, track_end))
    # stats_lat = dsstats['meanlat'].sel(tracks=slice(track_start, track_end))
    # ntracks = len(coord_tracks)
    time_res_hour = dsstats.attrs['time_resolution_hour']
    dsstats.close()

    print(f'Total Number of Tracks: {ntracks}')

    # Convert time resolution of data to minutes
    time_res_min = np.round(time_res_hour * 60).astype(int)
    ntimes_per_hour = np.round(60. / time_res_min).astype(int)

    # Select initiation time, and round to the nearest minute
    time0 = stats_basetime.isel(times=0).dt.round('min')
    # Get initiation lat/lon    
    stats_lon0 = stats_lon.isel(times=0).data
    stats_lat0 = stats_lat.isel(times=0).data

    # Make an array to store the full time series
    ntimes_prior = np.round(minutes_prior / freq_prior).astype(int)
    # ntimes_prior = np.round(nhours / time_res_hour).astype(int)
    # ntimes_full = np.ceil(ntimes_prior + ntimes_max).astype(int)
    ntimes_full = np.round(ntimes_prior + 1).astype(int)
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

    # Loop over each track
    for itrack in range(0, ntracks):
        # Calculate start/end times prior to initiation
        time0_start = stats_min0[itrack] - pd.offsets.Minute(minutes_prior)
        # time0_end = stats_min0[itrack] - pd.offsets.Minute(time_res_min)
        time0_end = pd.to_datetime(stats_mins[itrack,0])
        # Generate hourly time series leading up to initiation
        # prior_times = np.array(pd.date_range(time0_start, time0_end, freq=f'{time_res_min:.0f}min'))
        prior_times = np.array(pd.date_range(time0_start, time0_end, freq=f'{freq_prior:.0f}min'))

        # Save full history of times
        full_times[itrack,:] = prior_times
        # full_times[itrack,0:ntimes_prior] = prior_times
        # full_times[itrack,ntimes_prior] = stats_mins[itrack,0]

        # Convert full times to Epoch time
        itimes = full_times[itrack,:]
        # Find indices that is a time
        idx = ~np.isnat(itimes)
        # full_basetimes[itrack, idx] = np.array([tt.tolist()/1e9 for tt in itimes[idx]])
        full_basetimes[itrack, idx] = itimes[idx].astype('datetime64[s]').astype(np.float64)

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
        'comment': f'Multiply by {time_res_min:.0f}min to get physical time',
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
    uniq_basetimes = uniq_times.astype('datetime64[s]').astype(np.float64)
    # uniq_basetimes = np.array([tt.tolist()/1e9 for tt in uniq_times])


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

    # Loop over each pixel-file and call function to calculate
    for ifile in range(nfiles):
    # for ifile in range(0, 12):
        # Convert time string to match different files
        itime = uniq_times[ifile]
        itime_pixel = pd.to_datetime(str(itime)).strftime('%Y%m%d_%H%M%S')
        itime_wrfout = pd.to_datetime(str(itime)).strftime('%Y-%m-%d_%H_%M_%S')
        itime_met = pd.to_datetime(str(itime)).strftime('%Y%m%d.%H%M%S')
        itime_cld = pd.to_datetime(str(itime)).strftime('%Y%m%d.%H%M%S')

        # File names
        fname_pixel = f'{pixelfile_path}{pixel_filebase}{itime_pixel}.nc'
        if wrfout_path:
            fname_wrfout = f'{wrfout_path}wrfout_{domain}_{itime_wrfout}'
        # New MET file time format: yyyymmdd.hhmmss
        if met_filebase:
            fname_met = f'{metfile_path}{met_filebase}{itime_met}.nc'
        else:
            fname_met = fname_wrfout
        if cld_filebase:
            fname_cld = f'{cldfile_path}{cld_filebase}{itime_met}.nc'
        else:
            fname_cld = fname_wrfout
        # import pdb; pdb.set_trace()

        # Get all MCS tracks/times indices in the same time (file)
        # idx_track, idx_time = np.where(full_basetimes == uniq_basetimes[ifile])
        idx_track, idx_time = np.where(np.abs(full_basetimes - uniq_basetimes[ifile]) < dt_thresh)
        if len(idx_track) > 0:
            # Save matchindices for the current pixel file to the overall list
            trackindices_all.append(idx_track)
            timeindices_all.append(idx_time)
            # import pdb; pdb.set_trace()

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
                    idx_track, 
                    _lat,
                    _lon,
                    config,
                )
                results.append(result)
            # Parallel
            elif run_parallel >= 1:
                result = dask.delayed(extract_env_prof)(
                    fname_pixel, 
                    # fname_wrfout,
                    fname_met,
                    fname_cld,
                    idx_track, 
                    _lat,
                    _lon,
                    config,
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
    while counter > 0:
        if final_results[counter] is not None:
            var_names3d = list(final_results[counter][0].keys())
            var_names2d = list(final_results[counter][1].keys())
            var_attrs = final_results[counter][2]
            var_coords = final_results[counter][3]
            break
        counter -= 1
    
    # Loop over variable list to create the dictionary entry
    print(f'Creating output arrays ...')
    out_dict = {}
    out_dict_attrs = {}
    # Spatial coordinates
    out_nz, out_ny, out_nx = var_coords['dims']
    xcoords = var_coords['xcoords']
    ycoords = var_coords['ycoords']
    zcoords = var_coords['zcoords']
    xcoords_attrs = var_coords['xcoords_attrs']
    ycoords_attrs = var_coords['ycoords_attrs']
    zcoords_attrs = var_coords['zcoords_attrs']

    var_names = var_names3d + var_names2d
    # 3D variables 
    for ivar in var_names3d:
        out_dict[ivar] = np.full((ntracks, out_ntimes, out_nz, out_ny, out_nx), np.nan, dtype=np.float32)
        out_dict_attrs[ivar] = var_attrs[ivar]
    # 2D variables
    for ivar in var_names2d:
        out_dict[ivar] = np.full((ntracks, out_ntimes, out_ny, out_nx), np.nan, dtype=np.float32)
        out_dict_attrs[ivar] = var_attrs[ivar]

    # Put the results to output track stats variables
    # Loop over each file (parallel return results)
    # for ifile in range(nfiles):
    # Loop over each returned results
    for ifile in range(len(final_results)):
    # for ifile in range(0, 12):
        # Get the return results for this pixel file
        if final_results[ifile] is not None:
            iVAR3d = final_results[ifile][0]
            iVAR2d = final_results[ifile][1]
            if iVAR3d is not None:
                trackindices = trackindices_all[ifile]
                timeindices = timeindices_all[ifile]
                # Loop over each variable and assign values to output dictionary
                for ivar in var_names3d:
                    if iVAR3d[ivar].ndim == 4:
                        out_dict[ivar][trackindices,timeindices,:,:,:] = iVAR3d[ivar]
            if iVAR2d is not None:
                trackindices = trackindices_all[ifile]
                timeindices = timeindices_all[ifile]
                # Loop over each variable and assign values to output dictionary
                for ivar in var_names2d:
                    if iVAR2d[ivar].ndim == 3:
                        out_dict[ivar][trackindices,timeindices,:,:] = iVAR2d[ivar]

    # Define a dataset containing all PF variables
    var_dict = {}
    print(f'Saving data to output arrays ...')
    # Define output variable dictionary
    for key, value in out_dict.items():
        if value.ndim == 4:
            var_dict[key] = (
                [tracks_dimname, times_dimname, y_dimname, x_dimname], value, out_dict_attrs[key],
            )
        if value.ndim == 5:
            var_dict[key] = (
                [tracks_dimname, times_dimname, z_dimname, y_dimname, x_dimname], value, out_dict_attrs[key],
            )
    # Add base_time to output dictionary
    var_dict['base_time'] = ([tracks_dimname, times_dimname], full_basetimes, full_basetimes_attrs)

    # Define coordinate list
    coord_dict = {
        tracks_dimname: ([tracks_dimname], coord_tracks.data, coord_tracks.attrs),
        times_dimname: ([times_dimname], coord_relativetimes, coord_relativetimes_attrs),
        z_dimname: ([z_dimname], zcoords, zcoords_attrs),
        y_dimname: ([y_dimname], ycoords, ycoords_attrs),
        x_dimname: ([x_dimname], xcoords, xcoords_attrs),
    }
    gattr_dict = {
        'Title': 'Extracted 3D environments for cell tracks',
        'Institution': 'Pacific Northwest National Laboratory',
        'Contact': 'zhe.feng@pnnl.gov',
        'Created_on': time.ctime(time.time()),
    }

    # Define xarray dataset
    dsout = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)

    # Set encoding/compression for all variables
    comp = dict(zlib=False, dtype='float32')
    encoding = {var: comp for var in dsout.data_vars}

    # Delete file if it already exists
    if os.path.isfile(output_filename):
        os.remove(output_filename)

    # Write to netcdf file
    print(f'Writing netCDF ...')
    dsout.to_netcdf(path=output_filename, mode='w', format='NETCDF4', 
                    unlimited_dims=tracks_dimname, encoding=encoding)
    print(f'Output saved as: {output_filename}')

    # import pdb; pdb.set_trace()