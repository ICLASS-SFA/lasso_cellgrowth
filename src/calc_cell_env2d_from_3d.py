"""
Calculates 2D environmental variables from 3D profiles and saves to a netCDF file.
"""
__author__ = "Zhe.Feng@pnnl.gov"

import os, sys
import time
import numpy as np
import xarray as xr
import yaml
import metpy.calc as mpcalc
from metpy.units import units
import diag_functions as afwa
from wrf import interplevel
from itertools import repeat
from multiprocessing import Pool

#-----------------------------------------------------------------------------------------
def calc_shear(level_shear, u, v, z_agl, u10, v10):
    """
    Calculates vertical wind shear and direction at specified AGL levels

    Args:
        level_shear: np.array(float)
            Vertical height level AGL (km).
        u: xr.DataArray
            U components of wind.
        v: xr.DataArray
            V components of wind.
        z_agl: xr.DataArray
            Height above ground level.
        u10: xr.DataArray
            10m U components of wind.
        v10: xr.DataArray
            10m V components of wind.    

    Returns:
        shear_mag: xr.DataArray
            Bulk wind shear magnitude.
        shear_dir: xr.DataArray
            Bulk wind shear direction.
    """
    # Interpolate U, V to specific AGL levels
    u_agl = interplevel(u, z_agl, level_shear)
    v_agl = interplevel(v, z_agl, level_shear)
    # Calculate wind speed
    wspd_agl = np.sqrt(u_agl**2 + v_agl**2)
    wspd10 = np.sqrt(u10**2 + v10**2)

    # Wind shear magnitude
    shear_mag = wspd_agl - wspd10

    # Wind shear direction
    ushear = u_agl - u10
    vshear = v_agl - v10
    # .metpy.dequantify() converts data back to a unit-naive array
    # https://unidata.github.io/MetPy/latest/tutorials/xarray_tutorial.html?highlight=dequantify#
    shear_dir = mpcalc.wind_direction(ushear * units('m/s'), vshear * units('m/s')).metpy.dequantify()

    # Assign attributes
    shear_mag_attrs = {'long_name':'Bulk wind shear magnitude', 'units':'m/s'}
    shear_dir_attrs = {'long_name':'Bulk wind shear direction', 'units':'degree'}
    shear_mag = shear_mag.assign_attrs(shear_mag_attrs)
    shear_dir = shear_dir.assign_attrs(shear_dir_attrs)

    return shear_mag, shear_dir

#-----------------------------------------------------------------------------------------
def calc_shear_2level(u, v, z, level_lower, level_upper):
    """
    Calculates vertical wind shear and direction between two levels

    Args:
        u: xr.DataArray
            U components of wind.
        v: xr.DataArray
            V components of wind.
        z: xr.DataArray
            Height above ground level.
        level_lower: float
            Lower height level.
        level_upper: float
            Upper height level.    

    Returns:
        shear_mag: xr.DataArray
            Bulk wind shear magnitude.
        shear_dir: xr.DataArray
            Bulk wind shear direction.
    """
    # Interpolate U, V at lower & upper levels (on HAMSL coordinate)
    u_lower = interplevel(u, z, level_lower)
    v_lower = interplevel(v, z, level_lower)
    u_upper = interplevel(u, z, level_upper)
    v_upper = interplevel(v, z, level_upper)
    # Calculate wind speed
    wspd_lower = np.sqrt(u_lower**2 + v_lower**2)
    wspd_upper = np.sqrt(u_upper**2 + v_upper**2)
    # Wind shear magnitude
    shear_mag = wspd_upper - wspd_lower
    # Wind shear direction
    ushear = u_upper - u_lower
    vshear = v_upper - v_lower
    # .metpy.dequantify() converts data back to a unit-naive array
    # https://unidata.github.io/MetPy/latest/tutorials/xarray_tutorial.html?highlight=dequantify#
    shear_dir = mpcalc.wind_direction(ushear * units('m/s'), vshear * units('m/s')).metpy.dequantify()

    # Assign attributes
    shear_mag_attrs = {'long_name':f'Bulk wind shear magnitude between {level_lower}m and {level_upper}m', 'units':'m/s'}
    shear_dir_attrs = {'long_name':f'Bulk wind shear direction between {level_lower}m and {level_upper}m', 'units':'degree'}
    shear_mag = shear_mag.assign_attrs(shear_mag_attrs)
    shear_dir = shear_dir.assign_attrs(shear_dir_attrs)

    return shear_mag, shear_dir

#-----------------------------------------------------------------------------------------
def calc_envs_track(in_filename, tracknumber, config):
    """
    Calculte 2D environments for a given track time series

    Args:
        in_filename: string
            Input 3D environment filename
        tracknumber: np.array
            Track number to work on
        config: dictionary
            Dictionary containing config parameters

    Returns:
        var_dict: dictionary
            Dictionary containing the track statistics data
        out_dict_attrs: dictionary
            Dictionary containing the attributes of track statistics data
    """

    # Get list of variables to pass through
    list_var_pass = config['list_var_pass']

    # Define levels (HAGL) to calculate surface wind shear
    level_shear = [2000, 4000, 6000, 8000, 10000]   # [m AGL]
    level_theta_e = [3000, 5000, 7000]   # [m AGL]
    level_pres = [925, 850, 700, 680, 660, 640, 620, 600, 580, 560, 540, 520, 500]

    # Read 3D environment
    ds = xr.open_dataset(in_filename).sel(tracks=tracknumber)
    ntimes = ds.dims['times']
    nz = ds.dims['z']
    ny = ds.dims['y']
    nx = ds.dims['x']
    # 3D variables
    tk = ds['temperature'].load()
    qv = ds['qv']
    rh = ds['rh']
    height = ds['height']
    pressure = ds['pressure'] * 100  # Convert unit to Pa
    # u = ds['u']
    # v = ds['v']
    # w = ds['w']
    # 2D variables
    # U10 = ds['U10']
    # V10 = ds['V10']
    # HGT = ds['HGT']

    # Create arrays to store outputs
    var2d_dims = (ntimes, ny, nx)
    mucape = np.full(var2d_dims, np.NaN, dtype=np.float32)
    mucin = np.full(var2d_dims, np.NaN, dtype=np.float32)
    lcl = np.full(var2d_dims, np.NaN, dtype=np.float32)
    lfc = np.full(var2d_dims, np.NaN, dtype=np.float32)
    el = np.full(var2d_dims, np.NaN, dtype=np.float32)
    lpl = np.full(var2d_dims, np.NaN, dtype=np.float32)
    hamsl = np.full(var2d_dims, np.NaN, dtype=np.float32)

    u_4km = np.full(var2d_dims, np.NaN, dtype=np.float32)
    v_4km = np.full(var2d_dims, np.NaN, dtype=np.float32)

    # qv_925mb = np.full(var2d_dims, np.NaN, dtype=np.float32)
    # qv_850mb = np.full(var2d_dims, np.NaN, dtype=np.float32)
    # qv_700mb = np.full(var2d_dims, np.NaN, dtype=np.float32)
    # qv_600mb = np.full(var2d_dims, np.NaN, dtype=np.float32)
    # qv_500mb = np.full(var2d_dims, np.NaN, dtype=np.float32)
    rh_925mb = np.full(var2d_dims, np.NaN, dtype=np.float32)
    rh_850mb = np.full(var2d_dims, np.NaN, dtype=np.float32)
    rh_700mb = np.full(var2d_dims, np.NaN, dtype=np.float32)
    rh_680mb = np.full(var2d_dims, np.NaN, dtype=np.float32)
    rh_660mb = np.full(var2d_dims, np.NaN, dtype=np.float32)
    rh_640mb = np.full(var2d_dims, np.NaN, dtype=np.float32)
    rh_620mb = np.full(var2d_dims, np.NaN, dtype=np.float32)
    rh_600mb = np.full(var2d_dims, np.NaN, dtype=np.float32)
    rh_580mb = np.full(var2d_dims, np.NaN, dtype=np.float32)
    rh_560mb = np.full(var2d_dims, np.NaN, dtype=np.float32)
    rh_540mb = np.full(var2d_dims, np.NaN, dtype=np.float32)
    rh_520mb = np.full(var2d_dims, np.NaN, dtype=np.float32)
    rh_500mb = np.full(var2d_dims, np.NaN, dtype=np.float32)

    # Loop over times
    for itime in range(0, ntimes):
        # Proceed if this time has valid data (track exists)
        # if np.nanmax(z_sfc[itime, :, :]) > 0:
        _rh = rh[itime, :, :, :]
        _z = height[itime, :, :, :]
        _pressure = pressure[itime, :, :, :]
        
        hamsl[itime,:,:] = _z[0,:,:]
        
        rh_pres = interplevel(_rh, _pressure/100, level_pres)
        rh_925mb[itime, :, :] = rh_pres.sel(level=925)
        rh_850mb[itime, :, :] = rh_pres.sel(level=850)
        rh_700mb[itime, :, :] = rh_pres.sel(level=700)
        rh_680mb[itime, :, :] = rh_pres.sel(level=680)
        rh_660mb[itime, :, :] = rh_pres.sel(level=660)
        rh_640mb[itime, :, :] = rh_pres.sel(level=640)
        rh_620mb[itime, :, :] = rh_pres.sel(level=620)
        rh_600mb[itime, :, :] = rh_pres.sel(level=600)
        rh_580mb[itime, :, :] = rh_pres.sel(level=580)
        rh_560mb[itime, :, :] = rh_pres.sel(level=560)
        rh_540mb[itime, :, :] = rh_pres.sel(level=540)
        rh_520mb[itime, :, :] = rh_pres.sel(level=520)
        rh_500mb[itime, :, :] = rh_pres.sel(level=500)
        
        if np.count_nonzero(~np.isnan(tk[itime, :, :, :])) > 0:
            _tk = tk[itime, :, :, :]
            _qv = qv[itime, :, :, :]
            
            # _u = u[itime, :, :, :]
            # _v = v[itime, :, :, :]
            # _w = w[itime, :, :, :]
            # _U10 = U10[itime, :, :]
            # _V10 = V10[itime, :, :]

            # Interpolate to specific levels
            # u_4km[itime, :, :] = interplevel(_u, _z, 4000.)
            # v_4km[itime, :, :] = interplevel(_v, _z, 4000.)
            # Pressure level variables (pressure unit is Pa, convert it to hPa)                    
            # qv_pres = interplevel(_qv, _pressure/100, level_pres)
            # qv_925mb[itime, :, :] = qv_pres.sel(level=925)
            # qv_850mb[itime, :, :] = qv_pres.sel(level=850)
            # qv_700mb[itime, :, :] = qv_pres.sel(level=700)
            # qv_600mb[itime, :, :] = qv_pres.sel(level=600)
            # qv_500mb[itime, :, :] = qv_pres.sel(level=500)

            # Call AFWA diagnostics on data filtered below surface            
            ostat, _mucape, _mucin, _lcl, _lfc, _el, _lpl = afwa.diag_functions.diag_map(_tk, _rh, _pressure, _z, 1, 1)
            if ostat == 1:
                # Replace undefined values with NaN
                _mucape[_mucape < 0] = np.NaN
                _mucin[_mucin < -999] = np.NaN
                _lcl[_lcl < 0] = np.NaN
                _lfc[_lfc < 0] = np.NaN
                _el[_el < 0] = np.NaN
                _lpl[_lpl < 0] = np.NaN
                # Save to output arrays
                mucape[itime, :, :] = _mucape
                mucin[itime, :, :] = _mucin
                lcl[itime, :, :] = _lcl
                lfc[itime, :, :] = _lfc
                el[itime, :, :] = _el
                lpl[itime, :, :] = _lpl

    
    # Group outputs in dictionaries
    var_dict = {
        'tracknumber': tracknumber,
        'MUCAPE': mucape,
        'MUCIN': mucin,
        'LCL': lcl,
        'LFC': lfc,
        'EL': el,
        'LPL': lpl,
        'hamsl': hamsl,
        # 'u_4km': u_4km,
        # 'v_4km': v_4km,
        # 'qv_925mb': qv_925mb,
        # 'qv_850mb': qv_850mb,
        # 'qv_700mb': qv_700mb,
        # 'qv_600mb': qv_600mb,
        # 'qv_500mb': qv_500mb,
        'rh_925mb': rh_925mb,
        'rh_850mb': rh_850mb,
        'rh_700mb': rh_700mb,
        'rh_680mb': rh_680mb,
        'rh_660mb': rh_660mb,
        'rh_640mb': rh_640mb,
        'rh_620mb': rh_620mb,
        'rh_600mb': rh_600mb,
        'rh_580mb': rh_580mb,
        'rh_560mb': rh_560mb,
        'rh_540mb': rh_540mb,
        'rh_520mb': rh_520mb,
        'rh_500mb': rh_500mb,
    }
    var_attrs = {
        'tracknumber': {
            'long_name': 'Track number'
        },
        'MUCAPE': {
            'long_name': 'Most unstable convective available potential energy',
            'units': 'J/kg',
            # '_FillValue': fillval,
        },
        'MUCIN': {
            'long_name': 'Most unstable convective inhibition',
            'units': 'J/kg',
            # '_FillValue': fillval,
        },
        'LCL': {
            'long_name': 'Lifted condensation level',
            'units': 'm',
            # '_FillValue': fillval,
        },
        'LFC': {
            'long_name': 'Level of free convection',
            'units': 'm',
            # '_FillValue': fillval,
        },
        'EL': {
            'long_name': 'Equilibrium level',
            'units': 'm',
            # '_FillValue': fillval,
        },
        'LPL': {
            'long_name': 'Most unstable lifted parcel level',
            'units': 'm',
            # '_FillValue': fillval,
        },
        'hamsl': {
            'long_name': 'Height Above Mean Sea Level Pressure',
            'units': 'm',
            # '_FillValue': fillval,
        },
#         'u_4km': {
#             'long_name': 'U wind at 4 km HAMSL',
#             'units': 'm/s',
#         }, 
#         'v_4km': {
#             'long_name': 'V wind at 4 km HAMSL',
#             'units': 'm/s',
#         }, 
#         'qv_925mb': {
#             'long_name': 'Water vapor mixing ratio at 925mb',
#             'units': 'kg/kg',
#         },
#         'qv_850mb': {
#             'long_name': 'Water vapor mixing ratio at 850mb',
#             'units': 'kg/kg',
#         },
#         'qv_700mb': {
#             'long_name': 'Water vapor mixing ratio at 700mb',
#             'units': 'kg/kg',
#         },
#         'qv_600mb': {
#             'long_name': 'Water vapor mixing ratio at 600mb',
#             'units': 'kg/kg',
#         },
#         'qv_500mb': {
#             'long_name': 'Water vapor mixing ratio at 500mb',
#             'units': 'kg/kg',
#         },
        'rh_925mb': {
            'long_name': 'Relative humidity at 925mb',
            'units': '%',
        },
        'rh_850mb': {
            'long_name': 'Relative humidity at 850mb',
            'units': '%',
        },
        'rh_700mb': {
            'long_name': 'Relative humidity at 700mb',
            'units': '%',
        },
        'rh_680mb': {
            'long_name': 'Relative humidity at 650mb',
            'units': '%',
        },
        'rh_660mb': {
            'long_name': 'Relative humidity at 650mb',
            'units': '%',
        },
        'rh_640mb': {
            'long_name': 'Relative humidity at 650mb',
            'units': '%',
        },
        'rh_620mb': {
            'long_name': 'Relative humidity at 650mb',
            'units': '%',
        },
        'rh_600mb': {
            'long_name': 'Relative humidity at 600mb',
            'units': '%',
        },
        'rh_580mb': {
            'long_name': 'Relative humidity at 580mb',
            'units': '%',
        },
        'rh_560mb': {
            'long_name': 'Relative humidity at 560mb',
            'units': '%',
        },
        'rh_540mb': {
            'long_name': 'Relative humidity at 540mb',
            'units': '%',
        },
        'rh_520mb': {
            'long_name': 'Relative humidity at 520mb',
            'units': '%',
        },
        'rh_500mb': {
            'long_name': 'Relative humidity at 500mb',
            'units': '%',
        },
    }

    # Create a dictionary with pass through variables
    var_dict_pass = {}
    var_attrs_pass = {}
    for ivar in list_var_pass:
        var_dict_pass[ivar] = ds[ivar].data
        var_attrs_pass[ivar] = ds[ivar].attrs
    
    # Merge dictionaries
    var_dict = {**var_dict, **var_dict_pass}
    var_attrs = {**var_attrs, **var_attrs_pass}

    # import pdb; pdb.set_trace()
    return var_dict, var_attrs

#-----------------------------------------------------------------------------------------
def work_for_tracks(in_filename, out_filename, config):
    """
    Split work for each track to process

    Args:
        in_filename: string
            Input 3D environment filename
        out_filename: string
            Input 2D environment filename
        config: dictionary
            Dictionary containing config parameters

    Returns:
        out_filename: string
            Input 2D environment filename
    """

    # Read config parameters
    run_parallel = config['run_parallel']
    n_workers = config['n_workers']

    # Get coordinates info
    ds = xr.open_dataset(in_filename)
    ntracks = ds.dims['tracks']
    tracks = ds['tracks']
    ntimes = ds.dims['times']
    nz = ds.dims['z']
    ny = ds.dims['y']
    nx = ds.dims['x']
    ds.close()

    final_result = []
    # Serial
    if run_parallel == 0:
        for itrack in range(0, ntracks):
        # for itrack in range(0, 5):
            tracknumber = tracks.data[itrack]
            result = calc_envs_track(file_env3d, tracknumber, config)
            final_result.append(result)
    # Parallel
    elif run_parallel >= 1:
        pool = Pool(n_workers)
        final_result = pool.starmap(calc_envs_track, zip(repeat(file_env3d), tracks.data, repeat(config)))
        pool.close()
    else:
        sys.exit('Valid parallelization flag not set.')

    # Make a variable list from one of the returned dictionaries
    var_names = list(final_result[0][0].keys())
    # Get variable attributes from one of the returned dictionaries
    var_attrs = final_result[0][1]

    # Remove tracknumbers from the list
    var_names.remove('tracknumber')
    var_attrs.pop('tracknumber', None)

    # Loop over variable list to create the dictionary entry
    out_dict = {}
    out_dict_attrs = {}
    var2d_dims = (ntracks, ntimes, ny, nx)
    for ivar in var_names:
        out_dict[ivar] = np.full(var2d_dims, np.nan, dtype=np.float32)
        out_dict_attrs[ivar] = var_attrs[ivar]

    # Collect results
    for itrack in range(0, ntracks):
#     for itrack in range(0, 5):
        if final_result[itrack] is not None:
            # Get the return results for this track
            # The result is a tuple: (out_dict, out_dict_attrs)
            # The first entry is the dictionary containing the variables
            iResult = final_result[itrack][0]
            tracknumber = iResult['tracknumber']

            # Double check tracknumber from return to make sure it matches the track
            if tracks.data[itrack] == tracknumber:
                # Loop over each variable and assign values to output dictionary
                for ivar in var_names:
                    if iResult[ivar].ndim == 3:
                        out_dict[ivar][itrack, :, :, :] = iResult[ivar]
                        # import pdb; pdb.set_trace()
            else:
                print(f'ERROR: tracknumber does not match: {tracknumber}!')
                sys.exit('Double check results!')

    # Define a dataset containing all variables
    var_dict = {}
    # Define output variable dictionary
    for key, value in out_dict.items():
        if value.ndim == 4:
            var_dict[key] = (['tracks', 'times', 'y', 'x'], value, out_dict_attrs[key])
    coord_dict = {
        'tracks': (['tracks'], tracks.data, tracks.attrs),
        'times': (['times'], ds['times'].data, ds['times'].attrs),
        'y': (['y'], ds['y'].data, ds['y'].attrs),
        'x': (['x'], ds['x'].data, ds['x'].attrs),
    }
    gattr_dict = {
        'Title': 'Computed 2D environment data for cell tracks',
        'Institution': 'Pacific Northwest National Laboratoy',
        'Contact': 'zhe.feng@pnnl.gov',
        'Created_on': time.ctime(time.time()),
    }

    # Define xarray dataset
    dsout = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)
    # Set encoding/compression for all variables
    comp = dict(zlib=True, dtype='float32')
    encoding = {var: comp for var in dsout.data_vars}
    # Write to netcdf file
    dsout.to_netcdf(path=out_filename, mode='w', format='NETCDF4', 
                    unlimited_dims='tracks', encoding=encoding)
    print(f'Output: {out_filename}')

    return out_filename


if __name__ == "__main__":

    # Get configuration file name from input
    config_file = sys.argv[1]
    # Read configuration from yaml file
    stream = open(config_file, 'r')
    config = yaml.full_load(stream)

    startdate = config['startdate']
    enddate = config['enddate']
    # stats_path = config['stats_path']
    input_path = config['output_path']
    output_path = config['output_path']

    # 3D environment filename
    file_env3d = f'{input_path}preCI_3d_env_{startdate}_{enddate}.nc'
    print(f'Input: {file_env3d}')

    # Output filename
    output_filename = f'{output_path}stats_2d_env_{startdate}_{enddate}.nc'
    os.makedirs(output_path, exist_ok=True)

    # Check input file
    fc_3d = os.path.isfile(file_env3d)
    if fc_3d:
        # Call function to calculate
        result = work_for_tracks(file_env3d, output_filename, config)
    else:
        print(f'No input file: {file_env3d}')