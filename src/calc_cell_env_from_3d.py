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

    # Read 3D environment
    ds = xr.open_dataset(in_filename).sel(tracks=tracknumber)
    ntimes = ds.dims['times']
    nz = ds.dims['z']
    ny = ds.dims['y']
    nx = ds.dims['x']
    # 3D variables
    tk = ds['temperature']
    qv = ds['qv']
    rh = ds['rh']
    height = ds['height']
    pressure = ds['pressure'] * 100  # Convert unit to Pa    

    # Create arrays to store outputs
    var2d_dims = (ntimes, ny, nx)
    mucape = np.full(var2d_dims, np.NaN, dtype=np.float32)
    mucin = np.full(var2d_dims, np.NaN, dtype=np.float32)
    lcl = np.full(var2d_dims, np.NaN, dtype=np.float32)
    lfc = np.full(var2d_dims, np.NaN, dtype=np.float32)
    el = np.full(var2d_dims, np.NaN, dtype=np.float32)
    lpl = np.full(var2d_dims, np.NaN, dtype=np.float32)

    # Loop over times
    for itime in range(0, ntimes):
        # Proceed if this time has valid data (track exists)
        # if np.nanmax(z_sfc[itime, :, :]) > 0:
        if np.count_nonzero(~np.isnan(tk[itime, :, :, :])) > 0:
            _tk = tk[itime, :, :, :]
            _qv = qv[itime, :, :, :]
            _rh = rh[itime, :, :, :]
            _z = height[itime, :, :, :]
            _pressure = pressure[itime, :, :, :]

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

            # import pdb; pdb.set_trace()

    # Group outputs in dictionaries
    var_dict = {
        'tracknumber': tracknumber,
        'MUCAPE': mucape,
        'MUCIN': mucin,
        'LCL': lcl,
        'LFC': lfc,
        'EL': el,
        'LPL': lpl,
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
    # for itrack in range(0, 5):
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

    # Define a dataset containing all PF variables
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
    stats_path = config['stats_path']

    # 3D environment filename
    file_env3d = f'{stats_path}stats_3d_env_{startdate}_{enddate}.nc'
    print(f'Input: {file_env3d}')

    # Output filename
    output_path = stats_path
    output_filename = f'{output_path}stats_2d_env_{startdate}_{enddate}.nc'

    # Check input file
    fc_3d = os.path.isfile(file_env3d)
    if fc_3d:
        # Call function to calculate
        result = work_for_tracks(file_env3d, output_filename, config)

    # import pdb; pdb.set_trace()