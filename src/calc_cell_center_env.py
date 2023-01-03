"""
Calculates environmental variables at cell center and saves to a netCDF file.
"""
__author__ = "Zhe.Feng@pnnl.gov"

import os, sys
import time
import numpy as np
import xarray as xr
import yaml
from wrf import interplevel
from itertools import repeat
from multiprocessing import Pool
# import dask
# from dask.distributed import Client, LocalCluster, wait

def calc_envs_track(file_env3d, file_env2d, tracknumber, config):

    print(f'track: {tracknumber}')
    nx_center = config['nx_center']
    ny_center = config['ny_center']

    # Specify vertical level to interpolate to (HAMSL)
    z_lev_interp = np.arange(0, 20000.1, 200)

    # Read 3D environment
    # Subset track and center box
    ds3d = xr.open_dataset(file_env3d).sel(
        tracks=tracknumber,
        y=slice(-ny_center, ny_center), 
        x=slice(-nx_center, nx_center),
    )
    # ntimes = ds3d.dims['times']
    # nz = ds3d.dims['z']
    # ny = ds3d.dims['y']
    # nx = ds3d.dims['x']
    # 3D variables
    tk = ds3d['temperature']
    qv = ds3d['qv']
    rh = ds3d['rh']
    height = ds3d['height']
    pressure = ds3d['pressure'] * 100  # Convert unit to Pa
    u = ds3d['u']
    v = ds3d['v']
    w = ds3d['w']

    # Interpolate to fixed height
    _tk = interplevel(tk, height, z_lev_interp)
    _qv = interplevel(qv, height, z_lev_interp)
    _rh = interplevel(rh, height, z_lev_interp)
    _p = interplevel(pressure, height, z_lev_interp)
    _u = interplevel(u, height, z_lev_interp)
    _v = interplevel(v, height, z_lev_interp)
    _w = interplevel(w, height, z_lev_interp)
    # Subset to the center point
    _tk = _tk.sel(y=0, x=0)
    _qv = _qv.sel(y=0, x=0)
    _rh = _rh.sel(y=0, x=0)
    _p = _p.sel(y=0, x=0)
    _u = _u.sel(y=0, x=0)
    _v = _v.sel(y=0, x=0)
    _w = _w.sel(y=0, x=0)

    # Put 3D variables to dictionary
    var3d_dict = {
        # 'height': z_lev_interp,
        'temperature': _tk,
        'qv': _qv,
        'rh': _rh,
        'pressure': _p, 
        'u': _u,
        'v': _v,
        'w': _w,
    }
    var3d_attrs = {
        'temperature': _tk.attrs,
        'qv': _qv.attrs,
        'rh': _rh.attrs,
        'pressure': _p.attrs, 
        'u': _u.attrs,
        'v': _v.attrs,
        'w': _w.attrs,
    }

    # Read 2D environment
    ds2d = xr.open_dataset(file_env2d).sel(
        tracks=tracknumber,
        y=slice(0, 0), 
        x=slice(0, 0),
        # y=slice(-ny_center, ny_center), 
        # x=slice(-nx_center, nx_center),
    ).squeeze()

    # Add 2D variables to the dictionary
    var2d_dict = {}
    var2d_attrs = {}
    for var_name, values in ds2d.items():
        var2d_dict[var_name] = values
        var2d_attrs[var_name] = values.attrs

    # import pdb; pdb.set_trace()
    return var3d_dict, var2d_dict, var3d_attrs, var2d_attrs, z_lev_interp

def work_for_tracks(file_env3d, file_env2d, output_filename, config):

    # Read config parameters
    run_parallel = config['run_parallel']
    n_workers = config['n_workers']

    # Get coordinates info
    ds3d = xr.open_dataset(file_env3d)
    ntracks = ds3d.dims['tracks']
    tracks = ds3d['tracks']
    ntimes = ds3d.dims['times']
    ds3d.close()

    results = []
    final_result = []
    # Serial
    if run_parallel == 0:
        for itrack in range(0, ntracks):
        # for itrack in range(0, 5):
            tracknumber = tracks.data[itrack]
            result = calc_envs_track(file_env3d, file_env2d, tracknumber, config)
            final_result.append(result)
    # Parallel
    elif run_parallel >= 1:
        pool = Pool(n_workers)
        final_result = pool.starmap(
            calc_envs_track, zip(repeat(file_env3d), repeat(file_env2d), tracks.data, repeat(config))
            )
        pool.close()

        # # Initialize dask
        # dask_tmp_dir = config.get("dask_tmp_dir", "/tmp")
        # dask.config.set({'temporary-directory': dask_tmp_dir})
        # cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
        # client = Client(cluster)

        # for itrack in range(0, ntracks):
        #     tracknumber = tracks.data[itrack]
        #     result = dask.delayed(calc_envs_track)(
        #         file_env3d, file_env2d, tracknumber, config,
        #     )
        #     results.append(result)

        # # Trigger dask computation
        # final_result = dask.compute(*results)
        # wait(final_result)
    else:
        sys.exit('Valid parallelization flag not set.')

    # Make a variable list from one of the returned dictionaries
    var3d_names = list(final_result[0][0].keys())
    var2d_names = list(final_result[0][1].keys())
    # Get variable attributes from one of the returned dictionaries
    var3d_attrs = final_result[0][2]
    var2d_attrs = final_result[0][3]
    height_coord = final_result[3][4]
    height_attrs = {
            'long_name': 'Height above mean sea level',
            'units': 'm',
    }

    # import pdb; pdb.set_trace()

    # # Remove tracknumbers from the list
    # var_names.remove('tracknumber')
    # var_attrs.pop('tracknumber', None)

    # Loop over variable list to create the dictionary entry
    out_dict = {}
    out_dict_attrs = {}
    nz = len(height_coord)
    var2d_dims = (ntracks, ntimes)
    var3d_dims = (ntracks, ntimes, nz)
    for ivar in var3d_names:
        out_dict[ivar] = np.full(var3d_dims, np.nan, dtype=np.float32)
        out_dict_attrs[ivar] = var3d_attrs[ivar]
    for ivar in var2d_names:
        out_dict[ivar] = np.full(var2d_dims, np.nan, dtype=np.float32)
        out_dict_attrs[ivar] = var2d_attrs[ivar]

    # Collect results
    for itrack in range(0, ntracks):
    # for itrack in range(0, 5):
        if final_result[itrack] is not None:
            # Get the return results for this track
            # The result is a tuple: (out_dict, out_dict_attrs)
            # The first entry is the dictionary containing the variables
            iResult3d = final_result[itrack][0]
            iResult2d = final_result[itrack][1]
            # tracknumber = iResult3d['tracknumber']

            # Loop over each variable and assign values to output dictionary
            for ivar in var3d_names:
                if iResult3d[ivar].ndim == 2:
                    out_dict[ivar][itrack, :, :] = iResult3d[ivar]
            for ivar in var2d_names:
                if iResult2d[ivar].ndim == 1:
                    out_dict[ivar][itrack, :] = iResult2d[ivar]
                    # import pdb; pdb.set_trace()


    # Define a dataset containing all variables
    var_dict = {}
    # Define output variable dictionary
    for key, value in out_dict.items():
        if value.ndim == 3:
            var_dict[key] = (['tracks', 'times', 'z'], value, out_dict_attrs[key])
        if value.ndim == 2:
            var_dict[key] = (['tracks', 'times'], value, out_dict_attrs[key])
    coord_dict = {
        'tracks': (['tracks'], tracks.data, tracks.attrs),
        'times': (['times'], ds3d['times'].data, ds3d['times'].attrs),
        'height': (['height'], height_coord, height_attrs),
    }
    gattr_dict = {
        'Title': 'Center environment data for cell tracks',
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
    dsout.to_netcdf(path=output_filename, mode='w', format='NETCDF4', 
                    unlimited_dims='tracks', encoding=encoding)
    print(f'Output: {output_filename}')

    # import pdb; pdb.set_trace()
    return

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
    file_env3d = f'{input_path}stats_3d_env_{startdate}_{enddate}.nc'
    # 2D environment filename
    file_env2d = f'{input_path}stats_2d_env_{startdate}_{enddate}.nc'
    print(f'Input: {file_env3d}')
    print(f'Input: {file_env2d}')

    # Output filename
    output_filename = f'{output_path}stats_1d_env_{startdate}_{enddate}.nc'
    os.makedirs(output_path, exist_ok=True)

    # Check input file
    fc_3d = os.path.isfile(file_env3d)
    fc_2d = os.path.isfile(file_env2d)
    if (fc_3d == True) & (fc_2d == True):
        # Call function to calculate
        result = work_for_tracks(file_env3d, file_env2d, output_filename, config)
    else:
        print(f'No input file: {file_env3d}')

