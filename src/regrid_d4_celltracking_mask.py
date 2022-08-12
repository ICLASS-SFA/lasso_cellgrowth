import glob
import os
import sys
import time
import yaml
import numpy as np
import xarray as xr
import dask
from dask.distributed import Client, LocalCluster
from pyflextrkr.ft_utilities import load_config

#-----------------------------------------------------------------------
def regrid_file(in_filename, out_dir, out_basename):

    # Read input data
    ds = xr.open_dataset(in_filename, decode_times=False, mask_and_scale=False)
    time_coord = ds['time']
    ny, nx = ds.sizes['lat'], ds.sizes['lon']
    # Create a coordinate to mimic subsampling a 5:1 ratio of the full coordinate
    ratio = 5
    xcoord = (np.linspace(2, nx*ratio+2, nx, endpoint=False, dtype=int))
    ycoord = (np.linspace(2, ny*ratio+2, ny, endpoint=False, dtype=int))
    # Replace the input data coordinate
    ds = ds.assign_coords({'lat':ycoord, 'lon':xcoord})    

    # Create a full coordinate
    xcoord_out = np.arange(0, nx*ratio, 1)
    ycoord_out = np.arange(0, ny*ratio, 1)

    # Get variables for regridding
    tracknumber = ds['tracknumber']
    tracknumber_cmask = ds['tracknumber_cmask']
    track_status = ds['track_status']
    cloudnumber = ds['cloudnumber']
    merge_tracknumber = ds['merge_tracknumber']
    split_tracknumber = ds['split_tracknumber']

    # Remap to the full coordinate
    tracknumber_out = tracknumber.interp(
        lon=xcoord_out, lat=ycoord_out, method='nearest', 
        assume_sorted=True, kwargs={"fill_value": "extrapolate"},
    ).data.astype(int)
    tracknumber_cmask_out = tracknumber_cmask.interp(
        lon=xcoord_out, lat=ycoord_out, method='nearest', 
        assume_sorted=True, kwargs={"fill_value": "extrapolate"},
    ).data.astype(int)
    track_status_out = track_status.interp(
        lon=xcoord_out, lat=ycoord_out, method='nearest', 
        assume_sorted=True, kwargs={"fill_value": "extrapolate"},
    ).data.astype(int)
    cloudnumber_out = cloudnumber.interp(
        lon=xcoord_out, lat=ycoord_out, method='nearest', 
        assume_sorted=True, kwargs={"fill_value": "extrapolate"},
    ).data.astype(int)
    merge_tracknumber_out = merge_tracknumber.interp(
        lon=xcoord_out, lat=ycoord_out, method='nearest', 
        assume_sorted=True, kwargs={"fill_value": "extrapolate"},
    ).data.astype(int)
    split_tracknumber_out = split_tracknumber.interp(
        lon=xcoord_out, lat=ycoord_out, method='nearest', 
        assume_sorted=True, kwargs={"fill_value": "extrapolate"},
    ).data.astype(int)

    # Make output filename
    nleadingchar = len(f'{in_basename}')
    fname = os.path.basename(in_filename)
    ftimestr = fname[nleadingchar:]
    out_filename = f'{out_dir}{out_basename}{ftimestr}'

    # Define variable list
    var_dict = {
        "base_time": (["time"], time_coord.data, time_coord.attrs),
        # "longitude": (["lat", "lon"], longitude),
        # "latitude": (["lat", "lon"], latitude),
        "nclouds": (["time"], ds['nclouds'].data, ds['nclouds'].attrs),
        # "comp_ref": (["time", "lat", "lon"], comp_ref),
        # "dbz_lowlevel": (["time", "lat", "lon"], dbz_lowlevel),
        # "conv_core": (["time", "lat", "lon"], conv_core),
        # "conv_mask": (["time", "lat", "lon"], conv_mask),
        "tracknumber": (["time", "lat", "lon"], tracknumber_out, tracknumber.attrs),
        "tracknumber_cmask": (["time", "lat", "lon"], tracknumber_cmask_out, tracknumber_cmask.attrs),
        "track_status": (["time", "lat", "lon"], track_status_out, track_status.attrs),
        "cloudnumber": (["time", "lat", "lon"], cloudnumber_out, cloudnumber.attrs),
        "merge_tracknumber": (["time", "lat", "lon"], merge_tracknumber_out, merge_tracknumber.attrs),
        "split_tracknumber": (["time", "lat", "lon"], split_tracknumber_out, split_tracknumber.attrs),
        # "echotop10": (["time", "lat", "lon"], echotop10),
        # "echotop20": (["time", "lat", "lon"], echotop20),
        # "echotop30": (["time", "lat", "lon"], echotop30),
        # "echotop40": (["time", "lat", "lon"], echotop40),
        # "echotop50": (["time", "lat", "lon"], echotop50),
    }
    # Define coordinate list
    coord_dict = {
        "time": (["time"], time_coord.data, time_coord.attrs),
        "lat": (["lat"], ycoord_out),
        "lon": (["lon"], xcoord_out),
    }
    # Define global attributes
    gattr_dict = {
        "title": "Pixel-level cell tracking data regridded back to d4 grid",
        "contact": "Zhe Feng, zhe.feng@pnnl.gov",
        "created_on": time.ctime(time.time()),
    }
    # Define xarray dataset
    ds_out = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)

    # Set encoding/compression for all variables
    comp = dict(zlib=True)
    encoding = {var: comp for var in ds_out.data_vars}

    # Write to netcdf file
    ds_out.to_netcdf(
        path=out_filename, mode='w', format='NETCDF4', unlimited_dims='time', encoding=encoding,
    )
    print(f'Output saved: {out_filename}')
    # import pdb; pdb.set_trace()
    return out_filename 


#-----------------------------------------------------------------------
if __name__ == "__main__":

    config_file = sys.argv[1]

    # Load configuration file
    config_file = sys.argv[1]
    config = load_config(config_file)

    in_dir = config['pixeltracking_outpath']
    in_basename = config['pixeltracking_filebase']
    out_basename = f'regrid_{in_basename}'
    out_dir = in_dir

    run_parallel = config['run_parallel']
    # run_parallel = 0
    nprocesses = config['nprocesses']
    dask_tmp_dir = config.get("dask_tmp_dir", "/tmp")

    # Find input files
    in_files = sorted(glob.glob(f'{in_dir}{in_basename}*.nc'))
    print(f'Number of files to process: {len(in_files)}')


    # Serial
    if run_parallel == 0:
        for ifile in in_files:
            print(ifile)
            result = regrid_file(ifile, out_dir, out_basename)
    # Parallel
    elif run_parallel >= 1:
        # Set Dask temporary directory for workers
        dask.config.set({'temporary-directory': dask_tmp_dir})
        # Local cluster
        cluster = LocalCluster(n_workers=nprocesses, threads_per_worker=1, silence_logs=False)
        client = Client(cluster)

        results = []
        for ifile in in_files:
            print(ifile)
            result = dask.delayed(regrid_file)(ifile, out_dir, out_basename)
            results.append(result)
        final_result = dask.compute(*results)
    else:
        sys.exit('Valid parallelization flag not provided')
    # import pdb; pdb.set_trace()