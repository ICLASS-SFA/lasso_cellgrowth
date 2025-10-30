import glob
import os
import sys
import time
import yaml
import numpy as np
from scipy import ndimage
import xarray as xr
import pandas as pd
import dask
from dask.distributed import Client, LocalCluster

#-----------------------------------------------------------------------
def convolve_reflectivity(in_reflectivity, kernel):
    """
    Apply convolution to reflectivity within a moving kernel.
    This is equivalent to averaging reflectivity within a moving window.

    Args:
        in_reflectivity: np.array
            Input reflectivity array, can be either 2D or 3D.
        kernel: np.array
            Kernel for weights.
    
    Returns:
        out_reflectivity: np.array
            Output reflectivity array.
    """
    # Convert reflectivity to linear unit
    linrefl = 10. ** (in_reflectivity / 10.)
    # Make an array for counting number of grids for convolution
    mask_goodvalues = (~np.isnan(in_reflectivity)).astype(float)

    # Apply convolution filter
    bkg_linrefl = ndimage.convolve(linrefl, kernel, mode='constant', cval=0.0)
    numPixs = ndimage.convolve(mask_goodvalues, kernel, mode='constant', cval=0.0)
    # Mask missing data area
    bkg_linrefl[mask_goodvalues==0] = 0
    numPixs[mask_goodvalues==0] = 0

    # Calculate average linear reflectivity and convert to log values
    out_reflectivity = np.full(in_reflectivity.shape, np.NaN, dtype=np.float32)
    out_reflectivity[numPixs>0] = 10.0 * np.log10(bkg_linrefl[numPixs>0] / numPixs[numPixs>0])

    # Remove pixels with 0 number of pixels
    out_reflectivity[mask_goodvalues==0] = np.NaN
    
    return out_reflectivity

#-----------------------------------------------------------------------
def regrid_file(in_filename, out_dir, out_basename):
    """
    Regrid a file containing reflectivity data.

    Args:
        in_filename: string
            Input file name.
        out_dir: string
            Output directory name.
        out_basename: string
            Output file basename.
    
    Returns:
        out_filename: string
            Output file name.
    """

    # Read input data
    ds = xr.open_dataset(in_filename)
    in_time = ds['Time']
    REFL_10CM = ds['REFL_10CM'].squeeze()
    REFL_10CM_MAX = ds['REFL_10CM_MAX'].squeeze()
    XLONG = ds['XLONG']
    XLAT = ds['XLAT']

    # Make a kernel for weights
    ratio = 5   # Must be odd number
    start_idx = int((ratio-1) / 2)
    kernel = np.zeros((ratio+1,ratio+1), dtype=int)
    kernel[1:ratio, 1:ratio] = 1

    # Make a 3D kernel
    kernel3d = kernel[None,:,:]
    # Call convlution function
    REFL_10CM_conv = convolve_reflectivity(REFL_10CM.data, kernel3d)
    REFL_10CM_MAX_conv = convolve_reflectivity(REFL_10CM_MAX.data, kernel)

    # Subsample every X grid points
    REFL_10CM_reg = REFL_10CM_conv[:,start_idx::ratio,start_idx::ratio]
    REFL_10CM_MAX_reg = REFL_10CM_MAX_conv[start_idx::ratio,start_idx::ratio]
    XLONG_reg = XLONG.data[start_idx::ratio,start_idx::ratio]
    XLAT_reg = XLAT.data[start_idx::ratio,start_idx::ratio]

    # Make output filename
    nleadingchar = len(f'{in_basename}{ensmember}_')
    fname = os.path.basename(in_filename)
    ftimestr = fname[nleadingchar:]
    out_filename = f'{out_dir}{out_basename}{ftimestr}'
    
    # Make output coordinate
    nz, ny, nx = REFL_10CM_reg.shape
    # Create a coordinate to mimic subsampling ratio of the full coordinate
    xcoord = (np.linspace(2, nx*ratio+2, nx, endpoint=False, dtype=int))
    ycoord = (np.linspace(2, ny*ratio+2, ny, endpoint=False, dtype=int))
    xcoord_attrs = {
        'long_name': 'X-coordinate grid index',
    }
    ycoord_attrs = {
        'long_name': 'Y-coordinate grid index',
    }

    # Define output variablesf
    var_dict = {
        'XLONG': (['south_north', 'west_east'], XLONG_reg, XLONG.attrs),
        'XLAT': (['south_north', 'west_east'], XLAT_reg, XLAT.attrs),
        'REFL_10CM': (['Time', 'HAMSL', 'south_north', 'west_east'], np.expand_dims(REFL_10CM_reg, axis=0), REFL_10CM.attrs),
        'REFL_10CM_MAX': (['Time', 'south_north', 'west_east'], np.expand_dims(REFL_10CM_MAX_reg, axis=0), REFL_10CM_MAX.attrs),
    }
    # Output coordinates
    coord_dict = {
        'Time': (['Time'], in_time.data),
        'HAMSL': (['HAMSL'], ds['HAMSL'].data, ds.HAMSL.attrs),
        'south_north': (['south_north'], ycoord, ycoord_attrs),
        'west_east': (['west_east'], xcoord, xcoord_attrs),
    }
    # Output global attributes
    gattr_dict = {
        'Title': 'Regrid reflectivity from LASSO',
        'SIMULATION_START_DATE': ds.attrs['SIMULATION_START_DATE'],
        'run_name': ds.attrs['run_name'],
        'DX': ds.attrs['DX']*ratio,
        'DY': ds.attrs['DY']*ratio,
        'Contact': 'Zhe Feng, zhe.feng@pnnl.gov',
        'Institution': 'Pacific Northwest National Laboratory',
        'Created_on': time.ctime(time.time()),
    }
    # Define xarray dataset
    ds_out = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)

    # Set encoding/compression for all variables
    comp = dict(zlib=True)
    encoding = {var: comp for var in ds_out.data_vars}

    # Write to netcdf file
    ds_out.to_netcdf(
        path=out_filename, mode='w', format='NETCDF4', unlimited_dims='Time', encoding=encoding,
    )
    print(f'Output saved: {out_filename}')
    return out_filename 

#-----------------------------------------------------------------------
if __name__ == "__main__":

    config_file = sys.argv[1]

    # Read configuration from yaml file
    stream = open(config_file, "r")
    config = yaml.full_load(stream)

    in_dir = config['in_dir']
    in_basename = config['in_basename']
    ensmember = config['ensmember']
    out_dir = config['out_dir']
    out_basename = config['out_basename']
    run_parallel = config['run_parallel']
    nprocesses = config['nprocesses']
    dask_tmp_dir = config.get("dask_tmp_dir", "/tmp")
    start_datetime = config['start_datetime']
    end_datetime = config['end_datetime']

    # Generate time marks within the start/end datetime
    datetimes = pd.date_range(start=start_datetime, end=end_datetime, freq='5min')
    init_datetime = datetimes[0].strftime('%Y%m%d00')
    forecast_datetimes = datetimes.strftime('%H%M%S')
    file_datetimes = init_datetime + '_f' + forecast_datetimes

    # Find files
    in_files = []
    for tt in range(0, len(file_datetimes)):
        in_files.extend(sorted(glob.glob(f'{in_dir}{in_basename}{ensmember}_{file_datetimes[tt]}*d4.nc')))
    print(f'Number of files to process: {len(in_files)}')

    # in_files = sorted(glob.glob(f'{in_dir}{in_basename}{ensmember}*d4.nc'))

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
