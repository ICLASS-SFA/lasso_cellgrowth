"""
Make masks of the largest updraft objects from the time-height W stats data
The steps include:
- Label the largest updraft objects within a specified CI time window
- Get updraft object top/base height, start/end times, cloud base updraft area
- Make linear fit for updraft top height
- Compute distance between updraft base height & LCL
- Save output to netCDF file
"""
import os, sys, glob
import yaml
import time
import numpy as np
import xarray as xr
from scipy.stats import linregress
from scipy.ndimage import label
import warnings

#-----------------------------------------------------------------------
def get_W_properties(W_array, tidx_start, tidx_end, tidx_end_CI, cbase_depth):
    """
    Get W object properties given a W 2D array
    
    Args:
        W_array: np.array [times, height]
            W 2D array.
        tidx_start: int
            Start time index to subset W object.
        tidx_end: int
            End time index to subset W object.
        tidx_end_CI: int
            End time index for CI sampling.
        cbase_depth: float
            Depth above cloud base to compute W statistics [km].
            
    Returns:
        out_dict: dictionary
            A dictionary containing W object property variables.    
    """
    
    # Make a binary array for W
    W_binary = np.zeros_like(W_array)
    W_binary[W_array > 0] = 1
    
    # Label the connected pixels
    label_image, n_obj = label(W_binary > 0)
    # Trim the times, then get the unique values of the labeled image, and the pixel counts
    label_num_, npix_ = np.unique(label_image[tidx_start:tidx_end,:], return_counts=True)
    # Trim the 0 (not an object), then count the number of W objects
    label_num_W = np.trim_zeros(label_num_)
    n_obj_W = len(label_num_W)

    # Make output arrays
    mask_time2d = np.zeros(label_image.shape)
    mask_large = np.full(label_image.shape, np.nan)
    mask_remove = np.full(label_image.shape, np.nan)
    # mask_large = np.full(label_image.shape, 0, dtype=np.int8)
    # mask_remove = np.full(label_image.shape, 0, dtype=np.int8)
    max_indices_z = np.full(label_image.shape, -1, dtype=int)
    mask_top = np.full(label_image.shape[0], np.nan)
    mask_base = np.full(label_image.shape[0], np.nan)
    Wbase_timeseries = np.full(label_image.shape[0], np.nan)
    Wtime_start = np.nan
    Wtime_end = np.nan
    Wlifetime = np.nan
    Wbase_mean = np.nan
    Wbase_median = np.nan
    Wbase_max = np.nan
    Wbase_max_time = np.nan
    Wbase_max_CI = np.nan
    Wbase_max_time_CI = np.nan

    # Get the size (pixel count) of the W objects
    if n_obj_W > 0:
        # Exclude the first value (0)
        npix_W = npix_[1:]
        # Get the largest object number
        big_cell = label_num_W[np.argmax(npix_W)]

        # Mask time window to keep object    
        mask_time2d[tidx_start:tidx_end+1,:] = 1
        # Make a binary mask of the largest object
        mask_large[(label_image==big_cell) & (mask_time2d==1)] = 1
        # Make a mask for removing object (for plotting purpose)
        mask_remove[(mask_large != 1) & (W_binary > 0)] = 1

        #------------------------------------
        # Get updraft top heights
        # Get indices of the last occurrence of the max value along the y-axis (height)
        max_indices_z = np.argmax(mask_large[:,::-1]==1, axis=1)
        # Invert the indices to get the correct position along the original y-axis
        max_indices_z = mask_large.shape[1] - 1 - max_indices_z
        # Filter out indices where the mask is False
        max_indices_z[np.isnan(np.nanmax(mask_large, axis=1))] = -1
        # Apply indices to height array using advanced indexing
        mask_top = height.data[max_indices_z]
        # Replace times when indices is invalid with NaN (< 0)
        mask_top[max_indices_z < 0] = np.nan
        #------------------------------------
        #------------------------------------
        # Get updraft base heights
        # Get indices of the last occurrence of the max value along the y-axis (height)
        min_indices_z = np.argmax(mask_large==1, axis=1)
        # Filter out indices where the mask is False
        min_indices_z[np.isnan(np.nanmax(mask_large, axis=1))] = -1
        # Apply indices to height array using advanced indexing
        mask_base = height.data[min_indices_z]
        # Replace times when indices is invalid with NaN (< 0)
        mask_base[min_indices_z < 0] = np.nan
        
        # Take median value for updraft base heights
        median_base = np.nanmedian(mask_base)
        # Get base height + depth
        median_base_top = median_base + cbase_depth
        # Get z indices
        median_base_z0 = np.argmin(np.abs(height.values - median_base))
        median_base_z1 = np.argmin(np.abs(height.values - median_base_top))
        # print(median_base, median_base_top)
        # print(median_base_z0, median_base_z1)
        # Mask out small cores
        W_array[mask_remove == 1] = np.nan
        # Check if layer top > base
        if (median_base_z1 > median_base_z0):
            _W_array = W_array[:,median_base_z0:median_base_z1+1]
            # Get the time & height index of the maximum W value within the layer
            t_idx_WbaseMax, z_idx_WbaseMax = np.unravel_index(np.nanargmax(_W_array), _W_array.shape)
            # Get time series of the maximum W value within the layer
            Wbase_timeseries = np.nanmax(_W_array, axis=1)
        else:
            _W_array = W_array[:,median_base_z0]
            # Get the time & height index of the maximum W value within the layer
            t_idx_WbaseMax = np.nanargmax(_W_array)
            # Get time series of the maximum W value within the layer
            Wbase_timeseries = _W_array.squeeze()
        # Get W statistics within the cloud-base layer
        Wbase_mean = np.nanmean(_W_array)
        Wbase_median = np.nanmedian(_W_array)
        Wbase_max = np.nanmax(_W_array)
        # # Get the time & height index of the maximum W value within the layer
        # t_idx_WbaseMax, z_idx_WbaseMax = np.unravel_index(np.nanargmax(_W_array), _W_array.shape)
        # Get the time value corresponding to the max W base value
        Wbase_max_time = time_coord.values[t_idx_WbaseMax].item()

        # Subset W_array for CI period
        if _W_array.ndim == 2:
            _W_array_CI = _W_array[:tidx_end_CI+1,:]
            Wbase_max_CI = np.nanmax(_W_array_CI)

            # Check if there are valid values in the CI period
            if np.isnan(_W_array_CI).all():
                Wbase_max_CI = np.nan
                Wbase_max_time_CI = np.nan
            else:
                t_idx_WbaseMax_CI, z_idx_WbaseMax_CI = np.unravel_index(np.nanargmax(_W_array_CI), _W_array_CI.shape)
                Wbase_max_time_CI = time_coord.values[t_idx_WbaseMax_CI].item()
        else:
            _W_array_CI = _W_array[:tidx_end_CI+1]
            Wbase_max_CI = np.nanmax(_W_array_CI)

            # Check if there are valid values in the CI period
            if np.isnan(_W_array_CI).all():
                Wbase_max_CI = np.nan
                Wbase_max_time_CI = np.nan
            else:
                t_idx_WbaseMax_CI = np.nanargmax(_W_array_CI)
                Wbase_max_time_CI = time_coord.values[t_idx_WbaseMax_CI].item()
        # import pdb; pdb.set_trace()
        #------------------------------------

        # Find start/end time indices of the largest object
        # tind, zind = np.where(label_image==big_cell)
        tind, zind = np.where(mask_large == 1)
        istart, iend = min(tind), max(tind)
        # Mark start/end time and lifetime
        Wtime_start = time_coord.values[istart].item()
        Wtime_end = time_coord.values[iend].item()
        Wlifetime = (iend - istart + 1) * time_res

    # Put outputs to a dictionary
    out_dict = {
        'mask_large': mask_large,
        'mask_remove': mask_remove,
        'mask_top': mask_top,
        'mask_base': mask_base,
        'Wtime_start': Wtime_start,
        'Wtime_end': Wtime_end,
        'Wlifetime': Wlifetime,
        'Wbase_mean': Wbase_mean,
        'Wbase_median': Wbase_median,
        'Wbase_max': Wbase_max,
        'Wbase_max_time': Wbase_max_time,
        'Wbase_max_CI': Wbase_max_CI,
        'Wbase_max_time_CI': Wbase_max_time_CI,
        'Wbase_timeseries': Wbase_timeseries,
    }
        
    return out_dict

#-----------------------------------------------------------------------
def make_W_mask(da_W, maxETH_10dbz, tidx_start, tidx_end, tidx_end_CI, cbase_depth=0.5):
    """
    Make W object masks for all tracks
    
    Args:
        da_W: DataArray [tracks, times, height]
            Data array containing W.
        maxETH_10dbz: DataArray [tracks, times]
            Max 10dBZ echo-top height.
        tidx_start: int
            Start time index to subset W object.
        tidx_end: int
            End time index to subset W object.
        tidx_end_CI: int
            End time index for CI sampling.
        cbase_depth: float
            Depth above cloud base to compute W statistics [km].
            
    Returns:
        out_dict: dictionary
            A dictionary containing W object property variables.    
    """

    # Make a binary array for W
    # binary_w = np.zeros_like(da_W)
    # binary_w[da_W > 0] = 1
    
    # Make output arrays
    _ntracks = da_W.sizes['tracks']
    _ntimes = da_W.sizes['times']
    Wmask = np.full(da_W.shape, np.nan)
    Wmask_remove = np.full(da_W.shape, np.nan)
    # Wmask = np.full(da_W.shape, 0, dtype=np.int8)
    # Wmask_remove = np.full(da_W.shape, 0, dtype=np.int8)
    Wtime_start = np.full(_ntracks, np.nan)
    Wtime_end = np.full(_ntracks, np.nan)
    Wlifetime = np.full(_ntracks, np.nan)
    Wtop = np.full((_ntracks, _ntimes), np.nan)
    Wbase = np.full((_ntracks, _ntimes), np.nan)
    Wbase_mean = np.full(_ntracks, np.nan)
    Wbase_median = np.full(_ntracks, np.nan)
    Wbase_max = np.full(_ntracks, np.nan)
    Wbase_max_time = np.full(_ntracks, np.nan)
    Wbase_max_CI = np.full(_ntracks, np.nan)
    Wbase_max_time_CI = np.full(_ntracks, np.nan)
    # maxETH_filter = np.copy(maxETH_10dbz.values)
    maxETH_filter = np.full(maxETH_10dbz.shape, np.nan)
    Wbase_timeseries = np.full((_ntracks, _ntimes), np.nan)
    # import pdb; pdb.set_trace()

    # Loop over each track
    for itrack in range(0, _ntracks):
        # # Get the binary mask for this track
        # iW_binary = binary_w[itrack,:,:].squeeze()
        # # Call function to get W object properties
        # iW_object_dict = get_W_properties(iW_binary, tidx_start, tidx_end)
        
        # Get W for this track
        iW = (da_W.isel(tracks=itrack).squeeze().values).copy()
        iETH = (maxETH_10dbz.isel(tracks=itrack).squeeze().values).copy()
        # Call function to get W object properties
        iW_object_dict = get_W_properties(iW, tidx_start, tidx_end, tidx_end_CI, cbase_depth)
        
        # Save values to output arrays
        Wmask[itrack,:,:] = iW_object_dict['mask_large']
        Wmask_remove[itrack,:,:] = iW_object_dict['mask_remove']
        Wtop[itrack,:] = iW_object_dict['mask_top']
        Wbase[itrack,:] = iW_object_dict['mask_base']
        Wtime_start[itrack] = iW_object_dict['Wtime_start']
        Wtime_end[itrack] = iW_object_dict['Wtime_end']
        Wlifetime[itrack] = iW_object_dict['Wlifetime']
        Wbase_mean[itrack] = iW_object_dict['Wbase_mean']
        Wbase_median[itrack] = iW_object_dict['Wbase_median']
        Wbase_max[itrack] = iW_object_dict['Wbase_max']
        Wbase_max_time[itrack] = iW_object_dict['Wbase_max_time']
        Wbase_max_CI[itrack] = iW_object_dict['Wbase_max_CI']
        Wbase_max_time_CI[itrack] = iW_object_dict['Wbase_max_time_CI']
        Wbase_timeseries[itrack,:] = iW_object_dict['Wbase_timeseries']
        # Filter ETH after Wtime_end
        if Wtime_end[itrack] > 0:
            _Wtime_end = int(Wtime_end[itrack])
            # Find matching index from the time_coord
            _tidx = np.where(time_coord == _Wtime_end)[0].item()
            iETH[_tidx+1:] = np.nan
            maxETH_filter[itrack,:] = iETH
            # maxETH_filter[itrack,Wend_tidx+1:] = np.nan

    # Convert to DataArrays
    Wmask = xr.DataArray(Wmask, coords=da_W.coords, dims=('tracks','times','z'))
    Wmask_remove = xr.DataArray(Wmask_remove, coords=da_W.coords, dims=('tracks','times','z'))
    Wtop = xr.DataArray(Wtop, coords={'tracks':da_W.coords['tracks'], 'times':da_W.coords['times']}, dims=('tracks','times'))
    Wbase = xr.DataArray(Wbase, coords={'tracks':da_W.coords['tracks'], 'times':da_W.coords['times']}, dims=('tracks','times'))
    Wtime_start = xr.DataArray(Wtime_start, coords={'tracks':da_W.coords['tracks']}, dims=('tracks'))
    Wtime_end = xr.DataArray(Wtime_end, coords={'tracks':da_W.coords['tracks']}, dims=('tracks'))
    Wlifetime = xr.DataArray(Wlifetime, coords={'tracks':da_W.coords['tracks']}, dims=('tracks'))
    Wbase_mean = xr.DataArray(Wbase_mean, coords={'tracks':da_W.coords['tracks']}, dims=('tracks'))
    Wbase_median = xr.DataArray(Wbase_median, coords={'tracks':da_W.coords['tracks']}, dims=('tracks'))
    Wbase_max = xr.DataArray(Wbase_max, coords={'tracks':da_W.coords['tracks']}, dims=('tracks'))
    Wbase_max_time = xr.DataArray(Wbase_max_time, coords={'tracks':da_W.coords['tracks']}, dims=('tracks'))
    Wbase_max_CI = xr.DataArray(Wbase_max_CI, coords={'tracks':da_W.coords['tracks']}, dims=('tracks'))
    Wbase_max_time_CI = xr.DataArray(Wbase_max_time_CI, coords={'tracks':da_W.coords['tracks']}, dims=('tracks'))
    Wbase_timeseries = xr.DataArray(Wbase_timeseries, coords={'tracks':da_W.coords['tracks'], 'times':da_W.coords['times']}, dims=('tracks','times'))
    maxETH_filter = xr.DataArray(maxETH_filter, coords={'tracks':maxETH_10dbz.coords['tracks'], 'times':maxETH_10dbz.coords['times']}, dims=('tracks','times'))
    # Put outputs in a dictionary
    out_dict = {
        'Wmask': Wmask,
        'Wmask_remove': Wmask_remove,
        'Wtop': Wtop,
        'Wbase': Wbase,
        'Wtime_start': Wtime_start,
        'Wtime_end': Wtime_end,
        'Wlifetime': Wlifetime,
        'Wbase_mean': Wbase_mean,
        'Wbase_median': Wbase_median,
        'Wbase_max': Wbase_max,
        'Wbase_max_time': Wbase_max_time,
        'Wbase_max_CI': Wbase_max_CI,
        'Wbase_max_time_CI': Wbase_max_time_CI,
        'Wbase_timeseries': Wbase_timeseries,
        'maxETH_10dbz_filter': maxETH_filter,
    }
    return out_dict

#-----------------------------------------------------------------------
def fit_linear(xtime, Wtop, tidx_end, min_nsample=3):
    """
    Perform linear fit to a 1D time series array
    
    Args:
        xtime: np.array
            Array containing time values
        Wtop: np.array
            Array containing values to be fitted
        tidx_end: int
            Max time index to subset the time series before fitting
        min_nsample: int
            Minimum number of valid samples for fitting
            
    Returns:
        out_dict: dictionary
            A dictionary containing W object property variables.    
    """
    # Set default values
    slope, intercept, r_value, p_value, std_err = np.nan, np.nan, np.nan, np.nan, np.nan
    nt = len(xtime)
    predicted_x = np.full(nt, np.nan, dtype=np.float32)
    predicted_y = np.full(nt, np.nan, dtype=np.float32)
    
    # Subset Wtop time period
    Wtop_sub = Wtop[0:tidx_end+1]
    # Make sure there are valid samples
    if np.count_nonzero(~np.isnan(Wtop_sub)) > 0:
        # Find time index for max Wtop within the subet time period
        _Wtop_max_tidx = Wtop_sub.argmax().item()

        # Subset times to Wtop max
        _xtime = xtime.data[0:_Wtop_max_tidx+1]
        _Wtop = Wtop.data[0:_Wtop_max_tidx+1]

        # Find valid value time indices
        tidx_valid = _Wtop > 0
        n_valid = np.count_nonzero(tidx_valid)
        # print(n_valid)
        if (n_valid >= min_nsample):
            _xtime = _xtime[tidx_valid]
            _Wtop = _Wtop[tidx_valid]

            # Perform linear regression
            slope, intercept, r_value, p_value, std_err = linregress(_xtime, _Wtop)

            # Find indices matching the subsetted time to the full time
            t_idx = np.where(np.isin(xtime, _xtime))[0]
            # Save the valid subsetted time
            predicted_x[t_idx] = _xtime
            # Predict y values using the linear fit
            predicted_y[t_idx] = slope * _xtime + intercept
        
    # Put output values in a dictionary
    out_dict = {
        'slope': slope,
        'intercept': intercept,
        'r_value': r_value,
        'p_value': p_value,
        'std_err': std_err,
        'predicted_x': predicted_x,
        'predicted_y': predicted_y,
    }
    return out_dict


if __name__ == "__main__":

    # Get configuration file name from input
    config_file = sys.argv[1]
    # Read configuration from yaml file
    stream = open(config_file, 'r')
    config = yaml.full_load(stream)

    stats_path = config['stats_path']
    startdate = config['startdate']
    enddate = config['enddate']
    resolution = config['resolution']

    # Input files
    in_basename = 'trackstats_'
    # in_basename_w = 'stats_3d_w_'
    in_basename_w = 'stats_3d_w_fixshell_'
    # Environment file basename based on resolution
    if resolution == 'les':
        # in_basename_env = 'stats_1d_env_2location_'
        in_basename_env = 'stats_avg1d_env21x21_'
        # in_basename_env = 'stats_avg1d_env9x9_'
    elif resolution == 'meso':
        in_basename_env = 'stats_avg1d_env9x9_'
    tfiles = f'{stats_path}{in_basename}{startdate}_{enddate}.nc'
    wfiles = f'{stats_path}{in_basename_w}{startdate}_{enddate}.nc'
    envfiles = f'{stats_path}{in_basename_env}{startdate}_{enddate}.nc'
    # Output file
    # out_basename = 'stats_2d_wmask_'
    out_basename = 'stats_2d_wmask_ci15min_'
    # out_basename = 'stats_2d_wmask_2h_'
    output_filename = f'{stats_path}{out_basename}{startdate}_{enddate}.nc'


    # Buffer height above which to filter updraft variables [km]
    ETH_buffer = 1.0
    # Time for representative CI environment
    # time_env = -3     # for avg1d_env_2location
    time_env = -1       # for avg1d_env21x21
    # Define a time window to sample updraft
    time_start = 0  # [min]
    time_end = 60  # [min]
    # time_end = 120  # [min]
    time_end_CI = 15  # [min] End time for CI sampling (for cloud-base updraft width)
    # Minimum number of sample to fit updraft top height
    # min_nsample_fit = 3
    min_nsample_fit = 2
    # Threshold for max distance between updraft base and LCL height [km]
    max_Wbase_LCL_dist = 2.0
    # Threshold for updraft top fit line slope
    min_Wtop_slope = 0
    # Depth above cloud base to get W statistics [km]
    cbase_depth = 0.5


    # Read track data
    dst = xr.open_dataset(tfiles, engine='netcdf4')
    ntracks = dst.sizes['tracks']
    tracks_coord = dst.coords['tracks']
    times_coord = dst.coords['times']
    print(f'Number of tracks: {ntracks}')

    # Read W data
    drop_vars = ['base_time']
    dsw = xr.open_dataset(wfiles, drop_variables=drop_vars).isel(core=0)

    # Read ENV data (select CI location and representative time before CI, drop 'height' dimension)
    # dse = xr.open_dataset(envfiles).sel(location=1).drop_dims(['height']).sel(times=time_env)
    dse = xr.open_dataset(envfiles).drop_dims(['height']).sel(times=time_env)

    # Combine datasets by coordinates
    ds = xr.combine_by_coords([dst, dsw, dse], combine_attrs='drop_conflicts')

    # Get some constents common for all Datasets
    ntimes = ds.sizes['times']
    nz = ds.sizes['z']
    # ncores = ds.sizes['core']

    # time_res = 5.0  # [min]
    # Convert time resolution to [minute]
    time_res = np.round(dst.attrs['time_resolution_hour']*60)
    time_coord = ds.times
    z_coord = ds.z
    xtime = ds.times * time_res
    height = z_coord / 1000.


    # Filter largest updrafts above 10dBZ ETH + buffer
    maxETH_10dbz = ds['maxETH_10dbz'].load()
    CoreArea_up = ds['CoreArea_up'].where(height <= maxETH_10dbz+ETH_buffer).load()
    CoreMaxW_up = ds['CoreMaxW_up'].where(height <= maxETH_10dbz+ETH_buffer).load()

    # Convert AGL to HAMSL
    LCL_mu = (ds['LCL_height_mu'] + ds['Z_surface']) / 1000.
    LFC_mu = (ds['LFC_height_mu'] + ds['Z_surface']) / 1000.


    # Get updraft track coordinates
    c_times = CoreMaxW_up.coords['times'].data
    c_z = CoreMaxW_up.coords['z'].data

    # Get start/end time window as indices
    tidx_start = np.where(xtime == time_start)[0].item()
    tidx_end = np.where(xtime == time_end)[0].item()
    tidx_end_CI = np.where(xtime == time_end_CI)[0].item()
    # print(tidx_start, tidx_end)
    print(f"Updraft mask time window: {xtime[tidx_start].item()} - {xtime[tidx_end].item()} min")
    print(f"Updraft CI sampling time window: {xtime[tidx_start].item()} - {xtime[tidx_end_CI].item()} min")
    CI_time_window = [xtime[tidx_start].item(), xtime[tidx_end_CI].item()]
    # import pdb; pdb.set_trace()

    # Make 2D updraft object masks for each track 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        Wmask_dict = make_W_mask(CoreArea_up, maxETH_10dbz, tidx_start, tidx_end, tidx_end_CI, cbase_depth)
        Wspeed_dict = make_W_mask(CoreMaxW_up, maxETH_10dbz, tidx_start, tidx_end, tidx_end_CI, cbase_depth)

    # Rename updraft base keys to be more descriptive
    # Updraft area variables
    Wmask_dict['Warea_mean'] = Wmask_dict.pop('Wbase_mean')
    Wmask_dict['Warea_median'] = Wmask_dict.pop('Wbase_median')
    Wmask_dict['Warea_max'] = Wmask_dict.pop('Wbase_max')
    Wmask_dict['Warea_max_time'] = Wmask_dict.pop('Wbase_max_time')
    Wmask_dict['Warea_max_CI'] = Wmask_dict.pop('Wbase_max_CI')
    Wmask_dict['Warea_max_time_CI'] = Wmask_dict.pop('Wbase_max_time_CI')
    Wmask_dict['Warea_cloudbase'] = Wmask_dict.pop('Wbase_timeseries')

    # Updraft speed variables
    Wspeed_dict['Wspeed_mean'] = Wspeed_dict.pop('Wbase_mean')
    Wspeed_dict['Wspeed_median'] = Wspeed_dict.pop('Wbase_median')
    Wspeed_dict['Wspeed_max'] = Wspeed_dict.pop('Wbase_max')
    # import pdb; pdb.set_trace()

    # Make arrays to store data (narrow cells)
    ntracks = Wmask_dict['Wtop'].sizes['tracks']
    Wtop_slope = np.full(ntracks, np.nan, dtype=np.float32)
    Wtop_fit = np.full((ntracks, ntimes), np.nan, dtype=np.float32)

    # Loop over each track (narrow cells)
    for itrack in range(ntracks):
        # print(itrack)
        _Wtop = Wmask_dict['Wtop'].isel(tracks=itrack)
        fit_dict = fit_linear(xtime, _Wtop, tidx_end, min_nsample=min_nsample_fit)
        Wtop_slope[itrack] = fit_dict['slope']
        Wtop_fit[itrack,:] = fit_dict['predicted_y']

    # Calculate distance between minimum updraft base height and MU LCL height (HAMSL)
    Wbase_LCL_diff = np.absolute(Wmask_dict['Wbase'].min(dim='times') - LCL_mu)

    # True/False flag for positive slope
    Wtop_Pslope = Wtop_slope > min_Wtop_slope
    # True/False flag for updraft base height from LCL
    Wbase_LCL = Wbase_LCL_diff <= max_Wbase_LCL_dist

    # Convert boolean arrays to integer arrays (True -> 1, False -> 0)
    Wtop_Pslope_int = Wtop_Pslope.astype(np.int8)
    Wbase_LCL_int = Wbase_LCL.astype(np.int8)


    ##########################################################
    # Write to netcdf
    print('Writing output netcdf ... ')

    tracks_dimname = 'tracks'
    times_dimname = 'times'
    z_dimname = 'z'

    out_dict_attrs = {
        'Wmask': {
            'long_name': 'Updraft core mask to keep',
            'units': 'unitless',
        },
        'Wmask_remove': {
            'long_name': 'Updraft core masks to remove (for plotting purpose)',
            'units': 'unitless',
        },
        'Wtop': {
            'long_name': 'Updraft top height',
            'units': 'km',
        },
        'Wbase': {
            'long_name': 'Updraft base height',
            'units': 'km',
        },
        'Wtime_start': {
            'long_name': 'Updraft core start time',
            'units': 'unitless',
        },
        'Wtime_end': {
            'long_name': 'Updraft core end time',
            'units': 'unitless',
        },
        'Wlifetime': {
            'long_name': 'Updraft core lifetime',
            'units': 'minute',
        },
        # Cloud-base updraft area variables
        'Warea_mean': {
            'long_name': 'Cloud base mean updraft core area',
            'units': 'km^2',
        },
        'Warea_median': {
            'long_name': 'Cloud base median updraft core area',
            'units': 'km^2',
        },
        'Warea_max': {
            'long_name': 'Cloud base max updraft core area',
            'units': 'km^2',
        },
        'Warea_max_time': {
            'long_name': 'Time of cloud base max updraft core area',
            'units': 'unitless',
        },
        'Warea_max_CI': {
            'long_name': 'Cloud base max updraft core area within CI period',
            'units': 'km^2',
            'CI_time_window': CI_time_window,
        },
        'Warea_max_time_CI': {
            'long_name': 'Time of cloud base max updraft core area within CI period',
            'units': 'unitless',
            'CI_time_window': CI_time_window,
        },
        'Warea_cloudbase': {
            'long_name': 'Cloud base updraft area time series',
            'units': 'km^2',
        },
        # Cloud-base updraft speed variables
        'Wspeed_mean': {
            'long_name': 'Cloud base mean updraft speed',
            'units': 'm/s',
        },
        'Wspeed_median': {
            'long_name': 'Cloud base median updraft speed',
            'units': 'm/s',
        },
        'Wspeed_max': {
            'long_name': 'Cloud base max updraft speed',
            'units': 'm/s',
        },
        # Fits and misc
        'Wtop_fit': {
            'long_name': 'Updraft top height linear fit values',
            'units': 'km',
        },
        'Wtop_slope': {
            'long_name': 'Updraft top height linear fit slope',
            'units': 'km/time',
        },
        'Wbase_LCL_diff': {
            'long_name': 'Updraft base height distance from LCL',
            'units': 'km',
        },
        'maxETH_10dbz_filter': {
            'long_name': 'Maximum 10dBZ echo-top height filtered by updraft core',
            'units': 'km',
        },
    }

    # Define variable dictionary
    var_dict = {}
    # Define output variable dictionary
    for key, value in Wmask_dict.items():
        if value.ndim == 1:
            var_dict[key] = ([tracks_dimname], value.data, out_dict_attrs[key])
        if value.ndim == 2:
            var_dict[key] = ([tracks_dimname, times_dimname], value.data, out_dict_attrs[key])
        if value.ndim == 3:
            var_dict[key] = ([tracks_dimname, times_dimname, z_dimname], value.data, out_dict_attrs[key])
            
    # Add Wtop fit variables to the dictionary
    var_dict['Wtop_slope'] = ([tracks_dimname], Wtop_slope, out_dict_attrs['Wtop_slope'])
    var_dict['Wtop_fit'] = ([tracks_dimname, times_dimname], Wtop_fit, out_dict_attrs['Wtop_fit'])
    var_dict['Wbase_LCL_diff'] = ([tracks_dimname], Wbase_LCL_diff.data, out_dict_attrs['Wbase_LCL_diff'])

    # Add Wspeed variables to the dictionary
    var_dict['Wspeed_mean'] = ([tracks_dimname], Wspeed_dict['Wspeed_mean'].data, out_dict_attrs['Wspeed_mean'])
    var_dict['Wspeed_median'] = ([tracks_dimname], Wspeed_dict['Wspeed_median'].data, out_dict_attrs['Wspeed_median'])
    var_dict['Wspeed_max'] = ([tracks_dimname], Wspeed_dict['Wspeed_max'].data, out_dict_attrs['Wspeed_max'])
    # import pdb; pdb.set_trace()

    # Define coordinates
    coord_dict = {
        tracks_dimname: ([tracks_dimname], tracks_coord.data, tracks_coord.attrs),
        times_dimname: ([times_dimname], times_coord.data, times_coord.attrs),
        z_dimname: ([z_dimname], z_coord.data, z_coord.attrs),
    }

    # Define global attributes
    gattr_dict = {
        'title':  'Tracked cell updraft core masks & statistics',
        'Institution': 'Pacific Northwest National Laboratoy',
        'Contact': 'Zhe Feng, zhe.feng@pnnl.gov',
        'Created_by': f'{os.path.abspath(__file__)}',
        'Created_on': time.ctime(time.time()),
        'source_trackfile': tfiles,
        'source_W_file': wfiles,
        'source_env_file': envfiles,
        'time_start': time_start,
        'time_end': time_end,
        'time_environment': time_env,
        'ETH_buffer': ETH_buffer,
        'cbase_depth': cbase_depth,
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
    dsout.to_netcdf(path=output_filename, mode="w", format="NETCDF4", unlimited_dims=tracks_dimname, encoding=encoding)
    print(f'Output saved: {output_filename}')
