import numpy as np
import os, sys, glob
import time
import xarray as xr
import pandas as pd
from pyflextrkr.ft_utilities import subset_files_timerange
from pyflextrkr.ft_utilities import load_config

#--------------------------------------------------------------
def subset_ds(ds, geolimits):
    """
    Subset DataSet by geolimits.

    Args:
        ds: Xarray DataSet
            DataSet containing (time, latitude, longitude) coordinates.
        geolimits: list
            Subset geolimit: [lat_min, lon_min, lat_max, lon_max]

    Return:
        dsout: Xarray DataSet
            Subsetted DataSet.
    """
    # Get 2D x/y coordinates
    xcoord = ds['longitude']
    ycoord = ds['latitude']
    # Subset a single time
    if ('time' in xcoord.dims): xcoord = ds['longitude'].isel(time=0).squeeze()
    if ('time' in ycoord.dims): ycoord = ds['latitude'].isel(time=0).squeeze()
    # Double check coordinate dimension sizes
    if (xcoord.ndim != 2) | (ycoord.ndim != 2):
        print(f"ERROR: xcoord or ycoord dimension size != 2")
        print(f"x ndim: {xcoord.ndim}, y ndim: {ycoord.ndim}")
        sys.exit()
    # Get lat/lon limits
    buffer = 0
    latmin, latmax = geolimits[0]-buffer, geolimits[2]+buffer
    lonmin, lonmax = geolimits[1]-buffer, geolimits[3]+buffer
    # Make a 2D mask
    mask = ((xcoord >= lonmin) & (xcoord <= lonmax) & \
            (ycoord >= latmin) & (ycoord <= latmax)).squeeze()
    # Get y/x indices limits from the mask
    y_idx, x_idx = np.where(mask == True)
    xmin, xmax = np.min(x_idx), np.max(x_idx)
    ymin, ymax = np.min(y_idx), np.max(y_idx)
    # Subset DataSet
    dsout = ds.isel(lon=slice(xmin,xmax+1), lat=slice(ymin,ymax+1))
    return dsout

#--------------------------------------------------------------
def combine_ds(ds_m, ds_tb, max_time_diff=1):
    """
    Combine two Xarray DataSets by finding the nearest times

    Args:
        ds_m: Xarray DataSet
            DataSet for cell masks (reference time).
        ds_tb: Xarray DataSet
            DataSet for Tb.
        max_time_diff: long [minute]
            Maximum time difference allowed to find match.

    Return:
        ds: Xarray DataSet
            Combined DataSet.
    """
    # Extract time coordinates as Pandas DataFrames
    df_m = pd.DataFrame({'time_m': ds_m['time'].values})
    df_tb = pd.DataFrame({'time_tb': ds_tb['time'].values})

    # Ensure times are sorted
    df_m = df_m.sort_values(by='time_m')
    df_tb = df_tb.sort_values(by='time_tb')

    # Define tolerance in [minutes]
    tolerance_pd = pd.Timedelta(minutes=max_time_diff)

    # Find nearest times in ds_tb that match times in ds_m
    matched_df = pd.merge_asof(df_m, df_tb, left_on='time_m', right_on='time_tb', direction='nearest', tolerance=tolerance_pd)
    # Drop NaNs (if no match found)
    matched_df = matched_df.dropna()

    # Use `time_m` as the reference time
    valid_common_times = matched_df['time_m'].values

    # Select ds_m using `valid_common_times` (no change)
    ds_m_common = ds_m.sel(time=valid_common_times)

    # Select ds_tb using `time_tb` from `matched_df` (nearest matched times)
    matched_times_tb = matched_df['time_tb'].values
    ds_tb_common = ds_tb.sel(time=matched_times_tb)

    # Force ds_tb_common to use ds_m_common's exact time coordinate
    ds_tb_common = ds_tb_common.assign_coords(time=ds_m_common['time'])

    # Ensure time alignment before merging
    assert np.array_equal(ds_m_common['time'].values, ds_tb_common['time'].values), "Time mismatch after alignment!"

    # Merge datasets safely
    ds = xr.merge([ds_m_common, ds_tb_common], combine_attrs='drop_conflicts')
    return ds

#--------------------------------------------------------------
def calc_cellarea_stats(cellmask, quantiles=None):
    """
    Calculate cell area statistics

    Args:
        cellmask: np.array
            2D cell mask array with unique numbers.
        quantiles: list
            Quantile values to compute.

    Return:
        cellarea_avg: np.array
            Cell area mean.
        cellarea_quantile: np.array
            Cell area quantiles.
        ncells: long
            Number of cells.
    """
    # Get unique cellmask and their size (pixel counts)
    cellmask1d_uniq, cellmask1d_counts = np.unique(cellmask, return_counts=True)
    # Exclude 0 (non-cell mask)
    idx_valid = np.where(cellmask1d_uniq > 0)
    cellmask1d_uniq = cellmask1d_uniq[idx_valid]
    cellmask1d_counts = cellmask1d_counts[idx_valid]
    ncells = len(cellmask1d_uniq)
    # Convert pixel counts to cell area
    cellarea = cellmask1d_counts * pixel_radius**2
    # Compute statistics
    cellarea_avg = np.nanmean(cellarea)
    cellarea_quantile = np.nanquantile(cellarea, quantiles)
    return (cellarea_avg, cellarea_quantile, ncells)


#-----------------------------------------------------------------
if __name__ == '__main__':

    # Load configuration file
    config_file = sys.argv[1]
    config = load_config(config_file)
    # Get source from config filename (e.g., )
    source = os.path.basename(config_file).split('_')[1]

    celltrack_dir = config['pixeltracking_outpath']
    start_basetime = config.get("start_basetime", None)
    end_basetime = config.get("end_basetime", None)
    startdate = config.get('startdate')
    enddate = config.get('enddate')
    domain = config.get("domain", None)
    stats_outpath = config.get("stats_outpath")

    # Subset boundary [lat_min, lon_min, lat_max, lon_max]
    geolimits = [-33.3, -64.9, -30.8, -63.4]

    # Define max time difference allowed to match cell mask & Tb DataSets
    max_time_diff = 2  # [minute]

    # # TODO: Manual testing (20181129-20:00)
    # start_basetime = 1543521600
    # end_basetime = 1543525200
    
    # Handle OBS & LASSO differences
    if domain is None:
        # OBS (CSAPR)
        # Tb/rainrate file basename
        # tb_basename = 'corvisstpx2drectg16v4minnisX1.regrid2csapr2gridded.c1.'
        # tb_dir = '/gpfs/wolf2/arm/atm131/proj-shared/zfeng/cacti/csapr/corvisstpx2drectg16v4minnisX1.parallaxcorrected_regrid2csapr2gridded.c1/'
        tb_basename = config.get('tb_basename')
        tb_dir = config.get('tb_path')
        tb_varname = 'temperature_ir'
        # Cell mask files
        celltrack_basename = config.get('pixeltracking_filebase')
        pixel_radius = config.get("pixel_radius")
        time_format = config.get("time_format")
    else:
        # LASSO
        # Tb/rainrate file basename
        tb_basename = 'tb_rainrate_'
        # tb_dir = config['clouddata_path']
        tb_dir = config.get('tbrainrate_path')
        tb_varname = 'tb'
        time_format = config.get("wrfout_time_format", None)

        # Cell mask files
        if domain == 'd4':
            celltrack_basename = 'regrid_celltracks_'
            pixel_radius = 0.1  # 100m cell tracks are run at 500m, specify native grid spacing here [km]
        else:
            celltrack_basename = config.get('pixeltracking_filebase')
            pixel_radius = config.get("pixel_radius")

    # Output timeseries filename
    out_basename = 'domain_timeseries_cellstats_'
    outfilename = f"{stats_outpath}{out_basename}{startdate}_{enddate}.nc"


    # Find Tb files
    tbfiles_info = subset_files_timerange(
        tb_dir,
        tb_basename,
        start_basetime=start_basetime,
        end_basetime=end_basetime,
        time_format=time_format,
    )
    # Get file list
    filelist_tb = tbfiles_info[0]
    nfiles_tb = len(filelist_tb)
    print(f'Number of Tb files: {nfiles_tb}')

    # Find cell mask files
    maskfiles_info = subset_files_timerange(
        celltrack_dir,
        celltrack_basename,
        start_basetime=start_basetime,
        end_basetime=end_basetime,
        time_format="yyyymodd_hhmmss",
    )
    # Get file list
    filelist_mask = maskfiles_info[0]
    nfiles_mask = len(filelist_mask)
    print(f'Number of cell mask files: {nfiles_mask}')


    # Read and combine cell mask files
    ds_m = xr.open_mfdataset(filelist_mask, combine='by_coords')
    # Subset domain
    ds_m = subset_ds(ds_m, geolimits)
    xcoord_m = ds_m['lon']
    ycoord_m = ds_m['lat']
    # Get 2D lat/lon arrays
    lon2d_m = ds_m['longitude']
    lat2d_m = ds_m['latitude']
    # Only keep 1 time as lat/lon do not change
    if ('time' in lon2d_m.dims) | ('time' in lat2d_m.dims):
        lon2d_m = lon2d_m.isel(time=0).squeeze()
        lat2d_m = lat2d_m.isel(time=0).squeeze()
        ds_m['longitude'] = lon2d_m
        ds_m['latitude'] = lat2d_m
    ny_m = ds_m.sizes['lat']
    nx_m = ds_m.sizes['lon']

    # Read and combine Tb files
    ds_tb = xr.open_mfdataset(filelist_tb, combine='by_coords')
    # Rename dimensions to be consistent with cell mask file
    if domain is None:
        if (source == 'csapr500m'):
            ds_tb = ds_tb.rename({'lat': 'latitude', 'lon': 'longitude'})
            ds_tb = ds_tb.rename({'y': 'lat', 'x': 'lon'})
    else:
        ds_tb = ds_tb.rename({'Time': 'time', 'south_north': 'lat', 'west_east': 'lon'})
    # Subset domain
    ds_tb = subset_ds(ds_tb, geolimits)
    ny_tb = ds_tb.sizes['lat']
    nx_tb = ds_tb.sizes['lon']
    # Check dimensions again after subset
    if (nx_m == nx_tb) & (ny_m == ny_tb):
        # Drop 2D latitude/longitude coordinate variable
        ds_tb = ds_tb.reset_coords(names=['latitude', 'longitude'], drop=True)
        # Assign 1D lat/lon coordinate from mask
        ds_tb = ds_tb.assign_coords({'lon':xcoord_m, 'lat':ycoord_m})
    else:
        print(f'ERROR: Inconsistent number of grids between mask and Tb files.')
        print(f'ny_tb: {ny_tb}, ny_m: {ny_m}, nx_tb: {nx_tb}, nx_m: {nx_m}')
        sys.exit()
    

    # Combine two DataSets
    ds = combine_ds(ds_m, ds_tb, max_time_diff=max_time_diff)
    time_coord = ds['time']
    ntimes_out = ds.sizes['time']
    print(f'Finished combining DataSets.')

    # Apply inflated tracked cell masks to convective cell masks
    tracknumber = ds.tracknumber
    conv_mask = ds.conv_mask > 0
    cellmask = tracknumber.where(conv_mask > 0)

    # Specify quantile values to save
    quantiles = [0.25, 0.5, 0.75]
    nquantiles = len(quantiles)

    # Define array to save outputs
    cellarea_avg = np.full(ntimes_out, np.NaN, dtype=np.float32)
    cellarea_quantile = np.full((nquantiles, ntimes_out), np.NaN, dtype=np.float32)
    ncells = np.full(ntimes_out, 0, dtype=int)
    # Loop over time
    for tt in range(ntimes_out):
        # Get cell mask at tt
        _cellmask = cellmask.isel(time=tt).values
        # Calculate cell area stats
        cellarea_stats = calc_cellarea_stats(_cellmask, quantiles=quantiles)
        # Save to output arrays
        cellarea_avg[tt] = cellarea_stats[0]
        cellarea_quantile[:,tt] = cellarea_stats[1]
        ncells[tt] = cellarea_stats[2]

    # import matplotlib.pyplot as plt
    # import pdb; pdb.set_trace()

    # Filter variables by cell mask
    tb_cell = ds[tb_varname].where(cellmask > 0)
    echotop10_cell = ds.echotop10.where(cellmask > 0)

    # Dimension names to compute statistics
    dim_stats = ('lat', 'lon')
    dim_rechunk = {'lat': -1, 'lon': -1}  # Rechunk dimensions for Dask
    # Average
    ETH10_avg = echotop10_cell.mean(dim=dim_stats, keep_attrs=True).compute()
    TB_avg = tb_cell.mean(dim=dim_stats, keep_attrs=True).compute()
    # Percentiles
    ETH10_quantile = echotop10_cell.chunk(dim_rechunk).quantile(quantiles, dim=dim_stats, keep_attrs=True).compute()
    TB_quantile = tb_cell.chunk(dim_rechunk).quantile(quantiles, dim=dim_stats, keep_attrs=True).compute()

    #------------------------------------------------------------
    # Write output
    #------------------------------------------------------------
    # Variable attributes
    quantile_attrs = {
        'long_name': 'Quantile values',
    }
    cellarea_attrs = {
        'long_name': 'Convective cell area',
        'units': 'km^2',
        '_FillValues': np.NaN,
    }
    ncells_attrs = {
        'long_name': 'Number of convective cells',
        'units': 'count',
    }

    # Define xarray dataset
    var_dict = {
        'echotop10_mean': (['time'], ETH10_avg.data, ETH10_avg.attrs),
        'tb_mean': (['time'], TB_avg.data, TB_avg.attrs),
        'echotop10_quantile': (['quantile', 'time'], ETH10_quantile.data, ETH10_quantile.attrs),
        'tb_quantile': (['quantile', 'time'], TB_quantile.data, TB_quantile.attrs),
        'cellarea_mean': (['time'], cellarea_avg, cellarea_attrs),
        'cellarea_quantile': (['quantile', 'time'], cellarea_quantile, cellarea_attrs),
        'ncells': (['time'], ncells, ncells_attrs),
    }
    coord_dict = {
        'time': (['time'], ETH10_avg.time.data),
        'quantile': (['quantile'], quantiles, quantile_attrs)
    }
    gattr_dict = {
        'Title': 'Domain cell statistics time-series',
        'startdate': startdate, 
        'enddate': enddate, 
        'ensmember': config.get('ensmember', 'None'),
        'domain': config.get('domain', 'None'),
        'Contact': 'Zhe Feng: zhe.feng@pnnl.gov',
        'Institution': 'Pacific Northwest National Laboratory',
        'created on': time.ctime(time.time()),
    }
    dsout = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)

    # Set encoding/compression for all variables
    comp = dict(zlib=True, dtype='float32')
    encoding = {var: comp for var in dsout.data_vars}
    # Write to netcdf file
    dsout.to_netcdf(path=outfilename, mode='w', format='NETCDF4', unlimited_dims='time', encoding=encoding)
    print(f"Output saved: {outfilename}")
    # import pdb; pdb.set_trace()