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

    celltrack_dir = config['pixeltracking_outpath']
    start_basetime = config.get("start_basetime", None)
    end_basetime = config.get("end_basetime", None)
    startdate = config.get('startdate')
    enddate = config.get('enddate')
    stats_outpath = config.get("stats_outpath")

    # Subset boundary [lat_min, lon_min, lat_max, lon_max]
    geolimits = [-33.3, -64.9, -30.8, -63.4]

    # Cell mask files
    celltrack_basename = config.get('pixeltracking_filebase')
    pixel_radius = config.get("pixel_radius")
    time_format = config.get("wrfout_time_format", None)

    # Output timeseries filename
    out_basename = 'domain_timeseries_cellstats_'
    outfilename = f"{stats_outpath}{out_basename}{startdate}_{enddate}.nc"


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

    # Use mask dataset directly (no Tb data)
    ds = ds_m
    time_coord = ds['time']
    ntimes_out = ds.sizes['time']
    print(f'Number of time steps: {ntimes_out}')

    # Apply inflated tracked cell masks to convective cell masks
    tracknumber = ds.tracknumber
    conv_mask = ds.conv_mask > 0
    cellmask = tracknumber.where(conv_mask > 0)

    # Specify quantile values to save
    quantiles = [0.25, 0.5, 0.75]
    nquantiles = len(quantiles)

    # Define array to save outputs
    cellarea_avg = np.full(ntimes_out, np.nan, dtype=np.float32)
    cellarea_quantile = np.full((nquantiles, ntimes_out), np.nan, dtype=np.float32)
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

    # Filter echotop10 by cell mask
    echotop10_cell = ds.echotop10.where(cellmask > 0)

    # Dimension names to compute statistics
    dim_stats = ('lat', 'lon')
    dim_rechunk = {'lat': -1, 'lon': -1}  # Rechunk dimensions for Dask
    # Average
    ETH10_avg = echotop10_cell.mean(dim=dim_stats, keep_attrs=True).compute()
    # Percentiles
    ETH10_quantile = echotop10_cell.chunk(dim_rechunk).quantile(quantiles, dim=dim_stats, keep_attrs=True).compute()

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
        '_FillValues': np.nan,
    }
    ncells_attrs = {
        'long_name': 'Number of convective cells',
        'units': 'count',
    }

    # Define xarray dataset
    var_dict = {
        'echotop10_mean': (['time'], ETH10_avg.data, ETH10_avg.attrs),
        'echotop10_quantile': (['quantile', 'time'], ETH10_quantile.data, ETH10_quantile.attrs),
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
