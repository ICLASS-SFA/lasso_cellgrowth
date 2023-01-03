import os
import glob
import numpy as np
import xarray as xr
import pandas as pd

def list_dirs(in_dir):
    """
    List sub-directories.
    """
    dirs = []
    for it in os.scandir(in_dir):
        if it.is_dir():
            dirs.append(it.path)
    return dirs

def subset_stats(datafile_obs, datafile_m, out_dir, out_basename, hour_window):
    """
    Subset CSAPR track stats file between LASSO tracking period.
    """
    # Read CSAPR data
    ds_obs = xr.open_dataset(datafile_obs, mask_and_scale=True)
    # Read LASSO data
    ds_m = xr.open_dataset(datafile_m, mask_and_scale=True)
    # Get LASSO start/end datetime from attribute
    startdate = ds_m.attrs['startdate']
    enddate = ds_m.attrs['enddate']
    sdate = pd.to_datetime(f'{startdate[0:4]}-{startdate[4:6]}-{startdate[6:8]}T{startdate[9:11]}:{startdate[11:13]}')
    edate = pd.to_datetime(f'{enddate[0:4]}-{enddate[4:6]}-{enddate[6:8]}T{enddate[9:11]}:{enddate[11:13]}')
    # Subset CSAPR track stats by time window
    sdate_ext = sdate - pd.DateOffset(hours=hour_window)
    edate_ext = edate + pd.DateOffset(hours=hour_window)
    ds_out = ds_obs.where((ds_obs.start_basetime >= sdate_ext) & (ds_obs.end_basetime <= edate_ext), drop=True)
    # Update attributes
    sdate_str = sdate_ext.strftime('%Y%m%d.%H%M')
    edate_str = edate_ext.strftime('%Y%m%d.%H%M')
    ds_out.attrs['startdate'] = sdate_str
    ds_out.attrs['enddate'] = edate_str
    ds_out.attrs['hour_window'] = hour_window

    # Write output
    out_filename = f'{out_dir}{out_basename}{sdate_str}_{edate_str}.nc'
    ds_out.to_netcdf(out_filename, mode="w", format="NETCDF4", unlimited_dims="tracks",)

    return out_filename

if __name__ == "__main__":
    # CSAPR track stats file
    datadir_obs = '/gpfs/wolf/atm131/proj-shared/zfeng/cacti/csapr/stats/'
    datafile_obs = f'{datadir_obs}trackstats_20181015.0000_20190303.0000.nc'

    # LASSO tracking directory
    root_datadir_m = '/gpfs/wolf/atm131/proj-shared/zfeng/cacti/les/'
    in_basename = 'trackstats_20'

    # Output directory
    out_dir = '/gpfs/wolf/atm131/proj-shared/zfeng/cacti/csapr/stats4lasso/'
    out_basename = 'csapr_trackstats_'
    # LASSO domain (d3, d4)
    domain = 'd3'

    # Hour window to extend beyond the model start/end time
    # hour_window = 3
    hour_window = 0

    # LASSO case dates
    start_dates = [
        "20181129", 
        "20181204",
        "20181205", 
        "20181219", 
        "20190122", 
        "20190123",
        "20190125",
        "20190129",
        "20190208",
    ]

    # Make output directory
    os.makedirs(out_dir, exist_ok=True)

    # Loop over dates
    for ii in range(0, len(start_dates)):
        # Get sub-directories (separate runs)
        idate = start_dates[ii]
        idirs = list_dirs(f'{root_datadir_m}/{idate}/')

        if len(idirs) > 0:
            # Find track stats file
            datafiles_m = sorted(glob.glob(f'{idirs[0]}/{domain}/stats/{in_basename}*nc'))
            if len(datafiles_m) > 0:
                # Run subset
                out_filename = subset_stats(datafile_obs, datafiles_m[0], out_dir, out_basename, hour_window)
                print(out_filename)

# import pdb; pdb.set_trace()


