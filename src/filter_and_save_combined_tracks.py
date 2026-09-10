#!/usr/bin/env python
"""
Filter and save combined cell track datasets for LASSO simulations.

This script reads multiple cell tracking statistics files from different case dates,
combines them, filters tracks based on spatial/temporal criteria and merge/split 
conditions, and saves the filtered datasets to netCDF files.

Usage:
    python filter_and_save_combined_tracks.py --domain d2
    python filter_and_save_combined_tracks.py --domain d3
    python filter_and_save_combined_tracks.py --domain d4
    python filter_and_save_combined_tracks.py --domain all  # Process all domains
    python filter_and_save_combined_tracks.py --domain all --output-dir /path/to/output --time-offset +1.0

Author: Generated from plot_narrow_wide_cell_composite_profiles_avg_env.ipynb
Date: 2025-12-02
"""

import os
import glob
import argparse
import numpy as np
import xarray as xr
import pandas as pd
import warnings


def list_dirs(in_dir):
    """
    List sub-directories.
    """
    dirs = []
    for it in os.scandir(in_dir):
        if it.is_dir():
            dirs.append(it.path)
    return dirs


def find_lasso_files(start_dates, root_datadir, in_basename, domain):
    """
    Find LASSO data files for specified dates and domain.
    
    Args:
        start_dates: list
            List of case dates (format: 'YYYYMMDD')
        root_datadir: str
            Root directory containing data
        in_basename: str
            Base filename pattern to search for
        domain: str
            Domain name (e.g., 'd2', 'd3', 'd4')
    
    Returns:
        files: list
            List of file paths found
    """
    files = []
    # Loop over dates
    for ii in range(0, len(start_dates)):
        # Get sub-directories (cases)
        idate = start_dates[ii]
        case_dirs = list_dirs(f'{root_datadir}/{idate}/')
        if len(case_dirs) > 0:
            # Loop over cases
            for icase in case_dirs:
                # Find config sub-directories
                config_dirs = [f'{icase}/base/']
                if len(config_dirs) > 0:
                    # Loop over configs
                    for iconfig in config_dirs:
                        # Find track stats file
                        fname = sorted(glob.glob(f'{iconfig}{domain}/stats/{in_basename}*nc'))
                        if len(fname) > 0:
                            files.append(fname[0])
    return files


def get_casename(in_file):
    """
    Get case date, ensemble member, and config from full filename.
    
    Args:
        in_file: str
            Full file path
    
    Returns:
        caseconfig: str
            Configuration name
        ensmember: str
            Ensemble member name
        casedate: str
            Case date (YYYYMMDD)
    """
    # Example file name with file directory:
    # .../meso//20181129/gefs09/base/d2/stats/trackstats_20181129.1200_20181130.0000.nc
    caseconfig = (in_file).split(os.path.sep)[-4]
    ensmember = (in_file).split(os.path.sep)[-5]
    casedate = (in_file).split(os.path.sep)[-6]
    return caseconfig, ensmember, casedate


def subset_tracks(ds, time_res, lon_range, lat_range, lifetime_range, hours_range):
    """
    Subset tracks based on spatial and temporal criteria.
    
    Args:
        ds: xarray.Dataset
            Track statistics dataset
        time_res: float
            Time resolution [minutes]
        lon_range: tuple
            Longitude range (min, max)
        lat_range: tuple
            Latitude range (min, max)
        lifetime_range: tuple
            Lifetime range [minutes] (min, max)
        hours_range: tuple
            Start hour range (min, max)
    
    Returns:
        dsout: xarray.Dataset
            Subsetted dataset
        track_idx: numpy.ndarray
            Track indices that meet criteria
    """
    # Get track min/max lat/lon
    minlon = ds['meanlon'].min(dim='times').load()
    maxlon = ds['meanlon'].max(dim='times').load()
    minlat = ds['meanlat'].min(dim='times').load()
    maxlat = ds['meanlat'].max(dim='times').load()
    # Get track lifetime in [minutes]
    lifetime = (ds['track_duration'] * time_res).load()
    # Get track start time in decimal [hours]
    start_hour = (ds['start_basetime'].dt.hour + ds['start_basetime'].dt.minute/60.).load()
    # Get track indices
    track_idx = np.where(
        (minlon >= np.min(lon_range)) & (maxlon <= np.max(lon_range)) & \
        (minlat >= np.min(lat_range)) & (maxlat <= np.max(lat_range)) & \
        (lifetime >= min(lifetime_range)) & (lifetime <= np.max(lifetime_range)) & \
        (start_hour >= np.min(hours_range)) & (start_hour <= np.max(hours_range))
    )[0]
    # Subset tracks
    dsout = ds.isel(tracks=track_idx)
    return dsout, track_idx


def read_subset_combine(tfiles, wfiles, wmaskfiles, efiles, case_hours_dict, 
                        lon_range, lat_range, lifetime_range, time_res, time_env):
    """
    Read and subset track statistics files and combine multiple DataSets.
    
    Subsets based on:
    - Track inititial lat/lon
    - Track duration (lifetime)
    - Track initial times (different for each case)
    
    Args:
        tfiles: list
            List of track statistics files.
        wfiles: list
            List of W statistics files.
        wmaskfiles: list
            List of W mask statistics files.
        efiles: list
            List of environment statistics files.
        case_hours_dict: dictionary
            Case hour range dictionary.
        lon_range: list/tuple
            Longitude range to subset tracks.
        lat_range: list/tuple
            Latitude range to subset tracks.
        lifetime_range: list/tuple
            Lifetime range to subset tracks.
        time_res: float
            Time resolution of tracks [minutes].
        time_env: int
            Relative time index with respect to CI time for representative environment.
        
    Returns:
        ds: Xarray DataSet
            Combined Xarray DataSet.
    """
    
    # Suppress the warning within this block only
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
    
        # Make lists for the DataSets
        dst_list = []
        dsw_list = []
        dsm_list = []
        dse_list = []
        # List to keep track of original track numbers
        original_tracks_list = []

        # Loop over the number of files
        for ii in range(len(tfiles)):
            tfile = tfiles[ii]
            wfile = wfiles[ii]
            mfile = wmaskfiles[ii]
            efile = efiles[ii]
            caseconfig, ensmember, casedate = get_casename(tfile)
            # Make casename (date-ensmember) and look up hours
            casename = f"{casedate}-{ensmember}"
            # Get hours range for the case
            hours_range = case_hours_dict[casename]
            print(f"Processing: {casename}, hours range: {hours_range}")

            # Read tracks file
            dst = xr.open_dataset(tfile)
            # Add a variable case (date-forcing) to the DataSet
            dst['case'] = casename
            dst['date'] = pd.to_datetime(casedate, format='%Y%m%d')
            # Subset tracks
            dst_s, track_idx = subset_tracks(dst, time_res, lon_range, lat_range, lifetime_range, hours_range)
            # Store original track numbers for this subset
            original_tracks_list.append(dst_s['tracks'].values.copy())
            # Add subsetted DataSet to the list
            dst_list.append(dst_s)

            # Read W file
            dsw = xr.open_dataset(wfile)
            # Subset W tracks (only keep largest core)
            dsw_s = dsw.isel(tracks=track_idx, core=0)
            dsw_list.append(dsw_s)

            # Read W mask file
            dsm = xr.open_dataset(mfile)
            # Subset W mask tracks
            dsm_s = dsm.isel(tracks=track_idx)
            dsm_list.append(dsm_s)

            # Read ENV file
            dse = xr.open_dataset(efile)
            # Subset ENV tracks, select representative time before CI
            dse_s = dse.isel(tracks=track_idx).sel(times=time_env)
            dse_list.append(dse_s)

        # Concatenate original track numbers from all cases
        original_tracks = np.concatenate(original_tracks_list)

        # Concetenante subsetted DataSets
        dst = xr.concat(dst_list, dim='tracks', data_vars='all')
        # Renumber tracks to make it continuous
        ntracks = dst.sizes['tracks']
        dst['tracks'] = np.arange(0, ntracks)

        dsw = xr.concat(dsw_list, dim='tracks')
        dsw['tracks'] = np.arange(0, ntracks)

        dsm = xr.concat(dsm_list, dim='tracks')
        dsm['tracks'] = np.arange(0, ntracks)

        dse = xr.concat(dse_list, dim='tracks')
        dse['tracks'] = np.arange(0, ntracks)

        # Combine 2D and 3D datasets by coordinates
        ds = xr.combine_by_coords([dst, dsw, dsm, dse], combine_attrs='drop_conflicts')
        
        # Add original track numbers as a data variable (not a coordinate)
        ds['original_tracks'] = (('tracks',), original_tracks)
        ds['original_tracks'].attrs['long_name'] = 'Original track number from individual case files'
        ds['original_tracks'].attrs['description'] = 'Track numbers before renumbering for combined dataset'
        
        ntimes = ds.sizes['times']
        print(f'Number of tracks: {ntracks}')

    return ds


def find_merge_split_tracks(ds, time_window, time_res=5.0):
    """
    Find merge/split track indices.
    
    Args:
        ds: xarray.Dataset
            Track statistics dataset
        time_window: tuple
            Time window to constrain mergers [minutes] (start, end)
        time_res: float, default=5.0
            Time resolution [minutes]
    
    Returns:
        out_dict: dictionary
            Dictionary containing various sets of track indices
    """
    # Get track lifetime
    lifetime = ds.track_duration.load() * time_res

    # Non-split tracks
    idx_nsplit = np.isnan(ds.start_split_cloudnumber.load())

    # Non-merge tracks
    idx_nmerge = np.isnan(ds.end_merge_cloudnumber.load())

    # Merge after time window tracks
    idx_merge_after = ~np.isnan(ds.end_merge_cloudnumber.load()) & (lifetime > np.max(time_window))

    # Combine conditions (satisfy both):
    # - Non-merge or merge after time window
    # - Non-split
    idx_nms = np.logical_and(idx_nsplit, (np.logical_or(idx_nmerge, idx_merge_after)))

    # Group outputs to dictionary
    out_dict = {
        'ns': idx_nsplit,
        'nm': idx_nmerge,
        'ms': idx_merge_after,
        'nms': idx_nms,
    }
    return out_dict


def get_case_hours_dict(domain='d2', time_offset=2.0):
    """
    Get case hour dictionaries for specified domain with time offset applied.
    
    Args:
        domain: str, default='d2'
            Domain name ('d2', 'd3', 'd4')
        time_offset: float, default=2.0
            Time offset in hours to add to the end time of each case.
            Can be negative.
    
    Returns:
        case_hours_dict: dict
            Dictionary of hour ranges for each case with time offset applied
    """
    # Base reference for d2
    if domain == 'd2':
        base_dict = {
            '20181129-gefs00': (12.0, 19.5),
            '20181129-gefs03': (12.0, 20.0),
            '20181129-gefs09': (12.0, 18.5),
            '20181129-gefs18': (12.0, 18.5),
            '20181204-gefs18': (12.0, 17.5),
            '20181204-gefs19': (12.0, 19.0),
            '20181205-gefs01': (12.0, 18.0),
            '20181219-eda09': (12.0, 17.0),
            '20190122-gefs01': (12.0, 21.5),
            '20190122-gefs18': (12.0, 20.25),
            '20190123-eda05': (12.0, 16.0),
            '20190123-gefs18': (12.0, 17.5),
            '20190125-eda07': (12.0, 18.5),
            '20190125-gefs11': (12.0, 18.5),
            '20190129-eda09': (12.0, 18.0),
            '20190129-gefs11': (12.0, 17.5),
            '20190208-eda03': (12.0, 19.5),
            '20190208-eda08': (12.0, 19.75),
        }
    else:  # d3 or d4 use the same base
        base_dict = {
            '20181129-gefs00': (12.0, 19.5),
            '20181129-gefs03': (12.0, 20.0),
            '20181129-gefs09': (12.0, 19.0),
            '20181129-gefs18': (12.0, 19.0),
            '20181204-gefs18': (12.0, 17.5),
            '20181204-gefs19': (12.0, 19.0),
            '20181205-gefs01': (12.0, 18.0),
            '20181219-eda09': (12.0, 17.0),
            '20190122-gefs01': (12.0, 21.5),
            '20190122-gefs18': (12.0, 20.5),
            '20190123-eda05': (12.0, 16.0),
            '20190123-gefs18': (12.0, 17.5),
            '20190125-eda07': (12.0, 18.5),
            '20190125-gefs11': (12.0, 18.5),
            '20190129-eda09': (12.0, 18.5),
            '20190129-gefs11': (12.0, 18.25),
            '20190208-eda03': (12.0, 20.0),
            '20190208-eda08': (12.0, 20.0),
        }
    
    # Apply time offset to end times
    case_hours_dict = {
        # Ensure it does not exceed 23.99 hours (just before the next day)
        case: (start, min(end + time_offset, 23.99))
        for case, (start, end) in base_dict.items()
    }
    
    return case_hours_dict


def process_domain(domain, rootdir, output_dir, start_dates, time_offset=2.0):
    """
    Process a single domain: read, filter, and save combined datasets.
    
    Args:
        domain: str
            Domain name ('d2', 'd3', or 'd4')
        rootdir: str
            Root directory for data
        output_dir: str
            Output directory for saved files
        start_dates: list
            List of case dates
        time_offset: float, default=2.0
            Time offset in hours to use
    """
    print(f"\n{'='*80}")
    print(f"Processing domain: {domain.upper()}")
    print(f"{'='*80}\n")
    
    # Configuration
    in_basename = 'trackstats_20'
    in_basename_w = 'stats_3d_w_fixshell_'
    # in_basename_wmask = 'stats_2d_wmask_'
    in_basename_wmask = 'stats_2d_wmask_ci30min_'
    # in_basename_wmask = 'stats_2d_wmask_2h_'
    # Environment file basename depends on domain
    if 'd2' in domain:
        in_basename_env = 'stats_avg1d_env9x9_'
    else:  # d3 or d4
        if '2.5km' in domain:
            in_basename_env = 'stats_avg1d_env9x9_'
        else:
            in_basename_env = 'stats_avg1d_env21x21_'

    # # Sensitivity test with 10x10 km environment files
    # if 'd2' in domain:
    #     in_basename_env = 'stats_avg1d_env5x5_'
    # else:  # d3 or d4
    #     if '2.5km' in domain:
    #         in_basename_env = 'stats_avg1d_env5x5_'
    #     else:
    #         in_basename_env = 'stats_avg1d_env11x11_'

    # Domain 4 boundaries
    lon_range = [-65., -63.3]
    lat_range = [-33., -30.8]
    lifetime_range = [15, 1000]
    time_res = 5.0  # [minutes]
    
    # Time for representative environment
    time_env = -1  # relative time index with respect to CI time
    
    # Get case hours dictionary for this domain with time offset applied
    case_hours_dict = get_case_hours_dict(domain=domain, time_offset=time_offset)
    
    # Find data files
    print("Finding data files...")
    files = find_lasso_files(start_dates, rootdir, in_basename, domain)
    wfiles = find_lasso_files(start_dates, rootdir, in_basename_w, domain)
    wmaskfiles = find_lasso_files(start_dates, rootdir, in_basename_wmask, domain)
    efiles = find_lasso_files(start_dates, rootdir, in_basename_env, domain)

    print(f"Found {len(files)} track files")
    print(f"Found {len(wfiles)} W files")
    print(f"Found {len(wmaskfiles)} W mask files")
    print(f"Found {len(efiles)} environment files")
    
    if len(files) == 0:
        print(f"WARNING: No files found for domain {domain}")
        return
    
    # Read and combine data
    print("\nReading and combining datasets...")
    ds = read_subset_combine(files, wfiles, wmaskfiles, efiles, case_hours_dict,
                            lon_range, lat_range, lifetime_range, time_res, time_env)
    
    # Get time window from dataset attributes
    time_window = (ds.attrs['time_start'], ds.attrs['time_end'])
    print(f"Time window: {time_window} min")
    
    # Find merge/split tracks
    print("\nFinding merge/split tracks...")
    tid = find_merge_split_tracks(ds, time_window, time_res)
    
    # Get number of all tracks
    ntracks = ds.sizes['tracks']
    print(f"Total tracks: {ntracks}")
    
    # Filter non-merge-split tracks
    print("Filtering non-merge-split tracks...")
    dsnms = ds.isel(tracks=tid['nms'])
    
    # Get number of filtered tracks
    ntracks_nms = dsnms.sizes['tracks']
    print(f"Non-merge-split tracks: {ntracks_nms} ({100*ntracks_nms/ntracks:.0f}% of all tracks)")
    
    # Add attribute for time_offset used
    dsnms.attrs['coldpool_time_offset'] = time_offset
    
    # Save filtered dataset
    output_file = f'{output_dir}/stats_combined_filtered_{domain}.nc'
    print(f"\nSaving filtered dataset to:\n{output_file}")
    
    # Define compression encoding for all variables
    comp = {'zlib': True, 'complevel': 5}
    encoding = {var: comp for var in dsnms.data_vars}
    
    dsnms.to_netcdf(output_file, encoding=encoding)
    print(f"Successfully saved {ntracks_nms} tracks")
    print(f"File size: {os.path.getsize(output_file) / (1024**3):.2f} GB")


def main():
    """Main function to parse arguments and process domains."""
    parser = argparse.ArgumentParser(
        description='Filter and save combined cell track datasets for LASSO simulations.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single domain
  python filter_and_save_combined_tracks.py --domain d2
  
  # Process all domains
  python filter_and_save_combined_tracks.py --domain all
  
  # Use different time offset
  python filter_and_save_combined_tracks.py --domain d3 --time-offset +1h
        """
    )
    
    parser.add_argument(
        '--domain',
        type=str,
        required=True,
        choices=['d2', 'd3', 'd4', 'd3_5min_2.5km', 'd4_5min_2.5km', 'all'],
        help='Domain to process (d2, d3, d4, or all)'
    )
    
    parser.add_argument(
        '--time-offset',
        type=float,
        default=2.0,
        help='Time offset in hours to add to case end times (default: 2.0, can be negative)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/gpfs/wolf2/arm/atm131/proj-shared/zfeng/cacti/les/cell_stats_combined',
        help='Output directory for filtered datasets'
    )
    
    args = parser.parse_args()
    
    # LASSO case dates
    start_dates = [
        "20181129", 
        "20181204",
        "20181205", 
        # "20181219", 
        "20190122", 
        "20190123",
        "20190125",
        "20190129",
        "20190208",
    ]
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("LASSO Cell Track Filtering Script")
    print(f"{'='*80}")
    print(f"Output directory: {args.output_dir}")
    print(f"Time offset: {args.time_offset}")
    print(f"Case dates: {', '.join(start_dates)}")
    
    # Process domains
    if args.domain == 'all':
        domains = ['d2', 'd3', 'd4']
    else:
        domains = [args.domain]
    
    for domain in domains:
        # Set root directory based on domain
        if domain == 'd2':
            rootdir = '/gpfs/wolf2/arm/atm131/proj-shared/zfeng/cacti/meso/'
        else:  # d3 or d4
            rootdir = '/gpfs/wolf2/arm/atm131/proj-shared/zfeng/cacti/les/'
        
        try:
            process_domain(domain, rootdir, args.output_dir, start_dates, args.time_offset)
        except Exception as e:
            print(f"\nERROR processing domain {domain}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print("Processing complete!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
