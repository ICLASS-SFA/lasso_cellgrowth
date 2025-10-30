"""
Batch processing script for regridding LASSO subset files using Dask for parallelization.

This script processes multiple files in parallel using Dask LocalCluster.
"""
import argparse
import glob
import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
import dask
from dask.distributed import Client, LocalCluster
from regrid_func import regrid_file

#-----------------------------------------------------------------------
def setup_logging(log_file=None):
    """Set up logging configuration"""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers
    )

#-----------------------------------------------------------------------
def find_input_directory(case_date, ensemble, domain):
    """
    Find the input directory for the given case, checking multiple locations.
    
    Args:
        case_date: str - Case date (YYYYMMDD)
        ensemble: str - Ensemble member (e.g., 'gefs18')
        domain: str - Domain (d3 or d4)
    
    Returns:
        str - Input directory path, or None if not found
    """
    logger = logging.getLogger(__name__)
    
    # Define search paths based on domain
    if domain == 'd4':
        # d4 domain: check two locations
        search_paths = [
            f"/gpfs/wolf2/arm/cli120/world-shared/d3m088/cacti/staged_runs_2/{case_date}/{ensemble}/base/les/subset_{domain}/",
            f"/gpfs/wolf2/arm/cli120/world-shared/d3m088/cacti/staged_runs/{case_date}/{ensemble}/base/les/subset_{domain}/",
        ]
    elif domain == 'd3':
        # d3 domain: different location
        search_paths = [
            f"/gpfs/wolf2/arm/atm131/proj-shared/money/{case_date}/{ensemble}/base/les/subset_{domain}/",
        ]
    else:
        logger.error(f"Unknown domain: {domain}")
        return None
    
    # Check each path
    for path in search_paths:
        if os.path.isdir(path):
            logger.info(f"Found input directory: {path}")
            return path
        else:
            logger.debug(f"Directory not found: {path}")
    
    logger.error(f"No input directory found for case {case_date}, domain {domain}")
    return None

#-----------------------------------------------------------------------
def get_files_in_timewindow(base_dir, filebase, start_time, end_time):
    """
    Get list of files within a time window based on filename timestamps.
    
    Args:
        base_dir: str - Base directory containing files
        filebase: str - File base name pattern
        start_time: datetime - Start time
        end_time: datetime - End time
    
    Returns:
        list of file paths
    """
    logger = logging.getLogger(__name__)
    
    # Search pattern
    search_pattern = f"{base_dir}{filebase}*.nc"
    all_files = sorted(glob.glob(search_pattern))
    logger.info(f"Found {len(all_files)} total files matching pattern")
    
    # Filter by time window
    files_in_window = []
    for fpath in all_files:
        # Extract timestamp from filename
        # Example: corlasso_methamsl_2019012300gefs18d4_base_M1.m1.20190123.155000.nc
        fname = os.path.basename(fpath)
        try:
            # Extract datetime string (YYYYMMDD.HHMMSS)
            parts = fname.split('.')
            date_str = parts[-3]  # YYYYMMDD
            time_str = parts[-2]  # HHMMSS
            time_full = date_str + time_str
            file_time = datetime.strptime(time_full, '%Y%m%d%H%M%S')
            
            if start_time <= file_time <= end_time:
                files_in_window.append(fpath)
        except Exception as e:
            logger.warning(f"Could not parse timestamp from {fname}: {e}")
            continue
    
    logger.info(f"Found {len(files_in_window)} files within time window {start_time} to {end_time}")
    return files_in_window

#-----------------------------------------------------------------------
def process_single_file(in_filename, in_basename, out_dir, out_basename, config):
    """
    Wrapper function for processing a single file with error handling.
    
    Returns:
        dict with keys: file, success, error
    """
    logger = logging.getLogger(__name__)
    try:
        success = regrid_file(in_filename, in_basename, out_dir, out_basename, config)
        return {'file': in_filename, 'success': success, 'error': None}
    except Exception as e:
        logger.error(f"Error processing {in_filename}: {str(e)}")
        return {'file': in_filename, 'success': False, 'error': str(e)}

#-----------------------------------------------------------------------
def process_batch_files(file_list, config, n_workers=1):
    """
    Process a batch of files in serial or parallel mode.
    
    Args:
        file_list: list - List of tuples (in_file, in_base, out_dir, out_base)
        config: dict - Configuration dictionary
        n_workers: int - Number of parallel workers (default: 1 for serial)
                        If 1, runs in serial mode. If >1, uses Dask parallelization.
        
    Returns:
        list of result dictionaries
    """
    logger = logging.getLogger(__name__)
    
    if n_workers == 1:
        # Serial processing
        logger.info(f"Processing {len(file_list)} files in SERIAL mode")
        results = []
        for i, (in_file, in_base, out_dir, out_base) in enumerate(file_list, 1):
            logger.info(f"Processing file {i}/{len(file_list)}: {os.path.basename(in_file)}")
            result = process_single_file(in_file, in_base, out_dir, out_base, config)
            results.append(result)
            if result['success']:
                logger.info(f"  ✓ Success")
            else:
                logger.error(f"  ✗ Failed: {result['error']}")
    else:
        # Parallel processing with Dask
        logger.info(f"Processing {len(file_list)} files in PARALLEL mode with {n_workers} workers")
        
        # Set up Dask LocalCluster
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=1,  # Single-threaded workers to avoid nested parallelism
            memory_limit='40GB',   # Memory limit per worker
            dashboard_address=':8787',
        )
        client = Client(cluster)
        
        logger.info(f"Dask dashboard: {client.dashboard_link}")
        
        # Create delayed tasks
        tasks = []
        for in_file, in_base, out_dir, out_base in file_list:
            task = dask.delayed(process_single_file)(in_file, in_base, out_dir, out_base, config)
            tasks.append(task)
        
        # Execute in parallel
        results = dask.compute(*tasks)
        
        client.close()
        cluster.close()
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    logger.info(f"Completed: {successful} successful, {failed} failed")
    
    # Log failures
    if failed > 0:
        logger.error("Failed files:")
        for r in results:
            if not r['success']:
                logger.error(f"  - {r['file']}: {r['error']}")
    
    return results

#-----------------------------------------------------------------------
def main():

    parser = argparse.ArgumentParser(description='Batch regrid LASSO subset files')
    parser.add_argument('--case-date', required=True, help='Case date (YYYYMMDD)')
    parser.add_argument('--start-hour', type=int, required=True, help='Start hour (0-23)')
    parser.add_argument('--end-hour', type=int, required=True, help='End hour (0-23)')
    parser.add_argument('--file-type', required=True, choices=['methamsl', 'cldhamsl'], 
                       help='File type to process')
    parser.add_argument('--domain', required=True, choices=['d3', 'd4'], 
                       help='Domain (d3 or d4)')
    parser.add_argument('--ensemble', required=True, help='Ensemble member (e.g., gefs18, eda05)')
    parser.add_argument('--regrid-ratio', type=int, default=None, 
                       help='Regridding ratio (default: 5 for d3, 25 for d4 to achieve 2.5km grid spacing)')
    parser.add_argument('--n-workers', type=int, default=1, help='Number of parallel workers (default: 1 for serial mode, >1 for parallel)')
    parser.add_argument('--input-dir', help='Input directory (if different from default)')
    parser.add_argument('--output-dir', help='Output directory (if different from default)')
    
    args = parser.parse_args()
    
    # Create log directory if needed
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up logging
    log_file = f'{log_dir}/regrid_{args.case_date}_{args.file_type}_{args.domain}_{args.start_hour:02d}-{args.end_hour:02d}.log'
    setup_logging(log_file)
    logger = logging.getLogger(__name__)
    
    # Set default regrid_ratio based on domain if not specified
    if args.regrid_ratio is None:
        if args.domain == 'd3':
            args.regrid_ratio = 5   # 500m -> 2500m (2.5 km)
        elif args.domain == 'd4':
            args.regrid_ratio = 25  # 100m -> 2500m (2.5 km)
        logger.info(f"Using default regrid_ratio={args.regrid_ratio} for domain {args.domain} (target: 2.5 km)")
    
    # Configuration
    case_date = args.case_date
    
    # Calculate start and end times
    # Each case runs from 12:00:00 to 23:59:59
    # The last hour (23) extends to next day's 00:00:00 to include the final file
    start_time = datetime.strptime(f"{case_date}{args.start_hour:02d}0000", '%Y%m%d%H%M%S')
    
    # If end_hour is 23 (last hour), extend to next day 00:00:00
    case_date_obj = datetime.strptime(case_date, '%Y%m%d')
    if args.end_hour == 23:
        # Extend to next day 00:00:00
        next_day = case_date_obj + timedelta(days=1)
        end_time = datetime.strptime(f"{next_day.strftime('%Y%m%d')}000000", '%Y%m%d%H%M%S')
        logger.info(f"Last hour of case - extending to next day: {end_time}")
    else:
        end_time = datetime.strptime(f"{case_date}{args.end_hour:02d}5959", '%Y%m%d%H%M%S')
    
    # Determine input directory
    if args.input_dir:
        in_dir = args.input_dir
        if not os.path.isdir(in_dir):
            logger.error(f"Specified input directory does not exist: {in_dir}")
            sys.exit(1)
    else:
        in_dir = find_input_directory(case_date, args.ensemble, args.domain)
        if in_dir is None:
            logger.error("Could not find input directory")
            sys.exit(1)
    
    # Determine output directory with same subdirectory structure as input
    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir_root = f"/gpfs/wolf2/arm/atm131/proj-shared/zfeng/cacti/les/regrid_subsets"
        out_dir = f"{out_dir_root}/{case_date}/{args.ensemble}/base/les/subset_{args.domain}/"
    
    # Create output directory if needed
    os.makedirs(out_dir, exist_ok=True)
    
    # File base name
    if args.file_type == 'methamsl':
        filebase = f"corlasso_methamsl_{case_date}00{args.ensemble}{args.domain}_base_M1.m1."
    else:
        filebase = f"corlasso_cldhamsl_{case_date}00{args.ensemble}{args.domain}_base_M1.m1."
    
    logger.info(f"Processing {args.file_type} files for case {case_date}, domain {args.domain}")
    logger.info(f"Input directory: {in_dir}")
    logger.info(f"Output directory: {out_dir}")
    logger.info(f"Time window: {start_time} to {end_time}")
    logger.info(f"Regrid ratio: {args.regrid_ratio}")
    
    # Get files in time window
    files = get_files_in_timewindow(in_dir, filebase, start_time, end_time)
    
    if not files:
        logger.warning("No files found!")
        return
    
    # Prepare file list for processing
    file_list = [(f, filebase, out_dir, filebase) for f in files]
    
    # Config
    config = {
        'time_dimname': 'Time',
        'x_coordname': 'XLONG',
        'y_coordname': 'XLAT',
        'z_coordname': 'HAMSL',
        'x_dimname': 'west_east',
        'y_dimname': 'south_north',
        'z_dimname': 'HAMSL',
        'regrid_ratio': args.regrid_ratio,
    }
    
    # Process batch
    results = process_batch_files(file_list, config, n_workers=args.n_workers)
    
    logger.info("Batch processing complete!")
    
    # Return exit code based on failures
    failed_count = sum(1 for r in results if not r['success'])
    if failed_count > 0:
        logger.warning(f"{failed_count} files failed to process")
        sys.exit(1)
    else:
        logger.info("All files processed successfully")
        sys.exit(0)

#-----------------------------------------------------------------------
if __name__ == "__main__":
    main()
