#!/usr/bin/env python3
"""
Generate job list for processing tracks in batches using SLURM job arrays.
This script reads the track statistics file to determine the total number of tracks
and generates a list of commands to process tracks in specified batch sizes.
"""
import sys
import os
import argparse
import xarray as xr
import numpy as np
import yaml

def main():
    parser = argparse.ArgumentParser(description='Generate job list for batch track processing')
    parser.add_argument('config_file', help='Configuration YAML file')
    parser.add_argument('--batch_size', type=int, default=100, 
                       help='Number of tracks per batch (default: 100)')
    parser.add_argument('--output_dir', default='job_scripts',
                       help='Directory to save job scripts (default: job_scripts)')
    parser.add_argument('--script_name', default='calc_draft_stats_in_celltracks_bytracks.py',
                       help='Python script name to run (default: calc_draft_stats_in_celltracks_bytracks.py)')
    parser.add_argument('--slurm_template', default=None,
                       help='SLURM template file (optional)')
    parser.add_argument('--digits', type=int, default=4,
                       help='Number of digits for track start padding in filenames (default: 4)')
    
    args = parser.parse_args()
    
    # Read configuration
    with open(args.config_file, 'r') as f:
        config = yaml.full_load(f)
    
    # Get paths from config
    stats_path = config['stats_path']
    startdate = config['startdate']
    enddate = config['enddate']
    
    # Track statistics file
    stats_filebase = 'trackstats_'
    trackstats_file = f'{stats_path}{stats_filebase}{startdate}_{enddate}.nc'
    
    print(f"Reading track statistics from: {trackstats_file}")
    
    # Read track statistics file to get total number of tracks
    try:
        ds = xr.open_dataset(trackstats_file, decode_times=False)
        ntracks_total = ds.sizes['tracks']
        ds.close()
        print(f"Total number of tracks: {ntracks_total}")
    except Exception as e:
        print(f"ERROR: Could not read track statistics file: {e}")
        sys.exit(1)
    
    # Calculate batch ranges
    batch_size = args.batch_size
    batches = []
    
    for start_idx in range(0, ntracks_total, batch_size):
        end_idx = min(start_idx + batch_size - 1, ntracks_total - 1)
        batches.append((start_idx, end_idx))
    
    print(f"Will create {len(batches)} batches with batch size {batch_size}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate job commands
    job_commands_file = f"{args.output_dir}/job_commands_{startdate}_{enddate}.txt"
    job_list_file = f"{args.output_dir}/job_list_{startdate}_{enddate}.sh"
    slurm_array_file = f"{args.output_dir}/slurm_array_{startdate}_{enddate}.sh"
    
    # Write job commands (one per line for SLURM array)
    with open(job_commands_file, 'w') as f:
        for i, (start_idx, end_idx) in enumerate(batches):
            cmd = f"python {args.script_name} {args.config_file} {start_idx} {end_idx} {args.digits}"
            f.write(f"{cmd}\n")
    
    print(f"Job commands written to: {job_commands_file}")
    
    # Write individual job list (for manual execution)
    with open(job_list_file, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Job list for processing {ntracks_total} tracks in batches of {batch_size}\n")
        f.write(f"# Generated for {startdate}_{enddate}\n\n")
        
        for i, (start_idx, end_idx) in enumerate(batches):
            f.write(f"# Batch {i+1}/{len(batches)}: tracks {start_idx}-{end_idx}\n")
            f.write(f"python {args.script_name} {args.config_file} {start_idx} {end_idx} {args.digits}\n\n")
    
    print(f"Individual job list written to: {job_list_file}")
    
    # Generate SLURM job array script
    with open(slurm_array_file, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH -A atm123\n")
        f.write(f"#SBATCH -J tracks_{startdate}_{enddate}\n")
        f.write("#SBATCH --time=8:00:00\n")
        f.write("#SBATCH --nodes=1\n")
        f.write("#SBATCH --cpus-per-task=32\n")  # Reduced for batch processing
        # f.write("#SBATCH -p batch_all\n")
        f.write("#SBATCH -p batch_high_memory\n")
        f.write("#SBATCH --exclusive\n")
        f.write(f"#SBATCH --array=1-{len(batches)}\n")
        f.write(f"#SBATCH --output=logs/job_%A_%a_{startdate}_{enddate}.log\n")
        f.write("#SBATCH --mail-type=END,FAIL\n")
        f.write("#SBATCH --mail-user=zhe.feng@pnnl.gov\n\n")
        
        f.write("# Load environment\n")
        f.write("source /ccsopen/home/zhe1feng1/.bashrc\n")
        f.write("conda activate /ccsopen/home/zhe1feng1/anaconda3/envs/flextrkr\n\n")
        
        f.write("# Set environment variables\n")
        f.write("export OMP_NUM_THREADS=32\n")
        f.write("export NUMEXPR_MAX_THREADS=32\n\n")
        
        f.write("# Clean up /tmp space\n")
        f.write("rm -rf /tmp/dask-worker-space\n\n")
        
        f.write("# Create logs directory\n")
        f.write("mkdir -p logs\n\n")
        
        f.write("# Change to source directory\n")
        f.write("cd /ccsopen/home/zhe1feng1/program/lasso/cellgrowth/src\n\n")
        
        f.write("# Get the command for this array task\n")
        f.write(f"CMD=$(sed -n \"${{SLURM_ARRAY_TASK_ID}}p\" {job_commands_file})\n\n")
        
        f.write("# Print job info\n")
        f.write("date\n")
        f.write("echo \"Job ID: $SLURM_JOB_ID\"\n")
        f.write("echo \"Array Task ID: $SLURM_ARRAY_TASK_ID\"\n")
        f.write("echo \"Running: $CMD\"\n")
        f.write("echo \"Host: $(hostname)\"\n\n")
        
        f.write("# Execute the command\n")
        f.write("eval $CMD\n\n")
        
        f.write("date\n")
        f.write("echo \"Task $SLURM_ARRAY_TASK_ID completed\"\n")
    
    print(f"SLURM job array script written to: {slurm_array_file}")
    
    # Make scripts executable
    os.chmod(job_list_file, 0o755)
    os.chmod(slurm_array_file, 0o755)
    
    # Print summary
    print(f"\n=== SUMMARY ===")
    print(f"Total tracks: {ntracks_total}")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches: {len(batches)}")
    print(f"Track ranges:")
    for i, (start_idx, end_idx) in enumerate(batches[:5]):  # Show first 5
        print(f"  Batch {i+1}: tracks {start_idx}-{end_idx}")
    if len(batches) > 5:
        print(f"  ... ({len(batches)-5} more batches)")
        print(f"  Batch {len(batches)}: tracks {batches[-1][0]}-{batches[-1][1]}")
    
    print(f"\n=== TO RUN JOBS ===")
    print(f"Submit SLURM array: sbatch {slurm_array_file}")
    print(f"Or run individual jobs from: {job_list_file}")
    
    # Generate concatenation script
    concat_script = f"{args.output_dir}/concatenate_results_{startdate}_{enddate}.py"
    with open(concat_script, 'w') as f:
        f.write("#!/usr/bin/env python3\n")
        f.write('"""\n')
        f.write("Concatenate individual track batch results into a single file.\n")
        f.write('"""\n')
        f.write("import xarray as xr\n")
        f.write("import glob\n")
        f.write("import os\n\n")
        
        f.write("# Configuration\n")
        f.write(f"startdate = '{startdate}'\n")
        f.write(f"enddate = '{enddate}'\n")
        f.write(f"output_path = '{config['output_path']}'\n")
        f.write("output_basename = 'stats_3d_w_fixshell_'\n\n")
        
        f.write("# Find all batch result files\n")
        f.write(f"pattern = f'{{output_path}}{{output_basename}}{{startdate}}_{{enddate}}_t*.nc'\n")
        f.write("batch_files = sorted(glob.glob(pattern))\n")
        f.write("print(f'Found {len(batch_files)} batch files to concatenate')\n\n")
        
        f.write("if len(batch_files) == 0:\n")
        f.write("    print('No batch files found!')\n")
        f.write("    exit(1)\n\n")
        
        f.write("# Load and concatenate\n")
        f.write("print('Loading batch files...')\n")
        f.write("datasets = []\n")
        f.write("for f in batch_files:\n")
        f.write("    print(f'  Loading {os.path.basename(f)}')\n")
        f.write("    ds = xr.open_dataset(f)\n")
        f.write("    datasets.append(ds)\n\n")
        
        f.write("print('Concatenating along tracks dimension...')\n")
        f.write("combined = xr.concat(datasets, dim='tracks')\n\n")
        
        f.write("# Save combined file\n")
        f.write("output_file = f'{output_path}{output_basename}{startdate}_{enddate}.nc'\n")
        f.write("print(f'Saving combined file: {output_file}')\n")
        f.write("combined.to_netcdf(output_file)\n")
        f.write("combined.close()\n\n")
        
        f.write("# Close individual datasets\n")
        f.write("for ds in datasets:\n")
        f.write("    ds.close()\n\n")
        
        f.write("print('Concatenation completed successfully!')\n")
        f.write(f"print(f'Final file contains {{combined.sizes[\"tracks\"]}} tracks')\n")
    
    os.chmod(concat_script, 0o755)
    print(f"Concatenation script written to: {concat_script}")

if __name__ == '__main__':
    main()
