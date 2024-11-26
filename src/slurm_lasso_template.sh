#!/bin/bash
#SBATCH -A atm123
#SBATCH -J STARTDATEENSMEMBER
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH -p batch_short
##SBATCH -p batch_high_memory
#SBATCH --exclusive
#SBATCH --output=log_CONFIG_NAME.log
#SBATCH --mail-type=END
#SBATCH --mail-user=zhe.feng@pnnl.gov

source /ccsopen/home/zhe1feng1/.bashrc
# module load python
conda activate /ccsopen/home/zhe1feng1/anaconda3/envs/py310

date
export OMP_NUM_THREADS=128
export NUMEXPR_MAX_THREADS=128

# Clean up /tmp space
rm -rf /tmp/dask-worker-space

# Run Python
cd /ccsopen/home/zhe1feng1/program/lasso/cellgrowth/src
# Calculate W statistics
# python calc_draft_stats_in_celltracks.py CONFIG_NAME.yml

# # Extract 3D preCI environments (wallclock: 1-2 h)
# # For 100m runs, use: #SBATCH -p batch_high_memory, set n_workers=8 in config
# # For 100m runs, use: #SBATCH -p batch_short, set n_workers=8 in config
# # For 500m runs, use: #SBATCH -p batch_high_memory, set n_workers=8 in config
# # For 500m runs, use: #SBATCH -p batch_short, set n_workers=12 in config
# python extract_cell_env3d_preCI_met.py CONFIG_NAME.yml

# # Calculate 2D preCI environments
# # For 100m runs, use: #SBATCH -p batch_high_memory, set n_workers=16 in config
# python calc_cell_env2d_from_3d.py CONFIG_NAME.yml

# Extract 1D preCI evironments at 2 locations
# python extract_cell_env1d_2location_preCI_met.py CONFIG_NAME.yml

# # Calculate 1D preCI environments
# python calc_cell_center_env.py CONFIG_NAME.yml
python calc_cell_avg_env.py CONFIG_NAME.yml

# Combine environmental variables with profiles at 2 locations
# python combine_cell_env_2location.py CONFIG_NAME.yml

# Make largest updraft masks and stats
# python make_largest_updraft_stats.py CONFIG_NAME.yml

# Calculate satellite-like statistics
# python calc_sat_stats_in_celltracks.py CONFIG_NAME.yml

date