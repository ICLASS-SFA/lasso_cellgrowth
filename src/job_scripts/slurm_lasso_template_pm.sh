#!/bin/bash
#SBATCH -A m1657
#SBATCH -J STARTDATEENSMEMBER
#SBATCH --time=00:05:00
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --exclusive
#SBATCH --output=log_CONFIG_NAME.log
#SBATCH --mail-type=END
#SBATCH --mail-user=zhe.feng@pnnl.gov

# source /ccsopen/home/zhe1feng1/.bashrc
# module load python
# conda activate /ccsopen/home/zhe1feng1/anaconda3/envs/flextrkr

date
source activate /global/common/software/m1867/python/py310
# export OMP_NUM_THREADS=128
export NUMEXPR_MAX_THREADS=128

# Run Python
cd /global/homes/f/feng045/program/iclass/cacti/lasso/lasso_cellgrowth/src
# Calculate W statistics
# python calc_draft_stats_in_celltracks.py CONFIG_NAME.yml

# # Extract 3D preCI environments
# # For 100m runs, use: #SBATCH -p batch_high_memory, set n_workers=8 in config
# # For 500m runs, use: #SBATCH -p batch_high_memory, set n_workers=8 in config
# python extract_cell_env3d_preCI_met.py CONFIG_NAME.yml

# # Calculate 2D preCI environments
# # For 100m runs, use: #SBATCH -p batch_high_memory, set n_workers=16 in config
python calc_cell_env2d_from_3d.py CONFIG_NAME.yml

# # Calculate 1D preCI environments
# python calc_cell_center_env.py CONFIG_NAME.yml

date