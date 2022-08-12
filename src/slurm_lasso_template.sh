#!/bin/bash
#SBATCH -A atm123
#SBATCH -J STARTDATEENSMEMBER
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH -p batch_all
##SBATCH -p batch_high_memory
#SBATCH --exclusive
#SBATCH --output=log_CONFIG_NAME.log
#SBATCH --mail-type=END
#SBATCH --mail-user=zhe.feng@pnnl.gov

# source /ccsopen/home/zhe1feng1/.bashrc
# module load python
# conda activate /ccsopen/home/zhe1feng1/anaconda3/envs/flextrkr

date
export OMP_NUM_THREADS=128
export NUMEXPR_MAX_THREADS=128

# Run Python
cd /ccsopen/home/zhe1feng1/program/lasso/cellgrowth/src
# Calculate W statistics
# python calc_w_corestats_to_celltracks.py CONFIG_NAME.yml

# Extract 3D preCI environments
python extract_cell_env_3d_preCI_bytracks.py CONFIG_NAME.yml

date