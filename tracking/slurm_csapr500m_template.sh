#!/bin/bash
#SBATCH -A atm123
#SBATCH -J STARTDATE
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH -p batch_short
#SBATCH --exclusive
#SBATCH --output=log_CONFIG_NAME.log
#SBATCH --mail-type=END
#SBATCH --mail-user=zhe.feng@pnnl.gov

# source /ccsopen/home/zhe1feng1/.bashrc
# module load python
# conda activate /ccsopen/home/zhe1feng1/anaconda3/envs/flextrkr

date

# Run Python
python FLEXTRKR_DIR/run_celltracking.py CONFIG_FILE

date