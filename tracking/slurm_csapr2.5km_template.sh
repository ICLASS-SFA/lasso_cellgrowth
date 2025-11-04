#!/bin/bash
#SBATCH -A atm131
#SBATCH -J STARTDATE
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH -p batch_short
#SBATCH --exclusive
#SBATCH --output=LOG_FILE
#SBATCH --mail-type=END
#SBATCH --mail-user=zhe.feng@pnnl.gov

source /ccsopen/home/zhe1feng1/.bashrc
# module load python
conda activate /ccsopen/home/zhe1feng1/anaconda3/envs/flextrkr

date

# Run Python
cd /ccsopen/home/zhe1feng1/program/PyFLEXTRKR-dev
python ./runscripts/run_celltracking_lasso.py CONFIG_FILE

date