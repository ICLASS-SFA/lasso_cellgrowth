#!/bin/bash
#SBATCH -A atm131
#SBATCH -J STARTDATEENSMEMBER
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH -p batch_all
#SBATCH --exclusive
#SBATCH --output=LOG_FILE
#SBATCH --mail-type=END
#SBATCH --mail-user=zhe.feng@pnnl.gov

# source /ccsopen/home/zhe1feng1/.bashrc
# module load python
source activate /ccsopen/home/zhe1feng1/anaconda3/envs/pyflex26.4

date

# Run Python
python FLEXTRKR_DIR/run_celltracking_lasso.py CONFIG_FILE

date