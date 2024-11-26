#!/bin/bash
#SBATCH -A atm123
#SBATCH -J cp.d2_15min
#SBATCH -t 06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH -p batch_short
#SBATCH --exclusive
#SBATCH --output=log_coldpool_wrf2.5km_%A_%a.log
#SBATCH --mail-type=END
#SBATCH --mail-user=zhe.feng@pnnl.gov
#SBATCH --array=1-18

date
# Activate Python environment
# source /ccsopen/home/zhe1feng1/.bashrc
# module load python
# conda init bash
# conda activate /ccsopen/home/zhe1feng1/anaconda3/envs/flextrkr

# Takes a specified line ($SLURM_ARRAY_TASK_ID) from the task file
LINE=$(sed -n "$SLURM_ARRAY_TASK_ID"p tasks_coldpool_animation_wrf2.5km_base.txt)
echo $LINE
# Run the line as a command
$LINE

date
