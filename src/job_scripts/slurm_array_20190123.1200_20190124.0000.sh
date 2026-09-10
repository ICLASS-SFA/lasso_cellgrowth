#!/bin/bash
#SBATCH -A atm123
#SBATCH -J tracks_20190123.1200_20190124.0000
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH -p batch_high_memory
#SBATCH --exclusive
#SBATCH --array=1-29
#SBATCH --output=logs/job_%A_%a_20190123.1200_20190124.0000.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhe.feng@pnnl.gov

# Load environment
source /ccsopen/home/zhe1feng1/.bashrc
conda activate /ccsopen/home/zhe1feng1/anaconda3/envs/flextrkr

# Set environment variables
export OMP_NUM_THREADS=32
export NUMEXPR_MAX_THREADS=32

# Clean up /tmp space
rm -rf /tmp/dask-worker-space

# Create logs directory
mkdir -p logs

# Change to source directory
cd /ccsopen/home/zhe1feng1/program/lasso/cellgrowth/src

# Get the command for this array task
CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" job_scripts/job_commands_20190123.1200_20190124.0000.txt)

# Print job info
date
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running: $CMD"
echo "Host: $(hostname)"

# Execute the command
eval $CMD

date
echo "Task $SLURM_ARRAY_TASK_ID completed"
