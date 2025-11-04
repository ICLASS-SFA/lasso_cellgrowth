#!/bin/bash
#SBATCH --job-name=regrid_lasso
#SBATCH --account=atm131
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=01:00:00
#SBATCH --output=logs/regrid_%A_%a.out
#SBATCH --error=logs/regrid_%A_%a.err
#SBATCH --array=12-23  # 12 hours (12:00-23:59, plus next day 00:00:00 for hour 23)

# Description:
# This script processes LASSO subset files in 1-hour chunks using job arrays.
# Each case runs from 12:00:00 to 23:59:59 (12 hours total).
# Each array task processes files for one hour of the simulation.
# Default regrid_ratio: d3=5 (500m->2.5km), d4=25 (100m->2.5km)
# 
# Usage:
#   sbatch --export=CASE_DATE=20190123,FILE_TYPE=methamsl,DOMAIN=d4,ENSEMBLE=gefs18 slurm_regrid.sh
#   # Optional: Override regrid_ratio
#   sbatch --export=CASE_DATE=20190123,FILE_TYPE=methamsl,DOMAIN=d4,ENSEMBLE=gefs18,REGRID_RATIO=5 slurm_regrid.sh

# Create log directory if needed
mkdir -p logs

# Load modules (adjust for your system)
# module load python/3.9
# module load conda  # If using conda

# Activate your environment
source activate /ccsopen/home/zhe1feng1/anaconda3/envs/flextrkr
# Or use: conda activate your_env_name

# Set default values if not provided
CASE_DATE=${CASE_DATE:-"20190123"}
FILE_TYPE=${FILE_TYPE:-"methamsl"}
DOMAIN=${DOMAIN:-"d4"}
ENSEMBLE=${ENSEMBLE}  # Required - no default
REGRID_RATIO=${REGRID_RATIO:-""}  # Empty means auto-detect based on domain
N_WORKERS=${N_WORKERS:-6}

# Check required parameters
if [ -z "${ENSEMBLE}" ]; then
    echo "ERROR: ENSEMBLE is required but not set"
    echo "Usage: sbatch --export=CASE_DATE=...,FILE_TYPE=...,DOMAIN=...,ENSEMBLE=... slurm_regrid.sh"
    exit 1
fi

# Calculate start and end hours based on array task ID
START_HOUR=${SLURM_ARRAY_TASK_ID}
END_HOUR=$((START_HOUR + 1))

# Print job information
echo "=================================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Case Date: ${CASE_DATE}"
echo "File Type: ${FILE_TYPE}"
echo "Domain: ${DOMAIN}"
echo "Ensemble: ${ENSEMBLE}"
echo "Time Window: ${START_HOUR}:00 - ${END_HOUR}:00"
echo "Regrid Ratio: ${REGRID_RATIO}"
echo "Number of Workers: ${N_WORKERS}"
echo "=================================================="

# Change to script directory
cd /ccsopen/home/zhe1feng1/program/lasso/cellgrowth/regrid

# Build command
CMD="python regrid_lasso_batch.py \
    --case-date ${CASE_DATE} \
    --start-hour ${START_HOUR} \
    --end-hour ${END_HOUR} \
    --file-type ${FILE_TYPE} \
    --domain ${DOMAIN} \
    --ensemble ${ENSEMBLE} \
    --n-workers ${N_WORKERS}"

# Add regrid_ratio if specified (otherwise use domain default)
if [ -n "${REGRID_RATIO}" ]; then
    CMD="${CMD} --regrid-ratio ${REGRID_RATIO}"
fi

# Run processing
eval ${CMD}

# Check exit status
if [ $? -eq 0 ]; then
    echo "Successfully completed processing for hour ${START_HOUR}"
else
    echo "ERROR: Processing failed for hour ${START_HOUR}"
    exit 1
fi
