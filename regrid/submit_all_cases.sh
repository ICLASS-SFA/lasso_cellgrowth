#!/bin/bash
#
# Master script to submit regridding jobs for all cases
# This script submits SLURM job arrays for each case and file type combination.
#
# Usage: bash submit_all_cases.sh

# Array of case dates and ensemble members
# (matching the combinations from make_calc_stats_slurm_scripts_les.sh)
CASE_DATES=(
    # "20181129" "20181129" "20181129" "20181129"
    "20181129" "20181129" "20181129"
    "20181204" "20181204" 
    "20181205" 
    "20181219" 
    "20190122" "20190122"
    "20190123" "20190123"
    "20190125" "20190125"
    "20190129" "20190129"
    "20190208" "20190208"
)

ENSEMBLES=(
    # "gefs00" "gefs03" "gefs09" "gefs18"
    "gefs03" "gefs09" "gefs18"
    "gefs18" "gefs19" 
    "gefs01" 
    "eda09" 
    "gefs01" "gefs18"
    "eda05" "gefs18"
    "eda07" "gefs11"
    "eda09" "gefs11"
    "eda03" "eda08"
)

# File types to process
FILE_TYPES=("methamsl" "cldhamsl")

# Domains
DOMAINS=("d3" "d4")

# Other parameters (regrid_ratio is auto-detected: d3=5, d4=25)
N_WORKERS=6

# Create logs directory
mkdir -p logs

echo "=================================================="
echo "Submitting regridding jobs for all cases"
echo "Total case/ensemble combinations: ${#CASE_DATES[@]}"
echo "File types: ${FILE_TYPES[@]}"
echo "Domains: ${DOMAINS[@]}"
echo "Regrid ratio: auto (d3=5, d4=25 for 2.5km target)"
echo "N_WORKERS: ${N_WORKERS}"
echo "=================================================="

# Counter for submitted jobs
TOTAL_JOBS=0

# Loop over all case/ensemble combinations
for ((i = 0; i < ${#CASE_DATES[@]}; ++i)); do
    CASE=${CASE_DATES[$i]}
    ENSEMBLE=${ENSEMBLES[$i]}
    
    for FTYPE in "${FILE_TYPES[@]}"; do
        for DOMAIN in "${DOMAINS[@]}"; do
            echo ""
            echo "Submitting: Case=${CASE}, Ensemble=${ENSEMBLE}, FileType=${FTYPE}, Domain=${DOMAIN}"
            
            # Submit job (regrid_ratio auto-detected based on domain)
            JOB_ID=$(sbatch --parsable \
                --export=CASE_DATE=${CASE},FILE_TYPE=${FTYPE},DOMAIN=${DOMAIN},ENSEMBLE=${ENSEMBLE},N_WORKERS=${N_WORKERS} \
                slurm_regrid.sh)
            
            if [ $? -eq 0 ]; then
                echo "  Submitted job ID: ${JOB_ID}"
                ((TOTAL_JOBS++))
            else
                echo "  ERROR: Failed to submit job"
            fi
            
            # Small delay to avoid overwhelming the scheduler
            sleep 0.5
        done
    done
done

echo ""
echo "=================================================="
echo "Finished submitting jobs"
echo "Total jobs submitted: ${TOTAL_JOBS}"
echo "Expected job arrays: ${TOTAL_JOBS}"
echo "Expected total tasks: $((TOTAL_JOBS * 12))"
echo ""
echo "To check job status:"
echo "  squeue -u \$USER"
echo ""
echo "To monitor a specific job array:"
echo "  squeue -j <JOB_ID>"
echo ""
echo "To cancel all jobs:"
echo "  scancel -u \$USER"
echo "=================================================="
