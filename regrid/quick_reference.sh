#!/bin/bash
#
# Quick Reference Guide for LASSO Regridding
# Copy and modify these commands for your use case
#

# ========================================
# 1. TEST SINGLE FILE
# ========================================
# Edit test_regrid_single.py first, then run:
python test_regrid_single.py


# ========================================
# 2. INTERACTIVE BATCH PROCESSING
# ========================================
# Process files for one hour interactively (useful for testing):
python regrid_lasso_batch.py \
    --case-date 20190123 \
    --start-hour 0 \
    --end-hour 1 \
    --file-type methamsl \
    --subset d4 \
    --n-workers 6 \
    --regrid-ratio 25


# ========================================
# 3. SUBMIT SINGLE JOB ARRAY
# ========================================
# Process one case, one file type (creates 12 tasks, one per hour):
sbatch --export=CASE_DATE=20190123,FILE_TYPE=methamsl,SUBSET=d4,REGRID_RATIO=25 slurm_regrid.sh


# ========================================
# 4. SUBMIT ALL CASES
# ========================================
# First, edit submit_all_cases.sh to add all your case dates:
# CASES=(
#     "20190123"
#     "20190124"
#     ...
# )
# Then run:
bash submit_all_cases.sh


# ========================================
# 5. MONITOR JOBS
# ========================================
# Check job status
squeue -u $USER

# Watch job status (updates every 2 seconds)
watch -n 2 'squeue -u $USER'

# Check specific job array
squeue -j <JOB_ID>

# Check job efficiency
seff <JOB_ID>

# Check completed jobs
sacct -u $USER --starttime $(date -d '1 day ago' +%Y-%m-%d)

# Check detailed job info
scontrol show job <JOB_ID>


# ========================================
# 6. CHECK LOGS
# ========================================
# View most recent log
tail -f logs/regrid_*.out

# Check for errors
grep -i error logs/*.err
grep -i failed logs/*.out

# Count successful completions
grep -c "Successfully wrote" logs/*.out


# ========================================
# 7. CANCEL JOBS
# ========================================
# Cancel specific job
scancel <JOB_ID>

# Cancel all your jobs
scancel -u $USER

# Cancel specific job array tasks
scancel <JOB_ID>_[0-5]  # Cancel tasks 0-5


# ========================================
# 8. CHECK OUTPUT FILES
# ========================================
# Count output files
ls -1 /gpfs/wolf2/arm/atm131/proj-shared/zfeng/cacti/les/regrid_subsets/*.nc | wc -l

# Check file sizes
du -sh /gpfs/wolf2/arm/atm131/proj-shared/zfeng/cacti/les/regrid_subsets/

# List files for specific case
ls -lh /gpfs/wolf2/arm/atm131/proj-shared/zfeng/cacti/les/regrid_subsets/*20190123* | head

# Quick check of output file
ncdump -h /gpfs/wolf2/arm/atm131/proj-shared/zfeng/cacti/les/regrid_subsets/<filename>.nc | less


# ========================================
# 9. RESUBMIT FAILED TASKS
# ========================================
# Check which tasks failed
sacct -j <JOB_ID> --format=JobID,State,ExitCode | grep FAILED

# Resubmit specific hours (modify SLURM script array range):
# Edit slurm_regrid.sh: #SBATCH --array=3,7,10
sbatch --export=CASE_DATE=20190123,FILE_TYPE=methamsl,SUBSET=d4 slurm_regrid.sh


# ========================================
# 10. CUSTOM SETTINGS
# ========================================
# Use different regrid ratio
sbatch --export=CASE_DATE=20190123,FILE_TYPE=methamsl,SUBSET=d4,REGRID_RATIO=10 slurm_regrid.sh

# Use more workers (high-memory node)
sbatch --mem=512GB --export=CASE_DATE=20190123,FILE_TYPE=methamsl,SUBSET=d4,N_WORKERS=10 slurm_regrid.sh

# Process 2-hour chunks (edit slurm_regrid.sh first)
# Change: END_HOUR=$((START_HOUR * 2 + 2))
# Change: #SBATCH --array=0-5  # 6 chunks instead of 12


# ========================================
# 11. DEBUGGING
# ========================================
# Run interactively on compute node
salloc --nodes=1 --ntasks=1 --cpus-per-task=128 --mem=256GB --time=01:00:00

# Once allocated, run:
module load python/3.9
source /path/to/your/venv/bin/activate
cd /ccsopen/home/zhe1feng1/program/lasso/cellgrowth/regrid
python test_regrid_single.py

# Access Dask dashboard (from your local machine):
# ssh -L 8787:compute-node:8787 user@login-node
# Then open browser: http://localhost:8787


# ========================================
# 12. USEFUL CALCULATIONS
# ========================================
# Estimate total processing time for all cases
# 18 cases × 2 file types × 2 subsets × 144 files = 10,368 files
# At ~5 min per file with 6 workers = ~144 hours sequential
# With 72 parallel job arrays × 12 tasks = ~2 hours wall time

# Estimate total data size
# Input: 10,368 files × 9.5 GB = ~98 TB
# Output (ratio=25): 10,368 files × 15 MB = ~155 GB

# Memory per worker
# 256 GB / 6 workers = ~42 GB per worker (safe)
# 512 GB / 10 workers = ~51 GB per worker (high-mem)
