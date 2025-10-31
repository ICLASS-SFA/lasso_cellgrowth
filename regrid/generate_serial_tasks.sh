#!/bin/bash
#
# Generate a list of serial processing commands for all cases
# This creates a text file with all commands that can be run sequentially
# on a login node when SLURM queue is full.
#
# Usage: bash generate_serial_tasks.sh

# Array of case dates and ensemble members
# (matching the combinations from make_calc_stats_slurm_scripts_les.sh)
CASE_DATES=(
    "20181129"
    # "20181129" "20181129" "20181129" "20181129"
    # "20181204" "20181204" 
    # "20181205" 
    # "20181219" 
    # "20190122" "20190122"
    # "20190123" "20190123"
    # "20190125" "20190125"
    # "20190129" "20190129"
    # "20190208" "20190208"
)

ENSEMBLES=(
    "gefs00"
    # "gefs00" "gefs03" "gefs09" "gefs18"
    # "gefs18" "gefs19" 
    # "gefs01" 
    # "eda09" 
    # "gefs01" "gefs18"
    # "eda05" "gefs18"
    # "eda07" "gefs11"
    # "eda09" "gefs11"
    # "eda03" "eda08"
)

# File types to process
FILE_TYPES=("methamsl" "cldhamsl")

# Domains
DOMAINS=("d3" "d4")

# Other parameters (regrid_ratio is auto-detected: d3=5, d4=25)
# Use n_workers=1 for serial processing on login node
N_WORKERS=1

# Output file
OUTPUT_FILE="serial_tasks_test.txt"
WORK_DIR="/ccsopen/home/zhe1feng1/program/lasso/cellgrowth/regrid"

# Create logs directory
mkdir -p logs

# Clear output file
> ${OUTPUT_FILE}

echo "=================================================="
echo "Generating serial task list"
echo "Total case/ensemble combinations: ${#CASE_DATES[@]}"
echo "File types: ${FILE_TYPES[@]}"
echo "Domains: ${DOMAINS[@]}"
echo "Output file: ${OUTPUT_FILE}"
echo "=================================================="

# Counter for tasks
TOTAL_TASKS=0

# Loop over all case/ensemble combinations
for ((i = 0; i < ${#CASE_DATES[@]}; ++i)); do
    CASE=${CASE_DATES[$i]}
    ENSEMBLE=${ENSEMBLES[$i]}
    
    for FTYPE in "${FILE_TYPES[@]}"; do
        for DOMAIN in "${DOMAINS[@]}"; do
            # Each case has 12 hours (12-23)
            for HOUR in {12..23}; do
                echo "Adding: Case=${CASE}, Ensemble=${ENSEMBLE}, FileType=${FTYPE}, Domain=${DOMAIN}, Hour=${HOUR}"
                
                # Generate command
                CMD="cd ${WORK_DIR} && python regrid_lasso_batch.py --case-date ${CASE} --ensemble ${ENSEMBLE} --start-hour ${HOUR} --end-hour ${HOUR} --file-type ${FTYPE} --domain ${DOMAIN} --n-workers ${N_WORKERS}"
                
                # Add to file
                echo "${CMD}" >> ${OUTPUT_FILE}
                
                ((TOTAL_TASKS++))
            done
        done
    done
done

echo ""
echo "=================================================="
echo "Task list generation complete!"
echo "Total tasks: ${TOTAL_TASKS}"
echo "Output file: ${OUTPUT_FILE}"
echo ""
echo "To run all tasks in serial (in background with nohup):"
echo "  nohup bash run_serial_tasks.sh > serial_processing.log 2>&1 &"
echo ""
echo "To monitor progress:"
echo "  tail -f serial_processing.log"
echo ""
echo "To check if still running:"
echo "  ps aux | grep run_serial_tasks"
echo "=================================================="

# Also create the runner script
RUNNER_SCRIPT="run_serial_tasks.sh"
cat > ${RUNNER_SCRIPT} << 'EOF'
#!/bin/bash
#
# Run all serial tasks from the task list
# This script reads commands from serial_tasks.txt and executes them sequentially
#
# Usage: 
#   nohup bash run_serial_tasks.sh > serial_processing.log 2>&1 &

TASK_FILE="serial_tasks.txt"
WORK_DIR="/ccsopen/home/zhe1feng1/program/lasso/cellgrowth/regrid"

# Check if task file exists
if [ ! -f "${TASK_FILE}" ]; then
    echo "ERROR: Task file ${TASK_FILE} not found!"
    echo "Run generate_serial_tasks.sh first"
    exit 1
fi

# Load environment
source activate /ccsopen/home/zhe1feng1/anaconda3/envs/flextrkr

# Count total tasks
TOTAL_TASKS=$(wc -l < ${TASK_FILE})
CURRENT=0

echo "=================================================="
echo "Starting serial task execution"
echo "Total tasks: ${TOTAL_TASKS}"
echo "Started at: $(date)"
echo "=================================================="
echo ""

# Read and execute each command
while IFS= read -r CMD; do
    ((CURRENT++))
    echo "=================================================="
    echo "Task ${CURRENT}/${TOTAL_TASKS}"
    echo "Command: ${CMD}"
    echo "Started: $(date)"
    echo "=================================================="
    
    # Execute command
    eval ${CMD}
    EXIT_CODE=$?
    
    if [ ${EXIT_CODE} -eq 0 ]; then
        echo "✓ Task ${CURRENT} completed successfully"
    else
        echo "✗ Task ${CURRENT} FAILED with exit code ${EXIT_CODE}"
    fi
    
    echo "Finished: $(date)"
    echo ""
    
done < ${TASK_FILE}

echo "=================================================="
echo "All tasks completed!"
echo "Finished at: $(date)"
echo "Total tasks: ${TOTAL_TASKS}"
echo "=================================================="
EOF

chmod +x ${RUNNER_SCRIPT}

echo ""
echo "Created runner script: ${RUNNER_SCRIPT}"
echo "This script will execute all tasks sequentially"
