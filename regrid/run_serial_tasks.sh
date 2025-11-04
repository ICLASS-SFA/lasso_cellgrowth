#!/bin/bash
#
# Run all serial tasks from the task list
# This script reads commands from serial_tasks.txt and executes them sequentially
#
# Usage: 
#   nohup bash run_serial_tasks.sh > serial_processing.log 2>&1 &

TASK_FILE="serial_tasks_test.txt"
# TASK_FILE="serial_tasks.txt"
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
