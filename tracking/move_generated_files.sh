#!/bin/bash
# Script to move existing generated config and slurm files to the new organized structure
# Run this once to clean up the tracking directory

TRACKING_DIR="/ccsopen/home/zhe1feng1/program/lasso/cellgrowth/tracking"
OUTPUT_DIR="/ccsopen/home/zhe1feng1/program/lasso/cellgrowth/slurm"
LOG_DIR="${OUTPUT_DIR}/log"

# Create output directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

echo "Moving generated config and slurm files to ${OUTPUT_DIR}"
echo "Moving log files to ${LOG_DIR}"
echo ""

# Move config files (but NOT templates)
echo "Moving config files..."
mv ${TRACKING_DIR}/config_lasso_wrf*.yml ${OUTPUT_DIR}/ 2>/dev/null
mv ${TRACKING_DIR}/config_csapr*.yml ${OUTPUT_DIR}/ 2>/dev/null
# Move templates back to tracking directory
mv ${OUTPUT_DIR}/*_template.yml ${TRACKING_DIR}/ 2>/dev/null

# Move slurm files (but NOT templates)
echo "Moving slurm files..."
mv ${TRACKING_DIR}/slurm_lasso_wrf*_20*.sh ${OUTPUT_DIR}/ 2>/dev/null
mv ${TRACKING_DIR}/slurm_csapr*_20*.sh ${OUTPUT_DIR}/ 2>/dev/null

# Move log files
echo "Moving log files..."
mv ${TRACKING_DIR}/log_*.log ${LOG_DIR}/ 2>/dev/null

echo ""
echo "Done!"
echo ""
echo "Template files remain in: ${TRACKING_DIR}"
echo "Generated config/slurm files moved to: ${OUTPUT_DIR}"
echo "Log files moved to: ${LOG_DIR}"
echo ""
echo "Summary of moved files:"
echo "  Config files: $(ls -1 ${OUTPUT_DIR}/config_*.yml 2>/dev/null | wc -l)"
echo "  Slurm files: $(ls -1 ${OUTPUT_DIR}/slurm_*.sh 2>/dev/null | wc -l)"
echo "  Log files: $(ls -1 ${LOG_DIR}/log_*.log 2>/dev/null | wc -l)"
