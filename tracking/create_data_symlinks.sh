#!/bin/bash
# Script to create symbolic links for LASSO LES data
# This allows the tracking code to use a single consistent path regardless of where
# the actual data is stored (staged_runs or staged_runs_2)

# Base paths
SOURCE_BASE1="/gpfs/wolf2/arm/cli120/world-shared/d3m088/cacti/staged_runs"
SOURCE_BASE2="/gpfs/wolf2/arm/cli120/world-shared/d3m088/cacti/staged_runs_2"
LINK_BASE="/gpfs/wolf2/arm/atm131/proj-shared/zfeng/cacti/les/input_links"

# Configuration
CONFIG="base"

# Domain options: d3, d4
DOMAINS=("d3" "d4")

# Case dates and ensemble members
CASE_DATES=(
    "20181129" "20181129" "20181129" "20181129"
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
    "gefs00" "gefs03" "gefs09" "gefs18" 
    "gefs18" "gefs19" 
    "gefs01" 
    "eda09" 
    "gefs01" "gefs18"
    "eda05" "gefs18"
    "eda07" "gefs11"
    "eda09" "gefs11"
    "eda03" "eda08"
)

echo "=================================================="
echo "Creating symbolic links for LASSO LES input data"
echo "Link base directory: ${LINK_BASE}"
echo "=================================================="
echo ""

# Create base link directory
mkdir -p "${LINK_BASE}"

# Counters for summary
total_created=0
total_notfound=0

# Loop through all case/ensemble combinations
for ((i = 0; i < ${#CASE_DATES[@]}; ++i)); do
    CASE=${CASE_DATES[$i]}
    ENS=${ENSEMBLES[$i]}
    
    echo "Processing: ${CASE} / ${ENS}"
    
    # Process each domain
    for DOMAIN in "${DOMAINS[@]}"; do
        # Try both source locations and use whichever exists
        SOURCE_PATH1="${SOURCE_BASE1}/${CASE}/${ENS}/${CONFIG}/les/subset_${DOMAIN}"
        SOURCE_PATH2="${SOURCE_BASE2}/${CASE}/${ENS}/${CONFIG}/les/subset_${DOMAIN}"
        LINK_PATH="${LINK_BASE}/${CASE}/${ENS}/${CONFIG}/les/subset_${DOMAIN}"
        
        # Determine which source exists
        SOURCE_PATH=""
        SOURCE_LABEL=""
        if [[ -d "${SOURCE_PATH1}" ]]; then
            SOURCE_PATH="${SOURCE_PATH1}"
            SOURCE_LABEL="staged_runs"
        elif [[ -d "${SOURCE_PATH2}" ]]; then
            SOURCE_PATH="${SOURCE_PATH2}"
            SOURCE_LABEL="staged_runs_2"
        fi
        
        # Create link if source was found
        if [[ -n "${SOURCE_PATH}" ]]; then
            # Create parent directories for the link
            mkdir -p "$(dirname "${LINK_PATH}")"
            
            # Create symbolic link (remove existing link if present)
            if [[ -L "${LINK_PATH}" ]]; then
                rm "${LINK_PATH}"
            fi
            
            ln -s "${SOURCE_PATH}" "${LINK_PATH}"
            echo "  ✓ Created link for ${DOMAIN} (from ${SOURCE_LABEL})"
            ((total_created++))
        else
            echo "  ✗ Source not found for ${DOMAIN} in either location"
            ((total_notfound++))
        fi
    done
    echo ""
done

echo "=================================================="
echo "Symbolic link creation complete!"
echo ""
echo "Summary:"
echo "  Symbolic links created: ${total_created}"
echo "  Sources not found: ${total_notfound}"
echo ""
echo "To verify links:"
echo "  ls -lR ${LINK_BASE}"
echo ""
echo "To update your config template, change rawdata_path to:"
echo "  For d3: rawdata_path: '${LINK_BASE}/STARTDATE/ENSMEMBER/CONFIG/les/subset_d3/'"
echo "  For d4: rawdata_path: '${LINK_BASE}/STARTDATE/ENSMEMBER/CONFIG/les/subset_d4/'"
echo "=================================================="
