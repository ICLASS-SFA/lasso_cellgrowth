#!/bin/bash

# Script to check regridded file counts and output as CSV
# Usage: bash check_regrid_csv.sh > regrid_progress.csv

ROOT_DIR="/gpfs/wolf2/arm/atm131/proj-shared/zfeng/cacti/les/regrid_subsets"

# Case dates and ensemble members (matching submit_all_cases.sh)
CASE_DATES=(
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

DOMAINS=("d3" "d4")
# FILE_TYPES=("methamsl" "cldhamsl")
FILE_TYPES=("met" "cld")
EXPECTED_COUNT=144  # 12 hours × 12 files/hour

# Print CSV header
echo "Case,Ensemble,Domain,FileType,Count,Expected,Status"

# Iterate through all case/ensemble combinations
for ((i = 0; i < ${#CASE_DATES[@]}; ++i)); do
    case=${CASE_DATES[$i]}
    ensemble=${ENSEMBLES[$i]}
    case_dir="$ROOT_DIR/$case/$ensemble/base/les"
    
    for domain in "${DOMAINS[@]}"; do
        domain_dir="$case_dir/subset_$domain"
        
        for file_type in "${FILE_TYPES[@]}"; do
            if [[ ! -d "$domain_dir" ]]; then
                echo "$case,$ensemble,$domain,$file_type,0,$EXPECTED_COUNT,MISSING_DIR"
            else
                # Count files
                count=$(find "$domain_dir" -name "corlasso_${file_type}_*.nc" -type f 2>/dev/null | wc -l)
                
                if [[ $count -eq $EXPECTED_COUNT ]]; then
                    status="COMPLETE"
                elif [[ $count -eq 0 ]]; then
                    status="MISSING"
                else
                    status="INCOMPLETE"
                fi
                
                echo "$case,$ensemble,$domain,$file_type,$count,$EXPECTED_COUNT,$status"
            fi
        done
    done
done
