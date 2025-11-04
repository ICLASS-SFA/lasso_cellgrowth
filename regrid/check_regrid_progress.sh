#!/bin/bash

# Script to check regridded file counts for all cases
# Expected: 144 files per case/domain/filetype (12 hours × 12 files/hour)
#
# Usage: bash check_regrid_progress.sh [--detailed]
#        --detailed: Show detailed file counts for each hour

ROOT_DIR="/gpfs/wolf2/arm/atm131/proj-shared/zfeng/cacti/les/regrid_subsets"

# Case dates and ensemble members (matching submit_all_cases.sh)
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

DOMAINS=("d3" "d4")
FILE_TYPES=("methamsl" "cldhamsl")
EXPECTED_COUNT=144  # 12 hours × 12 files/hour

# Check if detailed mode
DETAILED=0
if [[ "$1" == "--detailed" ]]; then
    DETAILED=1
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Checking Regridded File Counts"
echo "Root Directory: $ROOT_DIR"
echo "Expected files per case/domain/filetype: $EXPECTED_COUNT"
echo "=========================================="
echo ""

# Summary counters
total_complete=0
total_incomplete=0
total_missing=0

# Iterate through all case/ensemble combinations
for ((i = 0; i < ${#CASE_DATES[@]}; ++i)); do
    case=${CASE_DATES[$i]}
    ensemble=${ENSEMBLES[$i]}
    
    echo "Case: $case / Ensemble: $ensemble"
    echo "----------------------------------------"
    
    case_dir="$ROOT_DIR/$case/$ensemble/base/les"
    
    if [[ ! -d "$case_dir" ]]; then
        echo "  ${RED}✗${NC} Directory not found: $case_dir"
        ((total_missing+=4))  # 2 domains × 2 file types
        echo ""
        continue
    fi
    
        for domain in "${DOMAINS[@]}"; do
            domain_dir="$case_dir/subset_$domain"
            
            if [[ ! -d "$domain_dir" ]]; then
                echo "  ${RED}✗${NC} $ensemble/$domain: Directory not found"
                ((total_missing+=2))  # 2 file types
                continue
            fi
            
            for file_type in "${FILE_TYPES[@]}"; do
                # Count files
                count=$(find "$domain_dir" -name "corlasso_${file_type}_*.nc" -type f | wc -l)
                
                if [[ $count -eq $EXPECTED_COUNT ]]; then
                    echo -e "  ${GREEN}✓${NC} $ensemble/$domain/$file_type: $count/$EXPECTED_COUNT files"
                    ((total_complete++))
                elif [[ $count -eq 0 ]]; then
                    echo -e "  ${RED}✗${NC} $ensemble/$domain/$file_type: $count/$EXPECTED_COUNT files (MISSING)"
                    ((total_missing++))
                else
                    echo -e "  ${YELLOW}⚠${NC} $ensemble/$domain/$file_type: $count/$EXPECTED_COUNT files (INCOMPLETE)"
                    ((total_incomplete++))
                    
                    # Show detailed hourly breakdown if requested
                    if [[ $DETAILED -eq 1 ]]; then
                        echo "     Hourly breakdown:"
                        for hour in {12..23}; do
                            hour_count=$(find "$domain_dir" -name "corlasso_${file_type}_*${case}${hour}????.nc" -type f | wc -l)
                            if [[ $hour_count -eq 12 ]]; then
                                echo -e "       Hour $hour: ${GREEN}$hour_count/12${NC}"
                            elif [[ $hour_count -eq 0 ]]; then
                                echo -e "       Hour $hour: ${RED}$hour_count/12${NC}"
                            else
                                echo -e "       Hour $hour: ${YELLOW}$hour_count/12${NC}"
                            fi
                        done
                    fi
                fi
            done
        done
    done
    echo ""
done

# Print summary
echo "=========================================="
echo "Summary"
echo "=========================================="
echo -e "${GREEN}Complete:${NC}   $total_complete combinations"
echo -e "${YELLOW}Incomplete:${NC} $total_incomplete combinations"
echo -e "${RED}Missing:${NC}    $total_missing combinations"
echo ""

total=$((total_complete + total_incomplete + total_missing))
if [[ $total -gt 0 ]]; then
    percent_complete=$((100 * total_complete / total))
    echo "Progress: $total_complete/$total ($percent_complete%)"
else
    echo "No data found!"
fi

echo ""
echo "Run with --detailed flag to see hourly breakdown for incomplete cases"
