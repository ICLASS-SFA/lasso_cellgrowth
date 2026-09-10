#!/bin/bash
#
# Verification script to check setup before running regridding jobs
#

echo "========================================================================"
echo "LASSO REGRIDDING - SETUP VERIFICATION"
echo "========================================================================"
echo ""

# Test case
CASE_DATE="20190123"
ENSEMBLE="gefs18"

echo "Testing with case: ${CASE_DATE}, ensemble: ${ENSEMBLE}"
echo ""

# Check d4 input directories
echo "1. Checking d4 input directories..."
echo "----------------------------------------"

D4_DIR1="/gpfs/wolf2/arm/cli120/world-shared/d3m088/cacti/staged_runs_2/${CASE_DATE}/${ENSEMBLE}/base/les/subset_d4/"
D4_DIR2="/gpfs/wolf2/arm/cli120/world-shared/d3m088/cacti/staged_runs/${CASE_DATE}/${ENSEMBLE}/base/les/subset_d4/"

if [ -d "$D4_DIR1" ]; then
    echo "✓ Found d4 in staged_runs_2: $D4_DIR1"
    D4_DIR=$D4_DIR1
    D4_FILE_COUNT=$(ls -1 ${D4_DIR}*.nc 2>/dev/null | wc -l)
    echo "  Files: $D4_FILE_COUNT"
elif [ -d "$D4_DIR2" ]; then
    echo "✓ Found d4 in staged_runs: $D4_DIR2"
    D4_DIR=$D4_DIR2
    D4_FILE_COUNT=$(ls -1 ${D4_DIR}*.nc 2>/dev/null | wc -l)
    echo "  Files: $D4_FILE_COUNT"
else
    echo "✗ d4 directory not found in either location"
    D4_DIR=""
fi
echo ""

# Check d3 input directory
echo "2. Checking d3 input directory..."
echo "----------------------------------------"

D3_DIR="/gpfs/wolf2/arm/atm131/proj-shared/money/${CASE_DATE}/${ENSEMBLE}/base/les/subset_d3/"

if [ -d "$D3_DIR" ]; then
    echo "✓ Found d3: $D3_DIR"
    D3_FILE_COUNT=$(ls -1 ${D3_DIR}*.nc 2>/dev/null | wc -l)
    echo "  Files: $D3_FILE_COUNT"
else
    echo "✗ d3 directory not found: $D3_DIR"
    D3_DIR=""
fi
echo ""

# Check output directory
echo "3. Checking output directory..."
echo "----------------------------------------"

OUT_DIR_BASE="/gpfs/wolf2/arm/atm131/proj-shared/zfeng/cacti/les/regrid_subsets/"

if [ -d "$OUT_DIR_BASE" ]; then
    echo "✓ Output base directory exists: $OUT_DIR_BASE"
    echo "  Writable: $([ -w $OUT_DIR_BASE ] && echo 'Yes' || echo 'No')"
else
    echo "✗ Output base directory not found: $OUT_DIR_BASE"
fi
echo ""

# Check if we can create subdirectories
echo "4. Testing output subdirectory creation..."
echo "----------------------------------------"

TEST_OUT_DIR="${OUT_DIR_BASE}${CASE_DATE}/${ENSEMBLE}/base/les/subset_d4/"

if mkdir -p "$TEST_OUT_DIR" 2>/dev/null; then
    echo "✓ Can create output subdirectories: $TEST_OUT_DIR"
    # Try to create a test file
    if touch "${TEST_OUT_DIR}test_write.txt" 2>/dev/null; then
        echo "✓ Can write to output directory"
        rm -f "${TEST_OUT_DIR}test_write.txt"
    else
        echo "✗ Cannot write to output directory"
    fi
else
    echo "✗ Cannot create output subdirectories"
fi
echo ""

# Check for sample files with next day timestamp
echo "5. Checking for next-day files (e.g., 20190124.000000.nc)..."
echo "----------------------------------------"

if [ -n "$D4_DIR" ]; then
    NEXT_DAY=$(date -d "${CASE_DATE} +1 day" +%Y%m%d)
    NEXT_DAY_FILES=$(ls -1 ${D4_DIR}*${NEXT_DAY}.000000.nc 2>/dev/null | wc -l)
    
    if [ $NEXT_DAY_FILES -gt 0 ]; then
        echo "✓ Found files with next day timestamp (${NEXT_DAY}.000000.nc)"
        echo "  Count: $NEXT_DAY_FILES"
    else
        echo "⚠ No files found with next day timestamp"
        echo "  This may be expected depending on the case"
    fi
else
    echo "⚠ Skipped (d4 directory not found)"
fi
echo ""

# Check Python environment
echo "6. Checking Python environment..."
echo "----------------------------------------"

if command -v python &> /dev/null; then
    echo "✓ Python found: $(python --version 2>&1)"
    
    # Check required packages
    echo "  Checking packages:"
    
    python -c "import numpy; print('    ✓ numpy:', numpy.__version__)" 2>/dev/null || echo "    ✗ numpy not found"
    python -c "import scipy; print('    ✓ scipy:', scipy.__version__)" 2>/dev/null || echo "    ✗ scipy not found"
    python -c "import xarray; print('    ✓ xarray:', xarray.__version__)" 2>/dev/null || echo "    ✗ xarray not found"
    python -c "import dask; print('    ✓ dask:', dask.__version__)" 2>/dev/null || echo "    ✗ dask not found"
    python -c "from dask.distributed import Client; print('    ✓ dask.distributed')" 2>/dev/null || echo "    ✗ dask.distributed not found"
else
    echo "✗ Python not found"
fi
echo ""

# Check script files
echo "7. Checking script files..."
echo "----------------------------------------"

SCRIPT_DIR="/ccsopen/home/zhe1feng1/program/lasso/cellgrowth/regrid"

if [ -d "$SCRIPT_DIR" ]; then
    cd "$SCRIPT_DIR"
    
    for FILE in regrid_func.py regrid_lasso_batch.py test_regrid_single.py slurm_regrid.sh submit_all_cases.sh; do
        if [ -f "$FILE" ]; then
            echo "  ✓ $FILE"
        else
            echo "  ✗ $FILE (missing)"
        fi
    done
    
    # Check if shell scripts are executable
    echo ""
    echo "  Executable scripts:"
    for FILE in slurm_regrid.sh submit_all_cases.sh; do
        if [ -x "$FILE" ]; then
            echo "    ✓ $FILE"
        else
            echo "    ✗ $FILE (not executable)"
        fi
    done
else
    echo "✗ Script directory not found: $SCRIPT_DIR"
fi
echo ""

# Summary
echo "========================================================================"
echo "SUMMARY"
echo "========================================================================"
echo ""

ISSUES=0

if [ -z "$D4_DIR" ]; then
    echo "⚠ d4 input directory not found for case ${CASE_DATE}"
    ISSUES=$((ISSUES + 1))
fi

if [ -z "$D3_DIR" ]; then
    echo "⚠ d3 input directory not found for case ${CASE_DATE}"
    ISSUES=$((ISSUES + 1))
fi

if [ ! -d "$OUT_DIR_BASE" ]; then
    echo "⚠ Output base directory not accessible"
    ISSUES=$((ISSUES + 1))
fi

if ! command -v python &> /dev/null; then
    echo "⚠ Python not found in PATH"
    ISSUES=$((ISSUES + 1))
fi

if [ $ISSUES -eq 0 ]; then
    echo "✓ All checks passed! Ready to run regridding."
    echo ""
    echo "Next steps:"
    echo "  1. Test single file:    python test_regrid_single.py"
    echo "  2. Test batch (1 hour): python regrid_lasso_batch.py --case-date ${CASE_DATE} --start-hour 12 --end-hour 13 --file-type methamsl --domain d4 --n-workers 4 --regrid-ratio 25"
    echo "  3. Submit job:          sbatch --export=CASE_DATE=${CASE_DATE},FILE_TYPE=methamsl,DOMAIN=d4,REGRID_RATIO=25 slurm_regrid.sh"
else
    echo "⚠ Found $ISSUES issue(s). Please resolve before running regridding."
fi

echo ""
echo "========================================================================"
