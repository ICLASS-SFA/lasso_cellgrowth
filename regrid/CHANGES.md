# Changes Made to Regridding Workflow

## Date: October 30, 2025

### Summary of Changes

The following updates were made to accommodate the actual data structure and requirements:

---

## 1. Extended Time Window to Include Next Day

**Change**: Each case now includes the first hour of the next day.

**Reason**: Each case runs from 12:00:00 to 23:59:59, and the last file is at 00:00:00 of the next day (e.g., case 20190123 hour 23 includes file 20190124.000000.nc)

**Implementation**:
- Cases run from hour 12 to hour 23 (12 hours total)
- When `--end-hour 23` (last hour), the time window extends to next day 00:00:00
- Example: Case 20190123, hour 23 → time window ends at 20190124 00:00:00

**Code Location**: `regrid_lasso_batch.py`, lines ~212-226

```python
if args.end_hour == 23:
    # Extend to next day 00:00:00
    next_day = case_date_obj + timedelta(days=1)
    end_time = datetime.strptime(f"{next_day.strftime('%Y%m%d')}000000", '%Y%m%d%H%M%S')
```

**SLURM Job Array**:
- Array range: `#SBATCH --array=12-23` (12 tasks, hours 12-23)

---

## 2. Added Serial/Parallel Processing Flexibility

**Change**: Renamed `process_batch_with_dask()` to `process_batch_files()` with automatic serial/parallel mode selection.

**Reason**: Some cases may need all available node memory (256GB), making parallel processing impractical.

**Implementation**:
- `--n-workers 1` (default): Serial processing, uses full memory, simpler
- `--n-workers >1`: Parallel processing with Dask LocalCluster
- Function automatically detects mode based on n_workers value

**Code Location**: `regrid_lasso_batch.py`, function `process_batch_files()`

```python
if n_workers == 1:
    # Serial processing
    logger.info(f"Processing {len(file_list)} files in SERIAL mode")
    results = []
    for i, (in_file, in_base, out_dir, out_base) in enumerate(file_list, 1):
        result = process_single_file(in_file, in_base, out_dir, out_base, config)
        results.append(result)
else:
    # Parallel processing with Dask
    logger.info(f"Processing {len(file_list)} files in PARALLEL mode with {n_workers} workers")
    # ... Dask setup and execution
```

**Use Cases**:
- Serial mode: When maximum memory per file is needed, or Dask overhead is not desired
- Parallel mode: When processing many smaller files or when speed is priority

---

## 3. Updated Case/Ensemble Combinations in Submission Script

**Change**: Updated `submit_all_cases.sh` to include all case_date and ensemble member combinations.

**Reason**: Each case date has different ensemble members (e.g., 20181129 has gefs00, gefs03, gefs09, gefs18).

**Implementation**:
- Now uses parallel arrays for CASE_DATES and ENSEMBLES
- Made `--ensemble` argument required (no default)
- Total: 18 case/ensemble combinations × 2 file types × 2 domains = 72 job arrays

**Code Location**: `submit_all_cases.sh`, `regrid_lasso_batch.py`

```bash
CASE_DATES=(
    "20181129" "20181129" "20181129" "20181129"
    "20181204" "20181204" 
    ...
)
ENSEMBLES=(
    "gefs00" "gefs03" "gefs09" "gefs18"
    "gefs18" "gefs19" 
    ...
)
```

---

## 4. Made `--ensemble` Required Argument

**Change**: Made `--ensemble` a required argument without default value

**Reason**: Each case date has different ensemble members, so a default value would be misleading and could cause errors

**Implementation**:
```python
# Old
parser.add_argument('--ensemble', default='gefs18', help='Ensemble member (default: gefs18)')

# New
parser.add_argument('--ensemble', required=True, help='Ensemble member (e.g., gefs18, eda05)')
```

**SLURM Script Update**:
```bash
# Check required parameters
if [ -z "${ENSEMBLE}" ]; then
    echo "ERROR: ENSEMBLE is required but not set"
    exit 1
fi
```

**Updated Files**:
- `regrid_lasso_batch.py`
- `slurm_regrid.sh`
- `submit_all_cases.sh` (uses correct ensemble for each case)

---

## 5. Changed Argument Name: `--subset` → `--domain`

**Change**: Renamed command-line argument from `--subset` to `--domain`

**Reason**: More intuitive naming (d3/d4 are domain identifiers)

**Updated Files**:
- `regrid_lasso_batch.py`
- `slurm_regrid.sh`
- `submit_all_cases.sh`
- `test_regrid_single.py`

**Usage**:
```bash
# Old
--subset d4

# New
--domain d4
```

---

## 6. Automatic `--regrid-ratio` Based on Domain

**Change**: Made `--regrid-ratio` optional with automatic domain-based defaults

**Reason**: Different domains have different native resolutions but should coarsen to the same 2.5 km target for comparison with mesoscale runs

**Implementation**:
- **d3 domain**: Default ratio = 5 (500m → 2500m)
- **d4 domain**: Default ratio = 25 (100m → 2500m)
- Can still override with `--regrid-ratio` if needed

```python
# Set default regrid_ratio based on domain if not specified
if args.regrid_ratio is None:
    if args.domain == 'd3':
        args.regrid_ratio = 5   # 500m -> 2500m (2.5 km)
    elif args.domain == 'd4':
        args.regrid_ratio = 25  # 100m -> 2500m (2.5 km)
```

**Updated Files**:
- `regrid_lasso_batch.py`: Auto-detection logic
- `slurm_regrid.sh`: Conditional regrid_ratio argument
- `submit_all_cases.sh`: Removed REGRID_RATIO variable

---

## 7. Multiple Input Directory Search

**Change**: Added automatic search of multiple input directories for d4 domain

**Reason**: d4 data exists in either `staged_runs` or `staged_runs_2` (not both)

**Implementation**: New function `find_input_directory()`

**Search Order for d4**:
1. `/gpfs/wolf2/arm/cli120/world-shared/d3m088/cacti/staged_runs_2/{case_date}/{ensemble}/base/les/subset_d4/`
2. `/gpfs/wolf2/arm/cli120/world-shared/d3m088/cacti/staged_runs/{case_date}/{ensemble}/base/les/subset_d4/`

**Search for d3**:
1. `/gpfs/wolf2/arm/atm131/proj-shared/money/{case_date}/{ensemble}/base/les/subset_d3/`

**Code Location**: `regrid_lasso_batch.py`, lines ~32-72

---

## 8. Updated Output Directory Structure

**Change**: Output now mirrors input subdirectory structure

**Old Behavior**:
```
/gpfs/wolf2/arm/atm131/proj-shared/zfeng/cacti/les/regrid_subsets/
└── all files in one directory
```

**New Behavior**:
```
/gpfs/wolf2/arm/atm131/proj-shared/zfeng/cacti/les/regrid_subsets/
├── 20190123/
│   └── gefs18/
│       └── base/
│           └── les/
│               ├── subset_d3/
│               │   └── regridded files
│               └── subset_d4/
│                   └── regridded files
├── 20190124/
│   └── gefs18/
│       └── ...
```

**Implementation**:
```python
out_dir = f"/gpfs/wolf2/arm/atm131/proj-shared/zfeng/cacti/les/regrid_subsets/{case_date}/{args.ensemble}/base/les/subset_{args.domain}/"
```

---

## Updated Command Examples

### Test Single File
```bash
python test_regrid_single.py
# (Edit file first to set paths and regrid_ratio)
```

### Batch Processing (d3 domain)
```bash
python regrid_lasso_batch.py \
    --case-date 20190123 \
    --start-hour 12 \
    --end-hour 13 \
    --file-type methamsl \
    --domain d3 \
    --n-workers 6 \
    --regrid-ratio 25
```

### Batch Processing (d4 domain)
```bash
### Batch Processing (d3 domain)
```bash
python regrid_lasso_batch.py \
    --case-date 20190123 \
    --ensemble gefs18 \
    --start-hour 12 \
    --end-hour 13 \
    --file-type methamsl \
    --domain d3
# Uses default: regrid_ratio=5 (500m -> 2.5km)
```

### Batch Processing (d4 domain)
```bash
python regrid_lasso_batch.py \
    --case-date 20190123 \
    --ensemble gefs18 \
    --start-hour 12 \
    --end-hour 13 \
    --file-type methamsl \
    --domain d4 \
    --n-workers 6
# Uses default: regrid_ratio=25 (100m -> 2.5km)
```

### Submit SLURM Job
```bash
# Uses domain defaults for regrid_ratio
sbatch --export=CASE_DATE=20190123,FILE_TYPE=methamsl,DOMAIN=d4,ENSEMBLE=gefs18 slurm_regrid.sh

# Or override regrid_ratio
sbatch --export=CASE_DATE=20190123,FILE_TYPE=methamsl,DOMAIN=d4,ENSEMBLE=gefs18,REGRID_RATIO=20 slurm_regrid.sh
```

### Submit All Cases
```bash
# Automatically uses correct ensemble for each case date and domain defaults (d3=5, d4=25)
bash submit_all_cases.sh
```

---

## Files Modified

1. **`regrid_lasso_batch.py`**
   - Added `find_input_directory()` function
   - Updated argument parsing (`--subset` → `--domain`)
   - Added automatic regrid_ratio selection based on domain (d3=5, d4=25)
   - Added serial/parallel processing flexibility (`process_batch_files()`)
   - Extended time window for last hour
   - Updated output directory structure

2. **`slurm_regrid.sh`**
   - Changed `#SBATCH --array=0-11` → `#SBATCH --array=12-23`
   - Made regrid_ratio optional (uses domain default)
   - Changed `SUBSET` → `DOMAIN`
   - Updated echo statements
   - Updated command line arguments

3. **`submit_all_cases.sh`**
   - Added all 18 case_date/ensemble combinations
   - Removed REGRID_RATIO (uses domain defaults)
   - Changed `SUBSETS` → `DOMAINS`
   - Updated loop variables
   - Updated export statements

4. **`test_regrid_single.py`**
   - Updated input paths for d3 domain
   - Updated output directory structure
   - Added comments about d4 paths

---

## Testing Checklist

After these changes, test in this order:

### 1. Test Single File (5 min)
```bash
cd /ccsopen/home/zhe1feng1/program/lasso/cellgrowth/regrid
python test_regrid_single.py
```

Expected: Output file created in subdirectory structure

### 2. Test d3 Domain Batch (10 min)
```bash
python regrid_lasso_batch.py \
    --case-date 20190123 \
    --ensemble eda05 \
    --start-hour 12 \
    --end-hour 13 \
    --file-type methamsl \
    --domain d3 \
    --n-workers 4
```

Expected: Finds files in `/gpfs/wolf2/arm/atm131/proj-shared/money/...`

### 3. Test d4 Domain Batch (10 min)
```bash
python regrid_lasso_batch.py \
    --case-date 20190123 \
    --ensemble gefs18 \
    --start-hour 12 \
    --end-hour 13 \
    --file-type methamsl \
    --domain d4 \
    --n-workers 4
```

Expected: 
- Automatically finds input directory (staged_runs or staged_runs_2)
- Creates output in proper subdirectory

### 4. Test Last Hour (Includes Next Day)
```bash
python regrid_lasso_batch.py \
    --case-date 20190123 \
    --ensemble gefs18 \
    --start-hour 23 \
    --end-hour 23 \
    --file-type methamsl \
    --domain d4 \
    --n-workers 4 \
    --regrid-ratio 25
```

Expected: 
- Time window extends to 20190124 00:00:00
- Includes file `*.20190124.000000.nc`

### 5. Test Job Submission
```bash
mkdir -p logs
sbatch --export=CASE_DATE=20190123,FILE_TYPE=methamsl,DOMAIN=d4,ENSEMBLE=gefs18 slurm_regrid.sh
```

Expected: Job submits successfully, check logs

---

## Verification Steps

1. **Check input directory discovery**:
   ```bash
   # Should log which directory was found
   tail -f logs/regrid_*.out | grep "Found input directory"
   ```

2. **Check output directory structure**:
   ```bash
   ls -la /gpfs/wolf2/arm/atm131/proj-shared/zfeng/cacti/les/regrid_subsets/
   # Should see: 20190123/gefs18/base/les/subset_d4/
   ```

3. **Check time window for last hour**:
   ```bash
   # In log file for hour 11, should see:
   grep "Last hour of case" logs/regrid_*.out
   grep "extending to next day" logs/regrid_*.out
   ```

4. **Verify file count**:
   ```bash
   # Hour 12-22: should have ~12 files each
   # Hour 23: should have ~13 files (includes next day 00:00:00)
   ```

---

## Important Notes

1. **Regrid Ratio**: Now required, must specify every time
2. **Directory Search**: Automatic for d4, checks both locations
3. **d3 Domain**: Different input location from d4
4. **Output Structure**: Now organized by case/ensemble/domain
5. **Time Window**: Cases run from hour 12 to 23; last hour (23) includes next day's first file
6. **Job Array**: SLURM array range is 12-23 (12 tasks)

---

## Questions or Issues?

Contact: Zhe Feng (zhe.feng@pnnl.gov)
