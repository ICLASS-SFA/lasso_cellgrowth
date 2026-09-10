# LASSO Subset Data Regridding

Efficient regridding of LASSO subset files using uniform filter for spatial coarsening.

## Directory Structure

```
regrid/
├── regrid_func.py              # Core regridding functions
├── regrid_lasso_batch.py       # Batch processing with Dask
├── test_regrid_single.py       # Test single file regridding
├── slurm_regrid.sh            # SLURM job array script
├── submit_all_cases.sh        # Master submission script
└── logs/                      # Log files (created automatically)
```

## Core Functions

### `regrid_func.py`
Contains the main regridding functions:
- `coarsen_variable_filter()`: Coarsen generic variables using uniform filter
- `coarsen_reflectivity_filter()`: Coarsen radar reflectivity (handles dBZ conversion)
- `regrid_file()`: Main function to regrid a single file

### Memory-Efficient Features
- Single-pass processing (no intermediate dictionaries)
- Automatic garbage collection
- Handles NaN values efficiently
- Optimized for large arrays (2800×2200×150)

## Usage

### 1. Test Single File
First, test with a single file to ensure everything works:

```bash
python test_regrid_single.py
```

Edit the script to change:
- Input directory and file
- Regrid ratio
- Variable subset (for testing)

### 2. Batch Processing (Interactive)
Process multiple files interactively:

**Serial mode (default, uses all available memory):**
```bash
python regrid_lasso_batch.py \
    --case-date 20190123 \
    --ensemble gefs18 \
    --start-hour 12 \
    --end-hour 13 \
    --file-type methamsl \
    --domain d4
```

**Parallel mode (multiple workers, recommended for faster processing):**
```bash
python regrid_lasso_batch.py \
    --case-date 20190123 \
    --ensemble gefs18 \
    --start-hour 12 \
    --end-hour 13 \
    --file-type methamsl \
    --domain d4 \
    --n-workers 6
```

Options:
- `--case-date`: Case date (YYYYMMDD) - **required**
- `--ensemble`: Ensemble member (e.g., gefs18, eda05) - **required**
- `--start-hour`: Start hour (12-23) - **required**
- `--end-hour`: End hour (12-23) - **required**
- `--file-type`: File type (methamsl or cldhamsl) - **required**
- `--domain`: Domain (d3 or d4) - **required**
- `--n-workers`: Number of workers (default: 1 for serial, >1 for parallel with Dask)
- `--regrid-ratio`: Coarsening ratio (default: 5 for d3, 25 for d4, both target 2.5km)

**Regrid Ratios:**
- **d3 domain**: 500m × 5 = 2500m (2.5 km)
- **d4 domain**: 100m × 25 = 2500m (2.5 km)
- Both domains coarsen to equivalent 2.5 km grid spacing for comparison with mesoscale runs

**Processing Modes:**
- `--n-workers 1` (default): Serial processing - uses full node memory, slower but simpler
- `--n-workers 6`: Parallel processing - 6 workers with 40GB each, faster but needs Dask

### 3. Submit SLURM Jobs

#### Single Job Array (One Case, One File Type)
```bash
# Uses default regrid_ratio based on domain
#### Single Job Array (One Case, One File Type)
```bash
# Uses default regrid_ratio based on domain
sbatch --export=CASE_DATE=20190123,FILE_TYPE=methamsl,DOMAIN=d4,ENSEMBLE=gefs18 slurm_regrid.sh

# Or override regrid_ratio
sbatch --export=CASE_DATE=20190123,FILE_TYPE=methamsl,DOMAIN=d4,ENSEMBLE=gefs18,REGRID_RATIO=20 slurm_regrid.sh
```

This creates a job array with 12 tasks (one per hour).

**Note**: Each case date may have different ensemble members. Check `submit_all_cases.sh` for the correct case/ensemble combinations.

#### Submit All Cases
The script automatically processes all 18 case/ensemble combinations:

```bash
bash submit_all_cases.sh
```

This submits jobs for:
- 18 case/ensemble combinations (see arrays in submit_all_cases.sh)
- 2 file types (methamsl, cldhamsl)
- 2 domains (d3, d4)
- Total: 18 × 2 × 2 = 72 job arrays
- Total tasks: 72 × 12 = 864 tasks

**Case/Ensemble Combinations**:
- 20181129: gefs00, gefs03, gefs09, gefs18
- 20181204: gefs18, gefs19
- 20181205: gefs01
- 20181219: eda09
- 20190122: gefs01, gefs18
- 20190123: eda05, gefs18
- 20190125: eda07, gefs11
- 20190129: eda09, gefs11
- 20190208: eda03, eda08

### 4. Monitor Jobs

```bash
# Check all your jobs
squeue -u $USER

# Check specific job array
squeue -j <JOB_ID>

# Watch job status (updates every 2 seconds)
watch -n 2 'squeue -u $USER'

# Check completed jobs
sacct -u $USER --starttime $(date -d '1 day ago' +%Y-%m-%d)
```

### 5. Check Results

```bash
# View log files
tail -f logs/regrid_<JOB_ID>_<TASK_ID>.out

# Check for errors
grep -i error logs/*.err

# Count successful outputs
ls -1 /path/to/output/dir/*.nc | wc -l
```

## Configuration

### Regrid Ratios
- `ratio=5`: 5×5 averaging → 1/25th the data size
- `ratio=25`: 25×25 averaging → 1/625th the data size

### Memory Requirements (per file)
For d4 domain (2775×2145×40):
- Input: ~9.5 GB
- Peak memory during processing: ~30-40 GB per file
- Output (ratio=25): ~15 MB per file

### Recommended Worker Settings

**Base Nodes (256 GB memory, 128 cores):**
- `--n-workers=6`: ~40 GB per worker → ~240 GB total (safe)
- `--cpus-per-task=128`

**High Memory Nodes (512 GB memory, 128 cores):**
- `--n-workers=10`: ~40 GB per worker → ~400 GB total
- Better for faster processing if nodes are available

## Processing Time Estimates

**Per File:**
- Small file (d3): ~2-3 minutes
- Large file (d4, ratio=25): ~3-5 minutes

**Per 1-Hour Chunk:**
- 12 files with 6 workers: ~6-10 minutes

**Full Case (12 hours):**
- 144 files (12×12) with job arrays: ~1.5-2 hours

**All 18 Cases:**
- If sufficient nodes available: ~2 hours
- Sequential processing: ~36 hours

## Output Format

Output files maintain:
- Original variable attributes
- Coordinate attributes with regridding metadata
- Global attributes including:
  - Original grid spacing
  - New grid spacing
  - Regrid ratio
  - Processing timestamp

Added attributes:
- `regridded`: Processing method description
- `regrid_ratio`: Coarsening factor

## Troubleshooting

### Memory Errors
- Reduce `--n-workers`
- Request high-memory nodes
- Increase `--mem` in SLURM script

### Files Not Found
- Check input directory path
- Verify file naming pattern
- Check time window parameters

### Dask Dashboard Not Accessible
- Dashboard runs on port 8787
- Forward port with SSH: `ssh -L 8787:localhost:8787 hostname`
- Access at: http://localhost:8787

### Failed Files
- Check error logs: `logs/regrid_*.err`
- Reprocess individual files with `test_regrid_single.py`
- Look for NaN/Inf values or corrupted input files

## Performance Tuning

1. **Adjust workers based on available memory:**
   ```bash
   # Conservative (240 GB)
   --n-workers=6
   
   # Aggressive (400 GB, high-mem nodes)
   --n-workers=10
   ```

2. **Adjust chunk size (edit slurm_regrid.sh):**
   ```bash
   # 1-hour chunks (default)
   #SBATCH --array=0-11
   
   # 2-hour chunks (fewer jobs)
   #SBATCH --array=0-5
   END_HOUR=$((START_HOUR * 2 + 2))
   ```

3. **Test with subset variables first:**
   ```python
   config = {
       'var_names': ['WA', 'HGT', 'UA', 'VA'],  # Subset for testing
       'regrid_ratio': 5,  # Lower ratio for testing
   }
   ```

## Contact

Zhe Feng, zhe.feng@pnnl.gov
Pacific Northwest National Laboratory
