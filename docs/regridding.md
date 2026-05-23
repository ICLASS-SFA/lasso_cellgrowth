# Regridding Methodology

Documents the approach for coarsening (regridding) LASSO high-resolution D3 (500 m) and D4 (100 m) data to match the D2 (2.5 km) resolution for cross-resolution sensitivity comparisons.

## Purpose

The paper compares convective cell statistics across three WRF nest resolutions:
- D2: 2.5 km (mesoscale)
- D3: 500 m (LES)
- D4: 100 m (LES)

To isolate the effect of resolution on tracked cell properties vs. the effect of resolved physics, D3 and D4 data are regridded (coarsened) to 2.5 km and cell tracking is re-run at the common grid spacing.

## Regrid Ratios

| Source Domain | Source Resolution | Ratio | Target Resolution |
|---|---|---|---|
| D3 | 500 m | 5 | 2500 m (2.5 km) |
| D4 | 100 m | 25 | 2500 m (2.5 km) |

---

## Core Method: Uniform Filter Averaging

### `regrid_func.py` — `coarsen_variable_filter(in_variable, ratio)`

Uses `scipy.ndimage.uniform_filter` for efficient spatial averaging:

```python
# 1. Create mask for valid (non-NaN) values
mask = ~np.isnan(in_variable)

# 2. Replace NaN with 0 for filtering
filtered_var = np.where(mask, in_variable, 0.0)

# 3. Set filter size (only in y,x dimensions for 3D arrays)
filter_size = (1, ratio, ratio)  # 3D: don't average in z

# 4. Apply uniform filter to both data and mask
var_sum = uniform_filter(filtered_var, size=filter_size, mode='constant')
count_sum = uniform_filter(mask.astype(dtype), size=filter_size, mode='constant')

# 5. Subsample at center of each averaging window
start_idx = ratio // 2
var_coarse = var_sum[:, start_idx::ratio, start_idx::ratio]
count_coarse = count_sum[:, start_idx::ratio, start_idx::ratio]

# 6. Compute average (handling NaN regions)
out_variable = var_coarse / count_coarse  # where count > 0
```

**Key features:**
- Handles NaN values correctly (excludes from average)
- Memory-efficient: uses float32 when possible
- Single-pass processing (no intermediate dictionaries)
- Works for both 2D (y,x) and 3D (z,y,x) arrays

---

### `regrid_func.py` — `coarsen_reflectivity_filter(in_reflectivity, ratio)`

Radar reflectivity requires special handling because dBZ is logarithmic:

```python
# 1. Convert dBZ to linear (Z = 10^(dBZ/10))
linear_refl = 10.0 ** (in_reflectivity / 10.0)

# 2. Average in linear space (same method as above)
linear_coarse = coarsen(linear_refl, ratio)

# 3. Convert back to dBZ (dBZ = 10 * log10(Z))
out_reflectivity = 10.0 * np.log10(linear_coarse)
```

This ensures physically meaningful averaging (power-weighted rather than log-averaged).

---

## Scripts

### `regrid_lasso_batch.py`

Batch processing for multiple files with Dask parallelization.

**Usage:**
```bash
# Serial (single worker, full node memory):
python regrid_lasso_batch.py \
    --case-date 20190123 --ensemble gefs18 \
    --start-hour 12 --end-hour 13 \
    --file-type methamsl --domain d4

# Parallel (multiple workers):
python regrid_lasso_batch.py \
    --case-date 20190123 --ensemble gefs18 \
    --start-hour 12 --end-hour 13 \
    --file-type methamsl --domain d4 --n-workers 6
```

**Arguments:**
| Argument | Required | Description |
|---|---|---|
| `--case-date` | Yes | Case date (YYYYMMDD) |
| `--ensemble` | Yes | Ensemble member (e.g., gefs18) |
| `--start-hour` | Yes | Start hour (12–23) |
| `--end-hour` | Yes | End hour (12–23) |
| `--file-type` | Yes | `methamsl` or `cldhamsl` |
| `--domain` | Yes | `d3` or `d4` |
| `--n-workers` | No | Workers (default: 1=serial, >1=parallel) |
| `--regrid-ratio` | No | Override ratio (default: 5 for d3, 25 for d4) |

### `regrid_d4_reflectivity.py`

Regrids D4 reflectivity fields using convolution-based approach:
- Applies `convolve_reflectivity()` (dBZ→linear→convolve→dBZ)
- Uses a kernel matrix for spatial averaging
- Subsamples at specified ratio

### `regrid_d4_celltracking_mask.py`

Regrids cell tracking pixel-level masks from regridded (2.5 km) tracking back to D4 native grid:
- Uses **nearest-neighbor interpolation** (not averaging) to preserve integer labels
- Handles: `tracknumber`, `tracknumber_cmask`, `track_status`, `cloudnumber`, `merge_tracknumber`, `split_tracknumber`

### `regrid_csapr_terrain.py`

Regrids CSAPR-2 terrain/range mask file from 500 m to 2.5 km:
- Terrain height: convolution averaging then subsample
- Range masks (binary): direct subsampling at center points

---

## SLURM Submission

```bash
# Single case:
sbatch --export=CASE_DATE=20190123,FILE_TYPE=methamsl,DOMAIN=d4,ENSEMBLE=gefs18 slurm_regrid.sh

# All cases:
bash submit_all_cases.sh
```

---

## File Types Processed

| File Type | Base Name Pattern | Variables |
|---|---|---|
| `methamsl` | `corlasso_methamsl_*` | T, qv, RH, P, U, V, W, height (on MSL levels) |
| `cldhamsl` | `corlasso_cldhamsl_*` | Cloud, rain, ice, snow mixing ratios (on MSL levels) |

---

## Notes

- D4 arrays are large (~2800×2200×150), requiring careful memory management
- Automatic garbage collection between files
- For full documentation of batch usage, see [regrid/README.md](../regrid/README.md)
- The older `regrid_d4_reflectivity.py` uses `scipy.ndimage.convolve` (kernel-based); the newer `regrid_func.py` uses `scipy.ndimage.uniform_filter` (more memory-efficient for large arrays)
