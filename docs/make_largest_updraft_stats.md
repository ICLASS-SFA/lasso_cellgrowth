# Largest Updraft Analysis: `make_largest_updraft_stats.py`

Makes masks of the largest updraft objects from the time-height W statistics data and computes updraft growth properties.

## Overview

For each tracked cell, this script:
1. Takes the 2D (time × height) W array from `calc_draft_stats_in_celltracks.py` output
2. Labels connected updraft regions using binary thresholding
3. Identifies the **largest** updraft object within a specified CI time window
4. Computes updraft top/base height time series
5. Fits a linear regression to updraft top height (growth rate)
6. Computes cloud-base updraft statistics and distance to LCL
7. Saves output to netCDF

## Key Functions

### `get_W_properties(W_array, tidx_start, tidx_end, tidx_end_CI, cbase_depth)`

Core analysis function that processes a single cell's W time-height array:

**Steps:**
1. **Binary mask**: Create `W_binary` where W > 0
2. **Connected-component labeling**: `scipy.ndimage.label()` on the binary mask
3. **Largest object selection**: Trim to time window `[tidx_start:tidx_end]`, find object with most pixels
4. **Top height extraction**: For each time step, find the highest height index where `mask_large == 1`
5. **Base height extraction**: For each time step, find the lowest height index
6. **Cloud-base W statistics**:
   - Compute median base height
   - Define a layer: `[median_base, median_base + cbase_depth]`
   - Extract max/mean/median W within this layer
   - Separate statistics for full lifetime vs CI period only
7. **Lifetime calculation**: Start/end time of the largest object

**Returns** a dictionary containing:
- `mask_large`: Binary mask of the largest updraft object (time × height)
- `mask_remove`: Mask of removed smaller objects
- `mask_top`, `mask_base`: Top/base height time series
- `Wtime_start`, `Wtime_end`, `Wlifetime`: Temporal extent
- `Wbase_mean`, `Wbase_median`, `Wbase_max`: Cloud-base W statistics
- `Wbase_max_time`: Time of maximum cloud-base W
- `Wbase_max_CI`, `Wbase_max_time_CI`: CI-period cloud-base W statistics
- `Wbase_timeseries`: Time series of max W within base layer

### `make_W_mask(da_W, maxETH_10dbz, tidx_start, tidx_end, tidx_end_CI, cbase_depth)`

Wrapper function that:
1. Calls `get_W_properties()` to get the largest updraft mask
2. Applies additional processing based on echo-top height data
3. Computes linear fit for updraft top height growth rate
4. Computes distance between updraft base and LCL

## Algorithm Details

### Largest Object Selection
```
1. Create binary array: W_binary[W > 0] = 1
2. Label connected pixels: label_image, n_obj = label(W_binary)
3. Trim to time window [tidx_start:tidx_end]
4. Count pixels per labeled object
5. Select object with maximum pixel count → "largest updraft"
```

### Linear Fit for Updraft Top Growth
```
1. Extract mask_top time series (highest height at each time)
2. Remove NaN values
3. Apply scipy.stats.linregress to get slope (growth rate, m/s)
4. Store slope, intercept, r-value, p-value
```

### Cloud-Base Layer Statistics
```
1. median_base = nanmedian(mask_base)  # Median base height across all times
2. median_base_top = median_base + cbase_depth  # Top of analysis layer
3. Extract W values within [median_base, median_base_top] layer
4. Compute mean, median, max W within layer
5. Separate statistics for full lifetime vs CI period
```

## Configuration

Uses the same config YAML as other scripts, with key parameters:
- `cbase_depth`: Depth above cloud base for W statistics (default: 0.5 km)
- Time window for CI sampling (from track statistics attributes)

## Inputs

- **W statistics file**: `stats_3d_w_fixshell_*.nc` (output from `calc_draft_stats_in_celltracks.py`)
- **Environmental variables**: `stats_avg1d_env*_*.nc` (output from `calc_cell_avg_env.py`) — provides LCL heights
- **Config YAML**: Specifies file paths and parameters

## Outputs

- **`stats_2d_wmask_ci30min_*.nc`**: Per-track updraft mask statistics:
  - `mask_large`: Largest updraft binary mask (times × height)
  - `mask_top`, `mask_base`: Top/base height time series
  - `Wbase_max`, `Wbase_mean`: Cloud-base W statistics
  - `Wtop_slope`: Linear growth rate of updraft top
  - `Wlifetime`: Updraft object lifetime
  - LCL-base distance metrics

## How to Run

```bash
cd src/
python make_largest_updraft_stats.py ./configs/config_lasso_wrf100m_20190123_gefs18_base.yml
```

## Notes

- Requires output from both `calc_draft_stats_in_celltracks.py` and `calc_cell_avg_env.py`
- The `ci30min` in filename refers to the 30-minute CI time window used for subsetting
- The `mask_remove` output is useful for visualization (shows removed small updraft objects)
- Uses global variables (`height`, `time_coord`, `time_res`) set in the main script body
