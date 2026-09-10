# Environment Extraction: `extract_cell_env3d_preCI_met.py` & `extract_cell_env1d_2location_preCI_met.py`

Extracts environmental profile data from WRF output files for tracked convective cells in the pre-convection initiation (CI) period.

## Overview

Two variants serve different analysis purposes:

| Script | Purpose | Output Dimensions |
|---|---|---|
| `extract_cell_env3d_preCI_met.py` | Full 3D box centered on cell (main modeling analysis) | (tracks, times, z, y, x) |
| `extract_cell_env1d_2location_preCI_met.py` | Profiles at 2 locations: CI + ARM AMF site (obs comparison) | (tracks, times, z, x=2) |

## 3D Extraction (`extract_cell_env3d_preCI_met.py`)

### Algorithm

For each tracked cell:
1. Determine the CI (convection initiation) time from track statistics
2. Sample at multiple times before CI (controlled by `minutes_prior` and `freq_prior`)
3. Center a box of size `[2*ny+1, 2*nx+1]` at the cell's lat/lon center
4. Extract 3D fields (z, y, x) within this box from Met and Cloud files
5. Handle domain boundaries via `pad_array()` (fills with NaN where box extends outside domain)

### Key Function: `pad_array(in_array, lat_idx, lon_idx, ny, nx, ny_d, nx_d, sub_y, sub_x, fillval)`

Pads a 2D or 3D array to fixed dimensions centered at a given lat/lon index:
- Handles edge cases where extraction box extends beyond domain boundary
- Supports sub-sampling (e.g., `sub_y=20` extracts every 20th grid point for D4)
- Works for both 2D (y,x) and 3D (z,y,x) arrays

### Key Function: `extract_env_prof(fname_pixel, fname_met, fname_cld, idx_track, _lat, _lon, config)`

Main extraction function per time step:
- Reads pixel-level tracking file to get cell locations
- Reads Met file (temperature, qv, RH, pressure, U, V, W, height)
- Reads Cloud file (cloud/rain/ice/snow mixing ratios)
- Returns dictionaries of 3D data, 2D data, attributes, and coordinates

### Configuration Parameters

| Parameter | D4 (100 m) | D3 (500 m) | D2 (2.5 km) | Description |
|---|---|---|---|---|
| `nx`, `ny` | 100 | 100 | — | Half-box size in grid points |
| `nz` | 149 | 149 | — | Number of vertical levels |
| `sub_x`, `sub_y` | 20 | 4 | 1 | Sub-sampling factor |
| `minutes_prior` | 30 | 30 | 30 | Minutes before CI to sample |
| `freq_prior` | 15 | 15 | 15 | Sampling frequency (min) |

### Inputs

- Track statistics file (`trackstats_*.nc`)
- Pixel-level tracking files (`regrid_celltracks_*`)
- Met files on height-MSL levels (`corlasso_methamsl_*`)
- Cloud files on height-MSL levels (`corlasso_cldhamsl_*`)

### Outputs

- `cell_env3d_*.nc` — 3D environmental profiles per track:
  - Variables: temperature, qv, RH, pressure, U, V, W, height, cloud/rain/ice mixing ratios
  - Dimensions: (tracks, times, z, y, x)

---

## 1D Two-Location Extraction (`extract_cell_env1d_2location_preCI_met.py`)

### Algorithm

For each tracked cell:
1. Extract a single vertical profile at the **CI location** (cell center lat/lon)
2. Extract a single vertical profile at the **fixed ARM AMF site** (`lat_site`, `lon_site` from config)
3. Store both profiles for comparison with ARM observations

### Key Differences from 3D Version

- No spatial box extraction — single column profiles only
- Two locations per time step (CI center + fixed site)
- No sub-sampling or padding needed
- Output dimension: `x=2` (location index: 0=CI, 1=AMF site)

### Key Function: `location_to_idx(lat, lon, center)`

Converts a (lat, lon) coordinate to the nearest grid indices:
```python
diff = abs(lat - center[0]) + abs(lon - center[1])
lat_idx, lon_idx = np.unravel_index(diff.argmin(), diff.shape)
```
Works with 2D lat/lon arrays (WRF curvilinear grids).

### Outputs

- 1D environmental profiles at 2 locations per track
- Used as input to `combine_cell_env_2location.py`

---

## How to Run

```bash
cd src/

# 3D extraction (memory-intensive, use high-memory partition for D4):
python extract_cell_env3d_preCI_met.py ./configs/config_lasso_wrf100m_20190123_gefs18_base.yml

# 1D two-location extraction:
python extract_cell_env1d_2location_preCI_met.py ./configs/config_lasso_wrf100m_20190123_gefs18_base.yml
```

## Notes

- 3D extraction wall-clock time: 1–2 hours per case (D4)
- For D4 (100 m): use `batch_high_memory` SLURM partition, set `n_workers=8`
- For D3 (500 m): use `batch_short` partition, set `n_workers=12`
- Uses Dask `LocalCluster` for parallel processing across tracks
- File existence checks prevent crashes when individual files are missing

## Variants

- `extract_cell_env3d_preCI.py` — Earlier version without Met-level files
- `extract_cell_env3d_preCI_bytracks.py` — Track-by-track processing for memory management
- `extract_cell_env3d_preCI_bytracks_joblib.py` — Uses joblib instead of Dask
