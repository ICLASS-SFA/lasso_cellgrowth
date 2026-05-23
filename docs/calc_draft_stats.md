# Updraft Statistics: `calc_draft_stats_in_celltracks.py`

Calculates 3D vertical velocity (W) core statistics from gridded Met data for tracked convective cells.

## Overview

For each tracked convective cell at each time step, this script:
1. Identifies updraft and downdraft cores using W and hydrometeor thresholds
2. Labels connected regions as individual cores
3. Ranks cores by vertical mass flux (VMF)
4. Computes height-resolved statistics for the largest cores
5. Expands core masks to capture the surrounding shell/perimeter region
6. Writes per-cell statistics to netCDF

## Key Functions

### `label_cores(W, W_thresh, Q, Q_thresh, VMF, ncores_min, min_core_npix, method)`
Labels updraft/downdraft cores at each vertical level using:
- **W threshold** (`W_up_thresh=2.0 m/s`, `W_down_thresh=-1.0 m/s`)
- **Q threshold** (hydrometeor mixing ratio, defines "moist" cores)
- **Connectivity**: `scipy.ndimage.label()` for connected-component analysis
- **Size filtering**: removes cores with fewer than `min_core_npix` pixels
- **Sorting**: cores ranked by total vertical mass flux (VMF = ρ × W × area), not pixel count

Returns the labeled core map and statistics for the top N cores.

### `make_dilation_structure(dilate_radius, DX, DY)`
Creates a circular dilation structure for expanding core masks:
- Converts radius (km) to grid points
- Builds a binary disk using ogrid distance formula
- Used to define the "shell" region around each core

### `calc_basetime(filelist, filebase)`
Parses filenames to extract timestamps as epoch time. File naming convention:
```
{filebase}{YYYYMMDD}.{HHMMSS}
```

## Configuration Parameters

From `src/configs/config_lasso_wrf*_template.yml`:

| Parameter | Value | Description |
|---|---|---|
| `W_up_thresh` | 2.0 m/s | Updraft threshold |
| `W_down_thresh` | -1.0 m/s | Downdraft threshold |
| `Q_up_thresh` | 0.00001 kg/kg | Moist updraft hydrometeor threshold |
| `Q_down_thresh` | 0.00005 kg/kg | Moist downdraft threshold |
| `min_core_npix` | 9 | Minimum pixels to define a core |
| `ncores_min` | 2 | Number of cores to save per cell |
| `core_expand_dist` | 6 | Pixels to expand core for perimeter |
| `core_shell_buffer_radius` | 10 km | Buffer for shell region |
| `n_workers` | 64 | Dask workers for parallel processing |

## Inputs

- **Track statistics file**: `trackstats_*.nc` (from PyFLEXTRKR)
- **Pixel-level tracking files**: `regrid_celltracks_*.nc` (cell masks with track numbers)
- **Met files**: `corlasso_drafthamsl_*.nc` or `corlasso_methamsl_*.nc` (W, Q, ρ on height-MSL levels)

## Outputs

- **`stats_3d_w_fixshell_*.nc`**: Per-cell W statistics including:
  - Core pixel counts, VMF at each height level
  - Maximum/mean W within cores
  - Core top/base heights
  - Shell-region statistics

## How to Run

```bash
cd src/
python calc_draft_stats_in_celltracks.py ./configs/config_lasso_wrf100m_20190123_gefs18_base.yml
```

Or via SLURM (uncomment in `job_scripts/slurm_lasso_template.sh`):
```bash
python calc_draft_stats_in_celltracks.py ./configs/CONFIG_NAME.yml
```

## Variants

- `calc_draft_stats_in_celltracks_bytracks.py` — Processes track-by-track (alternative memory management)
- `calc_draft_stats_in_celltracks_wrfout.py` — Reads directly from raw WRF output files
- `calc_draft_stats_in_celltracks_draftfiles.py` — Uses pre-extracted draft files

## Notes

- Uses Dask `LocalCluster` for parallelization across time steps
- Memory-intensive for D4 (100 m) runs; may need `batch_high_memory` partition
- Includes memory monitoring (`log_memory_usage()`) and cleanup functions
