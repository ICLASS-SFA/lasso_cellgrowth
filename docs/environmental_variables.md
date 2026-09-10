# Environmental Variables Pipeline

Documents the full pipeline from 3D environment extraction through derived 1D variables used in final analysis.

## Pipeline Overview

```
extract_cell_env3d_preCI_met.py     extract_cell_env1d_2location_preCI_met.py
        │                                       │
        ▼                                       ▼
  cell_env3d_*.nc                    env1d_2location_*.nc
        │                                       │
        ├──► calc_cell_env2d_from_3d.py         │
        │           │                           │
        │           ▼                           │
        │    stats_env2d_*.nc                   │
        │           │                           │
        ▼           ▼                           ▼
  calc_cell_avg_env.py              combine_cell_env_2location.py
        │                                       │
        ▼                                       ▼
  stats_avg1d_env*_*.nc             stats_avg1d_env*_*.nc
        │                                       │
        └───────────────┬───────────────────────┘
                        ▼
          filter_and_save_combined_tracks.py
                        │
                        ▼
           stats_combined_filtered_*.nc (for notebooks)
```

---

## Script Details

### `calc_cell_env2d_from_3d.py`

**Purpose**: Derives 2D environmental variables from 3D profiles.

**Key computations** (using MetPy and WRF-Python):
- **Wind shear**: Bulk shear magnitude and direction at specified AGL levels (0–1 km, 0–3 km, 0–6 km)
- **Thermodynamic indices**: Via `interplevel()` interpolation to fixed heights

**Key Function: `calc_shear(level_shear, u, v, z_agl, u10, v10)`**
```python
# Interpolate U, V to specific AGL level
u_agl = interplevel(u, z_agl, level_shear)
v_agl = interplevel(v, z_agl, level_shear)
# Compute bulk shear
shear_mag = sqrt((u_agl - u10)**2 + (v_agl - v10)**2)
shear_dir = arctan2(v_agl - v10, u_agl - u10)
```

**Input**: `cell_env3d_*.nc` (3D profiles)  
**Output**: `stats_env2d_*.nc` (2D derived variables per track)

---

### `calc_cell_avg_env.py`

**Purpose**: Computes 1D environmental variables averaged around the cell center from 3D data.

**Algorithm**:
1. Read 3D environment file, subset to center box (`nx_center × ny_center`)
2. Interpolate 3D fields to fixed height levels using `wrf.interplevel()`
3. Compute derived thermodynamic variables:
   - **Theta** (potential temperature): `T × (1e5/P)^0.286`
   - **ThetaE** (equivalent potential temperature): Bolton (1980) formula
4. Average horizontally over the center box
5. Read 2D environment file and Marquis metrics
6. Combine into output dataset

**Key Function: `theta_e_bolton(TEMPERATURE, QVAPOR, PRESSURE)`**

Bolton (1980) pseudoadiabatic equivalent potential temperature:
```python
es = PRESSURE * QVAPOR / (0.622 * QVAPOR)
TDEW = (35.86 * log(es) - 4947.2325) / (log(es) - 23.6837)
TLCL = 1/(1/(TDEW-56) + log(TEMPERATURE/TDEW)/800) + 56
THETAE = TEMPERATURE * (1e5/PRESSURE)^(0.2854*(1-0.28*QVAPOR)) * 
         exp((3.376/TLCL - 0.00254) * QVAPOR*1000 * (1+0.81*QVAPOR))
```
Error < 0.3 K between -35°C and 35°C.

**Input**: 3D env files + 2D env files + Marquis sounding metrics  
**Output**: `stats_avg1d_env{N}x{N}_*.nc` (where N indicates the averaging box size in grid points, e.g., 21×21 for D3/D4, 9×9 for D2)

---

### `calc_cell_center_env.py`

**Purpose**: Early experimental variant — computes 1D environment at the exact cell center point (no spatial averaging).

**Differences from `calc_cell_avg_env.py`**:
- Extracts single-column profile at cell center coordinates
- No horizontal averaging
- Used in early analysis; superseded by `calc_cell_avg_env.py` for the paper

---

### `combine_cell_env_2location.py`

**Purpose**: Combines environmental parameters and profiles from the 2-location extraction, integrating external (Marquis) sounding-based metrics.

**Algorithm**:
1. Read 3D environment at 2 locations (from `extract_cell_env1d_2location_preCI_met.py`)
2. Interpolate profiles to fixed height levels (0–20 km, 200 m spacing) using `metpy.interpolate.interpolate_1d()`
3. Read Marquis 2D metrics (CAPE, EL, LCL, LFC, shear, PW, etc.)
4. Pass through specified variables from Marquis dataset
5. Combine into single output file per track

**Input**: 1D 2-location env files + Marquis sounding metrics  
**Output**: Combined environment file with profiles at both locations

---

## External Dependency: James Marquis Metrics

The Marquis sounding-based environmental metrics are computed **outside this repository** on the 3D extracted environmental profiles. They provide:

| Variable | Description |
|---|---|
| CAPE | Convective Available Potential Energy |
| CIN | Convective Inhibition |
| EL | Equilibrium Level |
| LCL | Lifting Condensation Level |
| LFC | Level of Free Convection |
| PW | Precipitable Water |
| Shear (various levels) | Wind shear at 0–1, 0–3, 0–6 km |

These are read in by `calc_cell_avg_env.py` and `combine_cell_env_2location.py` as 2D environment files.

---

## Averaging Box Sizes

The averaging region controls what portion of the environment is sampled:

| Domain | Grid Spacing | Box (grid pts) | Physical Size |
|---|---|---|---|
| D2 (2.5 km) | 2500 m | 9×9 | ~22.5 × 22.5 km |
| D3 (500 m) | 500 m | 21×21 | ~10.5 × 10.5 km |
| D4 (100 m) | 100 m | 21×21 | ~2.1 × 2.1 km |

For regridded data (D3/D4 → 2.5 km): uses 9×9 box to match D2.

---

## How to Run

```bash
cd src/

# 2D from 3D (requires 3D env output):
python calc_cell_env2d_from_3d.py ./configs/CONFIG_NAME.yml

# Average environment (requires 3D + 2D env):
python calc_cell_avg_env.py ./configs/CONFIG_NAME.yml

# Center environment (experimental):
python calc_cell_center_env.py ./configs/CONFIG_NAME.yml

# Combine 2-location (requires 1D 2-location output + Marquis):
python combine_cell_env_2location.py ./configs/CONFIG_NAME.yml
```
