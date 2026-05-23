# LASSO Cell Tracking Analysis

Analysis pipeline for convective cell tracking applied to LASSO-CACTI simulations and CSAPR-2 radar observations, supporting the manuscript:

> **"Updraft Width Modulations on Convective Cloud Vertical Growth: Sensitivity to Model Resolution"**  
> Zhe Feng et al. (2026), *Journal of Geophysical Research: Atmospheres* (in revision)

This repository contains scripts and Jupyter notebooks to:
1. Run PyFLEXTRKR convective cell tracking on LASSO ensemble members and CSAPR-2 observations
2. Compute updraft statistics, environmental profiles, and satellite-like metrics for tracked cells
3. Regrid high-resolution data for resolution-sensitivity comparisons
4. Produce all analysis figures in the manuscript

---

## Repository Structure

```
lasso_cellgrowth/
├── tracking/          # PyFLEXTRKR config & SLURM script generation
├── src/               # Processing scripts (updraft stats, environments, filtering)
│   ├── configs/       # YAML config templates for processing scripts
│   └── job_scripts/   # SLURM job templates
├── regrid/            # Scripts to coarsen D3/D4 data to 2.5 km
├── notebooks/         # Jupyter notebooks for analysis & figure generation
└── docs/              # Detailed per-script documentation
```

---

## Computing Environment

All processing runs on **ORNL Cumulus** (SLURM-based HPC).

### Python Environments

| Environment | Path | Purpose |
|---|---|---|
| py310 | `/ccsopen/home/zhe1feng1/anaconda3/envs/py310` | `plot_domain_cellstats_timeseries_by_case_obs_lasso.ipynb` (older xarray lazy-load behavior) |
| py312 | `/ccsopen/home/zhe1feng1/anaconda3/envs/py312` | All other production notebooks and processing scripts |
| pyflex26.4 | `/ccsopen/home/zhe1feng1/anaconda3/envs/pyflex26.4` | PyFLEXTRKR cell tracking |
| Local (reference) | `/Users/feng045/opt/miniconda3/envs/py312-26.4` | Local Mac for notebook development |

To export environment specifications for reproducibility:
```bash
# On Cumulus:
conda env export -p /ccsopen/home/zhe1feng1/anaconda3/envs/py310 > environment_py310.yml
conda env export -p /ccsopen/home/zhe1feng1/anaconda3/envs/py312 > environment_py312.yml
conda env export -p /ccsopen/home/zhe1feng1/anaconda3/envs/pyflex26.4 > environment_pyflextrkr.yml

# On local Mac:
conda env export -p /Users/feng045/opt/miniconda3/envs/py312-26.4 > environment_local.yml
```

### Key Dependencies
- Python ≥ 3.10
- numpy, xarray, dask, scipy, pandas
- scikit-learn (Random Forest analysis)
- MetPy, wrf-python (thermodynamic/wind calculations)
- PyFLEXTRKR (cell tracking)
- matplotlib, seaborn, cartopy (visualization)

---

## Input Data

All LASSO files are "subsets" produced by the LASSO team.  
LASSO subset code: https://code.arm.gov/lasso/lasso-cacti/subsetwrf  
LASSO data DOI: https://doi.org/10.2172/1905845  
CSAPR-2 DOI: https://doi.org/10.5439/2440152  

| Dataset | Resolution | Location (ORNL Cumulus) |
|---|---|---|
| LASSO D4 | 100 m | `/pscratch/sd/f/feng045/lasso/staged_runs/STARTDATE/ENSMEMBER/CONFIG/les/subset_d4/` |
| LASSO D3 | 500 m | `/gpfs/wolf2/arm/atm131/proj-shared/money/STARTDATE/ENSMEMBER/CONFIG/les/subset_d3/` |
| LASSO D2 | 2.5 km | `/gpfs/wolf2/arm/atm131/proj-shared/money/STARTDATE/ENSMEMBER/CONFIG/meso/subset_d2/` |
| CSAPR-2 | 500 m | `/gpfs/wolf2/arm/atm131/proj-shared/zfeng/cacti/csapr/taranis_corcsapr2cfrppiqcM1_gridded.c1/` |


- **D3 (500 m)**: Re-run by Enoch Jo at 5-min output frequency (original LASSO D1–D3 outputs are 15-min only), then processed with the LASSO subset tool.
- **External dependency**: James Marquis (<james.marquis@pnnl.gov>) sounding-based environmental metrics (e.g., CAPE, EL, LCL, LFC, shear, PW) computed on the 3D extracted profiles (not produced by this repo).
- See `src/configs/config_*_template.yml` for full path templates.

### Case Days and Ensemble Members

9 LASSO-CACTI case days: `20181129`, `20181204`, `20181205`, `20181219`, `20190122`, `20190123`, `20190125`, `20190129`, `20190208`

Each case has multiple ensemble members (GEFS, EDA, ERA5 forcings) in two microphysics configurations:
- **base** (Thompson): ~18 members
- **morr** (Morrison): ~16 members (not used in this repo)

---

## End-to-End Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│  1. Cell Tracking (/tracking/)                                  │
│     PyFLEXTRKR on LASSO (D2, D3, D4) + CSAPR-2 observations   │
└───────────────────────────────┬─────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ 2a. Env (obs     │  │ 2b. Env (model   │  │ 2c. Tb/Rain      │
│   comparison)    │  │   analysis)      │  │   (separate)     │
│ extract_env1d    │  │ extract_env3d    │  │ wrf_tb_rainrate  │
│   → combine_env  │  │   → avg_env      │  │   → sat_stats    │
└────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
         │                     │                      │
         │              ┌──────┴──────┐               │
         │              ▼             ▼               │
         │  ┌──────────────────────────────┐          │
         │  │ 3. Updraft Stats             │          │
         │  │ draft_stats + avg_env output │          │
         │  │   → largest_updraft_stats    │          │
         │  └──────────────┬───────────────┘          │
         │                 │                          │
         └────────┬────────┴──────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. Filter & Combine (/src/filter_and_save_combined_tracks.py)  │
│     Combines all stats → filtered netCDF per resolution         │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. Analysis Notebooks (/notebooks/)                            │
│     Read filtered combined files → produce paper figures        │
└─────────────────────────────────────────────────────────────────┘

(Optional: /regrid/ coarsens D3/D4 to 2.5 km for sensitivity tests)
```

---

## 1. Cell Tracking (`/tracking/`)

Generate PyFLEXTRKR configuration files and SLURM scripts for each case/ensemble member.

| Script | Description |
|---|---|
| `make_lasso_config_slurm_scripts_cu2.sh` | Generate configs + SLURM for LASSO WRF (D2/D3/D4) |
| `make_csapr_config_slurm_scripts.sh` | Generate configs + SLURM for CSAPR-2 observations |

**Usage:**
```bash
cd tracking/
# Edit script to select resolution template, then:
bash make_lasso_config_slurm_scripts_cu2.sh
bash make_csapr_config_slurm_scripts.sh
```
Set `submit_job="yes"` to submit SLURM jobs immediately.

**Config templates** (3 resolutions × 2 frequencies + regrid variants):

| Resolution | 5-min | 15-min | Regrid to 2.5 km |
|---|---|---|---|
| D2 (2.5 km) | `config_lasso_wrf2.5km_template.yml` | `config_lasso_wrf2.5km_15min_template.yml` | — |
| D3 (500 m) | `config_lasso_wrf500m_template.yml` | `config_lasso_wrf500m_15min_template.yml` | `*_5min_regrid2.5km_template.yml` |
| D4 (100 m) | `config_lasso_wrf100m_template.yml` | `config_lasso_wrf100m_15min_template.yml` | `*_5min_regrid2.5km_template.yml` |

See [tracking/README.md](tracking/README.md) for additional details.

---

## 2. Processing Scripts (`/src/`)

### Batch Job Generation

| Script | Description |
|---|---|
| `make_calc_stats_slurm_scripts_les.sh` | Generate SLURM jobs for LES domains (D3/D4) |
| `make_calc_stats_slurm_scripts_meso.sh` | Generate SLURM jobs for mesoscale domain (D2) |

These use `job_scripts/slurm_lasso_template.sh` as a template, substituting start date, ensemble member, and config name via `sed`.

### Key Analysis Scripts

#### Cell Environments — Observation Comparison Pipeline

| Script | Function | Input | Output |
|---|---|---|---|
| `extract_cell_env1d_2location_preCI_met.py` | Extract vertical profiles at 2 locations (CI location + fixed ARM AMF site) for tracked cells | Pixel files, Met/Cloud files, track stats | `stats_env1d_2loc_*.nc` |
| `combine_cell_env_2location.py` | Combine environmental parameters and profiles, interpolate to fixed heights | 1D env files, Marquis metrics | `stats_avg1d_env*_*.nc` |

#### Cell Environments — Main Modeling Analysis Pipeline

| Script | Function | Input | Output |
|---|---|---|---|
| `extract_cell_env3d_preCI_met.py` | Extract 3D pre-CI environmental profiles centered on tracked cells | Pixel files, Met/Cloud files, track stats | `cell_env3d_*.nc` |
| `calc_cell_avg_env.py` | Compute 1D environmental variables averaged around cell center | 3D env files, 2D env files | `stats_avg1d_env*_*.nc` |

Note: `calc_cell_center_env.py` is an early experimental variant of the center-point environment extraction.

#### Cell Updraft Pipeline

| Script | Function | Input | Output |
|---|---|---|---|
| `calc_draft_stats_in_celltracks.py` | Label updraft/downdraft cores, compute height-resolved W statistics | Pixel files, Met files (W, Q, ρ), track stats | `stats_3d_w_*.nc` |
| `make_largest_updraft_stats.py` | Identify largest updraft object, compute top/base heights, linear fit, cloud-base W, LCL distance | W stats output + avg_env output | `stats_2d_wmask_*.nc` |

#### Model Tb Statistics (Separate Pipeline)

| Script | Function | Input | Output |
|---|---|---|---|
| `calc_wrf_tb_rainrate.py` | Calculate brightness temperature and rain rate from WRF output pairs | WRF output files (OLR, RAINNC) | `tb_rainrate_*.nc` |
| `calc_sat_stats_in_celltracks.py` | Compute satellite-like statistics (min Tb, OLR) for tracked cells | Tb/rainrate files, pixel files | Appended to track stats |

#### Final Combination & Filtering

| Script | Function | Input | Output |
|---|---|---|---|
| `filter_and_save_combined_tracks.py` | Combine track stats + environments + updraft stats + largest updraft stats from all cases. Apply spatial/temporal filtering and merge/split conditions. | All per-case stats files | `stats_combined_filtered_{d2,d3,d4}.nc` |

**This is the critical bridge between processing and notebooks.** All analysis notebooks read from these filtered combined files.

```bash
# Process all domains:
python filter_and_save_combined_tracks.py --domain all

# Process single domain with custom time offset:
python filter_and_save_combined_tracks.py --domain d3 --time-offset 2.0
```

### Processing Order Summary

```bash
# 1. Environments (obs comparison)
python extract_cell_env1d_2location_preCI_met.py ./configs/CONFIG_NAME.yml
python combine_cell_env_2location.py ./configs/CONFIG_NAME.yml

# 2. Environments (main modeling analysis)
python extract_cell_env3d_preCI_met.py ./configs/CONFIG_NAME.yml
python calc_cell_avg_env.py ./configs/CONFIG_NAME.yml

# 3. Updraft statistics (requires avg_env output)
python calc_draft_stats_in_celltracks.py ./configs/CONFIG_NAME.yml
python make_largest_updraft_stats.py ./configs/CONFIG_NAME.yml

# 4. Tb/rain rate (independent)
python calc_wrf_tb_rainrate.py ../tracking/CONFIG_NAME.yml
python calc_sat_stats_in_celltracks.py ./configs/CONFIG_NAME.yml

# 5. Combine and filter (after all above complete)
python filter_and_save_combined_tracks.py --domain all
```

For detailed documentation of individual scripts, see [`docs/`](docs/).

---

## 3. Regridding (`/regrid/`)

Coarsen LASSO D3 (500 m) and D4 (100 m) subset files to match D2 (2.5 km) resolution for cross-resolution comparisons.

| Domain | Native Resolution | Regrid Ratio | Target |
|---|---|---|---|
| D3 | 500 m | 5 | 2.5 km |
| D4 | 100 m | 25 | 2.5 km |

**Key scripts:**
- `regrid_func.py` — Core functions: `coarsen_variable_filter()` (uniform_filter averaging), `coarsen_reflectivity_filter()` (dBZ→linear→average→dBZ)
- `regrid_lasso_batch.py` — Batch processing with Dask parallelization
- `regrid_d4_reflectivity.py` — Regrid D4 reflectivity fields
- `regrid_d4_celltracking_mask.py` — Regrid cell tracking masks (nearest-neighbor)
- `regrid_csapr_terrain.py` — Regrid CSAPR terrain/range mask to 2.5 km

See [regrid/README.md](regrid/README.md) for detailed usage instructions.

---

## 4. Analysis Notebooks (`/notebooks/`)

**Prerequisite**: Run `src/filter_and_save_combined_tracks.py` first to produce the combined filtered netCDF files.

### Major Notebooks (Paper Figures)

| Notebook | Description | Key Figures |
|---|---|---|
| `plot_domain_cellstats_timeseries_by_case_obs_lasso.ipynb` | Domain-mean cell statistics time series comparing CSAPR-2 obs vs LASSO per case (cell counts, mean reflectivity, echo-top heights) | Time series panels per case day |
| `plot_cell_trackstats_obs_lasso_with_tb.ipynb` | Joint KDE of cell max diameter vs max reflectivity, max 10-dBZ echo-top height, and min Tb across D2/D3/D4 vs obs | Multi-panel KDE joint distributions |
| `plot_celltrack_counts_map_obs_lasso.ipynb` | Cell track count spatial maps and per-date bar plots comparing obs vs simulations | Maps + bar charts |
| `plot_track_correlations_lasso_by_celldiam_avg_env_load_filtered.ipynb` | Multiple Linear Regression (MLR) scatter/correlation plots of updraft top height vs environmental predictors, binned by cell diameter | Multi-panel scatter with regression lines |
| `plot_narrow_wide_cell_composite_profiles_avg_env_load_filtered.ipynb` | Composite vertical profiles for narrow vs wide cells across resolutions; cell count bar plots by date | Profile plots + bar charts |
| `plot_track_correlations_lasso_by_celldiam_avg_env_RF.ipynb` | Random Forest feature importance analysis as alternative to MLR | Feature importance bar plots |

### Variant Notebooks

Notebooks with `*_regrid2.5km.ipynb` suffix recreate the same analyses using D3/D4 data that has been regridded (coarsened) to 2.5 km, enabling direct resolution-sensitivity comparisons at matched grid spacing.

### Python Environment Notes

- `plot_domain_cellstats_timeseries_by_case_obs_lasso.ipynb` **requires the py310 environment** (older xarray lazy-load behavior needed for separating different case dates).
- All other production notebooks use **py312**.

---

## How to Reproduce

1. **Set up environments** — Install conda environments from exported YAML files (see [Computing Environment](#computing-environment))

2. **Run cell tracking** — Generate configs and submit SLURM jobs:
   ```bash
   cd tracking/
   bash make_lasso_config_slurm_scripts_cu2.sh   # LASSO
   bash make_csapr_config_slurm_scripts.sh       # CSAPR-2
   ```

3. **Compute cell updraft statistics**:
   ```bash
   cd src/
   bash make_calc_stats_slurm_scripts_les.sh    # D3/D4
   bash make_calc_stats_slurm_scripts_meso.sh   # D2
   ```
   The SLURM template executes scripts in the order described in [Processing Order](#processing-order-summary).

4. **Extract pre-CI environments**:
   - Obs comparison: `extract_cell_env1d_2location_preCI_met.py` → `combine_cell_env_2location.py`
   - Modeling analysis: `extract_cell_env3d_preCI_met.py` → `calc_cell_avg_env.py`

5. **Make largest updraft stats** (requires outputs from steps 3 + 4):
   ```bash
   python make_largest_updraft_stats.py ./configs/CONFIG_NAME.yml
   ```

6. **Compute model Tb/rain rate** (independent of steps 4–5):
   ```bash
   python calc_wrf_tb_rainrate.py ../tracking/CONFIG_NAME.yml
   python calc_sat_stats_in_celltracks.py ./configs/CONFIG_NAME.yml
   ```

7. **(Optional) Regrid D3/D4 to 2.5 km**:
   ```bash
   cd regrid/
   python regrid_lasso_batch.py --case-date 20190123 --ensemble gefs18 \
       --start-hour 12 --end-hour 13 --file-type methamsl --domain d4
   ```

8. **Filter and combine all track data**:
   ```bash
   cd src/
   python filter_and_save_combined_tracks.py --domain all
   ```

9. **Run analysis notebooks** to generate paper figures (in `/notebooks/`).

---

## Detailed Documentation

For in-depth documentation of complex processing scripts, see the [`docs/`](docs/) directory:

- [Updraft Statistics](docs/calc_draft_stats.md) — Core labeling, VMF sorting, height-resolved statistics
- [Environment Extraction](docs/extract_cell_environments.md) — 3D/1D extraction, pre-CI sampling, coordinate systems
- [Largest Updraft Analysis](docs/make_largest_updraft_stats.md) — Object identification, top/base heights, linear fit
- [Environmental Variables Pipeline](docs/environmental_variables.md) — Full 3D→1D pipeline, thermodynamic calculations
- [Regridding Methodology](docs/regridding.md) — Coarsening approach, dBZ handling, ratios

---

## Contact

Zhe Feng — [zhe.feng@pnnl.gov](mailto:zhe.feng@pnnl.gov)  
Pacific Northwest National Laboratory
