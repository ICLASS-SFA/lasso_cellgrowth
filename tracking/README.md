# **LASSO Cell Tracking Setup**

---

Scripts to create cell tracking config files for LASSO CACTI simulations.

* **Make LASSO config files and slurm job scripts:**

`bash make_lasso_config_slurm_scripts_cu2.sh`

* **Make CSAPR config files and slurm job scripts:**

`bash make_csapr_config_slurm_scripts_cu2.sh`

---

Modify the shell scripts above to point to the desired config templates and slurm templates to generate config files and slurm scripts for specific cases. Set `submit_job="yes"` in the shell script to submit the slurm jobs immediately.

There are 3 different tracking config templates for 3 resolutions:

For **5-min** tracking:

* `config_lasso_wrf2.5km_template.yml`
* `config_lasso_wrf500m_template.yml`
* `config_lasso_wrf100m_template.yml`

* `config_lasso_wrf500m_5min_regrid2.5km_template.yml`
* `config_lasso_wrf100m_5min_regrid2.5km_template.yml`

For **15-min** tracking:

* `config_lasso_wrf2.5km_15min_template.yml`
* `config_lasso_wrf500m_15min_template.yml`
* `config_lasso_wrf100m_15min_template.yml`

There are also 3 different slurm templates for 3 resolutions. The main difference is the cell tracking run script used:

* `slurm_lasso_wrf2.5km_template.sh`
* `slurm_lasso_wrf500m_template.sh`
* `slurm_lasso_wrf100m_template.sh`