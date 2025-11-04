#!/bin/bash
# Create LASSO cell tracking config and slurm scripts

submit_job="yes"
# Specify resolution: 'LES' or 'MESO'
# resolution="MESO"
# Specify configuration: 'base' or 'morr'
configuration="base"

# Directory for templates (config and slurm)
template_dir="/ccsopen/home/zhe1feng1/program/lasso/cellgrowth/tracking/"
# Output directory for generated config and slurm scripts
output_dir="/ccsopen/home/zhe1feng1/program/lasso/cellgrowth/tracking/slurm/"
# Log directory for slurm output
log_dir="${output_dir}log/"
# PyFLEXTRKR code directory
pyflex_dir="/ccsopen/home/zhe1feng1/program/PyFLEXTRKR-dev/runscripts"

# Create output directories if they don't exist
mkdir -p "${output_dir}"
mkdir -p "${log_dir}"

# D4 (100m) 5min tracking
# config_template=${template_dir}"config_lasso_wrf100m_template.yml"
# D4 (100m) 5min regrid to 2.5km tracking
config_template=${template_dir}"config_lasso_wrf100m_5min_regrid2.5km_template.yml"
# D4 (100m) 15min tracking
# config_template=${template_dir}"config_lasso_wrf100m_15min_template.yml"
# slurm_template=${template_dir}"slurm_lasso_wrf100m_template.sh"
# D4 (100m) 15min regrid to 2.5km tracking
# config_template=${template_dir}"config_lasso_wrf100m_15min_regrid2.5km_template.yml"
slurm_template=${template_dir}"slurm_lasso_wrf100m_template.sh"

# D3 (500m) 5min tracking
# config_template=${template_dir}"config_lasso_wrf500m_template.yml"
# D3 (500m) 5min regrid to 2.5km tracking
# config_template=${template_dir}"config_lasso_wrf500m_5min_regrid2.5km_template.yml"
# # D3 (500m) 15min tracking
# config_template=${template_dir}"config_lasso_wrf500m_15min_template.yml"
# slurm_template=${template_dir}"slurm_lasso_wrf500m_template.sh"
# D3 (500m) 15min regrid to 2.5km tracking
# config_template=${template_dir}"config_lasso_wrf500m_15min_regrid2.5km_template.yml"
# slurm_template=${template_dir}"slurm_lasso_wrf500m_template.sh"

# # D2 (2.5km) 5min tracking
# config_template=${template_dir}"config_lasso_wrf2.5km_template.yml"
# # D2 (2.5km) 15min tracking
# config_template=${template_dir}"config_lasso_wrf2.5km_15min_template.yml"
# slurm_template=${template_dir}"slurm_lasso_wrf2.5km_template.sh"

# config_basename="config_lasso_"
# slurm_basename="slurm_lasso_"
# Extract the filename using the basename command
config_fn="${config_template##*/}"
# Separate the string by "_" and get the first three parts
IFS="_" read -r part1 part2 part3 part4 rest <<< "$config_fn"
# Make basenames for config & slurm files
config_basename=config_${part2}_${part3}_
slurm_basename=slurm_${part2}_${part3}_

# Full list of case days
# Base runs
if [ ${configuration} == "base" ]
then
    start_dates=(
        "20181129" "20181129" "20181129" "20181129"
        "20181204" "20181204" 
        "20181205" 
        "20181219" 
        "20190122" "20190122"
        "20190123" "20190123"
        "20190125" "20190125"
        "20190129" "20190129"
        "20190208" "20190208"
    )
    # Long ensemble member names (for directory names)
    ens_members=(
        "gefs00" "gefs03" "gefs09" "gefs18" 
        "gefs18" "gefs19" 
        "gefs01" 
        "eda09" 
        "gefs01" "gefs18"
        "eda05" "gefs18"
        "eda07" "gefs11"
        "eda09" "gefs11"
        "eda03" "eda08"
    )
fi
# Morrison runs
if [ ${configuration} == "morr" ]
then
    start_dates=(
        "20181204" "20181204" "20181204"
        "20181205" "20181205" "20181205"
        "20181219" "20181219" "20181219" 
        "20190122" "20190122"
        "20190123" "20190123"
        "20190125"
        "20190129" "20190129"
        "20190208"
    )
    ens_members=(
        "gefs04" "gefs16" "gefs18" 
        "gefs01" "gefs02" "era5" 
        "eda09" "gefs10" "gefs11"
        "eda00" "gefs18"
        "eda05" "eda07"
        "eda07"
        "gefs11" "eda09"
        "eda03"
    )
fi

# # MESO ensemble member names
# if [ ${resolution} == "MESO" ]
# then
#     # Base member names
#     if [ ${configuration} == "base" ]
#     then
#         ens_members=(
#             "gefs_en00" "gefs_en03" 
#             "gefs_en18" "gefs_en19" 
#             "gefs_en01" 
#             "eda_en09" 
#             "gefs_en01" 
#             "eda_en05"
#             "eda_en07" "gefs_en11"
#             "eda_en09" "gefs_en11"
#             "eda_en03" "eda_en08"
#         )
#     fi
#     # Morrison member names
#     if [ ${configuration} == "morr" ]
#     then
#     fi
# fi

# Loop over case list
for ((i = 0; i < ${#start_dates[@]}; ++i)); do   
    sdate=${start_dates[$i]}
    # edate=${end_dates[$i]}
    edate="$((sdate+1))"
    ensmember=${ens_members[$i]}

    config_name=${config_basename}${sdate}_${ensmember}_${configuration}
    config_file=${output_dir}${config_name}.yml
    slurm_file=${output_dir}${slurm_basename}${sdate}_${ensmember}_${configuration}.sh
    log_file=${log_dir}log_${config_name}.log

    sed "s|STARTDATE|$sdate|g; s|ENDDATE|$edate|g; s|ENSMEMBER|$ensmember|g; s|CONFIG|$configuration|g" ${config_template} > ${config_file}
    sed "s|STARTDATE|$sdate|g; s|ENSMEMBER|$ensmember|g; s|CONFIG_FILE|$config_file|g; s|FLEXTRKR_DIR|$pyflex_dir|g; s|LOG_FILE|$log_file|g" ${slurm_template} > ${slurm_file}
    echo ${config_file}
    echo ${slurm_file}
    if [[ "${submit_job}" == "yes" ]]; then
        # echo ${slurm_file}
        sbatch ${slurm_file}
    fi
done