#!/bin/bash
# Create LASSO cell statistics config and slurm scripts

submit_job="no"

# Specify configuration: 'base' or 'morr'
configuration="base"

config_dir="/ccsopen/home/zhe1feng1/program/lasso/cellgrowth/src/"
slurm_dir="/ccsopen/home/zhe1feng1/program/lasso/cellgrowth/src/"
# D4 (100m) 5min tracking
config_template=${config_dir}"config_lasso_wrf100m_template.yml"
# D4 (100m) 15min tracking
# config_template=${config_dir}"config_lasso_wrf100m_15min_template.yml"
# D3 (500m) tracking
# config_template=${config_dir}"config_lasso_wrf500m_template.yml"
# config_template=${config_dir}"config_lasso_wrf500m_15min_template.yml"
# D3 (500m) 5min tracking
# config_template=${config_dir}"config_lasso_wrf500m_template.yml"
slurm_template=${slurm_dir}"slurm_lasso_template.sh"
config_basename="config_lasso_"
slurm_basename="slurm_lasso_"

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
    # start_dates=("20181219" )
    # ens_members=("gefs10" )
    # start_dates=("20181205" "20190122" "20190129")
    # ens_members=("era5" "gefs18" "eda09")
    start_dates=(
        "20181204" "20181204" "20181204"
        "20181205" "20181205" "20181205"
        "20181219" "20181219"
        "20190122" "20190122"
        "20190123" "20190123"
        "20190125"
        "20190129" "20190129"
        "20190208"
    )
    ens_members=(
        "gefs04" "gefs16" "gefs18"
        "gefs01" "gefs02" "era5"
        "eda09" "gefs10"
        "eda00" "gefs18"
        "eda05" "eda07"
        "eda07"
        "gefs11" "eda09"
        "eda03"
    )
fi

# Loop over list
for ((i = 0; i < ${#start_dates[@]}; ++i)); do   
    sdate=${start_dates[$i]}
    # edate=${end_dates[$i]}
    edate="$((sdate+1))"
    ensmember=${ens_members[$i]}

    config_name=${config_basename}${sdate}_${ensmember}_${configuration}
    config_file=${config_dir}${config_name}.yml
    slurm_file=${slurm_dir}${slurm_basename}${sdate}_${ensmember}_${configuration}.sh

    sed "s/STARTDATE/"${sdate}"/g;s/ENDDATE/"${edate}/g";s/ENSMEMBER/"${ensmember}"/g;s/CONFIG/"${configuration}"/g" ${config_template} > ${config_file}
    sed "s/STARTDATE/"${sdate}"/g;s/ENSMEMBER/"${ensmember}"/g;s/CONFIG_NAME/"${config_name}"/g" ${slurm_template} > ${slurm_file}
    echo ${config_file}
    echo ${slurm_file}
    if [[ "${submit_job}" == "yes" ]]; then
        # echo ${slurm_file}
        sbatch ${slurm_file}
    fi
done