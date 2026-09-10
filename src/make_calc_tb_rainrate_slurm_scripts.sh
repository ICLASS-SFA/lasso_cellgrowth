#!/bin/bash
# Create LASSO tb & rainrate calculation slurm scripts

# Flag to submit slurm jobs
submit_job="yes"

# Flag to make new config & slurm scripts (typically "yes")
# make_config="yes"
make_slurm="yes"

domain="wrf2.5km"
# domain="wrf500m"
# domain="wrf100m"

# Specify configuration: 'base' or 'morr'
configuration="base"

# Config file directory (from tracking)
config_dir="/ccsopen/home/zhe1feng1/program/lasso/cellgrowth/tracking/"
slurm_dir="/ccsopen/home/zhe1feng1/program/lasso/cellgrowth/src/"
# Config/slurm file basename
config_basename="config_lasso_${domain}_"
slurm_basename="slurm_lasso_${domain}_"

# Slurm template
slurm_template=${slurm_dir}"slurm_lasso_template.sh"

# Base runs
if [ ${configuration} == "base" ]
then
    # start_dates=("20190123")
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
    # ens_members=("gefs18")
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

# Loop over list
for ((i = 0; i < ${#start_dates[@]}; ++i)); do   
    sdate=${start_dates[$i]}
    # edate=${end_dates[$i]}
    edate="$((sdate+1))"
    ensmember=${ens_members[$i]}

    config_name=${config_basename}${sdate}_${ensmember}_${configuration}
    config_file=${config_dir}${config_name}.yml
    slurm_file=${slurm_dir}${slurm_basename}${sdate}_${ensmember}_${configuration}.sh

    if [[ "${make_slurm}" == "yes" ]]; then
        sed "s/STARTDATE/"${sdate}"/g;s/ENSMEMBER/"${ensmember}"/g;s/CONFIG_NAME/"${config_name}"/g" ${slurm_template} > ${slurm_file}
    fi
    # echo ${config_file}
    echo ${slurm_file}
    if [[ "${submit_job}" == "yes" ]]; then
        # echo ${slurm_file}
        sbatch ${slurm_file}
    fi
done