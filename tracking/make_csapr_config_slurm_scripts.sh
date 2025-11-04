#!/bin/bash
# Create CSAPR cell tracking config and slurm scripts

submit_job="yes"
# Specify resolution: '500m' or '2.5km'
resolution="2.5km"
# resolution="500m"

# Directory for templates (config and slurm)
template_dir="/ccsopen/home/zhe1feng1/program/lasso/cellgrowth/tracking/"
# Output directory for generated config and slurm scripts
output_dir="/ccsopen/home/zhe1feng1/program/lasso/cellgrowth/slurm/"
# Log directory for slurm output
log_dir="${output_dir}log/"
# PyFLEXTRKR code directory
pyflex_dir="/ccsopen/home/zhe1feng1/program/PyFLEXTRKR-dev/runscripts"

# Create output directories if they don't exist
mkdir -p "${output_dir}"
mkdir -p "${log_dir}"

# Slurm template
slurm_template=${template_dir}slurm_csapr${resolution}_template.sh
# Config template
config_template=${template_dir}config_csapr${resolution}_template.yml

# Full list of case days
start_dates=(
    "20181129" 
    "20181204" 
    "20181205" 
    "20181219" 
    "20190122" 
    "20190123"
    "20190125"
    "20190129"
    "20190208"
)

config_basename="config_csapr"${resolution}"_"
slurm_basename="slurm_csapr"${resolution}"_"

# Loop over case list
for ((i = 0; i < ${#start_dates[@]}; ++i)); do   
    sdate=${start_dates[$i]}
    # edate=${end_dates[$i]}
    edate="$((sdate+1))"
    # ensmember=${ens_members[$i]}

    config_name=${config_basename}${sdate}
    config_file=${output_dir}${config_name}.yml
    slurm_file=${output_dir}${slurm_basename}${sdate}.sh
    log_file=${log_dir}log_${config_name}.log

    sed "s|STARTDATE|$sdate|g; s|ENDDATE|$edate|g" ${config_template} > ${config_file}
    sed "s|STARTDATE|$sdate|g; s|CONFIG_NAME|$config_name|g; s|CONFIG_FILE|$config_file|g; s|FLEXTRKR_DIR|$pyflex_dir|g; s|LOG_FILE|$log_file|g" ${slurm_template} > ${slurm_file}
    echo ${config_file}
    echo ${slurm_file}
    if [[ "${submit_job}" == "yes" ]]; then
        # echo ${slurm_file}
        sbatch ${slurm_file}
    fi
done