"""
Make slurm scripts for domain mean cellstats timeseries.
"""
__author__ = "Zhe.Feng@pnnl.gov"
import numpy as np
import textwrap
import subprocess

if __name__ == "__main__":

    code_dir = '/ccsopen/home/zhe1feng1/program/lasso/cellgrowth/src/'
    slurm_dir = code_dir
    code_name = f'calc_domain_mean_cellstats_timeseries.py'

    # Submit slurm job
    submit_job = True

    # domain = "wrf2.5km"
    # domain = "wrf500m"
    # domain = "wrf100m"
    # domain = "csapr500m"
    domain = "csapr2.5km"

    # Specify configuration: 'base' or 'morr'
    configuration = "base"

    # Config file directory (from tracking)
    config_dir = "/ccsopen/home/zhe1feng1/program/lasso/cellgrowth/tracking/"
    slurm_dir = "/ccsopen/home/zhe1feng1/program/lasso/cellgrowth/src/job_scripts/"
    # Config/slurm file basename
    if "csapr" in domain:
        config_basename = f"config_{domain}_"
        slurm_basename = f"slurm_{domain}_"
    else:
        config_basename = f"config_lasso_{domain}_"
        slurm_basename = f"slurm_lasso_{domain}_"


    # Full list of runs
    if configuration == "base":
        start_dates = [
            "20181129", "20181129", "20181129", "20181129",
            "20181204", "20181204",
            "20181205", 
            "20181219", 
            "20190122", "20190122", 
            "20190123", "20190123",
            "20190125", "20190125",
            "20190129", "20190129",
            "20190208", "20190208",
        ]
        ens_members = [
            "gefs00", "gefs03", "gefs09", "gefs18",
            "gefs18", "gefs19",
            "gefs01", 
            "eda09", 
            "gefs01", "gefs18",
            "eda05", "gefs18",
            "eda07", "gefs11",
            "eda09", "gefs11",
            "eda03", "eda08",
        ]

    if "csapr" in domain:
        start_dates = np.unique(start_dates)
        # import pdb; pdb.set_trace()

    # Loop over date list
    for ii in range(len(start_dates)):
        sdate = start_dates[ii]
        ensmember = ens_members[ii]

        if "csapr" in domain:
            config_name = f'{config_basename}{sdate}'
            config_filename = f'{config_dir}{config_name}.yml'
            slurm_filename = f'{slurm_dir}{slurm_basename}{sdate}.sh'
            # import pdb; pdb.set_trace()
        else:
            config_name = f'{config_basename}{sdate}_{ensmember}_{configuration}'
            config_filename = f'{config_dir}{config_name}.yml'
            slurm_filename = f'{slurm_dir}{slurm_basename}{sdate}_{ensmember}.sh'

        # Create a SLURM submission script
        slurm_file = open(slurm_filename, "w")
        text = f"""\
            #!/bin/bash
            #SBATCH -A atm123
            #SBATCH -J {sdate}{ensmember}
            #SBATCH --time=00:10:00
            #SBATCH --nodes=1
            #SBATCH --ntasks=128
            #SBATCH -p batch_all
            ##SBATCH -p batch_high_memory
            #SBATCH --exclusive
            #SBATCH --output=logs/log_lasso_{domain}_{sdate}_{ensmember}.log
            #SBATCH --mail-type=END
            #SBATCH --mail-user=zhe.feng@pnnl.gov

            date
            source /ccsopen/home/zhe1feng1/.bashrc
            conda activate /ccsopen/home/zhe1feng1/anaconda3/envs/flextrkr

            export OMP_NUM_THREADS=128
            export NUMEXPR_MAX_THREADS=128

            python {code_dir}{code_name} {config_filename}
            date
        """
        slurm_file.writelines(textwrap.dedent(text))
        slurm_file.close()
        print(slurm_filename)

        # Run command
        if submit_job == True:
            cmd = f'sbatch {slurm_filename}'
            # print(cmd)
            subprocess.run(f'{cmd}', shell=True)

        # import pdb; pdb.set_trace()