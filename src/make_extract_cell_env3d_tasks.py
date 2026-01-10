"""
Make task list and slurm scripts for extracting cell 3D environments.
"""
__author__ = "Zhe.Feng@pnnl.gov"
import numpy as np
import xarray as xr
import yaml
import textwrap
import subprocess

if __name__ == "__main__":

    code_dir = '/ccsopen/home/zhe1feng1/program/lasso/cellgrowth/src/'
    slurm_dir = code_dir
    code_name = f'extract_cell_env3d_preCI_bytracks.py'
    config_basename = f'config_lasso_'
    slurm_basename = f'slurm_lasso_'

    # Number of jobs allowed to run simultaneously in job array
    # njobs_run = 60
    # Submit slurm job
    submit_job = False

    # Number of tracks to process per part
    ntracks_part = 50
    # Set the number of digits for 0 padding
    # This should be set to the digit for the maximum number of tracks
    # e.g., ntracks = 1387, digits = 4
    digits = 4

    # Full list of runs
    start_dates = [
        "20181129", "20181129",
        "20181204", "20181204",
        "20181205", 
        "20181219", 
        "20190122", 
        "20190123",
        "20190125", "20190125",
        "20190129", "20190129",
        "20190208", "20190208",
    ]
    # LES
    ens_members = [
        "gefs00", "gefs03",
        "gefs18", "gefs19",
        "gefs01", 
        "eda09", 
        "gefs01", 
        "eda05",
        "eda07", "gefs11",
        "eda09", "gefs11",
        "eda03", "eda08",
    ]

    # Loop over list
    # for ii in range(len(start_dates)):
    for ii in range(1, 2):
        sdate = start_dates[ii]
        ensmember = ens_members[ii]

        config_name = f'{config_basename}{sdate}_{ensmember}.yml'
        slurm_filename = f'{slurm_dir}{slurm_basename}{sdate}_{ensmember}.sh'

        # Get inputs from configuration file
        stream = open(config_name, 'r')
        config = yaml.full_load(stream)
        stats_path = config['stats_path']
        startdate = config['startdate']
        enddate = config['enddate']
        # Track statistics file
        stats_filebase = 'trackstats_'
        trackstats_file = f'{stats_path}{stats_filebase}{startdate}_{enddate}.nc'

        # Read track stats file
        dsm = xr.open_dataset(trackstats_file)
        ntracks_all = dsm.sizes['tracks']

        # Number of parts
        nparts = np.floor(ntracks_all / ntracks_part).astype(int)
        # Make a list for track start/end 
        track_start = []
        track_end = []
        for ii in range(0, nparts): 
            track_start.append(ii*ntracks_part)
            track_end.append((ii+1)*ntracks_part-1)
        # Add the last part to the list
        track_start.append(track_end[-1]+1)
        track_end.append(ntracks_all)
        # Update total number of parts
        nparts = len(track_start)

        # Create the list of job tasks needed by SLURM...
        # syear = track_period[0:4]
        task_filename = f'tasks_lasso_{sdate}_{ensmember}.txt'
        task_file = open(task_filename, "w")
        ntasks = 0
        # print(varname, '->', basename)
        for ipart in range(0, nparts): 
            cmd = f'python {code_name} {config_name} ' \
                  f'{track_start[ipart]} {track_end[ipart]} {digits}'
            # print(cmd)
            task_file.write(f"{cmd}\n")
            ntasks += 1
        task_file.close()
        print(task_filename)

         # Create a SLURM submission script for the above task list...
        slurm_file = open(slurm_filename, "w")
        text = f"""\
            #!/bin/bash
            #SBATCH -A atm123
            #SBATCH -J {sdate}{ensmember}
            #SBATCH --time=03:00:00
            #SBATCH --nodes=1
            #SBATCH --ntasks=128
            ##SBATCH -p batch_all
            #SBATCH -p batch_high_memory
            #SBATCH --exclusive
            #SBATCH --output=log_lasso_{sdate}_{ensmember}_%A_%a.log
            #SBATCH --mail-type=END
            #SBATCH --mail-user=zhe.feng@pnnl.gov
            #SBATCH --array=1-{ntasks}

            date
            # export OMP_NUM_THREADS=128
            export NUMEXPR_MAX_THREADS=128

            # conda activate /ccsopen/home/zhe1feng1/anaconda3/envs/py310
            cd {code_dir}

            # Takes a specified line ($SLURM_ARRAY_TASK_ID) from the task file
            LINE=$(sed -n "$SLURM_ARRAY_TASK_ID"p {task_filename})
            echo $LINE
            # Run the line as a command
            $LINE

            date
        """
        slurm_file.writelines(textwrap.dedent(text))
        slurm_file.close()
        print(slurm_filename)

        # Run command
        if submit_job == True:
            cmd = f'sbatch --array=1-{ntasks} {slurm_filename}'
            print(cmd)
            subprocess.run(f'{cmd}', shell=True)

        # import pdb; pdb.set_trace()