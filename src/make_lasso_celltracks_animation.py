"""
Make animations from LASSO cell tracking quicklook plots.
"""
import os
import sys
import subprocess

if __name__ == "__main__":

    # Specify resolution: 'les' or 'meso'
    resolution = 'meso'
    # Specify domain and framerate
    domain = 'd2'
    # domain = 'd3'
    # domain = 'd4'
    # domain = 'd4_15min'

    # start_dates = [
    #     "20181129",
    # ]
    # ens_members = [
    #     "gefs00",
    # ]

    # Full set of runs
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
    if resolution == 'les':
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
    elif resolution == 'meso':
        # MESO
        ens_members = [
            "gefs_en00", "gefs_en03",
            "gefs_en18", "gefs_en19",
            "gefs_en01", 
            "eda_en09", 
            "gefs_en01", 
            "eda_en05",
            "eda_en07", "gefs_en11",
            "eda_en09", "gefs_en11",
            "eda_en03", "eda_en08",
        ]
    else:
        sys.exit('Unknown resolution!')

    # Input figures root directory
    in_dir_root = f'/gpfs/wolf2/arm/atm131/proj-shared/zfeng/cacti/{resolution}/quicklooks_trackpaths/'
    # Output animation directory
    out_dir = f'{in_dir_root}animation/'
    os.makedirs(out_dir, exist_ok=True)

    # Determine framerate based on domain
    if domain == 'd4':
        framerate = 6  # for 5min
    elif (domain == 'd4_15min') | (domain == 'd3') | (domain == 'd2'):
        framerate = 2  # for 15min

    # Loop over dates
    for ii in range(0, len(start_dates)):
        idate = start_dates[ii]
        imember = ens_members[ii]
        in_dir = f'{in_dir_root}{idate}/{imember}/{domain}/'
        out_filename = f'{out_dir}{idate}_{imember}_{domain}.mp4'
        # Make ffmpeg command
        cmd = f"ffmpeg -framerate {framerate} -pattern_type glob -i '{in_dir}*.png' -c:v libx264 -r 10 -crf 20 -pix_fmt yuv420p -y {out_filename}"
        print(cmd)
        # Remove output file if it exists
        if os.path.isfile(out_filename):
            os.remove(out_filename)
        # Run command
        subprocess.run(f'{cmd}', shell=True)
        print(out_filename)