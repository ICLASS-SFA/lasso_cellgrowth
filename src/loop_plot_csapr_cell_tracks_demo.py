"""
Loop over CSAPR dates to plot quicklook plots for LASSO cases.
"""
import subprocess
import os

if __name__ == "__main__":

    # Specify resolution: 'les' or 'meso'
    resolution = '500m'

    # Flag to call Python codes to make plots
    make_plots = True
    make_mp4 = True

    # Full set of runs
    # start_dates = [
    #     "20181129",
    # ]
    start_dates = [
        # "20181129",
        "20181204",
        "20181205", 
        "20181219", 
        "20190122", 
        "20190123",
        "20190125",
        "20190129",
        "20190208",
    ]

    # Plotting code name
    code_name = '/ccsopen/home/zhe1feng1/program/lasso/cellgrowth/src/plot_subset_cell_tracks_ETH_terrain.py'
    # config_filename = f'../config/config_csapr{resolution}_cu2.yml'
    config_basename = f'/ccsopen/home/zhe1feng1/program/lasso/cellgrowth/tracking/config_csapr{resolution}_'
    out_dir_root = '/gpfs/wolf2/arm/atm131/proj-shared/zfeng/cacti/csapr/quicklooks_trackpaths_terrain/'
    # out_dir_root = '/gpfs/wolf2/arm/atm131/proj-shared/zfeng/cacti/csapr/quicklooks_trackpaths_ETH/'
    # out_dir_root = '/gpfs/wolf2/arm/atm131/proj-shared/zfeng/cacti/csapr/quicklooks_trackpaths_ETH_terrain/'

    # Output animation directory
    out_dir_mp4 = f'{out_dir_root}animation/'
    os.makedirs(out_dir_mp4, exist_ok=True)

    # figbasename = f'csapr{resolution}_eth10dbz_'
    figbasename = f'csapr{resolution}_dbz_comp_'
    figsize = [8,7]
    extent = [-65.9, -63.6, -33.1, -31.15]
    radar_lon, radar_lat = -64.7284, -32.1264
    vfscale = '1200:-1'
    parallel = 1
    framerate = 2  # for 15min

    # Loop over dates
    for ii in range(0, len(start_dates)):
        idate = start_dates[ii]
        year = idate[0:4]
        month = idate[4:6]
        day = idate[6:8]
        sdate = f'{year}-{month}-{day}T12:00'
        edate = f'{year}-{month}-{day}T23:55'
        config = f'{config_basename}{idate}.yml'
        out_dir = f'{out_dir_root}/{idate}/'
        # Make quicklook plots
        cmd = f'python {code_name} -s {sdate} -e {edate} -c {config} -p {parallel} --radar_lat {radar_lat} --radar_lon {radar_lon} ' + \
              f'--output {out_dir} --figbasename {figbasename} --figsize {figsize[0]} {figsize[1]} --extent {extent[0]} {extent[1]} {extent[2]} {extent[3]}'
        if make_plots == True:
            print(cmd)
            subprocess.run(cmd, shell=True)

        # Make animation
        video_filename = f'{out_dir_mp4}{figbasename}{idate}.mp4'
        # Make ffmpeg command
        cmd = f"ffmpeg -framerate {framerate} -pattern_type glob -i '{out_dir}{figbasename}*.png' -c:v libx264 -r 10 -crf 20 -pix_fmt yuv420p -vf scale={vfscale} -y {video_filename}"
        # print(cmd)
        if make_mp4 == True:
            print(video_filename)
            subprocess.run(cmd, shell=True)

    