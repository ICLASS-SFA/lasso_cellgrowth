"""
Calculates cloud statistics from satellite data for tracked convective cells.
The cloud statistics are written to netCDF file matching the cell track statistics file format.
"""
import numpy as np
import os, sys, glob
import time, datetime, calendar
from pytz import utc
import yaml
import xarray as xr
import dask
from dask.distributed import Client, LocalCluster

#########################################################
def calc_basetime(filelist, filebase):
    """
    Calculates basetime (Epoch Time) and a filename dictionary from a list of filenames.
    The basetime uses year, month, day, hour, minute but set all seconds to 0.

    Parameters:
    ===========
    filelist: list
        A list of input file names
    filebase: string
        Basename of the files

    Returns:
    ===========
    basetime: int array
        Epoch time corresponding to the input files.
    file_dict: dictionary
        Direction key by basetime and value is the file names
    """
    nfiles = len(filelist)
    prelength = len(filebase)
    basetime = np.full(nfiles, -999, dtype=int)
    file_dict = {}
    for ifile in range(nfiles):
        fname = os.path.basename(filelist[ifile])
        # File name format: basename_yyyymmdd.hhmm
        TEMP_filetime = datetime.datetime(int(fname[prelength:(prelength+4)]), 
                                            int(fname[prelength+4:(prelength+6)]), 
                                            int(fname[prelength+6:(prelength+8)]),
                                            int(fname[prelength+9:(prelength+11)]), 
                                            int(fname[prelength+11:(prelength+13)]), 0, tzinfo=utc)
        basetime[ifile] = calendar.timegm(TEMP_filetime.timetuple())
        file_dict[basetime[ifile]] = filelist[ifile]
    return basetime, file_dict

#########################################################
def calc_sat_cellstats_singlefile(
    pixel_filename, 
    sat_filename, 
    idx_track, 
    pixel_radius
    ):
    """
    Calculates cell statistics from matching satellite data in a single pixel file.

    Parameters:
    ===========
    pixel_filename: string
        Input cell pixel file name
    sat_filename: string
        Input satellite pixel file name
    idx_track: array
        Track indices in the current pixel file
    pixel_radius: float
        Pixel size.

    Returns:
    ===========
    nmatchcloud: <int>
        Number of matched cells in this file
    cell_area: <array>
        Cell area for each matched cell
    ctt_min: <array>
        Minimum cloud-top temperature for each matched cell
    
    If no matched cell exists in the file, returns None.
    """

    # Check if satellite data file exist
    if os.path.isfile(sat_filename):
        print(sat_filename)

        # Read satellite file
        dsv = xr.open_dataset(sat_filename)
        tir = dsv['temperature_ir'].squeeze().values
        ctt = dsv['cloud_top_temperature'].squeeze().values
        cth = dsv['cloud_top_height'].squeeze().values
        ctp = dsv['cloud_top_pressure'].squeeze().values
        # cep = dsv['cloud_effective_pressure'].squeeze().values
        phase = dsv['cloud_phase'].squeeze().values
        lwp_iwp = dsv['cloud_lwp_iwp'].squeeze().values
        dsv.close()

        # Read pixel-level track file
        ds = xr.open_dataset(pixel_filename, decode_times=False)
        # cloudid_basetime = ds['base_time'].values
        # tracknumbermap = ds['tracknumber'].squeeze().values
        tracknumbermap_cmask = ds['tracknumber_cmask'].squeeze().values
        ds.close()

        # Create arrays for output statistics
        nmatchcloud = len(idx_track)
        cell_area = np.full((nmatchcloud), np.nan, dtype=np.float32)
        ctt_min = np.full((nmatchcloud), np.nan, dtype=np.float32)
        tir_min = np.full((nmatchcloud), np.nan, dtype=np.float32)
        cth_max = np.full((nmatchcloud), np.nan, dtype=np.float32)
        ctp_min = np.full((nmatchcloud), np.nan, dtype=np.float32)
        area_liq = np.full((nmatchcloud), np.nan, dtype=np.float32)
        area_ice = np.full((nmatchcloud), np.nan, dtype=np.float32)
        lwp_max = np.full((nmatchcloud), np.nan, dtype=np.float32)
        iwp_max = np.full((nmatchcloud), np.nan, dtype=np.float32)

        if (nmatchcloud > 0):
            
            # Loop over each match tracked cloud
            for imatchcloud in range(nmatchcloud):

                # Intialize array for keeping data associated with the tracked cell
                # filtered_ctt = np.full((ydim, xdim), np.nan, dtype=float)

                # Track number needs to add 1
                itracknum = idx_track[imatchcloud] + 1

                # Count the number of pixels for the original cell mask
                inpix_cloud = np.count_nonzero(tracknumbermap_cmask == itracknum)

                # Index location of original cell mask
                idx_cloudorig = np.where(tracknumbermap_cmask == itracknum)
                # Index location of dilated cell mask
                # idx_clouddilated = np.where(tracknumbermap == itracknum)
                # Cell mask with liquid/ice phase
                idx_liq = np.where((tracknumbermap_cmask == itracknum) & (phase == 1))
                idx_ice = np.where((tracknumbermap_cmask == itracknum) & (phase == 2))

                # Get location indices of the cloud
                # icloudlocationy, icloudlocationx = np.where(tracknumbermap_cmask == itracknum)
                # inpix_cloud = len(icloudlocationy)

                # Proceed if the number matching cloud pixel > 0
                if inpix_cloud > 0:

                    # Subset variables to the current cell mask (original)
                    sub_ctt = ctt[idx_cloudorig]
                    sub_tir = tir[idx_cloudorig]
                    sub_cth = cth[idx_cloudorig]
                    sub_ctp = ctp[idx_cloudorig]
                    sub_phase = phase[idx_cloudorig]

                    # Subset variables to the current cell mask (dilated)
                    # sub_ctt = ctt[idx_clouddilated]
                    # sub_tir = tir[idx_clouddilated]
                    # sub_cth = cth[idx_clouddilated]
                    # sub_ctp = ctp[idx_clouddilated]
                    # sub_phase = phase[idx_clouddilated]
                    
                    cell_area[imatchcloud] = inpix_cloud * pixel_radius**2

                    # Calculate new statistics of the cloud
                    # Minimum/Maximum cloud-top variables
                    ctt_min[imatchcloud] = np.nanmin(sub_ctt)
                    tir_min[imatchcloud] = np.nanmin(sub_tir)
                    cth_max[imatchcloud] = np.nanmax(sub_cth)
                    ctp_min[imatchcloud] = np.nanmin(sub_ctp)
                    # Area with liquid/ice from cloud phase flags
                    # 0=clear with snow/ice, 1=water, 2=ice, 3=no retrieval, 4=clear, 5=bad retrieval, 6=weak water, 7=weak ice
                    area_liq[imatchcloud] = np.count_nonzero(sub_phase == 1) * pixel_radius**2
                    area_ice[imatchcloud] = np.count_nonzero(sub_phase == 2) * pixel_radius**2

                    if (len(idx_liq[0]) > 0):
                        lwp_max[imatchcloud] = np.nanmax(lwp_iwp[idx_liq])
                    if (len(idx_ice[0]) > 0):
                        iwp_max[imatchcloud] = np.nanmax(lwp_iwp[idx_ice])

            # Group outputs in dictionaries
            out_dict = {
                # "nmatchcloud": nmatchcloud,
                "cell_area": cell_area, 
                "cloud_top_temperature_min": ctt_min,
                "temperature_ir_min": tir_min,
                "cloud_top_height_max": cth_max,
                "cloud_top_pressure_min": ctp_min,
                "area_liquid": area_liq,
                "area_ice": area_ice,
                "lwp_max": lwp_max,
                "iwp_max": iwp_max,
            }
            out_dict_attrs = {
                # "nmatchcloud": nmatchcloud,
                "cell_area": {
                    "long_name": "Area of the convective cell in a track",
                    "units": "km^2",
                }, 
                "cloud_top_temperature_min": {
                    "long_name": "Minimum cloud top temperature in a track",
                    "units": "K",
                }, 
                "temperature_ir_min": {
                    "long_name": "Minimum IR temperature in a track",
                    "units": "K",
                }, 
                "cloud_top_height_max": {
                    "long_name": "Maximum cloud top height in a track",
                    "units": "km",
                }, 
                "cloud_top_pressure_min": {
                    "long_name": "Minimum cloud top pressure in a track",
                    "units": "hPa",
                }, 
                "area_liquid": {
                    "long_name": "Area of liquid cloud-top in a track",
                    "units": "km^2",
                }, 
                "area_ice": {
                    "long_name": "Area of ice cloud-top in a track",
                    "units": "km^2",
                }, 
                "lwp_max": {
                    "long_name": "Maximum liquid water path in a track",
                    "units": "g/m^2",
                }, 
                "iwp_max": {
                    "long_name": "Maximum ice water path in a track",
                    "units": "g/m^2",
                }, 
            }
            return out_dict, out_dict_attrs


#########################################################
if __name__ == '__main__':

    # Get configuration file name from input
    config_file = sys.argv[1]
    # Read configuration from yaml file
    stream = open(config_file, 'r')
    config = yaml.full_load(stream)

    run_parallel = config['run_parallel']
    n_workers = config['n_workers']
    threads_per_worker = config['threads_per_worker']
    startdate = config['startdate']
    enddate = config['enddate']
    time_window = config['time_window']
    stats_path = config['stats_path']
    pixelfile_path = config['pixelfile_path']
    satfile_path = config['satfile_path']

    output_path = stats_path

    # Input file basenames
    stats_filebase = 'trackstats_'
    pixel_filebase = 'celltracks_'
    sat_filebase = 'corvisstpx2drectg16v4minnisX1.regrid2csapr2gridded.c1.'

    # Output statistics filename
    output_filename = f'{output_path}stats_goes16_{startdate}_{enddate}.nc'

    # Track statistics file dimension names
    tracks_dimname = 'tracks'
    times_dimname = 'times'

    # Track statistics file
    trackstats_file = f'{stats_path}{stats_filebase}{startdate}_{enddate}.nc'
    # Find all pixel-level files
    pixelfilelist = sorted(glob.glob(f'{pixelfile_path}{pixel_filebase}*.nc'))
    nfiles = len(pixelfilelist)
    # Find all satellite files
    satfilelist = sorted(glob.glob(f'{satfile_path}{sat_filebase}*.nc'))
    nsatfiles = len(satfilelist)
    
    # Get basetime and from the satellite and pixel files
    sat_basetime, satfile_dict = calc_basetime(satfilelist, sat_filebase)
    pixel_basetime, pixelfile_dict = calc_basetime(pixelfilelist, pixel_filebase)

    # Find matching satellite files for each pixel file
    match_satfilelist = [''] * nfiles
    for ifile in range(nfiles):
        # Find satellite time closest to the pixel file time and get the index
        # Save the filename if time difference is < time_window
        idx = np.argmin(np.abs(sat_basetime - pixel_basetime[ifile]))        
        if np.abs(sat_basetime[idx] - pixel_basetime[ifile]) < time_window:
            match_satfilelist[ifile] = satfilelist[idx]
        else:
            print(f'No match file found for: {pixelfilelist[ifile]}')


    # Read track statistics file
    print(trackstats_file)
    dsstats = xr.open_dataset(trackstats_file, decode_times=False)
    ntracks = dsstats.dims[tracks_dimname]
    ntimes = dsstats.dims[times_dimname]
    stats_basetime = dsstats['base_time']
    cell_area = dsstats['cell_area']
    pixel_radius = dsstats.attrs['pixel_radius_km']
    dsstats.close()

    print(f'Total Number of Tracks: {ntracks}')
    

    ##############################################################
    # Call function to calculate statistics
    trackindices_all = []
    timeindices_all = []
    final_results = []
    if run_parallel==0:
        # Loop over each pixel-file and call function to calculate
        for ifile in range(nfiles):
            # Find all matching time indices from track stats file to the current pixel file
            matchindices = np.array(
                np.where(np.abs(stats_basetime.values - pixel_basetime[ifile]) < time_window)
            )
            # The returned match indices are for [tracks, times] dimensions respectively
            idx_track = matchindices[0]
            idx_time = matchindices[1]

            if len(idx_track) > 0:
                # Save matchindices for the current pixel file to the overall list
                trackindices_all.append(idx_track)
                timeindices_all.append(idx_time)

                iresult = calc_sat_cellstats_singlefile(
                            pixelfilelist[ifile], 
                            match_satfilelist[ifile],
                            idx_track, 
                            pixel_radius
                            )
                final_results.append(iresult)
                # import pdb; pdb.set_trace()

    elif run_parallel==1:
        print(f'Parallel version by dask')
        
        # Initialize dask
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker)
        client = Client(cluster)

        # Loop over each pixel-file and call function to calculate
        for ifile in range(nfiles):
            # Find all matching time indices from robust MCS stats file to the current pixel file
            matchindices = np.array(
                np.where(np.abs(stats_basetime.values - pixel_basetime[ifile]) < time_window)
            )
            # The returned match indices are for [tracks, times] dimensions respectively
            idx_track = matchindices[0]
            idx_time = matchindices[1]

            if len(idx_track) > 0:
                # Save matchindices for the current pixel file to the overall list
                trackindices_all.append(idx_track)
                timeindices_all.append(idx_time)

                iresult = dask.delayed(calc_sat_cellstats_singlefile)(
                            pixelfilelist[ifile], 
                            match_satfilelist[ifile], 
                            idx_track, 
                            pixel_radius
                            )
                final_results.append(iresult)
            
        # Collect results from Dask
        print("Computing statistics ...")
        final_results = dask.compute(*final_results)
    

    # Make a variable list from one of the returned dictionaries
    var_names = list(final_results[0][0].keys())
    # Get variable attributes from one of the returned dictionaries
    var_attrs = final_results[0][1]

    # Loop over variable list to create the dictionary entry
    out_dict = {}
    out_dict_attrs = {}
    for ivar in var_names:
        out_dict[ivar] = np.full((ntracks, ntimes), np.nan, dtype=np.float32)
        out_dict_attrs[ivar] = var_attrs[ivar]

    # The number of returned results
    nresults = len(final_results)

    # Now that all calculations for each pixel file is done, put the results back to the tracks format
    # Loop over the returned statistics list
    for ifile in range(nresults):
        # Get the results from the current file
        vars = final_results[ifile]
        if (vars is not None):
            # Get the return results for this pixel file
            # The result is a tuple: (out_dict, out_dict_attrs)
            # The first entry is the dictionary containing the variables
            iResult = final_results[ifile][0]

            # Get trackindices and timeindices for this file
            trackindices = trackindices_all[ifile]
            timeindices = timeindices_all[ifile]

            # Loop over each variable and assign values to output dictionary
            for ivar in var_names:
                out_dict[ivar][trackindices,timeindices] = iResult[ivar]


    ##################################
    # Write to netcdf
    print('Writing output netcdf ... ')

    # Define variable list
    varlist = {}
    # Define output variable dictionary
    for key, value in out_dict.items():
        varlist[key] = ([tracks_dimname, times_dimname], value, out_dict_attrs[key])
    # Define coordinate list
    coordlist = {tracks_dimname: ([tracks_dimname], np.arange(0, ntracks)), \
                 times_dimname: ([times_dimname], np.arange(0, ntimes)), \
                }
    # Define global attributes
    gattrlist = {'title':  'GOES16 cell track statistics', \
                 'Institution': 'Pacific Northwest National Laboratoy', \
                 'Contact': 'Zhe Feng, zhe.feng@pnnl.gov', \
                 'Created_on':  time.ctime(time.time()), \
                 'source_trackfile': trackstats_file, \
                 'startdate': startdate, \
                 'enddate': enddate, \
                }
    # Define xarray dataset
    dsout = xr.Dataset(varlist, coords=coordlist, attrs=gattrlist)

    # Add variables from cell track stats to the output
    dsout['base_time'] = stats_basetime
    # dsout['cell_area'] = cell_area

    # Delete file if it already exists
    if os.path.isfile(output_filename):
        os.remove(output_filename)
        
    # Set encoding/compression for all variables
    comp = dict(zlib=True)
    encoding = {var: comp for var in dsout.data_vars}

    # Write to netcdf file
    dsout.to_netcdf(path=output_filename, mode="w",
                    format="NETCDF4", unlimited_dims=tracks_dimname, encoding=encoding)
    print(f'Output saved: {output_filename}')