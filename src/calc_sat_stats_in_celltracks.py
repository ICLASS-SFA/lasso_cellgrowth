"""
Calculates satellite-like statistics for tracked convective cells.
The statistics are written to netCDF file matching the cell track statistics file format.
"""
import numpy as np
import os, sys, glob
import time
from datetime import datetime
from pytz import utc
import yaml
import xarray as xr
import dask
from dask.distributed import Client, LocalCluster
# import matplotlib.pyplot as plt

#-----------------------------------------------------------------------
def calc_basetime(filelist, filebase):
    """
    Calculates basetime (Epoch Time) and a filename dictionary from a list of filenames.
    The basetime uses year, month, day, hour, minute but set all seconds to 0.

    Args:
        filelist: list
            A list of input file names
        filebase: string
            Basename of the files

    Returns:
        file_basetime: int array
            Epoch time corresponding to the input files.
        file_dict: dictionary
            Direction key by basetime and value is the file names
    """
    nfiles = len(filelist)
    prelength = len(filebase)
    file_basetime = np.full(nfiles, -999, dtype=int)
    file_dict = {}
    for ifile in range(nfiles):
        fname = os.path.basename(filelist[ifile])
        # File name format: basename_yyyymmdd.hhmmss
        TEMP_filetime = datetime(
            int(fname[prelength:(prelength+4)]), 
            int(fname[prelength+4:(prelength+6)]), 
            int(fname[prelength+6:(prelength+8)]),
            int(fname[prelength+9:(prelength+11)]),
            int(fname[prelength+11:(prelength+13)]),
            int(fname[prelength+13:(prelength+15)]),
            tzinfo=utc,
            # int(fname[prelength+11:(prelength+13)]), 0, tzinfo=utc
        )
        # file_basetime[ifile] = calendar.timegm(TEMP_filetime.timetuple())
        file_basetime[ifile] = TEMP_filetime.timestamp()
        file_dict[file_basetime[ifile]] = filelist[ifile]
    return file_basetime, file_dict

#-----------------------------------------------------------------------
def olr_to_tb(OLR):
    """
    Convert OLR to IR brightness temperature.

    Args:
        OLR: np.array
            Outgoing longwave radiation
    
    Returns:
        tb: np.array
            Brightness temperature
    """
    # Calculate brightness temperature
    # (1984) as given in Yang and Slingo (2001)
    # Tf = tb(a+b*Tb) where a = 1.228 and b = -1.106e-3 K^-1
    # OLR = sigma*Tf^4 
    # where sigma = Stefan-Boltzmann constant = 5.67x10^-8 W m^-2 K^-4
    a = 1.228
    b = -1.106e-3
    sigma = 5.67e-8 # W m^-2 K^-4
    tf = (OLR/sigma)**0.25
    tb = (-a + np.sqrt(a**2 + 4*b*tf))/(2*b)
    return tb

#-----------------------------------------------------------------------
def calc_cellstats_singlefile(
    pixel_filename, 
    rad_filename, 
    idx_track, 
    config,
):
    # Get thresholds from config
    geolimits = config.get('geolimits', None)

    print(rad_filename)

    # Read RAD file
    dsm = xr.open_dataset(rad_filename)
    # Rename dimenensions
    dsm = dsm.rename_dims({'south_north':'lat', 'west_east':'lon'})
    # nz = dsm.sizes['HAMSL']
    ny = dsm.sizes['lat']
    nx = dsm.sizes['lon']
    # height = dsm['HAMSL'].data
    DX = dsm.attrs['DX']
    DY = dsm.attrs['DY']
    grid_area = DX * DY / 1e6
    # OLR = dsm['LWUPT'].squeeze().values

    # # Convert OLR to Tb
    # TB = olr_to_tb(OLR)

    # Read pixel-level track file
    ds = xr.open_dataset(pixel_filename, decode_times=False)
    ny_p = ds.sizes['lat']
    nx_p = ds.sizes['lon']
    cmask = ds['conv_mask'].squeeze().values
    tracknumbermap = ds['tracknumber'].squeeze().values
    # Get cell tracknumber mask
    # Convert convective cell mask to binary, then multiply by tracknumber
    tracknumbermap_cmask = (cmask > 0) * tracknumbermap
    # Replace background values with NaN
    tracknumbermap_cmask[tracknumbermap_cmask <= 0] = np.NaN
    ds.close()

    # Check dimensions between MET and pixel files
    if (ny_p < ny) | (nx_p < nx):
        # Get lat/lon limits
        buffer = 0
        latmin, latmax = geolimits[0]-buffer, geolimits[2]+buffer
        lonmin, lonmax = geolimits[1]-buffer, geolimits[3]+buffer
        # Make a 2D mask
        mask = ((dsm['XLONG'] >= lonmin) & (dsm['XLONG'] <= lonmax) & \
                (dsm['XLAT'] >= latmin) & (dsm['XLAT'] <= latmax)).squeeze()
        # Get y/x indices limits from the mask
        y_idx, x_idx = np.where(mask == True)
        xmin, xmax = np.min(x_idx), np.max(x_idx)
        ymin, ymax = np.min(y_idx), np.max(y_idx)
        # Subset 
        XLONG = dsm['XLONG'][ymin:ymax+1, xmin:xmax+1]
        XLAT = dsm['XLAT'][ymin:ymax+1, xmin:xmax+1]
        OLR = dsm['LWUPT'][:,ymin:ymax+1, xmin:xmax+1].squeeze()
    else:
        XLONG = dsm['XLONG']
        XLAT = dsm['XLAT']
        OLR = dsm['LWUPT'].squeeze()

    # Convert OLR to Tb
    TB = olr_to_tb(OLR)
    # import pdb; pdb.set_trace()

    # Create arrays for output statistics
    nmatchcloud = len(idx_track)
    cell_area = np.full((nmatchcloud), np.nan, dtype=np.float32)
    olr_min = np.full((nmatchcloud), np.nan, dtype=np.float32)
    tb_min = np.full((nmatchcloud), np.nan, dtype=np.float32)

    if (nmatchcloud > 0):
            
        # Loop over each match tracked cloud
        for imatchcloud in range(nmatchcloud):

            # Track number needs to add 1
            itracknum = idx_track[imatchcloud] + 1

            # Get current cell mask
            itrackcmask = tracknumbermap_cmask == itracknum

            # Count the number of pixels for the original cell mask
            inpix_cloud = np.count_nonzero(itrackcmask)

            # # Cell mask with liquid/ice phase
            # itrackcmask_liq = (itrackcmask) & (phase == 1)
            # itrackcmask_ice = (itrackcmask) & (phase == 2)
            # # Count the number of liquid/ice pixels
            # inpix_liq = np.count_nonzero(itrackcmask_liq)
            # inpix_ice = np.count_nonzero(itrackcmask_ice)

            # Proceed if the number matching cloud pixel > 0
            if inpix_cloud > 0:

                # Subset variables to the current cell mask (original)
                sub_OLR = OLR.values[itrackcmask]
                sub_TB = TB.values[itrackcmask]

                # Calculate new statistics of the cloud
                # Minimum/Maximum cloud-top variables
                olr_min[imatchcloud] = np.nanmin(sub_OLR)
                tb_min[imatchcloud] = np.nanmin(sub_TB)

                cell_area[imatchcloud] = inpix_cloud * pixel_radius**2

        # Group outputs in dictionaries
        out_dict = {
            # "nmatchcloud": nmatchcloud,
            "cell_area": cell_area, 
            "OLR_min": olr_min,
            "temperature_ir_min": tb_min,
        }
        out_dict_attrs = {
            # "nmatchcloud": nmatchcloud,
            "cell_area": {
                "long_name": "Area of the convective cell in a track",
                "units": "km^2",
            }, 
            "temperature_ir_min": {
                "long_name": "Minimum IR temperature in a track",
                "units": "K",
            }, 
            "OLR_min": {
                "long_name": "Minimum OLR in a track",
                "units": "W m-2",
            }, 
        }
        # import pdb; pdb.set_trace()
    return out_dict, out_dict_attrs


#-----------------------------------------------------------------------
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
    metfile_path = config['metfile_path']
    metfile_path_2 = config['metfile_path_2']
    output_path = config['output_path']
    rad_filebase = config['rad_filebase']
    # methamsl_filebase = config['methamsl_filebase']
    # cldhamsl_filebase = config['cldhamsl_filebase']
    pixel_filebase = config['pixel_filebase']
    ncores_min = config['ncores_min']

    # Replace directory (some LASSO data are staged in a different directory)
    if os.path.isdir(metfile_path_2):
        metfile_path = metfile_path_2

    # Add start/end date to pixel file path
    pixelfile_path = f'{pixelfile_path}{startdate}_{enddate}/'

    # Track stats file basename
    stats_filebase = 'trackstats_'

    # Output statistics filename
    output_filename = f'{output_path}stats_tb_{startdate}_{enddate}.nc'
    os.makedirs(output_path, exist_ok=True)

    # Track statistics file dimension names
    tracks_dimname = 'tracks'
    times_dimname = 'times'

    # Track statistics file
    trackstats_file = f'{stats_path}{stats_filebase}{startdate}_{enddate}.nc'
    # Find all pixel-level files
    pixelfilelist = sorted(glob.glob(f'{pixelfile_path}{pixel_filebase}*.nc'))
    nfiles = len(pixelfilelist)
    # Find all WRF files
    radfilelist = sorted(glob.glob(f'{metfile_path}{rad_filebase}*.nc'))
    nradfiles = len(radfilelist)
    print(f'Number of RAD files: {nradfiles}')

    # Get basetime from pixel files
    pixel_basetime, pixelfile_dict = calc_basetime(pixelfilelist, pixel_filebase)
    # Get basetime from rad files
    rad_basetime, radfile_dict = calc_basetime(radfilelist, rad_filebase)

    # Find matching WRF files for each pixel file
    match_radfilelist = [''] * nfiles
    for ifile in range(nfiles):
        # Find rad time closest to the pixel file time and get the index
        # Save the filename if time difference is < time_window
        idx = np.argmin(np.abs(rad_basetime - pixel_basetime[ifile]))
        if np.abs(rad_basetime[idx] - pixel_basetime[ifile]) < time_window:
            match_radfilelist[ifile] = radfile_dict[rad_basetime[idx]]
        else:
            print(f'No match rad file found for: {pixelfilelist[ifile]}')

    # Read track statistics file
    print(trackstats_file)
    dsstats = xr.open_dataset(trackstats_file, decode_times=False)
    ntracks = dsstats.sizes[tracks_dimname]
    ntimes = dsstats.sizes[times_dimname]
    tracks_coord = dsstats.coords[tracks_dimname]
    times_coord = dsstats.coords[times_dimname]
    stats_basetime = dsstats['base_time']
    stats_basetime_attrs = dsstats['base_time'].attrs
    # cell_area = dsstats['cell_area']
    pixel_radius = dsstats.attrs['pixel_radius_km']
    dsstats.close()

    print(f'Total Number of Tracks: {ntracks}')


    ##############################################################
    # Call function to calculate statistics
    trackindices_all = []
    timeindices_all = []
    final_results = []

    if run_parallel == 1:
        # Initialize dask
        dask_tmp_dir = config.get("dask_tmp_dir", "/tmp")
        dask.config.set({'temporary-directory': dask_tmp_dir})
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker)
        client = Client(cluster)

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
            # Serial
            if run_parallel == 0:
                iresult = calc_cellstats_singlefile(
                    pixelfilelist[ifile], 
                    match_radfilelist[ifile],
                    idx_track, 
                    config,
                )
            # Parallel
            elif run_parallel == 1:
                iresult = dask.delayed(calc_cellstats_singlefile)(
                    pixelfilelist[ifile], 
                    match_radfilelist[ifile],
                    idx_track, 
                    config,
                )
            final_results.append(iresult)

    if run_parallel == 1:
        # Trigger Dask computation
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

    # Define variable dictionary
    var_dict = {}
    # Define output variable dictionary
    for key, value in out_dict.items():
        var_dict[key] = ([tracks_dimname, times_dimname], value, out_dict_attrs[key])
    # Define coordinate dictionary
    coord_dict = {
        tracks_dimname: ([tracks_dimname], tracks_coord.data, tracks_coord.attrs),
        times_dimname: ([times_dimname], times_coord.data, times_coord.attrs),
    }
    # Define global attributes
    gattr_dict = {
        'title':  'Tracked cell cloud-top statistics',
        'Institution': 'Pacific Northwest National Laboratoy',
        'Contact': 'Zhe Feng, zhe.feng@pnnl.gov',
        'Created_on':  time.ctime(time.time()),
        'source_trackfile': trackstats_file,
        'startdate': startdate,
        'enddate': enddate,
    }
    # Define xarray dataset
    dsout = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)

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
    # import pdb; pdb.set_trace()