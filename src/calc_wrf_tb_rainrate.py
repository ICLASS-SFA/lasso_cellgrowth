import os
import sys
import yaml
import logging
import numpy as np
import time
import xarray as xr
import pandas as pd
import dask
from dask.distributed import Client, LocalCluster
from pyflextrkr.ft_utilities import subset_files_timerange
from pyflextrkr.ft_utilities import load_config, setup_logging
from pyflextrkr.ftfunctions import olr_to_tb

#-------------------------------------------------------------------------------------
def calc_rainrate_tb(filepairnames, outdir, inbasename, outbasename, config):
    """
    Calculates rain rates from a pair of WRF output files and write to netCDF

    Args:
        filepairnames: list
            A list of filenames in pair
        outdir: string
            Output file directory.
        inbasename: string
            Input file basename.
        outbasename: string
            Output file basename.
        config: dictionary
            Dictionary containing config parameters

    Returns:
        status: 0 or 1
            Returns status = 1 if success.
    """

    logger = logging.getLogger(__name__)
    status = 0

    regrid_input = config.get('regrid_input', False)
    write_native = config.get('write_native', False)
    rainrate_method = config.get('rainrate_method', 'standard')

    varname_OLR = 'LWUPT'
    varname_RAIN = 'RAINNC'
    
    # Filenames with full path
    filein_t1 = filepairnames[0]
    logger.debug(f'Reading input f1: {filein_t1}')

    filein_t2 = filepairnames[1]
    logger.debug(f'Reading input f2: {filein_t2}')

    # Get filename
    fname_t1 = os.path.basename(filein_t1)
    fname_t2 = os.path.basename(filein_t2)

    # Get basename string position
    idx0 = fname_t1.find(inbasename)
    ftime_t2 = fname_t2[idx0+len(inbasename):]
    # Output filename
    suffix = '' if '.nc' in ftime_t2 else '.nc'
    # 'Yes' if fruit == 'Apple' else 'No'
    fileout = f'{outdir}/{outbasename}{ftime_t2}{suffix}'
    # Output for native resolution
    if (regrid_input) & (write_native):
        outdir_native = f'{outdir}/native/'
        fileout_native = f'{outdir_native}/{outbasename}{ftime_t2}{suffix}'
        # Create output directory
        os.makedirs(outdir_native, exist_ok=True)

    # Read in WRF data files
    ds_in = xr.open_mfdataset([filein_t1, filein_t2], concat_dim='Time', combine='nested')
    Times = ds_in['Times'].load()
    XLONG = ds_in['XLONG'][0,:,:].squeeze()
    XLAT = ds_in['XLAT'][0,:,:].squeeze()
    ny, nx = np.shape(XLAT)
    DX = ds_in.attrs['DX']
    DY = ds_in.attrs['DY']
    
    ntimes = len(Times)
    Times_str = []
    basetimes = np.full(ntimes, np.NAN, dtype=float)
    for tt in range(0, ntimes):
        # Decode bytes to string with UTF-8 encoding, then replace "_" with "T"
        # to make time string: YYYY-MO-DDTHH:MM:SS
        tstring = Times[tt].item().decode("utf-8").replace("_", "T")
        Times_str.append(tstring)
        # Convert time string to Epoch time
        basetimes[tt] = pd.to_datetime(tstring).timestamp()
    # wrfdatetimes = pd.to_datetime(Times_str)

    # Calculate basetime difference in [seconds]
    delta_times = np.diff(basetimes)

    # Read and compute accumulated precipitation
    RAINALL = ds_in[varname_RAIN].load()
    # if rainrate_method == 'standard':
    #     RAINALL = ds_in[varname_RAIN].load()
    #     # RAINC = ds_in['RAINC']
    #     # Add grid-scale and convective precipitation
    #     # RAINALL = (RAINNC + RAINC).load()
    # elif rainrate_method == 'saag':
    #     RAINNC = ds_in['RAINNC']
    #     I_RAINNC = ds_in['I_RAINNC']
    #     # The total precipitation accumulation from the initial time is computed (units: mm)
    #     RAINALL = (RAINNC + I_RAINNC * 100).load()
    
    # Read OLR
    OLR_orig = ds_in[varname_OLR].load()

    # Create an array to store rainrates and brightness temperature
    rainrate = np.zeros((ntimes-1,ny,nx), dtype=float)
    OLR = np.zeros((ntimes-1,ny,nx), dtype = float)

    # Loop over all times-1
    for tt in range(0, ntimes-1):
        # Calculate rainrate, convert unit to [mm/h]
        rainrate[tt,:,:] = 3600. * (RAINALL.isel(Time=tt+1).data - RAINALL.isel(Time=tt).data) / delta_times[tt]
        OLR[tt,:,:] = OLR_orig.isel(Time=tt+1).data

    # Check regridding option
    if regrid_input:
        logger.info('Regridding input is requested.')

        weight_filename = config.get('weight_filename')
        regrid_method = config.get('regrid_method', 'conservative')

        # Make source & destination grid data for regridder
        grid_src, grid_dst = make_grid4regridder(filein_t1, config)
        # Retrieve Regridder
        regridder = xe.Regridder(grid_src, grid_dst, method=regrid_method, weights=weight_filename)

        # Make output coordinates
        x_coord_dst = grid_dst['lon']
        y_coord_dst = grid_dst['lat']
        if (x_coord_dst.ndim == 1) & (y_coord_dst.ndim == 1):
            x_coord_out, y_coord_out = np.meshgrid(x_coord_dst, y_coord_dst)
        else:
            x_coord_out = x_coord_dst
            y_coord_out = y_coord_dst
        # Regrid variables
        OLR_out = regridder(OLR)
        rainrate_out = regridder(rainrate)
    else:
        # Native resolution variables
        x_coord_out = XLONG.data
        y_coord_out = XLAT.data
        OLR_out = OLR
        rainrate_out = rainrate

    ds_in.close()
    
    # Convert OLR to IR brightness temperature
    tb = olr_to_tb(OLR)
    tb_out = olr_to_tb(OLR_out)

    # Write single time frame to netCDF output
    for tt in range(0, ntimes-1):

        # Use the next time to be consitent with output filename
        _basetime = basetimes[tt+1]
        _TimeStr = Times_str[tt+1]
        _tb = tb_out[tt,:,:]
        _rainrate = rainrate_out[tt,:,:]

        # Write output to file
        write_netcdf(_TimeStr, _basetime, _rainrate, _tb,
                     config, filein_t1, filein_t2, fileout, x_coord_out, y_coord_out)
        logger.info(f'{fileout}')

        # Write to native resolution file if regrid is requested
        if (regrid_input) & (write_native):
            __tb = tb[tt,:,:]
            __rainrate = rainrate[tt,:,:]

            # Write output to file
            write_netcdf(_TimeStr, _basetime, __rainrate, __tb,
                        config, filein_t1, filein_t2, fileout_native, XLONG.data, XLAT.data)
            logger.info(f'{fileout_native}')

        status = 1
    return (status)


#-------------------------------------------------------------------------------------
def write_netcdf(
        _TimeStr,
        _basetime,
        _rainrate,
        _tb,
        config,
        filein_t1,
        filein_t2,
        fileout,
        x_coord_out,
        y_coord_out,
):
    """
    Write output to netCDF file.

    Args:
        _TimeStr: string
            Time string for output file.
        _basetime: np.float
            Base time value in Epoch.
        _rainrate: np.array
            Rain rate array.
        _tb: np.array
            Tb array.
        config: dictionary
            Dictionary containing config parameters
        filein_t1: string
            Input filename for time1.
        filein_t2: string
            Input filename for time2.
        fileout: string
            Output filename.
        x_coord_out: np.array
            X coordinate array.
        y_coord_out: np.array
            Y coordinate array.

    Return:
        None.
    """
    # Get parameters from config
    time_dimname = config.get('time_dimname', 'time')
    x_dimname = config.get('x_dimname', 'lon')
    y_dimname = config.get('y_dimname', 'lat')
    x_coordname = config.get('x_coordname', 'longitude')
    y_coordname = config.get('y_coordname', 'latitude')
    tb_varname = config.get('tb_varname', 'tb')
    pcp_varname = config.get('pcp_varname', 'rainrate')
    # Define xarray dataset
    var_dict = {
        # 'Times': ([time_dimname,'char'], times_char_t1),
        x_coordname: ([y_dimname, x_dimname], x_coord_out),
        y_coordname: ([y_dimname, x_dimname], y_coord_out),
        tb_varname: ([time_dimname, y_dimname, x_dimname], np.expand_dims(_tb, axis=0)),
        pcp_varname: ([time_dimname, y_dimname, x_dimname], np.expand_dims(_rainrate, axis=0)),
    }
    coord_dict = {
        time_dimname: ([time_dimname], np.expand_dims(_basetime, axis=0)),
        # 'char': (['char'], np.arange(0, strlen_t1)),
    }
    gattr_dict = {
        'Title': 'WRF calculated rainrate and brightness temperature',
        'Contact': 'Zhe Feng: zhe.feng@pnnl.gov',
        'Institution': 'Pacific Northwest National Laboratory',
        'created on': time.ctime(time.time()),
        'Original_File1': filein_t1,
        'Original_File2': filein_t2,
        # 'DX': DX,
        # 'DY': DY,
    }
    dsout = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)
    # Specify attributes
    dsout[time_dimname].attrs['long_name'] = 'Epoch time (seconds since 1970-01-01 00:00:00)'
    dsout[time_dimname].attrs['units'] = 'seconds since 1970-01-01 00:00:00'
    # dsout[time_dimname].attrs['_FillValue'] = np.NaN
    dsout[time_dimname].attrs['tims_string'] = _TimeStr
    # dsout['Times'].attrs['long_name'] = 'WRF-based time'
    dsout[x_coordname].attrs['long_name'] = 'Longitude'
    dsout[x_coordname].attrs['units'] = 'degrees_east'
    dsout[y_coordname].attrs['long_name'] = 'Latitude'
    dsout[y_coordname].attrs['units'] = 'degrees_north'
    dsout[tb_varname].attrs['long_name'] = 'Brightness temperature'
    dsout[tb_varname].attrs['units'] = 'K'
    dsout[pcp_varname].attrs['long_name'] = 'Precipitation rate'
    dsout[pcp_varname].attrs['units'] = 'mm hr-1'
    # Write to netcdf file
    encoding_dict = {
        time_dimname: {'zlib': True, 'dtype': 'float'},
        # 'Times':{'zlib':True},
        x_coordname: {'zlib': True, 'dtype': 'float32'},
        y_coordname: {'zlib': True, 'dtype': 'float32'},
        tb_varname: {'zlib': True, 'dtype': 'float32'},
        pcp_varname: {'zlib': True, 'dtype': 'float32'},
    }
    dsout.to_netcdf(path=fileout, mode='w', format='NETCDF4', unlimited_dims=time_dimname, encoding=encoding_dict)
    return

if __name__ == '__main__':

    # Set the logging message level
    setup_logging()
    logger = logging.getLogger(__name__)

    # Load configuration file
    config_file = sys.argv[1]
    config = load_config(config_file)

    # # Read configuration from yaml file
    # stream = open(config_file, "r")
    # config = yaml.full_load(stream)

    # Dask parallel setup
    run_parallel = 1
    n_workers = 24

    # Get inputs from config
    # run_parallel = config['run_parallel']
    # n_workers = config['nprocesses']
    indir = config['wrfout_path']
    # indir = config['rawdata_path']
    outdir = config['clouddata_path']
    inbasename = config['wrfout_basename']
    # rawdatabasename = config['rawdatabasename']
    start_basetime = config.get("start_basetime", None)
    end_basetime = config.get("end_basetime", None)
    # time_format = config["time_format"]
    time_format = config["wrfout_time_format"]
    domain = config['domain']

    # # Get filename parts (e.g., 'corlasso_cldhamsl_2019012300eda05d4_base_M1.m1.')
    # parts = rawdatabasename.split('_')
    # # Add common parts to filenames
    # inbasename_olr = f'corlasso_rad_{parts[2]}_{parts[3]}_{parts[4]}'
    # inbasename_rain = f'corlasso_methamsl_{parts[2]}_{parts[3]}_{parts[4]}'
    outbasename = 'tb_rainrate_'

    # Identify files to process
    infiles_info = subset_files_timerange(
        indir,
        inbasename,
        start_basetime=start_basetime,
        end_basetime=end_basetime,
        time_format=time_format,
    )
    # Get file list
    filelist = infiles_info[0]
    nfiles = len(filelist)
    print(f'Number of WRF files: {nfiles}')

    # Create a list with a pair of WRF filenames that are adjacent in time
    filepairlist = []
    for ii in range(0, nfiles-1):
        ipair = [filelist[ii], filelist[ii+1]]
        filepairlist.append(ipair)
    nfilepairs = len(filepairlist)

    if run_parallel == 0:
        # Serial version
        for ifile in range(0, nfilepairs):
            status = calc_rainrate_tb(filepairlist[ifile], outdir, inbasename, outbasename, config)
    elif run_parallel >= 1:
        # Set Dask temporary directory for workers
        dask_tmp_dir = config.get("dask_tmp_dir", "./")
        dask.config.set({'temporary-directory': dask_tmp_dir})
        # Local cluster
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
        client = Client(cluster)
        client.run(setup_logging)

        results = []
        for ifile in range(0, nfilepairs):
            result = dask.delayed(calc_rainrate_tb)(filepairlist[ifile], outdir, inbasename, outbasename, config)
            results.append(result)
        final_result = dask.compute(*results)

    # import pdb; pdb.set_trace()