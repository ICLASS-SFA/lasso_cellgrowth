"""
Calculates 3D draft variables from LASSO subset MET and CLD files 
and writes to netCDF files.
"""
import numpy as np
import os, sys, glob
import time
from datetime import datetime
from pytz import utc
import yaml
import xarray as xr
import warnings
import dask
from dask.distributed import Client, LocalCluster
import psutil
import gc

def log_memory_usage(stage_name):
    """Log current memory usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"{stage_name}: Memory usage: {memory_info.rss / 1024**3:.2f} GB")

def cleanup_memory():
    """Enhanced memory cleanup function"""
    gc.collect()
    # Clear Dask cache more aggressively
    try:
        import dask
        import dask.array as da
        # Clear all caches
        da.core.clear_cache()
        # Force garbage collection in Dask
        from dask.base import clear_cache
        clear_cache()
        # Clear any remaining delayed objects
        dask.base.clear_cache()
    except (ImportError, AttributeError):
        pass

def monitor_memory_and_cleanup(threshold_gb=40):
    """Monitor memory usage and cleanup if threshold exceeded"""
    process = psutil.Process()
    memory_gb = process.memory_info().rss / 1024**3
    if memory_gb > threshold_gb:
        print(f"Memory usage ({memory_gb:.2f} GB) exceeds threshold, cleaning up...")
        cleanup_memory()
        return True
    return False

#-----------------------------------------------------------------------
def process_batch(batch_indices, metfilelist, match_cldfilelist, config):
    """Process a batch of files"""
    batch_results = []
    for ifile in batch_indices:
        if match_cldfilelist[ifile]:  # Only process if matching CLD file exists
            result = dask.delayed(calc_draft_singlefile)(
                metfilelist[ifile], 
                match_cldfilelist[ifile],
                config,
            )
            batch_results.append(result)
    return batch_results

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
def calc_rh_thompson(TEMPERATURE, PRESSURE, QVAPOR):
    """
    Calculate relative humidity following the Thompson scheme for supersaturation

    Args:
        TEMPERATURE: array-like
            Dry air temp [K]
        PRESSURE: array-like
            Air pressure [Pa]
        QVAPOR: array-like
            Water vapor mixing ratio [kg/kg]
   
    Returns:
        RH: array-like
            Relative humidity [%]
    """
    # RH (formula is used in Thompson scheme for supersaturation)
    C0 = 0.611583699e3
    C1 = 0.444606896e2
    C2 = 0.143177157e1
    C3 = 0.264224321e-1
    C4 = 0.299291081e-3
    C5 = 0.203154182e-5
    C6 = 0.702620698e-8
    C7 = 0.379534310e-11
    C8 = -0.321582393e-13
    X = TEMPERATURE - 273.16
    # X[X < -80] = -80  #setting values less than -80C to -80C
    X = xr.where(X < -80, -80, X) #setting values less than -80C to -80C
    ESL = C0 + X*(C1 + X*(C2 + X*(C3 + X*(C4 + X*(C5 + X*(C6 + X*(C7 + X*C8))))))) #saturation vapor pressure
    QVS = 0.622 * ESL / (PRESSURE - ESL) #saturation vapor mixing ratio
    RH = 1e2 * QVAPOR / QVS # %
    #SS = RH - 100 #supersaturation in %
    return RH

#-----------------------------------------------------------------------
def calc_theta_e(TEMPERATURE, PRESSURE, QVAPOR, Qliq, RH):
    """
    Calculate equivalent potential temperature following the Emanuel formula

    Args:
        TEMPERATURE: array-like
            Dry air temp [K]
        PRESSURE: array-like
            Air pressure [Pa]
        QVAPOR: array-like
            Water vapor mixing ratio [kg/kg]
        Qliq: array-like
            Total liquid condensate mixing ratio [kg/kg]
        RH: array-like
            Relative humidity [%]
    
    Returns:
        THETAE: array-like
            Equivalent potential temperature.
    """
    # constants:
    cpd = 1006 # J/kg K
    lv0 = 2501000
    Rv = 461.5
    Rd = 287.04
    cl = 4200

    teA = TEMPERATURE * (100000. / PRESSURE)**(Rd / (cpd + cl * Qliq))
    teB = np.exp((lv0 * QVAPOR) / ((cpd + Qliq * cl) * TEMPERATURE))
    teC = (RH/100)**((-QVAPOR * Rv) / (cpd + cl * Qliq))
    THETAE = teA * teB * teC
    return THETAE

#-----------------------------------------------------------------------
def theta_e_bolton(TEMPERATURE, QVAPOR, PRESSURE):
    """
    Calculate pseudoadiabatic equivalent potential temperature following the Bolton (1980) formula
    
    Error of < 0.3 K between -35 and 35C; from Thompson scheme

    Args:
        TEMPERATURE: array-like
            Dry air temp [K]
        PRESSURE: array-like
            Air pressure [Pa]
        QVAPOR: array-like
            Water vapor mixing ratio [kg/kg]

    Returns:
        THETAE_Bolton: array-like
            Equivalent potential temperature.
    """

    es = PRESSURE*QVAPOR/(0.622*QVAPOR)
    TDEW = (35.86*np.log(es) - 4947.2325)/(np.log(es) - 23.6837)
    TLCL = 1/(1/(TDEW - 56) + np.log(TEMPERATURE/TDEW)/800) + 56
    p1 = 3.376/TLCL - 0.00254
    p2 = 1e3*QVAPOR*(1 + 0.81*QVAPOR)
    THETAE_Bolton = (TEMPERATURE*(100000./PRESSURE)**(0.2854*(1 - 0.28*QVAPOR)))*np.exp(p1*p2) #K
    return THETAE_Bolton


def calc_draft_singlefile(metfile, cldfile, config):
    """
    Calculate draft variables from a single MET file and its matching CLD file.

    Args:
        metfile: str
            Path to the MET file.
        cldfile: str
            Path to the matching CLD file.
        config: dict
            Configuration dictionary with parameters for calculations.

    Returns:
        dict: Results of the draft variable calculations.
    """
    # Constants
    R_dry = 287.058   # J kg−1 K−1
    Cp_dry = 1005   # J kg-1 K−1

    # Get configuration parameters 
    draft_path = config.get('draft_path')
    drafthamsl_filebase = config.get('drafthamsl_filebase')

    print(f"Processing MET file: {metfile}")

    # Load MET data
    dsm = xr.open_dataset(metfile)
    # Coordinates
    Time = dsm['Time']
    XLONG = dsm['XLONG']
    XLAT = dsm['XLAT']
    HAMSL = dsm['HAMSL']
    # Load CLD data
    dsc = xr.open_dataset(cldfile)

    # Output file name
    outdatetime_str = Time.dt.strftime('%Y%m%d.%H%M%S').item()
    outfilename = f"{draft_path}{drafthamsl_filebase}{outdatetime_str}.nc"
    # import pdb; pdb.set_trace()

    # Get necessary variables from MET and CLD data
    PRESSURE = dsm['PRESSURE']*100  # [Pa]
    TEMPERATURE = dsm['TEMPERATURE']
    QVAPOR = dsm['QVAPOR']
    THETA = dsm['THETA']
    WA = dsm['WA']
    # Get cld variables
    QCLOUD = dsc['QCLOUD']
    QRAIN = dsc['QRAIN']
    QICE = dsc['QICE']
    QSNOW = dsc['QSNOW']
    QGRAUP = dsc['QGRAUP']
    
    # Total cloud condensates
    Qcld = QCLOUD + QICE + QSNOW
    # Total liquid condensates
    Qliq = QVAPOR + QCLOUD + QRAIN
    # Total condensates + vapor
    Qtotal = Qcld + QRAIN + QGRAUP + QVAPOR

    # Calculate virtual temperature
    TV = TEMPERATURE * (1 + QVAPOR / 0.622) / (1 + QVAPOR)
    # Calculate moist air density using virtual temperature
    RHO_MOIST = PRESSURE / (R_dry * TV)  # kg m-3
    # Calculate mass flux (kg m-2 s-1)
    MassFlux = (RHO_MOIST * WA)

    # Virtual Potential Temperature
    ThetaV = THETA * (QVAPOR + 0.622)/(0.622 * (1 + QVAPOR))
    # Density Potential Temperature
    ThetaRho = TEMPERATURE * (1 + QVAPOR/0.622) / (1 + Qtotal) * (100000/PRESSURE)**(R_dry/Cp_dry)
    # Thopmson RH
    RH = calc_rh_thompson(TEMPERATURE, PRESSURE, QVAPOR)
    # Emanuel ThetaE
    ThetaE = calc_theta_e(TEMPERATURE, PRESSURE, QVAPOR, Qliq, RH)
    # Bolton ThetaE
    ThetaE_Bolton = theta_e_bolton(TEMPERATURE, QVAPOR, PRESSURE)

    # Create output netCDF DataSet for draft variables
    dims2d = ['Time', 'south_north', 'west_east']
    dims3d = ['Time', 'HAMSL', 'south_north', 'west_east']

    # Define variable attributes
    var_attrs = {
        'XLONG': {'units': 'degrees_east', 'long_name': 'Longitude'},
        'XLAT': {'units': 'degrees_north', 'long_name': 'Latitude'},
        # 'PRESSURE': {'units': 'Pa', 'long_name': 'Pressure'},
        # 'TEMPERATURE': {'units': 'K', 'long_name': 'Temperature'},
        # 'QVAPOR': {'units': 'kg/kg', 'long_name': 'Water Vapor Mixing Ratio'},
        # 'THETA': {'units': 'K', 'long_name': 'Potential Temperature'},
        # 'TV': {'units': 'K', 'long_name': 'Virtual Temperature'},
        'WA': {'units': 'm/s', 'long_name': 'Vertical Velocity'},
        # 'QCLOUD': {'units': 'kg/kg', 'long_name': 'Cloud Mixing Ratio'},
        # 'QRAIN': {'units': 'kg/kg', 'long_name': 'Rain Mixing Ratio'},
        # 'QICE': {'units': 'kg/kg', 'long_name': 'Ice Mixing Ratio'},
        # 'QSNOW': {'units': 'kg/kg', 'long_name': 'Snow Mixing Ratio'},
        # 'QGRAUP': {'units': 'kg/kg', 'long_name': 'Graupel Mixing Ratio'},
        'Qcld': {'units': 'kg/kg', 'long_name': 'Total Cloud Condensates'},
        'Qliq': {'units': 'kg/kg', 'long_name': 'Total Liquid Condensates'},
        'Qtotal': {'units': 'kg/kg', 'long_name': 'Total Condensates + Vapor'},
        'MassFlux': {'units': 'kg m-2 s-1', 'long_name': 'Vertical mass Flux'},
        'ThetaV': {'units': 'K', 'long_name': 'Virtual Potential Temperature'},
        'ThetaRho': {'units': 'K', 'long_name': 'Density Potential Temperature'},
        'RH': {'units': '%', 'long_name': 'Relative Humidity'},
        'ThetaE': {'units': 'K', 'long_name': 'Emanuel Equivalent Potential Temperature'},
        'ThetaE_Bolton': {'units': 'K', 'long_name': 'Bolton Equivalent Potential Temperature'},
    }

    # Create variable dictionary
    var_dict = {
        # 'XLONG': (dims2d, XLONG.data, var_attrs['XLONG']),
        # 'XLAT': (dims2d, XLAT.data, var_attrs['XLAT']),
        # 'THETA': (dims3d, THETA.data, var_attrs['THETA']),
        # 'WA': (dims3d, WA.data, var_attrs['WA']),
        # Total cloud condensates
        'Qcld': (dims3d, Qcld.data, var_attrs['Qcld']),
        # Total liquid condensates
        'Qliq': (dims3d, Qliq.data, var_attrs['Qliq']),
        # Total condensates + vapor
        'Qtotal': (dims3d, Qtotal.data, var_attrs['Qtotal']),
        # Mass flux
        'MassFlux': (dims3d, MassFlux.data, var_attrs['MassFlux']),
        # Virtual potential temperature
        "ThetaV": (dims3d, ThetaV.data, var_attrs['ThetaV']),
        # Density potential temperature
        "ThetaRho": (dims3d, ThetaRho.data, var_attrs['ThetaRho']),
        # Relative humidity
        "RH": (dims3d, RH.data, var_attrs['RH']),
        # Emanuel equivalent potential temperature
        "ThetaE": (dims3d, ThetaE.data, var_attrs['ThetaE']),
        # Bolton equivalent potential temperature
        "ThetaE_Bolton": (dims3d, ThetaE_Bolton.data, var_attrs['ThetaE_Bolton']),
    }
    # Coordinates dictionary
    coord_dict = {
        'Time': (['Time'], dsm['Time'].data),
        'XLAT': (['south_north', 'west_east'], dsm['XLAT'].data),
        'XLONG': (['south_north', 'west_east'], dsm['XLONG'].data),
        'HAMSL': (['HAMSL'], dsm['HAMSL'].data, HAMSL.attrs),
    }
    # Global attributes
    gattr_dict = {
        'description': 'Draft variables calculated from MET and CLD files',
        'metfile': metfile,
        'cldfile': cldfile,
        'created_by': 'calc_draft_variables_from_lasso_subsets.py',
        'created_on': time.ctime(time.time()),
        # 'created_on': datetime.now(utc).isoformat(),
    }
    # Create the xarray Dataset
    ds_out = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)
    
    # Set encoding/compression for all variables
    comp = dict(zlib=True)
    encoding = {var: comp for var in ds_out.data_vars}

    # Write netCDF file
    print(f"Writing output file ...")
    ds_out.to_netcdf(path=outfilename, mode="w", format="NETCDF4", encoding=encoding)
    print(f"Output saved to {outfilename}")

    # import pdb; pdb.set_trace()
    return outfilename

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
    time_window = config['time_window']
    metfile_path = config['metfile_path']
    metfile_path_2 = config['metfile_path_2']
    draft_path = config['draft_path']
    methamsl_filebase = config['methamsl_filebase']
    cldhamsl_filebase = config['cldhamsl_filebase']
    drafthamsl_filebase = config['drafthamsl_filebase']

    # Replace directory (some LASSO data are staged in a different directory)
    if os.path.isdir(metfile_path_2):
        metfile_path = metfile_path_2

    # Create output directory
    os.makedirs(draft_path, exist_ok=True)

    # Find all Met files
    metfilelist = sorted(glob.glob(f'{metfile_path}{methamsl_filebase}*.nc'))
    cldfilelist = sorted(glob.glob(f'{metfile_path}{cldhamsl_filebase}*.nc'))
    nmetfiles = len(metfilelist)
    print(f'Number of MET files: {nmetfiles}')

    # Get basetime from met & cld files
    met_basetime, metfile_dict = calc_basetime(metfilelist, methamsl_filebase)
    cld_basetime, cldfile_dict = calc_basetime(cldfilelist, cldhamsl_filebase)

    # Find matching cld files for each met file
    match_cldfilelist = [''] * nmetfiles
    for ifile in range(nmetfiles):
        # Find cld time closest to the met file time and get the index
        # Save the filename if time difference is < time_window
        idx = np.argmin(np.abs(cld_basetime - met_basetime[ifile]))
        if np.abs(cld_basetime[idx] - met_basetime[ifile]) < time_window:
            match_cldfilelist[ifile] = cldfile_dict[cld_basetime[idx]]
        else:
            print(f'No match cld file found for: {met_basetime[ifile]}')


    if run_parallel == 1:
        # Enhanced Dask configuration for large datasets
        dask_tmp_dir = config.get("dask_tmp_dir", "/tmp")
        dask.config.set({
            'temporary-directory': dask_tmp_dir,
            'distributed.worker.memory.target': 0.8,  # Target 80% memory usage
            'distributed.worker.memory.spill': 0.85,  # Spill to disk at 85%
            'distributed.worker.memory.pause': 0.9,   # Pause at 90%
            'distributed.worker.memory.terminate': 0.95,  # Terminate at 95%
            'distributed.comm.timeouts.connect': '60s',
            'distributed.comm.timeouts.tcp': '60s',
            'distributed.worker.daemon': False,
            'distributed.scheduler.idle-timeout': '1h',
            'distributed.worker.lifetime.duration': '4h',  # Restart workers every 4h
            'distributed.worker.lifetime.stagger': '1h',   # Stagger restarts
        })
        
        try:
            print(f"Initializing Dask cluster with {n_workers} workers...")
            
            # Reduce memory per worker for this large dataset
            # memory_per_worker = min(48, int(256 / n_workers * 0.8))  # Use 80% of available memory

            # Get actual system memory
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
            # Leave some memory for the system and scheduler
            usable_memory_gb = total_memory_gb * 0.85  # Use 85% of total memory
            memory_per_worker = max(12, min(48, int(usable_memory_gb / n_workers)))
            print(f"Detected {total_memory_gb:.1f}GB total memory, allocating {memory_per_worker}GB per worker")
            
            cluster = LocalCluster(
                n_workers=n_workers, 
                threads_per_worker=threads_per_worker,
                memory_limit=f'{memory_per_worker}GB',
                processes=True,
                silence_logs=False,  # Keep logs for debugging
                dashboard_address=':8787',
                # Add worker resource limits
                worker_class='distributed.nanny.Nanny',  # Use nannies for better process management
                timeout='120s',
            )
            
            client = Client(cluster, timeout='120s')
            print(f"Dask cluster initialized successfully")
            print(f"Dashboard link: {client.dashboard_link}")
            print(f"Memory per worker: {memory_per_worker}GB")
            
        except Exception as e:
            print(f"Failed to initialize Dask cluster: {e}")
            print("Falling back to serial processing")
            run_parallel = 0

    # Enhanced processing loop with better monitoring
    print(f"Starting processing of {nmetfiles} files...")
    start_time = time.time()
    last_progress_time = start_time


    ##############################################################
    # Call function to calculate statistics
    final_results = []

    # Add batch processing to avoid creating too many delayed tasks
    batch_size = config.get('batch_size', min(n_workers * 2, 16))  # Process in batches

    # Process files in batches
    final_results = []
    for batch_start in range(0, nmetfiles, batch_size):
        batch_end = min(batch_start + batch_size, nmetfiles)
        batch_indices = range(batch_start, batch_end)
        
        print(f"Processing batch {batch_start//batch_size + 1}: files {batch_start} to {batch_end-1}")
        
        if run_parallel == 1:
            batch_tasks = process_batch(batch_indices, metfilelist, match_cldfilelist, config)
            batch_results = dask.compute(*batch_tasks)
            final_results.extend(batch_results)
        else:
            # Serial processing
            for ifile in batch_indices:
                if match_cldfilelist[ifile]:
                    result = calc_draft_singlefile(metfilelist[ifile], match_cldfilelist[ifile], config)
                    final_results.append(result)
        
        # Memory cleanup between batches
        cleanup_memory()

    # Close Dask client and cluster
    try:
        print("Shutting down Dask cluster...")
        client.close()
        cluster.close()
        print("Dask cluster shutdown complete")
    except Exception as e:
        print(f"Warning: Error during cluster shutdown: {e}")

    # Final cleanup
    cleanup_memory()
    elapsed_time = time.time() - start_time
    print(f"Total processing time: {elapsed_time/60:.1f} minutes")
    print(f"Successfully processed {len([r for r in final_results if r])} files")