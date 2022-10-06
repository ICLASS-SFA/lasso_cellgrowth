import os
import glob
import pandas as pd

#-----------------------------------------------------------------------------
def make_new_filename(fname):
    """
    Rename a file
    """

    # Example input filename:
    # 'corlasso_sub_cloudOnHamsl.M1.m1.gefs00_2018112900_f201500_d3.nc'

    # Split name by '_'
    fn_parts = fname.split('_')
    # Parse the parts
    fnlead = fn_parts[0]
    prefix = fn_parts[2]
    base_date = fn_parts[3]
    fx_time = fn_parts[4]
    domain = fn_parts[5].split('.')[0]

    # Split prefix by '.'
    prefix_parts = prefix.split('.')
    # Parse the parts
    dtype = prefix_parts[0]
    site = prefix_parts[1]
    data_level = prefix_parts[2]
    nameens = prefix_parts[3]

    # Convert base date to Pandas Timestamp
    basedate = f'{base_date[0:4]}-{base_date[4:6]}-{base_date[6:8]}T{base_date[8:10]}'
    basedate = pd.to_datetime(basedate)
    # Convert forecast time to Timedelta
    fxhour = pd.Timedelta(int(f'{fx_time[1:3]}'), 'hours')
    fxmin = pd.Timedelta(int(f'{fx_time[3:5]}'), 'minutes')
    fxsec = pd.Timedelta(int(f'{fx_time[5:7]}'), 'seconds')
    # Get full datetime
    full_time = basedate + fxhour + fxmin + fxsec
    # Make output datetime string
    out_time = full_time.strftime('%Y%m%d.%H%M%S')

    # Shorten data type string
    if dtype == 'cloudOnHamsl': dtype_new = 'cldhamsl'
    if dtype == 'metOnHamsl': dtype_new = 'methamsl'
    if dtype == 'coldpoolOnHamsl': dtype_new = 'cplhamsl'

    # New file name 
    fname_out = f'{fnlead}.{dtype_new}.{base_date}{nameens}{domain}.base.{site}.{data_level}.{out_time}.nc'
    return fname_out


def rename_files_dir(datadir, in_dtypes):
    """
    Rename files in a directory
    """
    # Loop over data types
    for itype in range(len(in_dtypes)):
        # Find all input files
        in_basename = f'corlasso_sub_{in_dtypes[itype]}.'
        in_files = sorted(glob.glob(f'{datadir}{in_basename}*nc'))
        if (len(in_files) > 0):
            # Loop over file
            for ifile in range(len(in_files)):
                # Strip file path to get the input filename
                in_filename = in_files[ifile]
                fname = os.path.basename(in_filename)
                # Make new filename
                fname_out = make_new_filename(fname)
                out_filename = f'{datadir}{fname_out}'
                # Rename file
                os.rename(in_filename, out_filename)
                print(out_filename)
                # import pdb; pdb.set_trace()
        else:
            print(f'No files found: {datadir}{in_basename}*nc')
    return True


if __name__ == "__main__":

    # case_dates = [
    #     '20181129',
    # ]
    # domains = ['d3']
    case_dates = [
        '20181129',
        '20181204',
        '20181205',
        '20181219',
        '20190122',
        '20190123',
        '20190125',
        '20190129',
        '20190208',
    ]
    # domains = ['d3', 'd4']
    domains = ['d2']
    # Input data types
    # in_dtypes = ['cloudOnHamsl', 'metOnHamsl', 'coldpoolOnHamsl']
    in_dtypes = ['cloudOnHamsl', 'metOnHamsl']
    # Root data directory
    # root_dir = '/gpfs/wolf/atm131/proj-shared/zfeng/cacti/les/'
    root_dir = '/gpfs/wolf/atm131/proj-shared/zfeng/cacti/meso/'

    # Loop over case dates
    for icase in range(len(case_dates)):
        # Find sub-directories (ensemble members)
        case_dir = f'{root_dir}{case_dates[icase]}/'
        ensemble_dirs = sorted(glob.glob(f'{case_dir}*/'))

        # Loop over ensemble members
        for iens in range(len(ensemble_dirs)):
            ens_dir = ensemble_dirs[iens]
            
            # Loop over domains
            for idom in range(len(domains)):
                data_dirs = sorted(glob.glob(f'{ens_dir}{domains[idom]}/run/'))
                if len(data_dirs) == 1:
                    datadir = data_dirs[0]
                    # Call rename function
                    status = rename_files_dir(datadir, in_dtypes)

