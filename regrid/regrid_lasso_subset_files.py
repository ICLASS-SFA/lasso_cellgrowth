import glob
import os
import sys
import time
import logging
import yaml
import numpy as np
import xarray as xr
import pandas as pd
from regrid_func import coarsen_variable_filter, coarsen_reflectivity_filter, create_semi_symmetric_array

#-----------------------------------------------------------------------
def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

#-----------------------------------------------------------------------
def regrid_file(in_filename, in_basename, out_dir, out_basename, config):
    logger = logging.getLogger(__name__)
    time_dimname = config.get('time_dimname', 'Time')
    x_coordname = config.get('x_coordname', 'XLONG')
    y_coordname = config.get('y_coordname', 'XLAT')
    z_coordname = config.get('z_coordname', 'HAMSL')
    x_dimname = config.get('x_dimname', 'west_east')
    y_dimname = config.get('y_dimname', 'south_north')
    z_dimname = config.get('z_dimname', 'HAMSL')
    dx = config['dx']
    dy = config['dy']
    regrid_ratio = config.get('regrid_ratio')
    var_names = config.get('var_names', None)

    success = False

    # Read input data
    logger.info(f"Regridding file: {in_filename}")
    ds = xr.open_dataset(in_filename)
    in_time = ds[time_dimname]
    # Get coordinates
    x_coord = ds[x_coordname]
    y_coord = ds[y_coordname]
    z_coord = ds[z_coordname]
    # Grid spacing after regridding
    DX_reg = ds.attrs['DX'] * regrid_ratio
    DY_reg = ds.attrs['DY'] * regrid_ratio

    # Subset variables if var_names is provided
    if var_names is not None:
        ds = ds[var_names]

    # Loop over each variable to regrid
    var_reg_dict = {}
    var_attrs_dict = {}
    for varname in ds.data_vars:
        # Store original attributes
        var_attrs_dict[varname] = ds[varname].attrs.copy()
        
        if 'REFL' in varname:
            # If variable name contains 'REFL', use reflectivity regridding
            var_reg_dict[varname] = coarsen_reflectivity_filter(ds[varname].squeeze().values, regrid_ratio)
        else:
            var_reg_dict[varname] = coarsen_variable_filter(ds[varname].squeeze().values, regrid_ratio)
            if var_reg_dict[varname].ndim == 3:
                # Get regridded z,y,x dimensions
                nz, ny, nx = var_reg_dict[varname].shape

    # Subsample coordinates
    start_idx = int((regrid_ratio-1) / 2)
    x_coord_reg = x_coord.data[start_idx::regrid_ratio,start_idx::regrid_ratio]
    y_coord_reg = y_coord.data[start_idx::regrid_ratio,start_idx::regrid_ratio]

    # Make output filename
    nleadingchar = len(f'{in_basename}')
    fname = os.path.basename(in_filename)
    ftimestr = fname[nleadingchar:]
    out_filename = f'{out_dir}{out_basename}{ftimestr}'

    # Make output coordinates
    # Create a coordinate to mimic subsampling ratio of the full coordinate
    x_coord_1d = create_semi_symmetric_array(nx) * DX_reg
    y_coord_1d = create_semi_symmetric_array(ny) * DY_reg
    
    # Create attributes for coordinates
    x_coord_attrs = {
        'long_name': 'Longitude',
        'units': 'degrees_east',
        'description': f'Coarsened by factor of {regrid_ratio}',
    }
    y_coord_attrs = {
        'long_name': 'Latitude',
        'units': 'degrees_north',
        'description': f'Coarsened by factor of {regrid_ratio}',
    }

    # Define output variables
    # Array dimensions
    dim4d = [time_dimname, z_dimname, y_dimname, x_dimname]
    dim3d = [time_dimname, y_dimname, x_dimname]
    dim2d = [y_dimname, x_dimname]
    # var_dict = {
    #     x_coordname: (dim2d, x_coord_reg, x_coord_attrs),
    #     y_coordname: (dim2d, y_coord_reg, y_coord_attrs),
    # }
    # Loop over var_reg_dict to add to output var_dict
    var_dict = {}
    for varname, vardata in var_reg_dict.items():
        # Get original attributes and add processing info
        var_attrs = var_attrs_dict[varname].copy()
        var_attrs['regridded'] = f'Coarsened by factor of {regrid_ratio} using uniform_filter'
        
        if vardata.ndim == 3:
            var_dict[varname] = (dim4d, np.expand_dims(vardata, axis=0), var_attrs)
        elif vardata.ndim == 2:
            var_dict[varname] = (dim3d, np.expand_dims(vardata, axis=0), var_attrs)
        else:
            logger.warning(f'Variable {varname} has unsupported number of dimensions: {vardata.ndim}')

    # Output coordinates
    coord_dict = {
        time_dimname: ([time_dimname], in_time.data, in_time.attrs),
        z_coordname: ([z_dimname], z_coord.data, z_coord.attrs),
        y_coordname: ([y_dimname, x_dimname], y_coord_reg, y_coord_attrs),
        x_coordname: ([y_dimname, x_dimname], x_coord_reg, x_coord_attrs),
    }
    # Output global attributes
    gattr_dict = {
        'Title': 'Regridded LASSO subset data',
        'SIMULATION_START_DATE': ds.attrs['SIMULATION_START_DATE'],
        'run_name': ds.attrs['run_name'],
        'DX': DX_reg,
        'DY': DY_reg,
        'regrid_ratio': regrid_ratio,
        'original_DX': ds.attrs['DX'],
        'original_DY': ds.attrs['DY'],
        'processing_script': os.path.basename(__file__),
        'Contact': 'Zhe Feng, zhe.feng@pnnl.gov',
        'Institution': 'Pacific Northwest National Laboratory',
        'Created_on': time.ctime(time.time()),
    }

    # Define xarray dataset
    # import pdb; pdb.set_trace()
    ds_out = xr.Dataset(var_dict, coords=coord_dict, attrs=gattr_dict)

    # Set encoding/compression for all variables
    comp = dict(zlib=True)
    encoding = {var: comp for var in ds_out.data_vars}

    # Write to netcdf file
    ds_out.to_netcdf(
        path=out_filename, mode='w', format='NETCDF4', unlimited_dims=time_dimname, encoding=encoding,
    )
    logger.info(f"Successfully wrote regridded data to {out_filename}")
    success = True

    return success


#-----------------------------------------------------------------------
def main():

    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Test data
    in_dir = "/gpfs/wolf2/arm/cli120/world-shared/d3m088/cacti/staged_runs_2/20190123/gefs18/base/les/subset_d3/"
    methamsl_filebase = "corlasso_methamsl_2019012300gefs18d3_base_M1.m1."
    cldhamsl_filebase = "corlasso_cldhamsl_2019012300gefs18d3_base_M1.m1."
    methamsl_filename = f"{in_dir}{methamsl_filebase}20190123.171500.nc"
    cldhamsl_filename = f"{in_dir}{cldhamsl_filebase}20190123.171500.nc"
    out_dir = "/gpfs/wolf2/arm/atm131/proj-shared/zfeng/cacti/les/regrid_subsets/"
    
    config = {
        'time_dimname': 'Time',
        'x_coordname': 'XLONG',
        'y_coordname': 'XLAT',
        'z_coordname': 'HAMSL',
        'x_dimname': 'west_east',
        'y_dimname': 'south_north',
        'z_dimname': 'HAMSL',
        'dx': 2500,
        'dy': 2500,
        'regrid_ratio': 5,
        'var_names': ['WA', 'HGT', 'UA', 'VA'],
    }

    regrid_file(methamsl_filename, methamsl_filebase, out_dir, methamsl_filebase, config)
    # import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()