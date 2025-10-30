import glob
import os
import sys
import time
import logging
import yaml
import numpy as np
import xarray as xr
from regrid_func import coarsen_variable_filter, coarsen_reflectivity_filter

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
    """
    Regrid a LASSO subset file containing reflectivity and other variables.
    Args:
        in_filename: string
            Input file name.
        in_basename: string
            Base name of input files to identify time strings.
        out_dir: string
            Output directory name.
        out_basename: string
            Base name for output files.
        config: dictionary
            Dictionary containing config parameters.

    Returns:
        success: boolean
    """
    logger = logging.getLogger(__name__)

    # Get configuration parameters
    time_dimname = config.get('time_dimname', 'Time')
    x_coordname = config.get('x_coordname', 'XLONG')
    y_coordname = config.get('y_coordname', 'XLAT')
    z_coordname = config.get('z_coordname', 'HAMSL')
    x_dimname = config.get('x_dimname', 'west_east')
    y_dimname = config.get('y_dimname', 'south_north')
    z_dimname = config.get('z_dimname', 'HAMSL')
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

    # Define output variable dimensions
    dim4d = [time_dimname, z_dimname, y_dimname, x_dimname]
    dim3d = [time_dimname, y_dimname, x_dimname]

    # Process each variable and directly create output dictionary
    var_dict = {}
    nz, ny, nx = None, None, None  # Will be set from first 3D variable
    
    for varname in ds.data_vars:
        # Get original attributes
        var_attrs = ds[varname].attrs.copy()
        
        # Check if variable has spatial dimensions (y_dimname and x_dimname)
        var_dims = ds[varname].dims
        has_spatial_dims = (y_dimname in var_dims) and (x_dimname in var_dims)
        
        if not has_spatial_dims:
            # Pass through variables without spatial dimensions (e.g., ITIMESTEP)
            logger.debug(f"Variable {varname} has no spatial dimensions - passing through without regridding")
            var_attrs['regridded'] = 'Not regridded - no spatial dimensions'
            vardata = ds[varname].squeeze().values
            
            if vardata.ndim == 0:
                # Scalar variable (e.g., Time-only dimension after squeeze)
                var_dict[varname] = ([time_dimname], np.expand_dims(vardata, axis=0), var_attrs)
            elif vardata.ndim == 1:
                # Already has time dimension
                var_dict[varname] = ([time_dimname], vardata, var_attrs)
            else:
                logger.warning(f'Pass-through variable {varname} has unexpected dimensions: {vardata.ndim}')
        
        else:
            # Regrid variables with spatial dimensions
            var_attrs['regridded'] = f'Coarsened by factor of {regrid_ratio} using uniform_filter'
            
            if 'REFL' in varname:
                # If variable name contains 'REFL', use reflectivity regridding
                vardata = coarsen_reflectivity_filter(ds[varname].squeeze().values, regrid_ratio)
            else:
                vardata = coarsen_variable_filter(ds[varname].squeeze().values, regrid_ratio)
            
            # Store dimensions from first 3D variable for coordinate creation
            if vardata.ndim == 3 and nz is None:
                nz, ny, nx = vardata.shape
            
            # Add to output dictionary
            if vardata.ndim == 3:
                var_dict[varname] = (dim4d, np.expand_dims(vardata, axis=0), var_attrs)
            elif vardata.ndim == 2:
                # Store dimensions from 2D variable if no 3D variable processed yet
                if ny is None:
                    ny, nx = vardata.shape
                var_dict[varname] = (dim3d, np.expand_dims(vardata, axis=0), var_attrs)
            else:
                logger.warning(f'Variable {varname} has unsupported number of dimensions: {vardata.ndim}')

    # Subsample coordinates
    start_idx = int((regrid_ratio-1) / 2)
    x_coord_reg = x_coord.data[start_idx::regrid_ratio, start_idx::regrid_ratio]
    y_coord_reg = y_coord.data[start_idx::regrid_ratio, start_idx::regrid_ratio]

    # Make output filename
    nleadingchar = len(f'{in_basename}')
    fname = os.path.basename(in_filename)
    ftimestr = fname[nleadingchar:]
    out_filename = f'{out_dir}{out_basename}{ftimestr}'

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
        # 'dx': 2500,
        # 'dy': 2500,
        'regrid_ratio': 5,
        # 'var_names': ['WA', 'HGT', 'UA', 'VA', 'ITIMESTEP'],
    }

    # regrid_file(methamsl_filename, methamsl_filebase, out_dir, methamsl_filebase, config)
    regrid_file(cldhamsl_filename, cldhamsl_filebase, out_dir, cldhamsl_filebase, config)
    # import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()