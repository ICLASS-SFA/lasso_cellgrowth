import numpy as np
import warnings
import os
import time
import logging
import xarray as xr
from scipy.ndimage import uniform_filter

#-----------------------------------------------------------------------
def coarsen_variable_filter(in_variable, ratio):
    """
    Coarsen a generic variable using uniform_filter with straight averaging.
    Optimized for large arrays with minimal memory usage.
    
    Args:
        in_variable: np.array (3D: z,y,x or 2D: y,x)
            Input variable array
        ratio: int
            Coarsening ratio (e.g., 5 means 5x5 averaging)
    
    Returns:
        out_variable: np.array
            Coarsened variable array
    """
    # Create mask for valid values (handle NaN efficiently)
    mask = ~np.isnan(in_variable)
    
    # Replace NaN with 0 for filtering (in-place to save memory)
    filtered_var = np.where(mask, in_variable, 0.0)
    
    # Set filter size based on dimensions
    if in_variable.ndim == 3:
        filter_size = (1, ratio, ratio)  # Don't average in z-direction
    else:
        filter_size = (ratio, ratio)
    
    # Apply uniform filter to both data and mask
    # Use float32 to save memory if input precision allows
    dtype = np.float32 if in_variable.dtype in [np.float32, np.int32, np.int16] else np.float64
    
    var_sum = uniform_filter(filtered_var.astype(dtype), size=filter_size, mode='constant')
    count_sum = uniform_filter(mask.astype(dtype), size=filter_size, mode='constant')
    
    # Subsample to get final coarsened grid
    start_idx = ratio // 2
    if in_variable.ndim == 3:
        var_coarse = var_sum[:, start_idx::ratio, start_idx::ratio]
        count_coarse = count_sum[:, start_idx::ratio, start_idx::ratio]
    else:
        var_coarse = var_sum[start_idx::ratio, start_idx::ratio]
        count_coarse = count_sum[start_idx::ratio, start_idx::ratio]
    
    # Calculate average where we have valid data
    out_variable = np.full_like(var_coarse, np.nan, dtype=np.float32)
    valid_mask = count_coarse > 0
    out_variable[valid_mask] = var_coarse[valid_mask] / count_coarse[valid_mask]
    
    return out_variable

#-----------------------------------------------------------------------
def coarsen_reflectivity_filter(in_reflectivity, ratio):
    """
    Coarsen radar reflectivity using uniform_filter.
    Radar reflectivity is converted to linear, coarsen, then convert back to dBZ.
    
    Args:
        in_reflectivity: np.array (3D: z,y,x or 2D: y,x)
            Input reflectivity array
        ratio: int
            Coarsening ratio (e.g., 5 means 5x5 averaging)
    
    Returns:
        out_reflectivity: np.array
            Coarsened reflectivity array
    """
    # Convert to linear
    mask = ~np.isnan(in_reflectivity)
    linear_refl = np.where(mask, 10.0 ** (in_reflectivity / 10.0), 0.0)
    
    if in_reflectivity.ndim == 3:
        filter_size = (1, ratio, ratio)  # Don't average in z-direction
    else:
        filter_size = (ratio, ratio)
    
    # Apply uniform filter
    linear_avg = uniform_filter(linear_refl.astype(np.float64), size=filter_size, mode='constant')
    count_avg = uniform_filter(mask.astype(np.float64), size=filter_size, mode='constant')
    
    # Subsample
    start_idx = ratio // 2
    if in_reflectivity.ndim == 3:
        linear_coarse = linear_avg[:, start_idx::ratio, start_idx::ratio]
        count_coarse = count_avg[:, start_idx::ratio, start_idx::ratio]
    else:
        linear_coarse = linear_avg[start_idx::ratio, start_idx::ratio]
        count_coarse = count_avg[start_idx::ratio, start_idx::ratio]
    
    # Convert back to dBZ
    out_reflectivity = np.full_like(linear_coarse, np.nan, dtype=np.float32)
    valid_mask = count_coarse > 0
    
    # Suppress expected warnings for log10 of very small values
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        out_reflectivity[valid_mask] = 10.0 * np.log10(linear_coarse[valid_mask] / count_coarse[valid_mask])
    
    return out_reflectivity

#-----------------------------------------------------------------------
def create_semi_symmetric_array(size):
    """
    Make a semi-symmetric array around 0 increment by 1

    If the size is even, the array starts from -(size // 2) + 1 and goes up to start + size - 1 
    (e.g., for size = 10, the array would be [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5]). 
    If the size is odd, the array starts from -(size // 2) and goes up to start + size - 1 
    (e.g., for size = 9, the array would be [-4, -3, -2, -1, 0, 1, 2, 3, 4]).

    Args:
        size: int
            Size of the array.

    Returns:
        np.array
    """
    if size % 2 == 0:
        start = -(size // 2) + 1
    else:
        start = -(size // 2)
    array = np.arange(start, start + size)
    return array

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
            Required keys:
                - time_dimname: str (default: 'Time')
                - x_coordname: str (default: 'XLONG')
                - y_coordname: str (default: 'XLAT')
                - z_coordname: str (default: 'HAMSL')
                - x_dimname: str (default: 'west_east')
                - y_dimname: str (default: 'south_north')
                - z_dimname: str (default: 'HAMSL')
                - regrid_ratio: int (required)
            Optional keys:
                - var_names: list of str (subset of variables to process)

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
        'processing_script': 'regrid_func.py',
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