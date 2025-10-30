import numpy as np
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