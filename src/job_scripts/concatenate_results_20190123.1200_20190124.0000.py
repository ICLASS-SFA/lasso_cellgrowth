#!/usr/bin/env python3
"""
Concatenate individual track batch results into a single file.
"""
import xarray as xr
import glob
import os

# Configuration
startdate = '20190123.1200'
enddate = '20190124.0000'
input_path = '/gpfs/wolf2/arm/atm131/proj-shared/zfeng/cacti/les/20190123/eda05/base/d4/stats/tmp/'
output_path = '/gpfs/wolf2/arm/atm131/proj-shared/zfeng/cacti/les/20190123/eda05/base/d4/stats/'
output_basename = 'stats_3d_w_fixshell_'

# Find all batch result files
pattern = f'{input_path}{output_basename}{startdate}_{enddate}_t*.nc'
batch_files = sorted(glob.glob(pattern))
print(f'Found {len(batch_files)} batch files to concatenate')

if len(batch_files) == 0:
    print('No batch files found!')
    exit(1)

# Load and concatenate
print('Loading batch files...')
datasets = []
for f in batch_files:
    print(f'  Loading {os.path.basename(f)}')
    ds = xr.open_dataset(f)
    datasets.append(ds)

print('Concatenating along tracks dimension...')
combined = xr.concat(datasets, dim='tracks')

# Remove duplicate index values along the 'tracks' dimension
print('Removing duplicate tracks indices ...')
combined = combined.drop_duplicates(dim='tracks', keep='first')

# Drop unnecessary variables
drop_vars = ['base_time']
combined = combined.drop_vars(drop_vars)

# Save combined file
output_file = f'{output_path}{output_basename}{startdate}_{enddate}.nc'
print(f'Saving combined file: {output_file}')
combined.to_netcdf(output_file)
combined.close()

# Close individual datasets
for ds in datasets:
    ds.close()

print('Concatenation completed successfully!')
print(f'Final file contains {combined.sizes["tracks"]} tracks')
