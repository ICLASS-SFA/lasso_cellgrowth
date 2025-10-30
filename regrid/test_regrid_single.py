"""
Simple test script for regridding a single file.
Use this to test the regridding functionality before running batch jobs.
"""
import logging
from regrid_func import regrid_file

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def main():
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Test data - small domain (d3)
    in_dir = "/gpfs/wolf2/arm/atm131/proj-shared/money/20190123/gefs18/base/les/subset_d3/"
    methamsl_filebase = "corlasso_methamsl_2019012300gefs18d3_base_M1.m1."
    cldhamsl_filebase = "corlasso_cldhamsl_2019012300gefs18d3_base_M1.m1."

    # Test data - large domain (d4)
    # Try staged_runs_2 first, then staged_runs
    # in_dir = "/gpfs/wolf2/arm/cli120/world-shared/d3m088/cacti/staged_runs_2/20190123/gefs18/base/les/subset_d4/"
    # methamsl_filebase = "corlasso_methamsl_2019012300gefs18d4_base_M1.m1."
    # cldhamsl_filebase = "corlasso_cldhamsl_2019012300gefs18d4_base_M1.m1."

    methamsl_filename = f"{in_dir}{methamsl_filebase}20190123.171500.nc"
    cldhamsl_filename = f"{in_dir}{cldhamsl_filebase}20190123.171500.nc"
    
    # Output follows same subdirectory structure
    out_dir = "/gpfs/wolf2/arm/atm131/proj-shared/zfeng/cacti/les/regrid_subsets/20190123/gefs18/base/les/subset_d3/"
    
    config = {
        'time_dimname': 'Time',
        'x_coordname': 'XLONG',
        'y_coordname': 'XLAT',
        'z_coordname': 'HAMSL',
        'x_dimname': 'west_east',
        'y_dimname': 'south_north',
        'z_dimname': 'HAMSL',
        'regrid_ratio': 5,  # Use smaller ratio for testing
        # 'var_names': ['WA', 'HGT', 'UA', 'VA', 'ITIMESTEP'],  # Subset for testing
    }

    # Test with methamsl file
    logger.info("Testing with methamsl file...")
    success = regrid_file(methamsl_filename, methamsl_filebase, out_dir, methamsl_filebase, config)
    
    if success:
        logger.info("Test completed successfully!")
    else:
        logger.error("Test failed!")

if __name__ == "__main__":
    main()
