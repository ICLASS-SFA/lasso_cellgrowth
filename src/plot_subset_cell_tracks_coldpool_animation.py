"""
Plot cell tracks on radar reflectivity, echo-top, cold pool, and low-level winds snapshots.

>python plot_subset_cell_tracks_coldpool_animation.py -s STARTDATE -e ENDDATE -c CONFIG.yml 
Optional arguments:
-p 0 (serial), 1 (parallel)
--extent lonmin lonmax latmin latmax (subset domain boundary)
--subset 0 (no), 1 (yes) (subset data before plotting)
--figbasename figure base name (output figure base name)
--figsize width height (figure size in inches)
--output output_directory (output figure directory)
"""
__author__ = "Zhe.Feng@pnnl.gov"
__created_date__ = "08-Jun-2022"

import argparse
import numpy as np
import os, sys
import xarray as xr
import pandas as pd
import math
from skimage.draw import disk
from scipy.ndimage import convolve, binary_erosion, generate_binary_structure
import datetime
import copy
import colorcet as cc
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
# For non-gui matplotlib back end
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
mpl.use('agg')
import dask
from dask.distributed import Client, LocalCluster
import warnings
warnings.filterwarnings("ignore")
from pyflextrkr.ft_utilities import load_config
from pyflextrkr.ft_utilities import load_config, subset_files_timerange

#-----------------------------------------------------------------------
def parse_cmd_args():
    # Define and retrieve the command-line arguments...
    parser = argparse.ArgumentParser(
        description="Plot cell tracks on radar reflectivity snapshots for a user-defined subset domain."
    )
    parser.add_argument("-s", "--start", help="first time in time series to plot, format=YYYY-mm-ddTHH:MM:SS", required=True)
    parser.add_argument("-e", "--end", help="last time in time series to plot, format=YYYY-mm-ddTHH:MM:SS", required=True)
    parser.add_argument("-c", "--config", help="yaml config file for tracking", required=True)
    parser.add_argument("-p", "--parallel", help="flag to run in parallel (0:serial, 1:parallel)", type=int, default=0)
    parser.add_argument("--radar_lat", help="radar latitude", type=float, required=False)
    parser.add_argument("--radar_lon", help="radar longitude", type=float, required=False)
    parser.add_argument("--extent", nargs='+', help="map extent (lonmin, lonmax, latmin, latmax)", type=float, default=None)
    parser.add_argument("--subset", help="flag to subset data (0:no, 1:yes)", type=int, default=0)
    parser.add_argument("--figbasename", help="output figure base name", default="")
    parser.add_argument("--figsize", nargs='+', help="figure size (width, height) in inches", type=float, default=[10,10])
    parser.add_argument("--output", help="ouput directory", default=None)
    args = parser.parse_args()

    # Put arguments in a dictionary
    args_dict = {
        'start_datetime': args.start,
        'end_datetime': args.end,
        'run_parallel': args.parallel,
        'config_file': args.config,
        'radar_lat': args.radar_lat,
        'radar_lon': args.radar_lon,
        'extent': args.extent,
        'subset': args.subset,
        'figbasename': args.figbasename,
        'figsize': args.figsize,
        'out_dir': args.output,
    }

    return args_dict

#-----------------------------------------------------------------------
def circular_mask(radius, resolution):
    """
    Generate a circular mask.
    
    Parameters:
    - radius: Radius of the circle.
    - resolution: The resolution of each pixel.
    
    Returns:
    - mask: 2D numpy array representing the circular mask.
    """
    # Calculate the radius in pixels
    radius_px = radius / resolution
    
    # Calculate the window size to fully contain the circle
    window_size = int(np.ceil(radius_px * 2)) + 1
    
    # Create the circular mask
    mask = np.zeros((window_size, window_size), dtype=np.float64)
    rr, cc = disk((window_size // 2, window_size // 2), radius_px)
    mask[rr, cc] = 1
    
    # Normalize the mask
    mask /= np.sum(mask)
    
    return mask

#-----------------------------------------------------------------------
def label_perimeter(tracknumber, dilationstructure):
    """
    Labels the perimeter on a 2D map from object tracknumber masks.
    """
    # Get unique tracknumbers that is no nan
    tracknumber_unique = np.unique(tracknumber[~np.isnan(tracknumber)]).astype(np.int32)

    # Make an array to store the perimeter
    tracknumber_perim = np.zeros(tracknumber.shape, dtype=np.int32)

    # Loop over each tracknumbers
    for ii in tracknumber_unique:
        # Isolate the cell mask
        itn = tracknumber == ii
        # Erode the cell by 1 pixel
        itn_erode = binary_erosion(itn, structure=dilationstructure).astype(itn.dtype)
        # Subtract the eroded area to get the perimeter
        iperim = np.logical_xor(itn, itn_erode)
        # Label the perimeter pixels with the track number
        tracknumber_perim[iperim == 1] = ii

    return tracknumber_perim

#-----------------------------------------------------------------------
def calc_cell_center(tracknumber, longitude, latitude, xx, yy):
    """
    Calculates the center location from labeled cells.
    """
    
    # Find unique tracknumbers
    tracknumber_uniqe = np.unique(tracknumber[~np.isnan(tracknumber)])
    num_tracknumber = len(tracknumber_uniqe)
    # Make arrays for cell center locations
    lon_c = np.full(num_tracknumber, np.nan, dtype=float)
    lat_c = np.full(num_tracknumber, np.nan, dtype=float)
    xx_c = np.full(num_tracknumber, np.nan, dtype=float)
    yy_c = np.full(num_tracknumber, np.nan, dtype=float)

    # Loop over each tracknumbers to calculate the mean lat/lon & x/y for their center locations
    for ii, itn in enumerate(tracknumber_uniqe):
        iyy, ixx = np.where(tracknumber == itn)
        lon_c[ii] = np.mean(longitude[tracknumber == itn])
        lat_c[ii] = np.mean(latitude[tracknumber == itn])
        xx_c[ii] = np.mean(xx[ixx])
        yy_c[ii] = np.mean(yy[iyy])
        
    return lon_c, lat_c, xx_c, yy_c, tracknumber_uniqe

#-----------------------------------------------------------------------
def calc_latlon(lon1, lat1, dist, angle):
    """
    Haversine formula to calculate lat/lon locations from distance and angle.
    
    lon1:   longitude in [degree]
    lat1:   latitude in [degree]
    dist:   distance in [km]
    angle:  angle in [degree]
    """

    # Earth radius
    # R_earth = 6378.39  # at Equator [km]
    R_earth = 6374.2  # at 40 degree latitude [km]
#     R_earth = 6356.91  # at the pole [km]

    # Conver degrees to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    bearing = math.radians(angle)

    lat2 = math.asin(math.sin(lat1) * math.cos(dist/R_earth) +
                     math.cos(lat1) * math.sin(dist/R_earth) * math.cos(bearing))
    lon2 = lon1 + math.atan2(math.sin(bearing) * math.sin(dist/R_earth) * math.cos(lat1),
                             math.cos(dist/R_earth) - math.sin(lat1) * math.sin(lat2))
    lat2 = math.degrees(lat2)
    lon2 = math.degrees(lon2)

    return lon2, lat2

#-----------------------------------------------------------------------
def get_track_stats(trackstats_file, start_datetime, end_datetime, dt_thres):
    """
    Subset tracks statistics data within start/end datetime

    Args:
        trackstats_file: string
            Track statistics file name.
        start_datetime: string
            Start datetime to subset tracks.
        end_datetime: dstring
            End datetime to subset tracks.
        dt_thres: timedelta
            A timedelta threshold to retain tracks.
            
    Returns:
        track_dict: dictionary
            Dictionary containing track stats data.
    """
    # Read track stats file
    dss = xr.open_dataset(trackstats_file)
    stats_starttime = dss.base_time.isel(times=0)
    # Convert input datetime to np.datetime64
    stime = np.datetime64(start_datetime)
    etime = np.datetime64(end_datetime)
    time_res = dss.attrs['time_resolution_hour']

    # Find track initiated within the time window
    idx = np.where((stats_starttime >= stime) & (stats_starttime <= etime))[0]
    ntracks = len(idx)
    print(f'Number of tracks within input period: {ntracks}')

    # Calculate cell lifetime
    lifetime = dss.track_duration.isel(tracks=idx) * time_res

    # # Select long-lived tracks
    # idx_long = np.where(lifetime > lifetime_longtracks)[0]
    # ntracks_long = len(idx_long)
    # lifetime_long = lifetime.isel(tracks=idx_long)
    # print(f'Number of long tracks within input period: {ntracks_long}')

    # Subset these tracks and put in a dictionary
    track_dict = {
        'ntracks' : ntracks,
        'lifetime' : lifetime,
        'cell_bt' : dss['base_time'].isel(tracks=idx),
        'cell_lon' : dss['cell_meanlon'].isel(tracks=idx),
        'cell_lat' : dss['cell_meanlat'].isel(tracks=idx),
        'start_split_tracknumber' : dss['start_split_tracknumber'].isel(tracks=idx),
        'end_merge_tracknumber' : dss['end_merge_tracknumber'].isel(tracks=idx),
        'dt_thres': dt_thres,
        'time_res': time_res,
        # # Long-lived tracks
        # 'lifetime_long': lifetime_long,
        # 'cell_bt_long' : dss['base_time'].isel(tracks=idx_long),
        # 'cell_lon_long' : dss['cell_meanlon'].isel(tracks=idx_long),
        # 'cell_lat_long' : dss['cell_meanlat'].isel(tracks=idx_long),
        # 'start_split_tracknumber_long' : dss['start_split_tracknumber'].isel(tracks=idx_long),
        # 'end_merge_tracknumber_long' : dss['end_merge_tracknumber'].isel(tracks=idx_long),
    }
    
    return track_dict

#-----------------------------------------------------------------------
def plot_map(pixel_dict, plot_info, map_info, track_dict):
    """
    Plotting function.

    Args:
        pixel_dict: dictionary
            Dictionary containing pixel-level variables
        plot_info: dictionary
            Dictionary containing plotting variables
        map_info: dictionary
            Dictionary containing mapping variables
        track_dict: dictionary
            Dictionary containing tracking variables

    Returns:
        fig: object
            Figure object.
    """
    
    # Get pixel data from dictionary
    pixel_bt = pixel_dict['pixel_bt']
    xx = pixel_dict['longitude']
    yy = pixel_dict['latitude']
    dbz_comp = pixel_dict['dbz_comp']
    echotop10 = pixel_dict['echotop10']/1000
    conv_mask = pixel_dict['conv_mask']
    tn_perim = pixel_dict['tn_perim']
    lon_tn = pixel_dict['lon_tn']
    lat_tn = pixel_dict['lat_tn']
    tracknumbers = pixel_dict['tracknumber_unique']
    XLONG = pixel_dict['XLONG']
    XLAT = pixel_dict['XLAT']
    buoy = pixel_dict['buoy']
    W_ll = pixel_dict['W_ll']
    U_ll = pixel_dict['U_ll']
    V_ll = pixel_dict['V_ll']
    HGT = pixel_dict['HGT']
    # Get track data from dictionary
    ntracks = track_dict['ntracks']
    lifetime = track_dict['lifetime']
    cell_bt = track_dict['cell_bt']
    cell_lon = track_dict['cell_lon']
    cell_lat = track_dict['cell_lat']
    dt_thres = track_dict['dt_thres']
    time_res = track_dict['time_res']
    # Get plot info from dictionary
    # var_scale = plot_info.get('var_scale', 1)
    # levels = plot_info['levels']
    # cmaps = plot_info['cmaps']
    # titles = plot_info['titles']
    # cblabels = plot_info['cblabels']
    # cbticks = plot_info['cbticks']
    fontsize = plot_info['fontsize']
    suptitle = plot_info['suptitle']
    timestr = plot_info['timestr']
    figname = plot_info['figname']
    figsize = plot_info['figsize']
    show_tracks = plot_info.get('show_tracks', True)

    marker_size = plot_info['marker_size']
    lw_centroid = plot_info['lw_centroid']
    cmap_tracks = plot_info['cmap_tracks']
    cblabel_tracks = plot_info['cblabel_tracks']
    cbticks_tracks = plot_info['cbticks_tracks']
    lev_lifetime = plot_info['lev_lifetime']
    # radii = plot_info['radii']
    # azimuths = plot_info['azimuths']
    # radar_lon = plot_info['radar_lon']
    # radar_lat = plot_info['radar_lat']
    buoy_z = plot_info.get('buoy_z')
    buoy_thres = plot_info.get('buoy_thres')
    W_z = plot_info.get('W_z')
    UV_z = plot_info.get('UV_z')
    nvecs = plot_info.get('nvecs')

    # Map domain, lat/lon ticks, background map features
    map_extent = map_info['map_extent']
    lonv = map_info['lonv']
    latv = map_info['latv']

    # Set up track lifetime colors
    cmap_lifetime = plt.get_cmap(cmap_tracks)
    norm_lifetime = mpl.colors.BoundaryNorm(lev_lifetime, ncolors=cmap_lifetime.N, clip=True)
    
    # Set up map projection
    proj = ccrs.PlateCarree()
    data_proj = ccrs.PlateCarree(central_longitude=0)

    # Set up figure
    mpl.rcParams['font.size'] = fontsize
    mpl.rcParams['font.family'] = 'Helvetica'
    fig = plt.figure(figsize=figsize, dpi=200, facecolor='w')
    # Two columns
    gs = gridspec.GridSpec(1, 2, height_ratios=[1], width_ratios=[0.5,0.5],)
    gs.update(left=0.06, right=0.94, bottom=0.05, top=0.92, wspace=0.35, hspace=0.1)
    # Two rows for each column
    gs_l = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0], height_ratios=[1,1], width_ratios=[1,0.02], wspace=0.02, hspace=0.15)
    gs_r = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[1], height_ratios=[1,1], width_ratios=[1,0.02], wspace=0.02, hspace=0.15)
    # Setup axes
    ax1 = plt.subplot(gs_l[0,0], projection=proj)
    ax2 = plt.subplot(gs_r[0,0], projection=proj)
    ax3 = plt.subplot(gs_l[1,0], projection=proj)
    ax4 = plt.subplot(gs_r[1,0], projection=proj)
    cax1 = plt.subplot(gs_l[0,1])
    cax2 = plt.subplot(gs_r[0,1])
    cax3 = plt.subplot(gs_l[1,1])
    cax4 = plt.subplot(gs_r[1,1])

    # Panel titles
    titles = ['Composite Reflectivity','10dBZ Echo-Top',f'{buoy_z}m AGL Buoyancy',f'{W_z}m AGL W']
    ax1.set_title(titles[0], loc='left')
    ax2.set_title(titles[1], loc='left')
    ax3.set_title(titles[2], loc='left')
    ax4.set_title(titles[3], loc='left')
    # Figure suptitle
    fig.suptitle(suptitle, fontweight='bold')

    # Set up terrain height
    cmap0 = copy.copy(mpl.cm.get_cmap('Greys'))
    HGT_lev = np.arange(500, 3001, 500)
    norm0 = mpl.colors.BoundaryNorm(boundaries=HGT_lev, ncolors=cmap0.N)

    #-----------------------------------------------
    # Upper left
    #-----------------------------------------------
    ax1.set_extent(map_extent, crs=data_proj)
    ax1.set_aspect('auto', adjustable=None)
    gl = ax1.gridlines(crs=data_proj, draw_labels=True, linestyle='--', linewidth=0)
    gl.right_labels = False
    gl.top_labels = False
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)
    # 
    cmap1 = copy.copy(mpl.cm.get_cmap('gist_ncar'))
    dbz_lev = np.arange(-10, 70.1, 5)
    norm = mpl.colors.BoundaryNorm(boundaries=dbz_lev, ncolors=cmap1.N)
    dbz_comp = np.ma.masked_where(dbz_comp < min(dbz_lev), dbz_comp)
    pcm0 = ax1.pcolormesh(XLONG, XLAT, HGT, shading='nearest', norm=norm0, cmap=cmap0, transform=data_proj, zorder=0)
    pcm1 = ax1.pcolormesh(xx, yy, dbz_comp, shading='nearest', norm=norm, cmap=cmap1, transform=data_proj, zorder=1, alpha=0.9)
    # Colorbar
    cb1 = plt.colorbar(pcm1, cax=cax1, label=f'Reflectivity (dBZ)', ticks=None, extend='both', orientation='vertical')

    # Overplot cell tracknumber perimeters
    Tn = np.ma.masked_where(tn_perim == 0, tn_perim)
    Tn[Tn > 0] = 10
    tn1 = ax1.pcolormesh(xx, yy, Tn, cmap='gray', transform=proj, zorder=3)

    # Plot track centroids and paths
    if show_tracks:
        marker_style_s = dict(edgecolor='k', linestyle='-', marker='o')
        marker_style_m = dict(edgecolor='k', linestyle='-', marker='o')
        marker_style_l = dict(edgecolor='k', linestyle='-', marker='o')
        for itrack in range(0, ntracks):
            # Get duration of the track
            ilifetime = lifetime.values[itrack]
            idur = (ilifetime / time_res).astype(int)
            # Get basetime of the track and the last time
            ibt = cell_bt.values[itrack,:idur]
            ibt_last = np.nanmax(ibt)
            # Compute time difference between current pixel-level data time and the last time of the track
            idt = (pixel_bt - ibt_last).astype('timedelta64[m]')
            # Proceed if time difference is <= threshold
            # This means for tracks that end longer than the time threshold are not plotted
            if (idt <= dt_thres):
                # Find times in track data <= current pixel-level file time
                idx_cut = np.where(ibt <= pixel_bt)[0]
                idur_cut = len(idx_cut)
                if (idur_cut > 0):
                    color_vals = np.repeat(ilifetime, idur_cut)
                    # Change centroid marker, linewidth based on track lifetime [hour]
                    if (ilifetime < 1):
                        lw_c = lw_centroid[0]
                        size_c = marker_size[0]
                        marker_style = marker_style_s
                    elif ((ilifetime >= 1) & (ilifetime < 2)):
                        lw_c = lw_centroid[1]
                        size_c = marker_size[1]
                        marker_style = marker_style_m
                    elif (ilifetime >= 2):
                        lw_c = lw_centroid[2]
                        size_c = marker_size[2]
                        marker_style = marker_style_l
                    else:
                        lw_c = 0
                        size_c = 0
                    size_vals = np.repeat(size_c, idur_cut)
                    size_vals[0] = size_c * 2   # Make CI symbol size larger
                    _cc = ax1.plot(cell_lon.values[itrack,idx_cut], cell_lat.values[itrack,idx_cut], lw=lw_c, ls='-', color='k', transform=proj, zorder=3)
                    _cl = ax1.scatter(cell_lon.values[itrack,idx_cut], cell_lat.values[itrack,idx_cut], s=size_vals, c=color_vals, 
                                      norm=norm_lifetime, cmap=cmap_lifetime, transform=proj, zorder=4, **marker_style)

        # Plot colorbar for tracks
        cax = inset_axes(ax1, width="100%", height="100%", bbox_to_anchor=(.04, .97, .3, .03), bbox_transform=ax1.transAxes)
        cbinset = mpl.colorbar.ColorbarBase(cax, cmap=cmap_lifetime, norm=norm_lifetime, orientation='horizontal', label=cblabel_tracks)
        cbinset.set_ticks(cbticks_tracks)
            
    # Overplot cell tracknumbers at current frame
    for ii in range(0, len(lon_tn)):
        if (lon_tn[ii] > map_extent[0]) & (lon_tn[ii] < map_extent[1]) & \
            (lat_tn[ii] > map_extent[2]) & (lat_tn[ii] < map_extent[3]):
            ax1.text(lon_tn[ii]+0.02, lat_tn[ii]+0.02, f'{tracknumbers[ii]:.0f}', color='k', size=fontsize*0.8, 
                    weight='bold', ha='left', va='center', transform=proj, zorder=4)

    # Plot range circles around radar
    # for ii in range(0, len(radii)):
    #     rr = ax1.tissot(rad_km=radii[ii], lons=radar_lon, lats=radar_lat, n_samples=100, facecolor='None', edgecolor='k', lw=0.4, zorder=3)
    # # Plot azimuth lines
    # for ii in range(0, len(azimuths)):
    #     lon2, lat2 = calc_latlon(radar_lon, radar_lat, 200, azimuths[ii])
    #     ax1.plot([radar_lon,lon2], [radar_lat,lat2], color='k', lw=0.4, transform=ccrs.Geodetic(), zorder=5)
    # Reflectivity colorbar
    # cb1 = plt.colorbar(cf1, cax=cax1, label=cblabels, ticks=cbticks, extend='both')
    # ax1.set_title(timestr)

    #-----------------------------------------------
    # Upper right
    #-----------------------------------------------
    ax2.set_extent(map_extent, crs=data_proj)
    ax2.set_aspect('auto', adjustable=None)
    gl = ax2.gridlines(crs=data_proj, draw_labels=True, linestyle='--', linewidth=0)
    gl.right_labels = False
    gl.top_labels = False
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)
    # 
    cmap2 = copy.copy(mpl.cm.get_cmap('nipy_spectral'))
    eth_lev = [1,1.5,2,2.5,3,3.5,4,4.5,5,6,7,8,9,10,12,14,16,18,20]
    norm = mpl.colors.BoundaryNorm(boundaries=eth_lev, ncolors=cmap2.N)
    var_fill = np.ma.masked_where(echotop10 < min(eth_lev), echotop10)
    pcm0 = ax2.pcolormesh(XLONG, XLAT, HGT, shading='nearest', norm=norm0, cmap=cmap0, transform=data_proj, zorder=0)
    pcm2 = ax2.pcolormesh(xx, yy, var_fill, shading='nearest', norm=norm, cmap=cmap2, transform=data_proj, zorder=1, alpha=0.9)
    # Overplot cell tracknumber perimeters
    Tn = np.ma.masked_where(tn_perim == 0, tn_perim)
    Tn[Tn > 0] = 10
    tn2 = ax2.pcolormesh(xx, yy, Tn, cmap='gray', transform=proj, zorder=3)
    # Overplot cell tracknumbers at current frame
    for ii in range(0, len(lon_tn)):
        if (lon_tn[ii] > map_extent[0]) & (lon_tn[ii] < map_extent[1]) & \
            (lat_tn[ii] > map_extent[2]) & (lat_tn[ii] < map_extent[3]):
            ax2.text(lon_tn[ii]+0.02, lat_tn[ii]+0.02, f'{tracknumbers[ii]:.0f}', color='k', size=fontsize*0.8, 
                    weight='bold', ha='left', va='center', transform=proj, zorder=4)
    # Colorbar
    cb2 = plt.colorbar(pcm2, cax=cax2, label=f'10dBZ ETH (km)', ticks=None, extend='both', orientation='vertical')

    #-----------------------------------------------
    # Lower left
    #-----------------------------------------------
    ax3.set_extent(map_extent, crs=data_proj)
    ax3.set_aspect('auto', adjustable=None)
    gl = ax3.gridlines(crs=data_proj, draw_labels=True, linestyle='--', linewidth=0)
    gl.right_labels = False
    gl.top_labels = False
    # if (lonv is not None) & (latv is not None):
    #     gl.xlocator = mpl.ticker.FixedLocator(lonv)
    #     gl.ylocator = mpl.ticker.FixedLocator(latv)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax3.xaxis.set_major_formatter(lon_formatter)
    ax3.yaxis.set_major_formatter(lat_formatter)

    cmap3 = copy.copy(mpl.cm.get_cmap('Blues_r'))
    buoy_lev = np.arange(-0.25, -0.05, 0.01)
    norm = mpl.colors.BoundaryNorm(boundaries=buoy_lev, ncolors=cmap3.N)
    pcm0 = ax3.pcolormesh(XLONG, XLAT, HGT, shading='nearest', norm=norm0, cmap=cmap0, transform=data_proj, zorder=0)
    pcm3 = ax3.pcolormesh(XLONG, XLAT, buoy, shading='nearest', norm=norm, cmap=cmap3, transform=data_proj, zorder=1, alpha=0.9)
    # Colorbar
    cb3 = plt.colorbar(pcm3, cax=cax3, label=f'{buoy_z}m'+' AGL Buoyancy (m s$^{-2}$)', ticks=None, extend='both', orientation='vertical')


    #-----------------------------------------------
    # Lower right
    #-----------------------------------------------
    ax4.set_extent(map_extent, crs=data_proj)
    ax4.set_aspect('auto', adjustable=None)
    gl = ax4.gridlines(crs=data_proj, draw_labels=True, linestyle='--', linewidth=0)
    gl.right_labels = False
    gl.top_labels = False

    # # cmap = copy.copy(mpl.cm.get_cmap('RdBu_r'))
    cmap4 = copy.copy(mpl.cm.get_cmap(cc.cm.CET_D3))
    W_lev = [-3,-2,-1,1,2,3]
    norm = mpl.colors.BoundaryNorm(boundaries=W_lev, ncolors=cmap4.N)
    _W_ll = np.ma.masked_where(np.absolute(W_ll) < 1, W_ll)
    pcm0 = ax4.pcolormesh(XLONG, XLAT, HGT, shading='nearest', norm=norm0, cmap=cmap0, transform=data_proj, zorder=0)
    pcm4 = ax4.pcolormesh(XLONG, XLAT, _W_ll, shading='nearest', norm=norm, cmap=cmap4, transform=data_proj, zorder=1)
    # Wind vectors
    # nvecs = [15,15]
    quiver_width = None  # 0.003
    qkscale = 10.0
    Q4 = ax4.quiver(XLONG[::nvecs[0],::nvecs[1]], XLAT[::nvecs[0],::nvecs[1]], \
                U_ll.values[::nvecs[0],::nvecs[1]], V_ll.values[::nvecs[0],::nvecs[1]], \
                # scale=vector_scale, \
                pivot='mid', width=quiver_width, transform=data_proj, zorder=2)
    qk4 = ax4.quiverkey(Q4, 0.85, 1.02, qkscale, f'{qkscale} m/s', labelpos='E', coordinates='axes')
    # 
    cb4 = plt.colorbar(pcm4, cax=cax4, label=f'{W_z}m'+' AGL W (m s$^{-1}$)', ticks=None, extend='both', orientation='vertical')

    # Thread-safe figure output
    canvas = FigureCanvas(fig)
    canvas.print_png(figname)
    fig.savefig(figname)
    
    return fig

#-----------------------------------------------------------------------
def work_for_time_loop(datafile, date_time, track_dict, map_info, plot_info):
    """
    Process data for a single frame and make the plot.

    Args:
        datafile: string
            Pixel-level data filename
        date_time: Timestamp
            Pandas Timestamp matching the pixel-level data file
        track_dict: dictionary
            Dictionary containing tracking variables
        map_info: dictionary
            Dictionary containing mapping variables
        plot_info: dictionary
            Dictionary containing plotting variables

    Returns:
        1.
    """
    
    map_extent = map_info.get('map_extent', None)
    figdir = plot_info.get('figdir')
    figbasename = plot_info.get('figbasename')
    vector_spacing = plot_info.get('vector_spacing')
    # varname_fill = plot_info.get('varname_fill')

    bkg_rad = plot_info.get('bkg_rad')
    buoy_z = plot_info.get('buoy_z')
    buoy_thres = plot_info.get('buoy_thres')
    W_z = plot_info.get('W_z')
    UV_z = plot_info.get('UV_z')

    dir_lasso = config['dir_lasso']
    basename_lasso = config['basename_lasso']
    # LASSO datetime string
    in_datetime = date_time.strftime('%Y%m%d.%H%M%S')
    file_lasso = f'{dir_lasso}{basename_lasso}{in_datetime}.nc'

    if os.path.isfile(file_lasso) != True:
        print(f'ERROR: LASSO file does not exist: {file_lasso}')
        sys.exit()

    # Read cell tracking pixel-level data
    ds = xr.open_dataset(datafile)
    pixel_bt = ds.time.data

    # Get map extent from data
    if map_extent is None:
        lonmin = ds['longitude'].min().item()
        lonmax = ds['longitude'].max().item()
        latmin = ds['latitude'].min().item()
        latmax = ds['latitude'].max().item()
        map_extent = [lonmin, lonmax, latmin, latmax]
        map_info['map_extent'] = map_extent
    else:
        map_extent = map_info['map_extent']
        buffer = 0  # buffer area for subset
        lonmin, lonmax = map_extent[0]-buffer, map_extent[1]+buffer
        latmin, latmax = map_extent[2]-buffer, map_extent[3]+buffer

    # Make dilation structure (larger values make thicker outlines)
    # perim_thick = 1
    # dilationstructure = np.zeros((perim_thick+1,perim_thick+1), dtype=int)
    # dilationstructure[1:perim_thick, 1:perim_thick] = 1
    dilationstructure = generate_binary_structure(2,1)

    # Get cell tracknumbers
    tn = ds['tracknumber'].squeeze()

    # Only plot if there is cell in the frame
    # if (np.nanmax(tn) > 0):
    # Subset pixel data within the map domain
    if subset == 1:
        # map_extent = map_info['map_extent']
        # buffer = 0.05  # buffer area for subset
        # lonmin, lonmax = map_extent[0]-buffer, map_extent[1]+buffer
        # latmin, latmax = map_extent[2]-buffer, map_extent[3]+buffer
        mask = (ds['longitude'] >= lonmin) & (ds['longitude'] <= lonmax) & \
                (ds['latitude'] >= latmin) & (ds['latitude'] <= latmax)
        xx_sub = mask.where(mask == True, drop=True).lon.data
        yy_sub = mask.where(mask == True, drop=True).lat.data
        lon_sub = ds['longitude'].where(mask == True, drop=True).squeeze()
        lat_sub = ds['latitude'].where(mask == True, drop=True).squeeze()
        dbz_comp = ds['dbz_comp'].where(mask == True, drop=True).squeeze()
        echotop10 = ds['echotop10'].where(mask == True, drop=True).squeeze()
        convmask_sub = ds['conv_mask'].where(mask == True, drop=True).squeeze()
        tracknumber_sub = ds['tracknumber'].where(mask == True, drop=True).squeeze()
    else:
        xx_sub = ds['lon'].data
        yy_sub = ds['lat'].data
        lon_sub = ds['longitude'].data.squeeze()
        lat_sub = ds['latitude'].data.squeeze()
        dbz_comp = ds['dbz_comp'].squeeze()
        echotop10 = ds['echotop10'].squeeze()
        convmask_sub = ds['conv_mask'].squeeze()
        tracknumber_sub = ds['tracknumber'].squeeze()

    # Get object perimeters
    tn_perim = label_perimeter(tracknumber_sub.data, dilationstructure)

    # Apply tracknumber to conv_mask
    tnconv = tracknumber_sub.where(convmask_sub > 0).data

    # Calculates cell center locations
    lon_tn, lat_tn, xx_tn, yy_tn, tnconv_unique = calc_cell_center(tnconv, lon_sub, lat_sub, xx_sub, yy_sub)

    # Read LASSO data
    dsl = xr.open_dataset(file_lasso)
    XLONG = dsl.XLONG.squeeze()
    XLAT = dsl.XLAT.squeeze()
    dx = dsl.attrs['DX']
    dy = dsl.attrs['DY']

    # Get number of vectors for specified spacing
    nvec_x = np.ceil(vector_spacing / dx).astype(int)
    nvec_y = np.ceil(vector_spacing / dy).astype(int)
    # Add to plot_info dictionary
    plot_info['nvecs'] = [nvec_x, nvec_y]

    # Calcuate virtual potential temperature. Original formula should have Qcloud, Qrain.
    # thetav = theta*(1. + 0.608*qv + qc + qr)  # virtual potential temperature (K)
    THETA_V = dsl['THETA'] * (1 + 0.608 * dsl['QVAPOR'])
    # Select a low-level
    THETA_V_ll = THETA_V.sel(HAGL=buoy_z).squeeze()
    # Make a circular mask for convolution
    mask_convol = circular_mask(bkg_rad, dx)
    # Virtual potential temperature horizontal mean over mesoscale
    THETA_V_ll_bar = convolve(THETA_V_ll.values, mask_convol, mode='nearest')
    # Calculate buoyancy in relation to mesoscale mean [m/s^2]
    buoy = 9.81*(THETA_V_ll - THETA_V_ll_bar)/THETA_V_ll_bar
    # Only keep negative buoyancy
    buoy_negative = buoy.where(buoy < buoy_thres).copy(deep=True)
    # Make a 2D mask for subsetting the LASSO domain
    mask_subset = ((XLONG >= lonmin) & (XLONG <= lonmax) & \
                    (XLAT >= latmin) & (XLAT <= latmax)).squeeze()
    # Get y/x indices limits from the mask
    y_idx, x_idx = np.where(mask_subset == True)
    xmin, xmax = np.min(x_idx), np.max(x_idx)
    ymin, ymax = np.min(y_idx), np.max(y_idx)

    # Subset buoyancy
    buoy_s = buoy_negative.where(mask_subset, drop=True).squeeze()
    THETA_V_ll = THETA_V_ll.where(mask_subset, drop=True).squeeze()
    # Subset DataSet
    dss = dsl.where(mask_subset, drop=True).squeeze()
    # Rename dimensions
    # y_dimname = 'south_north'
    # x_dimname = 'west_east'
    # dss = dss.rename({x_dimname: 'lon', y_dimname: 'lat'})
    XLONG = dss['XLONG']
    XLAT = dss['XLAT']

    # Get low-level variables
    W_ll = dss.WA.sel(HAGL=W_z)
    U_ll = dss.UA.sel(HAGL=UV_z)
    V_ll = dss.VA.sel(HAGL=UV_z)
    HGT = dss.HGT
    
    # titles = [timestr]
    timestr = ds['time'].squeeze().dt.strftime("%Y-%m-%d %H:%M:%S UTC").data
    suptitle = f"{timestr} ({ensmember.upper()} {domain.upper()})"
    fignametimestr = ds['time'].squeeze().dt.strftime("%Y%m%d_%H%M%S").data.item()
    figname = f'{figdir}{figbasename}{fignametimestr}.png'

    # Put variables in dictionaries
    pixel_dict = {
        'pixel_bt': pixel_bt,
        'longitude': lon_sub, 
        'latitude': lat_sub, 
        'dbz_comp': dbz_comp, 
        'echotop10': echotop10,
        'tn': tracknumber_sub,
        'conv_mask': convmask_sub,
        'tn_perim': tn_perim, 
        'lon_tn': lon_tn, 
        'lat_tn': lat_tn, 
        'tracknumber_unique': tnconv_unique,
        # LASSO data
        'XLONG': XLONG,
        'XLAT': XLAT,
        'buoy': buoy_s,
        # 'buoy': THETA_V_ll,
        'W_ll': W_ll,
        'U_ll': U_ll,
        'V_ll': V_ll,
        'HGT': HGT,
    }
    plot_info['timestr'] = timestr
    plot_info['figname'] = figname
    plot_info['suptitle'] = suptitle

    # Call plotting function
    fig = plot_map(pixel_dict, plot_info, map_info, track_dict)
    plt.close(fig)
    print(figname)

    ds.close()
    return 1



if __name__ == "__main__":

    # Get the command-line arguments...
    args_dict = parse_cmd_args()
    start_datetime = args_dict.get('start_datetime')
    end_datetime = args_dict.get('end_datetime')
    run_parallel = args_dict.get('run_parallel')
    config_file = args_dict.get('config_file')
    # radar_lat = args_dict.get('radar_lat')
    # radar_lon = args_dict.get('radar_lon')
    map_extent = args_dict.get('extent')
    subset = args_dict.get('subset')
    figbasename = args_dict.get('figbasename')
    figsize = args_dict.get('figsize')
    out_dir = args_dict.get('out_dir')

    # Specify plotting info
    # varname_fill = 'dbz_comp'
    # varname_fill = 'echotop10'
    # var_scale = 1     # scale factor for the variable
    # var_scale = 1e-3    # scale factor for the variable
    # Colorfill levels
    # levels = np.arange(-10, 60.1, 5)
    # levels = [1,1.5,2,2.5,3,3.5,4,4.5,5,6,7,8,9,10,12,14,16,18,20]
    lev_lifetime = np.arange(0.5, 4.01, 0.5)
    # Colorbar ticks & labels
    # cbticks = levels
    # cbticks = np.arange(-10, 60.1, 5)
    # cblabels = 'Composite Reflectivity (dBZ)'
    # cblabels = '10 dBZ ETH (km)'
    cblabel_tracks = 'Lifetime (hour)'
    cbticks_tracks = [1,2,3,4]
    # Colormaps
    # cmaps = 'gist_ncar'     # Reflectivity
    cmaps = 'nipy_spectral'     # Echo-top Height
    cmap_tracks = 'Spectral_r'  # Lifetime
    show_tracks = False

    bkg_rad = 50.0 * 1000  # radius for circular area to compute mean ThetaV [m]
    buoy_z = 50  # height for buoyancy [m]
    buoy_thres = -0.05  # buoyancy threshold to define cold pool [m/s^2]
    W_z = 500  # height for W [m]
    UV_z = 100  # height for U,V [m]
    
    # Put plot specifications in a dictionary
    plot_info = {
        # 'varname_fill': varname_fill,
        # 'var_scale': var_scale,
        # 'levels': levels,
        'lev_lifetime': lev_lifetime,
        # 'cbticks': cbticks,
        'cbticks_tracks': cbticks_tracks,
        # 'cblabels': cblabels,
        'cblabel_tracks': cblabel_tracks,
        'fontsize': 10,
        'cmaps': cmaps,
        'cmap_tracks': cmap_tracks,
        'show_tracks': show_tracks,
        'marker_size': [30,30,30],    # track centroid marker size (short, medium, long lived)
        'lw_centroid': [3,3,3],         # track path line width
        'radii': np.arange(20,101,20),  # radii for the radar range rings [km]
        'azimuths': np.arange(0,361,90),   # azimuth angles for HSRHI scans [degree]
        'figbasename': figbasename,
        'figsize': figsize,
        # 'radar_lon': radar_lon,
        # 'radar_lat': radar_lat,
        'bkg_rad': bkg_rad,
        'buoy_z': buoy_z,
        'buoy_thres': buoy_thres,
        'W_z': W_z,
        'UV_z': UV_z,
        'vector_spacing': 7500,  # spacing between wind vectors [m]
    }

    # Customize lat/lon labels
    lonv = None
    latv = None
    # Put map info in a dictionary
    map_info = {
        'map_extent': map_extent,
        'subset': subset,
        'lonv': lonv,
        'latv': latv,
    }

    # Tracks that end longer than this threshold from the current pixel-level frame are not plotted
    # This treshold controls the time window to retain previous tracks
    track_retain_time_min = 15

    # Create a timedelta threshold in minutes
    dt_thres = datetime.timedelta(minutes=track_retain_time_min)

    # Track stats file
    config = load_config(config_file)
    stats_path = config["stats_outpath"]
    pixeltracking_path = config["pixeltracking_outpath"]
    pixeltracking_filebase = config["pixeltracking_filebase"]
    trackstats_filebase = config["trackstats_filebase"]
    startdate = config["startdate"]
    enddate = config["enddate"]
    trackstats_file = f"{stats_path}{trackstats_filebase}{startdate}_{enddate}.nc"
    n_workers = config["nprocesses"]
    domain = config["domain"]

    casedate = startdate[0:8]
    clouddata_path = config['clouddata_path']
    ensmember = config['ensmember']
    cfig = 'base'
    basename_lasso = f'corlasso_methagl_{casedate}00{ensmember}{domain}_{cfig}_M1.m1.'
    # Get LASSO datapath
    # Example: /gpfs/wolf/atm131/proj-shared/money/STARTDATE/ENSMEMBER/CONFIG/les/subset_d3/
    dir_parts = clouddata_path.split(os.path.sep)
    if (domain == 'd3') | (domain == 'd4'):
        resolution = 'les'
    if (domain == 'd2'):
        resolution = 'meso'
    # LASSO subset data directory
    dir_lasso1 = f'/gpfs/wolf2/arm/cli120/world-shared/d3m088/cacti/staged_runs/{casedate}/{ensmember}/{cfig}/{resolution}/subset_{domain}/'
    dir_lasso2 = f'/gpfs/wolf2/arm/cli120/world-shared/d3m088/cacti/staged_runs_2/{casedate}/{ensmember}/{cfig}/{resolution}/subset_{domain}/'
    # Check which directory exists
    if os.path.isdir(dir_lasso1):
        dir_lasso = dir_lasso1
    elif os.path.isdir(dir_lasso2):
        dir_lasso = dir_lasso2
    else:
        print(f'WRF path does not exist: {dir_lasso1}')
        print(f'WRF path does not exist: {dir_lasso2}')
        print(f'Code will exit now.')
        sys.exit()
    # Add LASSO directory and basename to config
    config['dir_lasso'] = dir_lasso
    config['basename_lasso'] = basename_lasso
 
    # Convert datetime string to Epoch time (base time)
    start_basetime = pd.to_datetime(start_datetime).timestamp()
    end_basetime = pd.to_datetime(end_datetime).timestamp()

    # Find all pixel-level files that match the input datetime
    datafiles, \
    datafiles_basetime, \
    datafiles_datestring, \
    datafiles_timestring = subset_files_timerange(
        pixeltracking_path,
        pixeltracking_filebase,
        start_basetime,
        end_basetime,
        time_format="yyyymodd_hhmmss",
    )
    print(f'Number of pixel files: {len(datafiles)}')

    # Output figure directory
    if out_dir is None:
        figdir = f'{pixeltracking_path}quicklooks_trackpaths/'
    else:
        figdir = out_dir
    os.makedirs(figdir, exist_ok=True)
    # Add to plot_info dictionary
    plot_info['figdir'] = figdir

    # Get track stats data
    track_dict = get_track_stats(trackstats_file, start_datetime, end_datetime, dt_thres)

    # Serial option
    if run_parallel == 0:
        for ifile in range(len(datafiles)):
            print(datafiles[ifile])
            # Convert Epoch time to datetime
            idatetime = pd.to_datetime(datafiles_basetime[ifile], unit='s')
            result = work_for_time_loop(datafiles[ifile], idatetime, track_dict, map_info, plot_info)

    # Parallel option
    elif run_parallel == 1:
        # Set Dask temporary directory for workers
        # dask_tmp_dir = config.get("dask_tmp_dir", "./")
        # dask.config.set({'temporary-directory': dask_tmp_dir})
        # Initialize dask
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
        client = Client(cluster)
        results = []
        for ifile in range(len(datafiles)):
            print(datafiles[ifile])
            # Convert Epoch time to datetime
            idatetime = pd.to_datetime(datafiles_basetime[ifile], unit='s')
            result = dask.delayed(work_for_time_loop)(datafiles[ifile], idatetime, track_dict, map_info, plot_info)
            results.append(result)

        # Trigger dask computation
        final_result = dask.compute(*results)
        # cluster.close()
        # client.close()
    
    else:
        sys.exit('Valid parallelization flag not provided')
