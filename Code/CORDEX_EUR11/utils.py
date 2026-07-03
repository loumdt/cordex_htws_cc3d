import numpy as np
import xarray as xr
from datetime import datetime #create file history with creation date
from tqdm import tqdm #create a user-friendly feedback while script is running
from os import listdir, makedirs
from os.path import isfile, join, isdir, exists
import pandas as pd #handle dataframes
import cc3d #connected components patterns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.colors as clrs
import matplotlib.animation as animation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import json
import glob
from plots import *
import subprocess
import os
import mannkendall as mk
from scipy.spatial import cKDTree
from geopy.distance import geodesic as GD
import warnings
warnings.filterwarnings("ignore")
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from windrose import WindroseAxes
from matplotlib.ticker import FuncFormatter
import matplotlib.cm as cm
import pyproj
from ast import literal_eval
from functools import partial

# Split year is the last year of the historical period. In CORDEX runs based on CMIP5 experiment, it is 2005.
# Have to change value for a different version of CORDEX. 
split_year = 2005

def create_files_list_to_load(read_directory_historical=None,read_directory_rcp=None,start_year=1975,end_year=split_year):
    """Creates the list of files to load, and checks for the time consistency of the created list."""
    # Create sorted list of all files in directory
    if read_directory_rcp == None and read_directory_historical == None:
        raise ValueError("Both read_directory_rcp and read_directory_historical are None. At least one must be defined")
    elif read_directory_historical == None:
        existing_files = np.sort(glob.glob(join(read_directory_rcp,"*.nc")))
    elif read_directory_rcp == None:
        existing_files = np.sort(glob.glob(join(read_directory_historical,"*.nc")))
    else:
        existing_files = np.sort(glob.glob(join(read_directory_historical,"*.nc"))+glob.glob(join(read_directory_rcp,"*.nc")))
    # Select files matching the study period
    files_to_load = []
    for file in existing_files:
        file_start_year = int(file[-20:-16])
        file_end_year = int(file[-11:-7])
        if (file_start_year >= start_year and file_start_year <= end_year) or (file_end_year >= start_year and file_end_year <= end_year):
            files_to_load.append((str(file),file_start_year))

    # Sort files based on start year
    dtype = [('path','S1000'),('start year', int)]
    files_to_load = np.array(files_to_load, dtype=dtype)
    files_to_load_array = np.sort(files_to_load, order = ['start year'])
    files_to_load = [files_to_load_array[i][0].decode("utf-8") for i in range(len(files_to_load_array))]

    # Check time consistency
    if len(files_to_load)==0:
        raise ValueError("files_to_load is empty. Check directories and time boundaries.")

    first_file_start_year = int(files_to_load[0][-20:-16])
    if start_year < first_file_start_year:
        raise ValueError(f"Start year {start_year} is before beginning of first file {first_file_start_year}.")
    last_file_end_year = int(files_to_load[-1][-11:-7])
    if end_year > last_file_end_year:
        raise ValueError(f"End year {end_year} is after ending of last file {last_file_end_year}.")

    for i in range(1,len(files_to_load)):
        previous_file_end_year = int(files_to_load[i-1][-11:-7])
        file_start_year = int(files_to_load[i][-20:-16])
        if file_start_year != previous_file_end_year+1:
            raise ValueError(f"Time is not consistent in files_to_load. Missing year(s) between {previous_file_end_year} and {file_start_year}.")
    print(f"Time is consistent between start year {start_year} and end year {end_year}.")
    return(files_to_load)

def compute_seasonal_cycle(read_directory_historical,read_directory_rcp,write_directory,start_year_ref=1975,end_year_ref=2025,temp_variable='tasmax'):
    '''This function computes a seasonal_cycle for each calendar day of the year. 
    By default, the seasonal_cycle is computed over 1975-2025.
    This function can be used with several models and variables.'''

    #Create list of files to load
    if end_year_ref <= split_year:
        files_to_load = create_files_list_to_load(read_directory_historical=read_directory_historical,read_directory_rcp=None,start_year=start_year_ref,end_year=end_year_ref)
    else:
        files_to_load = create_files_list_to_load(read_directory_historical,read_directory_rcp,start_year_ref,end_year_ref)

    # Load multi-file dataset
    ds = xr.open_mfdataset(files_to_load, engine='netcdf4',data_vars='all',chunks={'time': 1461},join='outer')
    da = getattr(ds, temp_variable)

    #if 'x' in da.dims and 'y' in da.dims:
    #    da = da.chunk(chunks={'time':len(da.time),'x': 82,'y': 84})
    #elif 'rlat' in da.dims and 'rlon' in da.dims:
    #    da = da.chunk(chunks={'time':len(da.time),'rlat': 82,'rlon': 84})
    #elif 'lat' in da.dims and 'lon' in da.dims:
    #    da = da.chunk(chunks={'time':len(da.time),'lat': 36,'lon': 54})
    #else:
    #    raise ValueError("Dimensions are not as expected. Should be either ('lat','lon'), ('rlat','lon'), or ('x','y').")
    # Since the files generally cover several years, have to select sub-period (it may not exactly match the boundaries of loaded files)
    mask = (da.time.dt.year>=start_year_ref) & (da.time.dt.year<=end_year_ref)
    da = da.sel(time=mask)
    del mask

    # Drop Feb 29
    original_calendar = da.time.dt.calendar
    if original_calendar != '360_day': 
        da = da.convert_calendar("noleap")
    #da = da.convert_calendar("360_day",align_on='year')
    # Group using dayofyear and sum to compute mean at the end
    seasonal_cycle = da.groupby(da.time.dt.dayofyear).mean(dim="time")

    # Export data to netcdf file
    seasonal_cycle.to_netcdf(join(write_directory,f"seasonal_cycle_{start_year_ref}_{end_year_ref}.nc"))

    # Clear resources
    da.close()
    seasonal_cycle.close()
    ds.close()
    return

def compute_distrib_percentile(read_directory_historical,read_directory_rcp,write_directory,start_year_ref=1975,end_year_ref=2025,temp_variable='tasmax',threshold_value=95,distrib_window_size=15,anomaly=True):
    '''This function computes, for every calendar day, the n-th (n is the threshold_value, default 95) percentile of the corresponding distribution of daily variable. 
    By default, the distribution is computed over 1975-2025.'''

    if distrib_window_size%2==0:
        raise ValueError('distrib_window_size is even. It has to be odd so the window can be centered on the computed day.')

    #Create list of files to load
    if end_year_ref <= split_year: # If all files are in historical directory, ignore rcp directory
        files_to_load = create_files_list_to_load(read_directory_historical=read_directory_historical,read_directory_rcp=None,start_year=start_year_ref,end_year=end_year_ref)
    else:
        files_to_load = create_files_list_to_load(read_directory_historical,read_directory_rcp,start_year_ref,end_year_ref)

    # Load multi-file dataset
    ds = xr.open_mfdataset(files_to_load, engine='netcdf4',data_vars='all',chunks={'time': 1461},join='outer')#,parallel=True)
    da = getattr(ds, temp_variable)
    if 'x' in da.dims and 'y' in da.dims:
        da = da.chunk(chunks={'time':len(da.time),'x': 82,'y': 84})
    elif 'rlat' in da.dims and 'rlon' in da.dims:
        da = da.chunk(chunks={'time':len(da.time),'rlat': 82,'rlon': 84})
    elif 'lat' in da.dims and 'lon' in da.dims:
        da = da.chunk(chunks={'time':len(da.time),'lat': 36,'lon': 54})
    else:
        raise ValueError("Dimensions are not as expected. Should be either ('lat','lon'), ('rlat','lon'), or ('x','y').")
    # Since files generally cover several years, have to select sub-period (it may not exactly match the boundaries of loaded files)
    mask = (da.time.dt.year>=start_year_ref) & (da.time.dt.year<=end_year_ref)
    da = da.sel(time=mask)
    del mask
    
    # Load seasonal_cycle file to create output data structure and compute anomaly
    seasonal_cycle = xr.open_dataarray(join(write_directory,f"seasonal_cycle_{start_year_ref}_{end_year_ref}.nc"), engine='netcdf4')

    # Drop 29 Feb
    original_calendar = da.time.dt.calendar
    if original_calendar != '360_day':
        da = da.convert_calendar("noleap")
        year_length = 365
    else :
        year_length = 360
    # Create threshold table by copying seasonal_cycle table, values will be updated later
    threshold = seasonal_cycle.copy()

    if anomaly:
        for year in tqdm(range(len(da.time)//year_length)): # Iterate over the number of years
            da[year*year_length:(year+1)*year_length,:,:] = da[year*year_length:(year+1)*year_length,:,:] - seasonal_cycle.data # Compute anomaly
    else:
        seasonal_cycle.close()
    
    for day in tqdm(range(1,year_length+1)): # Calendar days ranging from 1 to year_length
        if day - (distrib_window_size//2) <= 0: # day is at the beginning of January, window overlapping with December
            distrib_start_bound = year_length + day - (distrib_window_size//2) # start boundary of the distribution window
            distrib_end_bound = day + (distrib_window_size//2) # end boundary of the distribution window
            mask = (da.time.dt.dayofyear >= distrib_start_bound) + (da.time.dt.dayofyear <= distrib_end_bound) # Create a boolean mask for days between distrib_start_bound and distrib_end_bound
        elif day + (distrib_window_size//2) > year_length: # day is at the end of December , window overlapping with January
            distrib_start_bound = day - (distrib_window_size//2)
            distrib_end_bound = day + (distrib_window_size//2) - year_length
            mask = (da.time.dt.dayofyear >= distrib_start_bound) + (da.time.dt.dayofyear <= distrib_end_bound) # Create a boolean mask for days between distrib_start_bound and distrib_end_bound
        else:
            distrib_start_bound = day - (distrib_window_size//2)
            distrib_end_bound = day + (distrib_window_size//2)
            mask = (da.time.dt.dayofyear >= distrib_start_bound) & (da.time.dt.dayofyear <= distrib_end_bound) # Create a boolean mask for days between distrib_start_bound and distrib_end_bound
        # Apply the mask to select the corresponding days and compute percentile. Take day-1 because day ranges from 1 to 365(or 360) but python indexing ranges from 0 to 364 (or 359)
        threshold.data[day-1,:,:] = np.percentile(da.sel(time=mask).data,threshold_value,axis=0)

    # Export data to netcdf file
    threshold.to_netcdf(join(write_directory,f"distrib_threshold_{threshold_value}_{da.time.dt.calendar}.nc"))

    # Clear resources
    da.close()
    threshold.close()
    if anomaly:
        seasonal_cycle.close()
    ds.close()
    return

def cc3d_scan_heatwaves(read_directory_historical,read_directory_rcp,write_directory,other_data_path,start_year=1975,end_year=2099,start_year_ref=1975,end_year_ref=2025,temp_variable='tasmax',threshold_value=95,relative_threshold=True,distrib_window_size=15,anomaly=True,nb_days=4,resolution_CORDEX=0.11,connectivity=26,dust_threshold=775,bias_adjusted=False):
    '''This function carries out a cc3d scan (https://pypi.org/project/connected-components-3d/) to detect heatwaves in the meteorological database (default ERA5, t2m, tg).
    The heatwaves point are labeled with a number corresponding to a heatwave identifier.
    Otherwise, values are set to -9999.'''
    # Set dust threshold to supress small amount of points. The threshold of 775 have been established empirically with ERA5 0.25° (see Mandonnet et al. 2026 https://hal.science/hal-05495839v1), and is translated according to CORDEX resolution, hence the following calculation
    #if 'ERA5' not in read_directory_historical:#bias_adjusted==False:
    #    dust_threshold = int(dust_threshold * (0.25/resolution_CORDEX)**2) # resolution_CORDEX is given in °

    # Create list of files to load
    if end_year <= split_year: # If all files are in historical directory, ignore rcp directory
        files_to_load = create_files_list_to_load(read_directory_historical=read_directory_historical,read_directory_rcp=None,start_year=start_year,end_year=end_year)
    elif start_year > split_year: # If all files are in rcp directory, ignore historical directory
        files_to_load = create_files_list_to_load(read_directory_historical=None,read_directory_rcp=read_directory_rcp,start_year=start_year,end_year=end_year)
    else:
        files_to_load = create_files_list_to_load(read_directory_historical,read_directory_rcp,start_year,end_year)

    original_calendar = xr.open_dataset(files_to_load[0]).time.dt.calendar
    if original_calendar=='360_day':
        JJA_beg = 151 #150 is June 1st, 240 is August 30th
        JJA_end = 240
    else: # noleap calendar
        original_calendar = "noleap"
        JJA_beg = 152 #152 is June 1st, 243 is August 31st
        JJA_end = 243

    if anomaly:
        seasonal_cycle = xr.open_dataarray(join(write_directory,f"seasonal_cycle_{start_year_ref}_{end_year_ref}.nc"), engine='netcdf4')
        # Keep only JJA values
        mask = (seasonal_cycle.dayofyear>=JJA_beg) & (seasonal_cycle.dayofyear<=JJA_end) # dayofyear ranges from 1 to 365 (or 360)
        seasonal_cycle = seasonal_cycle.sel(dayofyear=mask)

    if relative_threshold: # Load temperature threshold for reference period:
        threshold = xr.open_dataarray(join(other_data_path,'threshold',f"distrib_threshold_{threshold_value}_{original_calendar}.nc"), engine='netcdf4')
        # Keep only JJA values
        mask = (threshold.dayofyear>=JJA_beg) & (threshold.dayofyear<=JJA_end) 
        threshold = threshold.sel(dayofyear=mask)
    else: # If absolute threshold, only need a scalar, not a 3D array
        threshold = threshold_value

    N_labels = 0 # Count the numbers of patterns
    N_heatwaves = 0 # Count real number of heatwaves
    print("Computing cc3d.connected_components labels and dusting...")
    for file in tqdm(files_to_load):
        ds = xr.open_dataset(file,engine='netcdf4')
        da = getattr(ds, temp_variable)
        # Convert calendar if needed, either 360_day and leave it like it, otherwise make sure it is noleap
        if original_calendar == '360_day':
            JJA_duration = 90 # Only 90 days in JJA season if 360_day calendar
        else:
            da = da.convert_calendar("noleap")
            JJA_duration = 92
        # Keep only JJA values
        mask = (da.time.dt.season=='JJA')
        da = da.sel(time=mask)
        file_start_year = da.time.dt.year.data[0]
        for year in range(len(da.time)//JJA_duration): # Iterate over the years, number of years depends on the file
            da_year = da[year*JJA_duration:(year+1)*JJA_duration,:,:] # Select data for the given year
            if anomaly: # Substract seasonal_cycle to compute anomaly
                da_year = da_year - seasonal_cycle.data
            da_year = da_year*(da_year>threshold.data)-9999*(da_year<=threshold.data) # Set to -9999 the values that do not exceed threshold
            stack_temp = -9999*np.ones((JJA_duration,np.shape(da_year)[1],np.shape(da_year)[2])) # Create a 3D array that will hold the temperature values when and where there are heatwaves
            stack_where = np.zeros((np.shape(da_year)[1],np.shape(da_year)[2])) # Create a 2D array that holds the number of consecutive hot days for each location, computed for each day
            for day in range(JJA_duration):
                stack_where[:,:] = stack_where[:,:] + np.ones((np.shape(da_year)[1],np.shape(da_year)[2]))*(da_year[day,:,:]!=-9999) # Add one day to each potential heatwave location
                stack_where[:,:] = stack_where[:,:]*(da_year[day,:,:]!=-9999) # When not adding a day, have to set back the duration to zero
                if day>=nb_days-1: #record the last four days for the corresponding scanning window, add new consecutive hot days and set not hot days to -9999
                    stack_temp[day-(nb_days-1):day+1,:,:] = stack_temp[day-(nb_days-1):day+1,:,:]*(stack_temp[day-(nb_days-1):day+1,:,:]!=-9999)+da_year[day-(nb_days-1):day+1,:,:]*((stack_where>=nb_days)*stack_temp[day-(nb_days-1):day+1,:,:]==-9999)+(-9999*((stack_where<nb_days)*(stack_temp[day-(nb_days-1):day+1,:,:]==-9999))) 

            # Compute connected components for the remaining values of stack_temp
            labels_in = cc3d.dust((stack_temp!=-9999),dust_threshold,connectivity=connectivity)
            labels_out, N_added = cc3d.connected_components(labels_in,connectivity=connectivity,return_N=True) # Return the table of labels and the number of added patterns
            # Initialize output array 
            label = da_year.copy()
            # Record labels and add N_labels offset where labels are nonzero
            label.data = labels_out + N_labels*(labels_out>0)

            #Remove sea heatwaves
            #try:
            #    label.data = remove_sea_heatwaves(read_directory=join(other_data_path,"mask"),labels=label.data,grid_mapping=da.grid_mapping,connectivity=connectivity)
            #except:
            label.data = remove_sea_heatwaves(read_directory=join(other_data_path,"mask"),labels=label.data,grid_mapping='ERA5',connectivity=connectivity)
            
            label = merge_heatwaves_cc3d(label)

            # Get real number of remaining heatwaves
            labels_list = np.unique(label.data)
            labels_list = labels_list[np.where(labels_list!=0)]
            N_heatwaves += len(labels_list)
            
            # Set DataArray to Dataset to set variable name
            label = label.to_dataset(name="label")
            # Save to netCDF 
            label.to_netcdf(join(write_directory,f"labels_cc3d_year_{file_start_year+year}_ref_{start_year_ref}_{end_year_ref}.nc"))

            # Update N_labels
            N_labels += N_added

        # Clear resources
        da.close()
        ds.close()
    label.close()
    threshold.close()
    if anomaly:
        seasonal_cycle.close()
    print(N_heatwaves,"heatwaves detected")
    return

def merge_heatwaves_cc3d(da_label):
    for time_step in da_label.time.data: # Iterate over days
        da_label_day = da_label.sel(time=time_step)
        labels_day = np.unique(da_label_day.data)
        labels_day = labels_day[labels_day!=0]
        active_labels = set(labels_day)
        if len(labels_day)>=2: # Only check if several point clouds on given day
            for k,label in enumerate(labels_day[:-1]): # Ignore last cloud
                if label in active_labels: # Check if value still in data
                    for k_other in range(k+1,len(labels_day)):
                        da_label_day = da_label.sel(time=time_step)
                        lab_other = labels_day[k_other]
                        point_cloud_1 = np.argwhere(da_label_day.data==label)
                        point_cloud_2 = np.argwhere(da_label_day.data==lab_other)
                        if lab_other in active_labels: # Check if value still in data and compute minimal distance between two point clouds
                            tree = cKDTree(point_cloud_1)
                            dist, idx = tree.query(point_cloud_2, k=1)
                            min_dist = dist.min()
                            if min_dist <= 2: # If minimal distance under threshold, replace labels
                                print(lab_other, 'merged with',label)
                                da_label = da_label.where(da_label.data!=lab_other,other=label)
                                active_labels.remove(lab_other)
    return da_label

def remove_sea_heatwaves(read_directory,labels,grid_mapping='ERA5',resolution_CORDEX=0.11,connectivity=26,dust_threshold=775,bias_adjusted=False):
    '''
    '''
    # Set dust threshold to supress small amount of points. The threshold of 775 have been established with ERA5 0.25°, and is translated according to CORDEX resolution, hence the following calculation
    #if grid_mapping!='ERA5':#bias_adjusted==False:
    #    dust_threshold = int(dust_threshold * (0.25/resolution_CORDEX)**2) # resolution_CORDEX is given in °

    land_sea_mask = xr.open_dataset(join(read_directory,f"mask_Europe_land_only_CORDEX_EUR11_{grid_mapping}.nc"),engine='netcdf4').mask # ERA5 land-sea mask

    # Remove sea points and non-European points
    labels = labels * (land_sea_mask.data==0) #mask is 0 for European countries, 1 elsewhere (sea and non-European countries)

    # Remove small heatwaves without using the cc3d dust function
    labels_list = np.unique(labels)
    labels_list = labels_list[np.where(labels_list!=0)] # Remove 0 which is not a heatwave label

    for lab in labels_list:
        if (labels==lab).sum() < dust_threshold:
            labels = labels*((labels!=lab)) # Remove every point where label is one of a small heatwave
        
    return labels

def compute_regional_warming_levels(read_directory_historical,read_directory_rcp,write_directory,other_data_path,start_year=1975,end_year=2099,start_year_ref=1986,end_year_ref=2005,temp_variable='tasmax',ref_period_offset=0.72,running_mean_window_size=20,regional_warming_levels_list=[2.1,2.6,4.0,5.1],bias_adjusted=False):
    # RWL values (regional_warming_levels_list) gathered from https://interactive-atlas.ipcc.ch/regional-information by selecting the four reference regions overlapping with EUR CORDEX domain, and taking the median temperature of each GWL period in the Table Summary
    # Default RWL 0.72°C of the 1986-2005 period is computed in compute_regional_offset.ipynb
    if running_mean_window_size % 2 != 0:
        raise ValueError(f"running_mean_window_size must be an even integer, found {running_mean_window_size}")

    # Create list of files to load
    if end_year <= split_year: # If all files are in historical directory, ignore rcp directory
        files_to_load = create_files_list_to_load(read_directory_historical=read_directory_historical,read_directory_rcp=None,start_year=start_year,end_year=end_year)
    elif start_year > split_year: # If all files are in rcp directory, ignore historical directory
        files_to_load = create_files_list_to_load(read_directory_historical=None,read_directory_rcp=read_directory_rcp,start_year=start_year,end_year=end_year)
    else:
        files_to_load = create_files_list_to_load(read_directory_historical,read_directory_rcp,start_year,end_year)
    
    for i in range(len(files_to_load)): # RWL are computed on average temperature ('tas'), regardless of the chosen temp_variable
        tas_file = files_to_load[i].replace(temp_variable,'tas'+'Adjust'*bias_adjusted)
        files_to_load[i] = tas_file

    # Load multi-file dataset
    ds = xr.open_mfdataset(files_to_load, engine='netcdf4',chunks={'time': 1461},data_vars='all',join='outer')
    # Variable is necessarily 'tas' for the computation of warming levels
    da = getattr(ds,'tas'+'Adjust'*bias_adjusted)
    # Since files generally cover several years, have to select sub-period (it may not exactly match the boundaries of loaded files)
    mask = (da.time.dt.year>=start_year) & (da.time.dt.year<=end_year)
    da = da.sel(time=mask)
    del mask
    da = da.groupby(da.time.dt.year).mean() # Compute annual mean
    da = da.chunk(chunks={'year':len(da.year)})
    # Create reference period (default 1986-2005) average
    mask = (da.year >= start_year_ref)&(da.year <= end_year_ref)
    da_ref = da.sel(year=mask)
    da_ref = da_ref.mean(dim="year") # Average over time, 1 time step remaining

    da = da.rolling(year=running_mean_window_size, center=True).mean() # Compute running mean
    
    # Get grid_mapping to load correct cell_area file
    #try:
    #    grid_mapping = da.grid_mapping
    #except:
    grid_mapping = 'ERA5'
    # Load cell area
    ds_cell_area = xr.open_dataset(join(other_data_path,"cellarea",f"gridarea_CORDEX_EUR11_{grid_mapping}.nc"),engine='netcdf4') # Area of each grid cell, in m²
    da_cell_area = ds_cell_area.cell_area/1e6 # Load DataArray and convert to km²

    # Compute area-weighted mean to obtain 1D-array of annual mean
    weights = da_cell_area
    weights.name = "weights"
    da_weighted = da.weighted(weights)
    da_ref = da_ref.weighted(weights)
    
    # Have to take into account the fact that grid_mapping is not always the same in CORDEX outputs
    if 'x' in da.dims and 'y' in da.dims:
        da_weighted = da_weighted.mean(("x","y")) # 1D-array of annual values
        da_ref = da_ref.mean(("x","y")) # one single value
    elif 'rlat' in da.dims and 'rlon' in da.dims:
        da_weighted = da_weighted.mean(("rlat","rlon")) # 1D-array of annual values
        da_ref = da_ref.mean(("rlat","rlon")) # one single value
    elif 'lat' in da.dims and 'lon' in da.dims:
        da_weighted = da_weighted.mean(("lat","lon")) # 1D-array of annual values
        da_ref = da_ref.mean(("lat","lon")) # one single value
    else:
        raise ValueError("Dimensions are not as expected. Should be either ('lat','lon'), ('rlat','lon'), or ('x','y').")
    da_weighted = da_weighted - da_ref # 1D-array of annual values reprenting the european warming since the reference period (default 1986-2005)
    da_weighted = da_weighted.compute()

    # Create dictionary to hold the data of
    warming_levels_dict = {}
    for level in regional_warming_levels_list:
        level_diff = level - ref_period_offset # Since reference period is not 1850-1900, have to introduce offset (default is 0.61°C for 1986-2005 reference period)
        # Find years warmer than level and get the first matching year
        bool_array = (da_weighted - level_diff > 0)
        if not bool_array.any(): # If the corresponding level has not been reached by the model
            warming_levels_dict[level] = None
        else:
            central_year = int(bool_array.idxmax())
            warming_levels_dict[level] = {"start_year": int(central_year - running_mean_window_size/2), "end_year": int(central_year + (running_mean_window_size/2 - 1))}
    
    with open(join(write_directory,'regional_warming_levels.json'), 'w') as fp:
        json.dump(warming_levels_dict, fp)

    return

#%%
def compute_Russo_HWMId(read_directory_historical,read_directory_rcp,write_directory,start_year=1975,end_year=2099,start_year_ref=1975,end_year_ref=2025,temp_variable='tasmax'):
    """Compute the HWMId index.
    Based on HWMId defined by Russo et al (2015, https://dx.doi.org/10.1088/1748-9326/10/12/124003 )."""

    # Create list of files to load
    if end_year <= split_year: # If all files are in historical directory, ignore rcp directory
        files_to_load = create_files_list_to_load(read_directory_historical=read_directory_historical,read_directory_rcp=None,start_year=start_year,end_year=end_year)
    elif start_year > split_year: # If all files are in rcp directory, ignore historical directory
        files_to_load = create_files_list_to_load(read_directory_historical=None,read_directory_rcp=read_directory_rcp,start_year=start_year,end_year=end_year)
    else:
        files_to_load = create_files_list_to_load(read_directory_historical,read_directory_rcp,start_year,end_year)

    # Load multi-file dataset
    ds = xr.open_mfdataset(files_to_load, engine='netcdf4',data_vars='all',join='outer')#,parallel=True)
    da = getattr(ds,temp_variable)
    original_calendar = da.time.dt.calendar
    if 'x' in da.dims and 'y' in da.dims:
        da = da.chunk(chunks={'time':len(da.time),'x': 82,'y': 84})
    elif 'rlat' in da.dims and 'rlon' in da.dims:
        da = da.chunk(chunks={'time':len(da.time),'rlat': 82,'rlon': 84})
    elif 'lat' in da.dims and 'lon' in da.dims:
        da = da.chunk(chunks={'time':len(da.time),'lat': 36,'lon': 54})
    else:
        raise ValueError("Dimensions are not as expected. Should be either ('lat','lon'), ('rlat','lon'), or ('x','y').")
    if original_calendar != '360_day': # Either calendar is 360_day and leave it this way, or it is other and make sure it is noleap
        da = da.convert_calendar("noleap")
    
    mask = (da.time.dt.year>=start_year) & (da.time.dt.year<=end_year)
    da = da.sel(time=mask)
    mask = (da.time.dt.year>=start_year_ref) & (da.time.dt.year<=end_year_ref)
    da_ref = da.sel(time=mask)
    da = da.sel(time=(da.time.dt.season=='JJA'))
    
    del mask
    
    # Compute values needed for HWMId
    temp_25p = np.percentile(da_ref.groupby(da_ref.time.dt.year).max(), 25, axis=0)
    temp_75p = np.percentile(da_ref.groupby(da_ref.time.dt.year).max(), 75, axis=0)

    # Copy da to create output data structure
    Russo_HWMId = da.copy()

    # Compute HWMId
    Russo_HWMId.data = np.maximum((da - temp_25p)/(temp_75p - temp_25p), 0)

    Russo_HWMId = Russo_HWMId.to_dataset(name="HWMId")

    try:
        Russo_HWMId.to_netcdf(join(write_directory,f"Russo_HWMId.nc")) # Save to netCDF 
    except:
        # Ensure _FillValue is the same as missing_value to avoid encoding error
        Russo_HWMId.HWMId.encoding['_FillValue'] = Russo_HWMId.HWMId.encoding['missing_value']
        Russo_HWMId.to_netcdf(join(write_directory,f"Russo_HWMId.nc")) # Save to netCDF 
        
    Russo_HWMId.close()
    da.close()
    ds.close()
    return

def create_heatwaves_indices_database(read_directory_historical,read_directory_rcp,write_directory,pop_data_path,other_data_path,start_year=1975,end_year=2099,start_year_ref=1975,end_year_ref=2025,temp_variable='tasmax',threshold_value=95,anomaly=True,split_year=2005):
    '''This function is used to create the dataset of the indices of the detected heatwaves. The set of detected heatwaves depends on all the parameters.'''
    # Load dictionary holding the time boundaries of each RWL
    with open(join(write_directory,'regional_warming_levels.json'), 'r') as f:
        RWL_dict = json.load(f)

    # Get grid_mapping
    #try:
    #    grid_mapping = xr.open_dataset(join(write_directory,f"labels_cc3d_year_{start_year}_ref_{start_year_ref}_{end_year_ref}.nc"), engine="netcdf4").label.grid_mapping
    #except:
    grid_mapping = 'ERA5'

    # Create list of temperature files to load
    if end_year <= split_year: # If all files are in historical directory, ignore rcp directory
        files_to_load = create_files_list_to_load(read_directory_historical=read_directory_historical,read_directory_rcp=None,start_year=start_year,end_year=end_year)
    elif start_year > split_year: # If all files are in rcp directory, ignore historical directory
        files_to_load = create_files_list_to_load(read_directory_historical=None,read_directory_rcp=read_directory_rcp,start_year=start_year,end_year=end_year)
    else:
        files_to_load = create_files_list_to_load(read_directory_historical,read_directory_rcp,start_year,end_year)

    original_calendar = getattr(xr.open_dataset(files_to_load[0], engine="netcdf4"),temp_variable).time.dt.calendar
    if original_calendar=='360_day':
        JJA_beg = 151 #151 is June 1st, 240 is August 30th
        JJA_end = 240
    else: # noleap calendar
        original_calendar = "noleap"
        JJA_beg = 152 #152 is June 1st, 243 is August 31st
        JJA_end = 243

    da_threshold = xr.open_dataarray(join(other_data_path,"threshold",f"distrib_threshold_{threshold_value}_{original_calendar}.nc"),engine='netcdf4')
    mask = (da_threshold.dayofyear>=JJA_beg) & (da_threshold.dayofyear<=JJA_end) # dayofyear ranges from 1 to 365 (or 360)
    da_threshold = da_threshold.sel(dayofyear=mask)

    if anomaly:
        seasonal_cycle = xr.open_dataarray(join(write_directory,f"seasonal_cycle_{start_year_ref}_{end_year_ref}.nc"), engine='netcdf4')
        seasonal_cycle = seasonal_cycle.sel(dayofyear=mask) # Keep only JJA values

    da_HWMId = xr.open_dataarray(join(write_directory,"Russo_HWMId.nc"),engine='netcdf4')

    # Load population files
    # FPOP files
    ssp_file_dict = {}
    for ssp in range(1,6):
        ssp_file_dict[f"ds_pop_ssp{ssp}"] = xr.open_dataset(join(pop_data_path,"FPOP",f"FPOP_SSP{ssp}_2020_2100_{grid_mapping}.nc"),engine='netcdf4')
    # GHS-POP file
    ds_ghs_pop = xr.open_dataset(join(pop_data_path,"GHS-POP",f"GHS_POP_1975_2030_{grid_mapping}.nc"),engine='netcdf4')

    # Load cell area
    ds_cell_area = xr.open_dataset(join(other_data_path,"cellarea",f"gridarea_CORDEX_EUR11_{grid_mapping}.nc"),engine='netcdf4') # Area of each grid cell, in m²
    da_cell_area = ds_cell_area.cell_area/1e6 # Load DataArray and convert to km²

    # Create DataFrame
    df_htws = pd.DataFrame(data=None,columns=['Year','Start Date','End Date','RWL_1','RWL_2','RWL_3','RWL_4', # Account for the fact that RWL can overlap. Create 4 RWL columns to avoid edge cases but RWL_3 and RWL_4 will very likely remain empty
    'Intensity','Spatial extent','Accumulated area','Duration','Max','HWMId_sum',
    'Exposed_population_ghs','HWMId_pop_ghs','Exposed_population_ssp1','HWMId_pop_ssp1',
    'Exposed_population_ssp2','HWMId_pop_ssp2','Exposed_population_ssp3','HWMId_pop_ssp3',
    'Exposed_population_ssp4','HWMId_pop_ssp4','Exposed_population_ssp5','HWMId_pop_ssp5',
    'Distance','Speed','Total_exposed_population_ghs','Total_exposed_population_ssp1','Total_exposed_population_ssp2',
    'Total_exposed_population_ssp3','Total_exposed_population_ssp4','Total_exposed_population_ssp5','Global_centroid','Global_centroid_date','Centroid_p1','Centroid_p2',
    'GCM','RCM','simulation','version','ensemble','version_date','calendar','bias-adjusted','grid_mapping','temp_file_path']) # Create DataFrame

    df_summers = pd.DataFrame(data=None,columns=['Year','Frequency','Nb hot days','Accumulated area summer','Mean duration','Max duration',
    'Mean spatial extent', 'Max spatial extent', 'Mean accumulated area', 'Max accumulated area','Mean speed','Max speed','Mean distance','Max distance','temp_file_path'],index=range((end_year-start_year+1)))
    # Initialize variables used to find the correct temperature file to load for each year
    loaded_temp_file = None
    old_loaded_temp_file = None

    heatwaves_year = 0 

    for n_summer,year in tqdm(enumerate(range(start_year,end_year+1))): # Iterate over the years
        # Load labels if there are heatwaves on given year
        try:
            ds_labels = xr.open_dataset(join(write_directory,f"labels_cc3d_year_{year}_ref_{start_year_ref}_{end_year_ref}.nc"), engine="netcdf4") # Load the corresponding nc file
            labels_exist = True
            heatwaves_year += 1
        except:
            labels_exist = False
        df_summers.loc[n_summer,'Year'] = year
        if labels_exist:
            da_labels = ds_labels.label
            # Find temperature file to load 
            found_temp_file = False
            i = 0
            while i<len(files_to_load) and (found_temp_file==False): # Find the file containing the given year
                file_start_year = int(files_to_load[i][-20:-16])
                file_end_year = int(files_to_load[i][-11:-7])
                if year>=file_start_year and year<=file_end_year:
                    loaded_temp_file = files_to_load[i]
                    found_temp_file = True
                i+=1
            if found_temp_file:
                if loaded_temp_file != old_loaded_temp_file: # Only load file if change of file
                    ds_temp = xr.open_dataset(loaded_temp_file,engine='netcdf4')
                    if original_calendar!='360_day':
                        ds_temp = ds_temp.convert_calendar("noleap") # Either calendar is 360_day and we leave it this way, or it is otherwise and we make sure it is noleap
                old_loaded_temp_file = loaded_temp_file # Update old_loaded_temp_file value for next iteration
                da_temp = getattr(ds_temp, temp_variable)
                da_temp = da_temp.sel(time=(da_temp.time.dt.year==year)) # Keep only correct year
                da_temp = da_temp.sel(time=(da_temp.time.dt.season=='JJA')) # Keep only JJA
                if anomaly:
                    da_temp = da_temp - seasonal_cycle.data # Compute anomaly
                da_temp = da_temp - da_threshold.data # Compute threshold exceedance
            else: # If file not found, raise error
                raise ValueError(f"File not found for year {year} in following list of files:\n{files_to_load}")

            # Select correct year for HWMId and keep only JJA
            da_HWMId_year = da_HWMId.sel(time=(da_HWMId.time.dt.year==year))
            da_HWMId_year = da_HWMId_year.sel(time=(da_HWMId_year.time.dt.season=='JJA'))

            # Create list of labels to iterate over
            labels_list = np.unique(da_labels.data)
            labels_list = labels_list[np.where(labels_list!=0)] # Remove 0 which is not a heatwave label
            df_summers.loc[n_summer,'Nb hot days'] = ((da_labels.where(da_labels>0,drop=False).fillna(0).sum(dim=('lat','lon')).data)>0).sum()
            df_summers.loc[n_summer,'Frequency'] = len(labels_list) # Number of heatwaves on the given summer
            df_summers.loc[n_summer,'temp_file_path'] = loaded_temp_file
            cumul_area_summer = 0
            for label in labels_list: # Iterate over the heatwaves (one label = one heatwave).
                df_htws.loc[label,"Year"]=year
                df_htws.loc[label,'temp_file_path'] = loaded_temp_file
                
                # Find the corresponding RWL(s)
                rwl_count=0
                for rwl in RWL_dict:
                    if (RWL_dict[rwl] != None) and (year in range(RWL_dict[rwl]["start_year"],RWL_dict[rwl]["end_year"]+1)):
                        rwl_count += 1 # Find the correct column to fill
                        df_htws.loc[label,f"RWL_{rwl_count}"] = float(rwl) # Record the corresponding RWL
                da = da_labels
                da_bool_htw = da.where(da==label, drop=False).fillna(0)>0 # Select days and grid points for the heatwave of interest and convert to bool array
                da_area_htw = da_bool_htw*da_cell_area.data
                # Workaround bug in da_temp.where(), could not find the explanation of the behaviour
                temp_copy = da.copy()
                temp_copy.data = da_temp.data
                da_temp_htw = temp_copy.where(da==label, drop=True) # This should work but does not: da_temp_htw = da_temp.where(da==label, drop=True)
                # Workaround bug in da_HWMId_year.where(), could not find the explanation of the behaviour
                da_HWMId_htw = da.copy()
                da_HWMId_htw.data = da_HWMId_year.data
                da_HWMId_htw = da_HWMId_htw.where(da==label, drop=True) # This should work but does not: da_HWMId_htw = da_HWMId_year.where(da==label, drop=True)

                df_htws.loc[label,'Year'] = year
                df_htws.loc[label,'Start Date'] = da_temp_htw.time.data[0]
                df_htws.loc[label,'End Date'] = da_temp_htw.time.data[-1]

                labels_bool_2D = np.max(da==label,axis=0) # Squeeze heatwave labels on a boolean 2D-map to see maximum spatial extension

                # Compute weights for area-weighted mean
                weights = da_cell_area.where(labels_bool_2D, drop=True)
                weights.name = "weights"

                df_htws.loc[label,'Intensity'] = da_temp_htw.weighted(weights.fillna(0)).mean().data
                df_htws.loc[label,'Spatial extent'] = (da_cell_area*labels_bool_2D).sum().data
                df_htws.loc[label,'Accumulated area'] = da_area_htw.sum().data
                cumul_area_summer += df_htws.loc[label,'Accumulated area']
                df_htws.loc[label,'Max'] = da_temp_htw.max().data
                df_htws.loc[label,'HWMId_sum'] = da_HWMId_htw.weighted(weights.fillna(0)).sum().data
                if year<=2030: #GHS-POP covers 1975-2030
                    da_pop = ds_ghs_pop.Band1
                    da_pop_year = da_pop.sel(time=(da_pop.time.dt.year==year))
                    
                    # Workaround bug in da_pop_htw in some cases
                    pop_copy = (da_temp.isel(time=0)).copy()
                    pop_copy.data = da_pop_year.data[0]
                    da_pop_htw = pop_copy

                    da_pop_htw = da_pop_htw.where(labels_bool_2D,drop=True)
                    df_htws.loc[label,'Exposed_population_ghs'] = da_pop_htw.sum().data
                    df_htws.loc[label,'HWMId_pop_ghs'] = (da_HWMId_htw*da_pop_htw.data).sum().data
                    df_htws.loc[label,'Total_exposed_population_ghs'] = (da_bool_htw*da_pop_year.data).sum().data
                if year>=2020: # FPOP covers 2020-2100
                    for ssp in range(1,6):
                        da_pop = ssp_file_dict[f"ds_pop_ssp{ssp}"].Band1
                        da_pop_year = da_pop.sel(time=(da_pop.time.dt.year==year))
                    
                        # Workaround bug in da_pop_htw in some cases
                        pop_copy = (da_temp.isel(time=0)).copy()
                        pop_copy.data = da_pop_year.data[0]
                        da_pop_htw = pop_copy

                        da_pop_htw = da_pop_htw.where(labels_bool_2D,drop=True)
                        
                        df_htws.loc[label,f'Exposed_population_ssp{ssp}'] = da_pop_htw.sum().data
                        df_htws.loc[label,f'HWMId_pop_ssp{ssp}'] = (da_HWMId_htw*da_pop_htw.data).sum().data
                        df_htws.loc[label,f'Total_exposed_population_ssp{ssp}'] = (da_bool_htw*da_pop_year.data).sum().data
                
                label_coords = np.where(da_bool_htw) # time,lat,lon
                df_htws.loc[label,'Duration'] = len(np.unique(label_coords[0]))
                lat_where = da.lat[label_coords[1]].data # convert list of indices to value of latitudes
                lon_where = da.lon[label_coords[2]].data # convert list of indices to value of longitudes
                centroid_time = int(np.nansum(label_coords[0]*da_area_htw.data[label_coords[0],label_coords[1],label_coords[2]])/df_htws.loc[label,'Accumulated area'])
                centroid_lat = np.nansum(lat_where*da_area_htw.data[label_coords[0],label_coords[1],label_coords[2]])/df_htws.loc[label,'Accumulated area']
                centroid_lon = np.nansum(lon_where*da_area_htw.data[label_coords[0],label_coords[1],label_coords[2]])/df_htws.loc[label,'Accumulated area']
                df_htws.loc[label,'Global_centroid'] = str((centroid_lat,centroid_lon))
                df_htws.loc[label,'Global_centroid_date'] = da.time.data[centroid_time]

                labels_p1 = da.where(da==label, drop=False).where(da.time<=da.time[centroid_time]).fillna(0)>0
                labels_p2 = da.where(da==label, drop=False).where(da.time>da.time[centroid_time]).fillna(0)>0

                cumul_area_p1 = (labels_p1*da_area_htw.data).fillna(0).sum().data
                cumul_area_p2 = (labels_p2*da_area_htw.data).fillna(0).sum().data

                # Compute Centroid_p1
                label_coords = np.where(labels_p1) # time,lat,lon
                lat_where = da.lat[label_coords[1]].data # convert list of indices to value of latitudes
                lon_where = da.lon[label_coords[2]].data # convert list of indices to value of longitudes
                centroid_lat = np.nansum(lat_where*da_area_htw.data[label_coords[0],label_coords[1],label_coords[2]])/cumul_area_p1
                centroid_lon = np.nansum(lon_where*da_area_htw.data[label_coords[0],label_coords[1],label_coords[2]])/cumul_area_p1
                centroid_p1 = (centroid_lat,centroid_lon)
                df_htws.loc[label,'Centroid_p1'] = str(centroid_p1)
                # Compute Centroid_p2
                label_coords = np.where(labels_p2) # time,lat,lon
                lat_where = da.lat[label_coords[1]].data # convert list of indices to value of latitudes
                lon_where = da.lon[label_coords[2]].data # convert list of indices to value of longitudes
                centroid_lat = np.nansum(lat_where*da_area_htw.data[label_coords[0],label_coords[1],label_coords[2]])/cumul_area_p2
                centroid_lon = np.nansum(lon_where*da_area_htw.data[label_coords[0],label_coords[1],label_coords[2]])/cumul_area_p2
                centroid_p2 = (centroid_lat,centroid_lon)
                df_htws.loc[label,'Centroid_p2'] = str(centroid_p2)
                # Compute distance and speed
                df_htws.loc[label,'Distance'] = GD(centroid_p1,centroid_p2).km
                df_htws.loc[label,'Speed'] = df_htws.loc[label,'Distance']/df_htws.loc[label,'Duration']
                
                if year <= split_year:
                    df_htws['simulation'] = "historical"
                else:
                    df_htws['simulation'] = read_directory_rcp.split("/")[-7] #rcp45 or rcp85
            df_summers.loc[n_summer,'Accumulated area summer'] = cumul_area_summer
        else:
            df_summers.loc[n_summer,'Frequency'] = 0
            df_summers.loc[n_summer,'Nb hot days'] = 0
            df_summers.loc[n_summer,'Accumulated area summer'] = 0
        df_htws_year = df_htws[df_htws["Year"]==year]
        df_summers.loc[n_summer,'Mean duration'] = df_htws_year['Duration'].mean()
        df_summers.loc[n_summer,'Max duration'] = df_htws_year['Duration'].max()
        df_summers.loc[n_summer,'Mean spatial extent'] = df_htws_year['Spatial extent'].mean()
        df_summers.loc[n_summer,'Max spatial extent'] = df_htws_year['Spatial extent'].max()
        df_summers.loc[n_summer,'Mean accumulated area'] = df_htws_year['Accumulated area'].mean()
        df_summers.loc[n_summer,'Max accumulated area'] = df_htws_year['Accumulated area'].max()
        df_summers.loc[n_summer,'Mean speed'] = df_htws_year['Speed'].mean()
        df_summers.loc[n_summer,'Max speed'] = df_htws_year['Speed'].max()
        df_summers.loc[n_summer,'Mean distance'] = df_htws_year['Distance'].mean()
        df_summers.loc[n_summer,'Max distance'] = df_htws_year['Distance'].max()


    print(f'{heatwaves_year} years with heatwaves out of {end_year-start_year+1} years.')

    #TODO Update with bias-adjusted case
    df_htws.loc[:,'GCM'] = read_directory_historical.split("/")[-8]
    df_htws.loc[:,'RCM'] = read_directory_historical.split("/")[-5]
    df_htws.loc[:,'version'] = read_directory_historical.split("/")[-4]
    df_htws.loc[:,'ensemble'] = read_directory_historical.split("/")[-6]
    df_htws.loc[:,'version_date'] = read_directory_historical.split("/")[-1]
    df_htws.loc[:,'calendar'] = original_calendar
    df_htws.loc[:,'bias-adjusted'] = ("CDFt" in read_directory_rcp)
    df_htws.loc[:,'grid_mapping'] = grid_mapping

    
    #Save dataframe
    df_htws.to_csv(join(write_directory,"df_htws.csv"))
    df_summers.to_csv(join(write_directory,"df_summers.csv"))

    # Clear resources
    da.close()
    da_labels.close()
    ds_labels.close()
    da_temp.close()
    ds_temp.close()
    ds_ghs_pop.close()
    for ssp in range(1,6):
        ssp_file_dict[f"ds_pop_ssp{ssp}"].close()
    return

def mylog(x):
    if x>0:
        res = np.log10(x)
    else:
        res = np.nan
    return res

def compute_grid_points_stats(write_directory,start_year=1975,end_year=2099,start_year_ref=1975,end_year_ref=2025):
    for i,year in tqdm(enumerate(range(start_year,end_year+1))):
        da_label = xr.open_dataset(join(write_directory,f"labels_cc3d_year_{year}_ref_{start_year_ref}_{end_year_ref}.nc"),engine='netcdf4').label
        if i==0:
            time = pd.date_range(str(start_year), periods=end_year-start_year+1,freq='YE')
            reference_time = pd.Timestamp(f"{start_year}-01-01")
            lat = da_label.lat.data
            lon = da_label.lon.data
            da_nb_hot_days = xr.DataArray(
            data=None,
            dims=["time", "lat", "lon"],
            coords=dict(
                lon=(["lon"], lon),
                lat=(["lat"], lat),
                time=time,
                reference_time=reference_time,
                ),
            attrs=dict(
                description="Number of hot days",
                units="days",
                ),
            )
            da_nb_max_continuous_hot_days = da_nb_hot_days.copy()
            da_nb_mean_continuous_hot_days = da_nb_hot_days.copy()
            da_nb_min_rest_hot_days = da_nb_hot_days.copy()
            da_nb_max_rest_hot_days = da_nb_hot_days.copy()
            da_nb_mean_rest_hot_days = da_nb_hot_days.copy()
        
        bool_labels = da_label.where(da_label>0,drop=False)
        bool_labels.data = (bool_labels.data>0)
        da_nb_hot_days.data[i,:,:] = bool_labels.sum(dim='time')
        JJA_duration = np.shape(bool_labels.data)[0]

        stack_where = da_label.copy() # Create a 3D array that holds the number of consecutive hot days for each location, computed for each day
        stack_where.data[1:,:,:] = 0
        stack_where.data[0,:,:] = bool_labels.data[0,:,:]
        
        ever_positive_bool = bool_labels.data[0,:,:]
        stack_where_rest = stack_where.copy() # Create a 3D array that holds the number of consecutive hot days for each location, computed for each day
        stack_where_rest.data[:] = 0
        for day in range(1,JJA_duration):
            stack_where[day,:,:] = stack_where[day-1,:,:] + bool_labels.data[day,:,:] # Add one day to each potential heatwave location
            stack_where[day,:,:] = stack_where[day,:,:]*bool_labels.data[day,:,:] # When not adding a day, have to set back the duration to zero
            
            ever_positive_bool = (ever_positive_bool+stack_where_rest[day,:,:])>0
            stack_where_rest[day,:,:] = ever_positive_bool*(stack_where_rest[day-1,:,:]+(bool_labels.data[day,:,:]==0))
            stack_where_rest[day,:,:] = ever_positive_bool*stack_where_rest[day,:,:]*(bool_labels.data[day,:,:]==0)
        for day in range(JJA_duration-1):
            stack_where[day,:,:] = stack_where[day,:,:]*(stack_where[day+1,:,:]==0) # Remove values that are not the last day of continuous sequence
            stack_where_rest[day,:,:] = stack_where_rest[day,:,:]*(stack_where_rest[day+1,:,:]==0) # Remove values that are not the last day of continuous sequence
        stack_where_rest[-1,:,:] = 0 # Remove last day of summer since it should not be considered rest until next heatwave
        stack_where_rest = stack_where_rest.where(stack_where_rest>0)
        da_nb_max_continuous_hot_days.data[i,:,:] = stack_where.max(dim='time').data
        da_nb_mean_continuous_hot_days.data[i,:,:] = stack_where.mean(dim='time',skipna=True).data
        da_nb_max_rest_hot_days.data[i,:,:] = stack_where_rest.max(dim='time').fillna(JJA_duration).data
        da_nb_min_rest_hot_days.data[i,:,:] = stack_where_rest.min(dim='time').fillna(JJA_duration).data
        da_nb_mean_rest_hot_days.data[i,:,:] = stack_where_rest.mean(dim='time',skipna=True).fillna(JJA_duration).data

    da_nb_hot_days.to_netcdf(join(write_directory,'da_nb_hot_days.nc'))
    da_nb_max_continuous_hot_days.to_netcdf(join(write_directory,'da_nb_max_continuous_hot_days.nc'))
    da_nb_mean_continuous_hot_days.to_netcdf(join(write_directory,'da_nb_mean_continuous_hot_days.nc'))
    da_nb_min_rest_hot_days.to_netcdf(join(write_directory,'da_nb_min_rest_hot_days.nc'))
    da_nb_max_rest_hot_days.to_netcdf(join(write_directory,'da_nb_max_rest_hot_days.nc'))
    da_nb_mean_rest_hot_days.to_netcdf(join(write_directory,'da_nb_mean_rest_hot_days.nc'))

    da_dict = {'da_nb_hot_days':da_nb_hot_days,'da_nb_max_continuous_hot_days':da_nb_max_continuous_hot_days,'da_nb_mean_continuous_hot_days':da_nb_mean_continuous_hot_days,
    'da_nb_min_rest_hot_days':da_nb_min_rest_hot_days,'da_nb_max_rest_hot_days':da_nb_max_rest_hot_days,'da_nb_mean_rest_hot_days':da_nb_mean_rest_hot_days}

    for da_name in tqdm(da_dict.keys()):
        da = da_dict[da_name]
        da = da.astype(float)
        if 'ERA5' in write_directory.split('/') or 'output_ERA5' in write_directory.split('/'): # Only for 1975-2025 period
            func = partial(
                mk_wrapper,
                dates=np.array([pd.Timestamp(i).to_pydatetime() for i in da.time.data]),
                resolution=0.9,
            )

            p, ss, slope, lcl, ucl = xr.apply_ufunc(
                func,
                da,
                input_core_dims=[["time"]],
                output_core_dims=[[], [], [], [], []],
                output_dtypes=[float, float, float, float, float],
                vectorize=True,
                dask="parallelized",
            )

            mk_da = xr.DataArray(
                data=[p.data, ss.data, slope.data, lcl.data, ucl.data],
                dims=["value", "lat", "lon"],
                coords=dict(
                    lon=(["lon"], lon),
                    lat=(["lat"], lat),
                    value=['p', 'ss', 'slope', 'lcl', 'ucl'],
                        ),

                attrs=dict(
                    description="Mann-Kendall trend test",
                ),
            )
            mk_da.to_netcdf(join(write_directory,f"mk_{da_name}_{start_year_ref}_{end_year_ref}.nc"))
        else: # Compute trends for both period
            for (start,end) in [(start_year_ref,end_year_ref),(end_year_ref+1,end_year)]:
                sub_da = da.sel(time=(da.time.dt.year>=start)&(da.time.dt.year<=end))
                func = partial(
                    mk_wrapper,
                    dates=np.array([pd.Timestamp(i).to_pydatetime() for i in sub_da.time.data]),
                    resolution=0.01,
                )

                p, ss, slope, lcl, ucl = xr.apply_ufunc(
                    func,
                    sub_da,
                    input_core_dims=[["time"]],
                    output_core_dims=[[], [], [], [], []],
                    output_dtypes=[float, float, float, float, float],
                    vectorize=True,
                    dask="parallelized",
                )

                mk_da = xr.DataArray(
                    data=[p.data, ss.data, slope.data, lcl.data, ucl.data],
                    dims=["value", "lat", "lon"],
                    coords=dict(
                        lon=(["lon"], lon),
                        lat=(["lat"], lat),
                        value=['p', 'ss', 'slope', 'lcl', 'ucl'],
                            ),

                    attrs=dict(
                        description="Mann-Kendall trend test",
                    ),
                )
                mk_da.to_netcdf(join(write_directory,f"mk_{da_name}_{start_year_ref}_{end_year_ref}.nc"))
    return

def mk_wrapper(obs,dates,resolution):
    res = mk.mk_temp_aggr(
        multi_obs_dts=dates,
        multi_obs=obs,
        resolution=resolution,
    )
    res = res[1]
    return res['p'], res['ss'], res['slope'], res['lcl'], res['ucl']

def merge_heatwaves_dataframes(read_directory,write_directory,start_year_ref=1975,end_year_ref=2025,regional_warming_levels_list=[2.1,2.6,4.0,5.1]):
    dir_list = [item for item in listdir(read_directory) if isdir(join(read_directory,item))] # List all subdirectories
    dir_list = [_dir for _dir in dir_list if 'figs' not in _dir]
    dataframe_path_list = [join(read_directory,subdir,"df_htws.csv") for subdir in dir_list if exists(join(read_directory,subdir,"df_htws.csv"))] # Get all df_htws.csv files, containing heatwaves indices and data
    dataframe_summers_path_list = [join(read_directory,subdir,"df_summers.csv") for subdir in dir_list if exists(join(read_directory,subdir,"df_summers.csv"))] # Get all df_summers.csv files, containing heatwaves indices and data

    df_global_htws = pd.DataFrame(data=None,columns=['Year','Start Date','End Date','model','original_label','RWL_1','RWL_2','RWL_3','RWL_4',
    'Intensity','Spatial extent','Accumulated area','Duration','Max','HWMId_sum',
    'Exposed_population_ghs','HWMId_pop_ghs','Exposed_population_ssp1','HWMId_pop_ssp1',
    'Exposed_population_ssp2','HWMId_pop_ssp2','Exposed_population_ssp3','HWMId_pop_ssp3',
    'Exposed_population_ssp4','HWMId_pop_ssp4','Exposed_population_ssp5','HWMId_pop_ssp5',
    'Distance','Speed','Total_exposed_population_ghs','Total_exposed_population_ssp1','Total_exposed_population_ssp2',
    'Total_exposed_population_ssp3','Total_exposed_population_ssp4','Total_exposed_population_ssp5',
    'GCM','RCM','simulation','version','ensemble','version_date','calendar','bias-adjusted','grid_mapping','temp_file_path'])
    
    df_global_summers = pd.DataFrame(data=None,columns=['Year','Frequency','Nb hot days','Accumulated area summer','Mean duration','Max duration',
    'Mean spatial extent', 'Max spatial extent', 'Mean accumulated area', 'Max accumulated area','Mean speed','Max speed','Mean distance','Max distance','temp_file_path'])

    for df_path in tqdm(dataframe_path_list):
        df_htws = pd.read_csv(df_path,index_col=0,header=0,parse_dates=["Start Date", "End Date"],date_format="%Y/%m/%d")
        df_htws.insert(loc=2,column='model',value=df_path.split("/")[-2])
        df_htws.insert(loc=3,column='original_label',value=df_htws.index)
        df_global_htws = pd.concat([df_global_htws,df_htws],ignore_index=True)
    
    for df_path in tqdm(dataframe_summers_path_list):
        df_htws = pd.read_csv(df_path,index_col=0,header=0)
        model = df_path.split("/")[-2]
        df_htws.insert(loc=0,column='model',value=model)
        df_htws.insert(loc=1,column='calendar',value=df_global_htws[df_global_htws['model']==model]['calendar'].iloc[0])
        df_global_summers = pd.concat([df_global_summers,df_htws],ignore_index=True)
    
    df_global_htws["Period RWL 1"] = False
    df_global_htws["Period RWL 2"] = False
    df_global_htws["Period RWL 3"] = False
    df_global_htws["Period RWL 4"] = False
    df_global_htws["Reanalysis"] = False
    df_global_htws["Historical"] = False

    for idx in tqdm(df_global_htws.index):
        for i in range(4):
            rwl = regional_warming_levels_list[i]
            df_global_htws.loc[idx,f"Period RWL {i+1}"] = (df_global_htws.loc[idx,"RWL_1"]==rwl) + (df_global_htws.loc[idx,"RWL_2"]==rwl) + (df_global_htws.loc[idx,"RWL_3"]==rwl) + (df_global_htws.loc[idx,"RWL_4"]==rwl)>0
        if df_global_htws.loc[idx,"model"]=='ERA5': #TODO Check this condition when data is ready
            df_global_htws.loc[idx,"Reanalysis"] = True
        elif df_global_htws.loc[idx,'Year']>=start_year_ref and df_global_htws.loc[idx,'Year']<=end_year_ref:
            df_global_htws["Historical"]=True

    df_global_htws.to_csv(join(write_directory,"df_global_htws.csv"))
    df_global_summers.to_csv(join(write_directory,"df_global_summers.csv"))
    return

def remap_labels_for_comparison(read_directory='/scratchu/tmandonnet/CORDEX',write_directory='/scratchu/tmandonnet/CORDEX/remapped_labels_for_figs',target_grid_directory='/data/tmandonnet/CORDEX/cellarea',mapping_target='rotated_pole',overwrite=False):
    df_htws = pd.read_csv(join(read_directory,'figs_tasmax_period_1975_2099_ref_1975_2025_threshold_95_4days_connec_26_ano','df_global_htws.csv'),header=0,index_col=0)
    for grid_mapping in ['ERA5','Lambert_Conformal','rotated_pole','rotated_latitude_longitude']:
        if grid_mapping != mapping_target: # Need to remap
            mapping_model_list = np.unique(df_htws[df_htws['grid_mapping']==grid_mapping]['model'])
            for model in tqdm(mapping_model_list):
                print(model)
                labels_files = glob.glob(join(read_directory,model,'labels*.nc'))
                for file in labels_files:
                    if overwrite == True or exists(file) == False:
                        makedirs(join(write_directory,model),exist_ok=True)
                        subprocess.call(f"cdo -remapnn,{join(target_grid_directory,f'gridarea_CORDEX_EUR11_{mapping_target}.nc')} {file} {join(write_directory,model,file.split('/')[-1])}",shell=True)
        else: # i.e grid_mapping is mapping_target, just copy file
            mapping_model_list = np.unique(df_htws[df_htws['grid_mapping']==grid_mapping]['model'])
            for model in tqdm(mapping_model_list):
                print(model)
                labels_files = glob.glob(join(read_directory,model,'labels*.nc'))
                for file in labels_files:
                    if overwrite == True or exists(file) == False:
                        makedirs(join(write_directory,model),exist_ok=True)
                        subprocess.call(f"cp {file} {join(write_directory,model,file.split('/')[-1])}",shell=True)

def plot_4_panel_hot_days(read_directory,write_directory,start_year=2025,end_year=2099,need_to_compute_labels=False):
    """Plot hot days per location"""
    if need_to_compute_labels:
        files_list = np.sort(glob.glob(join(read_directory,'*Adjust*','labels*.nc')))
        # Initialize arrays to hold data
        labels_days_adj = xr.open_dataset(files_list[0],engine='netcdf4').label.copy().sum(dim='time')
        labels_days_adj.data = np.zeros(np.shape(labels_days_adj))
        labels_htws_adj = labels_days_adj.copy()
        labels_days_adj_92 = labels_days_adj.copy()
        labels_days_adj_60 = labels_days_adj.copy()
        labels_days_adj_30 = labels_days_adj.copy()
        
        total_summer_days = 0

        df_htws = pd.read_csv(join(write_directory,'df_htws_BC_results.csv'),header=0,index_col=0)
        model_list = np.unique(df_htws['model'])
        model_list = model_list[model_list!='ERA5']

        # Iterate over files
        for file in tqdm(files_list):
            year = int(file[-21:-17])
            model = file.split("/")[-2]
            if model in model_list:
                calendar = df_htws[df_htws['model']==model]['calendar'].iloc[0]
                if calendar=='360_day':
                    JJA_duration = 90
                else:
                    JJA_duration = 92
                if year >= start_year and year <= end_year:
                    total_summer_days += JJA_duration
                    file_labels = xr.open_dataset(file,engine='netcdf4').label
                    bool_labels = (file_labels>0)
                    bool_labels_sum = bool_labels.sum(dim='time').data
                    labels_days_adj_30.data += (bool_labels_sum>=30)
                    labels_days_adj_60.data += (bool_labels_sum>=60) # Add 1 to every point that is hot at least 60 days of summer
                    labels_days_adj_92.data += (bool_labels_sum>=JJA_duration) # Add 1 to every point that is hot during the entire summer
                    labels_days_adj.data += bool_labels_sum
                    labels_list = np.unique(file_labels.data)
                    labels_list = labels_list[labels_list!=0]
                    for lab in labels_list:
                        bool_labels = np.max(file_labels==lab,axis=0)
                        labels_htws_adj.data = labels_htws_adj.data + bool_labels.data


        labels_days_adj_frac = labels_days_adj.copy()
        labels_days_adj_frac.data = labels_days_adj_frac.data/total_summer_days

        # Divide by number of model to get an average
        labels_days_adj.data = labels_days_adj.data/len(model_list)
        labels_htws_adj.data = labels_htws_adj.data/len(model_list)
        labels_days_adj_92.data = labels_days_adj_92.data/len(model_list)
        labels_days_adj_60.data = labels_days_adj_60.data/len(model_list)
        labels_days_adj_30.data = labels_days_adj_30.data/len(model_list)
                
        # Export label files to netcdf
        labels_days_adj.to_dataset(name="label")
        labels_htws_adj.to_dataset(name="label")
        labels_days_adj_92.to_dataset(name="label")
        labels_days_adj_60.to_dataset(name="label")
        labels_days_adj_30.to_dataset(name="label")
        labels_days_adj_frac.to_dataset(name="label")

        labels_days_adj.to_netcdf(join(write_directory,"labels_days_adj.nc"))
        labels_htws_adj.to_netcdf(join(write_directory,"labels_htws_adj.nc"))
        labels_days_adj_92.to_netcdf(join(write_directory,"labels_days_adj_92.nc"))
        labels_days_adj_60.to_netcdf(join(write_directory,"labels_days_adj_60.nc"))
        labels_days_adj_30.to_netcdf(join(write_directory,"labels_days_adj_30.nc"))
        labels_days_adj_frac.to_netcdf(join(write_directory,"labels_days_adj_frac.nc"))
    else:
        labels_days_adj = xr.open_dataset(join(write_directory,"labels_days_adj.nc"),engine='netcdf4').label
        labels_htws_adj = xr.open_dataset(join(write_directory,"labels_htws_adj.nc"),engine='netcdf4').label
        labels_days_adj_92 = xr.open_dataset(join(write_directory,"labels_days_adj_30.nc"),engine='netcdf4').label
        labels_days_adj_60 = xr.open_dataset(join(write_directory,"labels_days_adj_60.nc"),engine='netcdf4').label
    
    # Plot figure
    labels_list = [
        [labels_days_adj,labels_htws_adj],
        [labels_days_adj_92,labels_days_adj_60]
        ]

    proj_pc = ccrs.PlateCarree()

    fig,axes = plt.subplots(
    2, 2, figsize=(14,10), sharex=True, sharey=True,
    subplot_kw={'projection': proj_pc, "aspect": 1},
    gridspec_kw = {'wspace':0.02, 'hspace':0.07},
)

    titles_list = [
        ['Number of hot days','Number of heatwaves occurrences'],
        ['Years with at least 30 hot days','Years with at least 60 hot days']
        ]
    
    for i in [0,1]:
        for j in [0,1]:
            ax = axes[i][j]
            labels = labels_list[i][j]
            # Removing 2 small islands that break the scale of plot
            labels = labels.where(~((labels.lat>=69)&(labels.lon>=-12)&(labels.lon<=-5))).where(~((labels.lat<=30)&(labels.lon>=-16)&(labels.lon<=-12))).fillna(0)
            labels = labels.where(labels>0)

            img = labels.plot(cmap='YlOrRd',ax=ax,levels=7,add_labels=False,transform=proj_pc)
            ax.set_extent([-28, 46, 35, 75], crs=proj_pc)
            ax.set_title(titles_list[i][j])
            ax.add_feature(cfeature.COASTLINE,linewidth=0.3)
            ax.add_feature(cfeature.OCEAN)

            gl = ax.gridlines(crs=proj_pc, linewidth=1, color='black', alpha=0.2, linestyle="--")

            gl.ylocator = mticker.FixedLocator(np.arange(-90,90,20))
            gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 25))

            if i==1:
                gl.bottom_labels = True
            if j==0:
                gl.left_labels = True
    fig_title = 'hot_days_4_panels_bias_adj_models'
    print(f"Saving {fig_title}.pdf ...")
    plt.savefig(join(write_directory,f'{fig_title}.pdf'),dpi=1200)
    plt.savefig(join(write_directory,f'{fig_title}.png'))
    print(f"Saved.")
    return

def plot_4_panel_hot_days_RWL(read_directory,write_directory,need_to_compute_labels=True,regional_warming_levels_list=[2.1,2.6,4.0,5.1]):
    """Plot hot days per location"""
    if need_to_compute_labels:
        
        df_htws = pd.read_csv(join(write_directory,'df_htws_BC_results.csv'),header=0,index_col=0)
        model_list = np.unique(df_htws['model'])
        model_list = model_list[model_list!='ERA5']
        # Iterate over files
        for rwl in [0,1,2,3]:
            if rwl==0:
                files_list = np.sort(glob.glob(join(read_directory,'ERA5','labels*.nc')))
                # Initialize arrays to hold data
                labels_mean_hist = xr.open_dataset(files_list[0],engine='netcdf4').label.copy().sum(dim='time')
                labels_mean_hist.data = np.zeros(np.shape(labels_mean_hist))
                labels_mean_rwl1 = labels_mean_hist.copy()
                labels_mean_rwl2 = labels_mean_hist.copy()
                labels_mean_rwl3 = labels_mean_hist.copy()

                labels_max_hist = labels_mean_hist.copy()
                labels_max_rwl1 = labels_max_hist.copy()
                labels_max_rwl2 = labels_max_hist.copy()
                labels_max_rwl3 = labels_max_hist.copy()

                labels_max_continuous_hist = labels_mean_hist.copy()
                labels_max_continuous_rwl1 = labels_max_continuous_hist.copy()
                labels_max_continuous_rwl2 = labels_max_continuous_hist.copy()
                labels_max_continuous_rwl3 = labels_max_continuous_hist.copy()

                labels_min_rest_hist = labels_mean_hist.copy()
                labels_min_rest_rwl1 = labels_min_rest_hist.copy()
                labels_min_rest_rwl2 = labels_min_rest_hist.copy()
                labels_min_rest_rwl3 = labels_min_rest_hist.copy()

                df_mean_dict = {0:labels_mean_hist,1:labels_mean_rwl1,2:labels_mean_rwl2,3:labels_mean_rwl3}
                df_max_dict = {0:labels_max_hist,1:labels_max_rwl1,2:labels_max_rwl2,3:labels_max_rwl3}
                df_max_continuous_dict = {0:labels_max_continuous_hist,1:labels_max_continuous_rwl1,2:labels_max_continuous_rwl2,3:labels_max_continuous_rwl3}
                df_min_rest_dict = {0:labels_min_rest_hist,1:labels_min_rest_rwl1,2:labels_min_rest_rwl2,3:labels_min_rest_rwl3}
                for file in tqdm(files_list):
                    year = int(file[-21:-17])
                    if year >= 1975 and year <= 2025:
                        file_labels = xr.open_dataset(file,engine='netcdf4').label
                        bool_labels = (file_labels>0)
                        JJA_duration = np.shape(bool_labels.data)[0]
                        labels_mean_hist.data += bool_labels.sum(dim='time').data
                        labels_max_hist.data = np.max([labels_max_hist.data,bool_labels.sum(dim='time').data],axis=0)
                        stack_where = np.zeros(np.shape(bool_labels.data)) # Create a 3D array that holds the number of consecutive hot days for each location, computed for each day
                        stack_where[0,:,:] = bool_labels.data[0,:,:]
                        ever_positive_bool = bool_labels.data[0,:,:]
                        stack_where_rest = np.zeros(np.shape(bool_labels.data)) # Create a 3D array that holds the number of consecutive hot days for each location, computed for each day
                        for day in range(1,JJA_duration):
                            stack_where[day,:,:] = stack_where[day-1,:,:] + bool_labels.data[day,:,:] # Add one day to each potential heatwave location
                            stack_where[day,:,:] = stack_where[day,:,:]*bool_labels.data[day,:,:] # When not adding a day, have to set back the duration to zero
                            
                            ever_positive_bool = (ever_positive_bool+stack_where_rest[day,:,:])>0
                            stack_where_rest[day,:,:] = ever_positive_bool*(stack_where_rest[day-1,:,:]+(bool_labels.data[day,:,:]==0))
                            stack_where_rest[day,:,:] = ever_positive_bool*stack_where_rest[day,:,:]*(bool_labels.data[day,:,:]==0)

                        stack_where = np.max(stack_where,axis=0)
                        labels_max_continuous_hist.data = np.max([labels_max_continuous_hist.data,stack_where],axis=0)
                        #labels_min_rest_hist.data = np.min([labels_min_rest_hist.data,stack_where_rest],axis=0)
                labels_mean_hist.data = labels_mean_hist.data/(2025-1975+1) # Compute mean number of hot days per summer for each grid point
            else:
                count = 0
                count_models = 0
                for model in model_list:
                    with open(join(read_directory,model,'regional_warming_levels.json'), 'r') as f:
                        RWL_dict = json.load(f)
                    files_list = np.sort(glob.glob(join(read_directory,model,'labels*.nc')))
                    temp_rwl = regional_warming_levels_list[rwl-1]
                    if RWL_dict[str(temp_rwl)] is not None:
                        count_models += 1
                        start_year_rwl = RWL_dict[str(temp_rwl)]['start_year']
                        end_year_rwl = RWL_dict[str(temp_rwl)]['start_year']
                        stack_model = np.zeros((np.shape(df_max_continuous_dict[rwl].data)))
                        stack_model_rest = np.zeros((np.shape(df_min_rest_dict[rwl].data)))
                        for file in tqdm(files_list):
                            year = int(file[-21:-17])
                            if year >= start_year_rwl and year <= end_year_rwl:
                                file_labels = xr.open_dataset(file,engine='netcdf4').label
                                bool_labels = (file_labels>0)
                                JJA_duration = np.shape(bool_labels.data)[0]
                                df_mean_dict[rwl].data += bool_labels.sum(dim='time').data
                                df_max_dict[rwl].data = np.max([df_max_dict[rwl].data,bool_labels.sum(dim='time').data],axis=0)
                                count+=1 # Since not all models reach a given RWL, need to count
                                stack_where = np.zeros(np.shape(bool_labels.data)) # Create a 3D array that holds the number of consecutive hot days for each location, computed for each day
                                stack_where[0,:,:] = bool_labels.data[0,:,:]
                                ever_positive_bool = bool_labels.data[0,:,:]
                                stack_where_rest = np.zeros(np.shape(bool_labels.data))
                                for day in range(1,JJA_duration):
                                    stack_where[day,:,:] = stack_where[day-1,:,:] + bool_labels.data[day,:,:] # Add one day to each potential heatwave location
                                    stack_where[day,:,:] = stack_where[day,:,:]*bool_labels.data[day,:,:] # When not adding a day, have to set back the duration to zero

                                    ever_positive_bool = (ever_positive_bool+stack_where_rest[day,:,:])>0
                                    stack_where_rest[day,:,:] = ever_positive_bool*(stack_where_rest[day-1,:,:]+(bool_labels.data[day,:,:]==0))
                                    stack_where_rest[day,:,:] = ever_positive_bool*stack_where_rest[day,:,:]*(bool_labels.data[day,:,:]==0)

                                stack_where = np.max(stack_where,axis=0)
                                stack_where_rest = np.min(stack_where_rest,axis=0)
                                stack_model = np.max([stack_model,stack_where],axis=0)
                                #stack_model_rest = np.min([stack_model_rest,stack_where_rest],axis=0)
                        df_max_continuous_dict[rwl].data = np.max([df_max_continuous_dict[rwl].data,stack_model],axis=0)
                        #df_min_rest_dict[rwl].data += stack_model_rest
                #df_max_continuous_dict[rwl].data = df_max_continuous_dict[rwl].data/count_models
                df_min_rest_dict[rwl].data = df_min_rest_dict[rwl].data/count_models
                df_mean_dict[rwl].data = df_mean_dict[rwl].data/count
                                 
        # Export label files to netcdf
        labels_mean_hist.to_dataset(name="label")
        labels_mean_rwl1.to_dataset(name="label")
        labels_mean_rwl2.to_dataset(name="label")
        labels_mean_rwl3.to_dataset(name="label")

        labels_max_hist.to_dataset(name="label")
        labels_max_rwl1.to_dataset(name="label")
        labels_max_rwl2.to_dataset(name="label")
        labels_max_rwl3.to_dataset(name="label")

        labels_max_continuous_hist.to_dataset(name="label")
        labels_max_continuous_rwl1.to_dataset(name="label")
        labels_max_continuous_rwl2.to_dataset(name="label")
        labels_max_continuous_rwl3.to_dataset(name="label")

        labels_min_rest_hist.to_dataset(name="label")
        labels_min_rest_rwl1.to_dataset(name="label")
        labels_min_rest_rwl2.to_dataset(name="label")
        labels_min_rest_rwl3.to_dataset(name="label")

        labels_mean_hist.to_netcdf(join(write_directory,"labels_mean_nb_days_hist.nc"))
        labels_mean_rwl1.to_netcdf(join(write_directory,"labels_mean_nb_days_rwl1.nc"))
        labels_mean_rwl2.to_netcdf(join(write_directory,"labels_mean_nb_days_rwl2.nc"))
        labels_mean_rwl3.to_netcdf(join(write_directory,"labels_mean_nb_days_rwl3.nc"))

        labels_max_hist.to_netcdf(join(write_directory,"labels_max_nb_days_hist.nc"))
        labels_max_rwl1.to_netcdf(join(write_directory,"labels_max_nb_days_rwl1.nc"))
        labels_max_rwl2.to_netcdf(join(write_directory,"labels_max_nb_days_rwl2.nc"))
        labels_max_rwl3.to_netcdf(join(write_directory,"labels_max_nb_days_rwl3.nc"))

        labels_max_continuous_hist.to_netcdf(join(write_directory,"labels_max_continuous_nb_days_hist.nc"))
        labels_max_continuous_rwl1.to_netcdf(join(write_directory,"labels_max_continuous_nb_days_rwl1.nc"))
        labels_max_continuous_rwl2.to_netcdf(join(write_directory,"labels_max_continuous_nb_days_rwl2.nc"))
        labels_max_continuous_rwl3.to_netcdf(join(write_directory,"labels_max_continuous_nb_days_rwl3.nc"))

        labels_min_rest_hist.to_netcdf(join(write_directory,"labels_min_rest_nb_days_hist.nc"))
        labels_min_rest_rwl1.to_netcdf(join(write_directory,"labels_min_rest_nb_days_rwl1.nc"))
        labels_min_rest_rwl2.to_netcdf(join(write_directory,"labels_min_rest_nb_days_rwl2.nc"))
        labels_min_rest_rwl3.to_netcdf(join(write_directory,"labels_min_rest_nb_days_rwl3.nc"))

    else:
        labels_mean_hist = xr.open_dataset(join(write_directory,"labels_mean_nb_days_hist.nc"),engine='netcdf4').label
        labels_mean_rwl1 = xr.open_dataset(join(write_directory,"labels_mean_nb_days_rwl1.nc"),engine='netcdf4').label
        labels_mean_rwl2 = xr.open_dataset(join(write_directory,"labels_mean_nb_days_rwl2.nc"),engine='netcdf4').label
        labels_mean_rwl3 = xr.open_dataset(join(write_directory,"labels_mean_nb_days_rwl3.nc"),engine='netcdf4').label

        labels_max_hist = xr.open_dataset(join(write_directory,"labels_max_nb_days_hist.nc"),engine='netcdf4').label
        labels_max_rwl1 = xr.open_dataset(join(write_directory,"labels_max_nb_days_rwl1.nc"),engine='netcdf4').label
        labels_max_rwl2 = xr.open_dataset(join(write_directory,"labels_max_nb_days_rwl2.nc"),engine='netcdf4').label
        labels_max_rwl3 = xr.open_dataset(join(write_directory,"labels_max_nb_days_rwl3.nc"),engine='netcdf4').label

        labels_max_continuous_hist = xr.open_dataset(join(write_directory,"labels_max_continuous_nb_days_hist.nc"),engine='netcdf4').label
        labels_max_continuous_rwl1 = xr.open_dataset(join(write_directory,"labels_max_continuous_nb_days_rwl1.nc"),engine='netcdf4').label
        labels_max_continuous_rwl2 = xr.open_dataset(join(write_directory,"labels_max_continuous_nb_days_rwl2.nc"),engine='netcdf4').label
        labels_max_continuous_rwl3 = xr.open_dataset(join(write_directory,"labels_max_continuous_nb_days_rwl3.nc"),engine='netcdf4').label

        labels_min_rest_hist = xr.open_dataset(join(write_directory,"labels_min_rest_nb_days_hist.nc"),engine='netcdf4').label
        labels_min_rest_rwl1 = xr.open_dataset(join(write_directory,"labels_min_rest_nb_days_rwl1.nc"),engine='netcdf4').label
        labels_min_rest_rwl2 = xr.open_dataset(join(write_directory,"labels_min_rest_nb_days_rwl2.nc"),engine='netcdf4').label
        labels_min_rest_rwl3 = xr.open_dataset(join(write_directory,"labels_min_rest_nb_days_rwl3.nc"),engine='netcdf4').label
    
    # Plot figures
    labels_mean_list = [
        [labels_mean_hist,labels_mean_rwl1],
        [labels_mean_rwl2,labels_mean_rwl3]
        ]

    labels_max_list = [
        [labels_max_hist,labels_max_rwl1],
        [labels_max_rwl2,labels_max_rwl3]
        ]

    labels_max_continuous_list = [
        [labels_max_continuous_hist,labels_max_continuous_rwl1],
        [labels_max_continuous_rwl2,labels_max_continuous_rwl3]
        ]
    
    labels_min_rest_list = [
        [labels_min_rest_hist,labels_min_rest_rwl1],
        [labels_min_rest_rwl2,labels_min_rest_rwl3]
        ]

    proj_pc = ccrs.PlateCarree()

    fig,axes = plt.subplots(
    2, 2, figsize=(14,10), sharex=True, sharey=True,
    subplot_kw={'projection': proj_pc, "aspect": 1},
    gridspec_kw = {'wspace':0.02, 'hspace':0.07},
)

    titles_list = [
        ['ERA5 1975-2025','RWL 1 (+2.1°C)'],
        ['RWL 2 (+2.6°C)','RWL 3 (+4.0°C)']
        ]
    
    the_levels=[1,4,7,10,13,16,19,22,25,28,31,34,37,40,43,46,49,52]
    for i in [0,1]:
        for j in [0,1]:
            ax = axes[i][j]
            labels = labels_mean_list[i][j]
            # Removing 2 small islands that break the scale of plot
            labels = labels.where(~((labels.lat>=69)&(labels.lon>=-12)&(labels.lon<=-5))).where(~((labels.lat<=30)&(labels.lon>=-16)&(labels.lon<=-12))).fillna(0)
            labels = labels.where(labels>0)

            img = labels.plot(cmap='YlOrRd',ax=ax,levels=the_levels,add_labels=False,transform=proj_pc)
            ax.set_extent([-28, 46, 35, 75], crs=proj_pc)
            ax.set_title(titles_list[i][j])
            ax.add_feature(cfeature.COASTLINE,linewidth=0.3)
            ax.add_feature(cfeature.OCEAN)

            gl = ax.gridlines(crs=proj_pc, linewidth=1, color='black', alpha=0.2, linestyle="--")

            gl.ylocator = mticker.FixedLocator(np.arange(-90,90,20))
            gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 25))

            if i==1:
                gl.bottom_labels = True
            if j==0:
                gl.left_labels = True
    fig_title = 'mean_nb_hot_days_rwl'
    print(f"Saving {fig_title}.pdf ...")
    plt.savefig(join(write_directory,f'{fig_title}.pdf'),dpi=1200)
    plt.savefig(join(write_directory,f'{fig_title}.png'))
    print(f"Saved.")

# Plot figure for max number of hot days in summer

    fig,axes = plt.subplots(
    2, 2, figsize=(14,10), sharex=True, sharey=True,
    subplot_kw={'projection': proj_pc, "aspect": 1},
    gridspec_kw = {'wspace':0.02, 'hspace':0.07},
)

    titles_list = [
        ['ERA5 1975-2025','RWL 1 (+2.1°C)'],
        ['RWL 2 (+2.6°C)','RWL 3 (+4.0°C)']
        ]
    
    the_levels=[1,4,7,10,13,16,19,22,25,28,31,34,37,40,43,46,49,52]
    for i in [0,1]:
        for j in [0,1]:
            ax = axes[i][j]
            labels = labels_max_list[i][j]
            # Removing 2 small islands that break the scale of plot
            labels = labels.where(~((labels.lat>=69)&(labels.lon>=-12)&(labels.lon<=-5))).where(~((labels.lat<=30)&(labels.lon>=-16)&(labels.lon<=-12))).fillna(0)
            labels = labels.where(labels>0)

            img = labels.plot(cmap='YlOrRd',ax=ax,levels=the_levels,add_labels=False,transform=proj_pc)
            ax.set_extent([-28, 46, 35, 75], crs=proj_pc)
            ax.set_title(titles_list[i][j])
            ax.add_feature(cfeature.COASTLINE,linewidth=0.3)
            ax.add_feature(cfeature.OCEAN)

            gl = ax.gridlines(crs=proj_pc, linewidth=1, color='black', alpha=0.2, linestyle="--")

            gl.ylocator = mticker.FixedLocator(np.arange(-90,90,20))
            gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 25))

            if i==1:
                gl.bottom_labels = True
            if j==0:
                gl.left_labels = True
    fig_title = 'max_nb_hot_days_rwl'
    print(f"Saving {fig_title}.pdf ...")
    plt.savefig(join(write_directory,f'{fig_title}.pdf'),dpi=1200)
    plt.savefig(join(write_directory,f'{fig_title}.png'))
    print(f"Saved.")

# Plot figure for max continuous number of hot days in summer

    fig,axes = plt.subplots(
    2, 2, figsize=(14,10), sharex=True, sharey=True,
    subplot_kw={'projection': proj_pc, "aspect": 1},
    gridspec_kw = {'wspace':0.02, 'hspace':0.07},
)

    titles_list = [
        ['ERA5 1975-2025','RWL 1 (+2.1°C)'],
        ['RWL 2 (+2.6°C)','RWL 3 (+4.0°C)']
        ]
    
    the_levels=[1,4,7,10,13,16,19,22,25,28,31,34,37,40,43,46,49,52]
    for i in [0,1]:
        for j in [0,1]:
            ax = axes[i][j]
            labels = labels_max_continuous_list[i][j]
            # Removing 2 small islands that break the scale of plot
            labels = labels.where(~((labels.lat>=69)&(labels.lon>=-12)&(labels.lon<=-5))).where(~((labels.lat<=30)&(labels.lon>=-16)&(labels.lon<=-12))).fillna(0)
            labels = labels.where(labels>0)

            img = labels.plot(cmap='YlOrRd',ax=ax,levels=the_levels,add_labels=False,transform=proj_pc)
            ax.set_extent([-28, 46, 35, 75], crs=proj_pc)
            ax.set_title(titles_list[i][j])
            ax.add_feature(cfeature.COASTLINE,linewidth=0.3)
            ax.add_feature(cfeature.OCEAN)

            gl = ax.gridlines(crs=proj_pc, linewidth=1, color='black', alpha=0.2, linestyle="--")

            gl.ylocator = mticker.FixedLocator(np.arange(-90,90,20))
            gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 25))

            if i==1:
                gl.bottom_labels = True
            if j==0:
                gl.left_labels = True
    fig_title = 'max_continuous_nb_hot_days_rwl'
    print(f"Saving {fig_title}.pdf ...")
    plt.savefig(join(write_directory,f'{fig_title}.pdf'),dpi=1200)
    plt.savefig(join(write_directory,f'{fig_title}.png'))
    print(f"Saved.")
    return

def animation_hottest_summer(label_directory,temp_file,other_data_path,write_directory,year,proj=ccrs.PlateCarree()):
    # Load label data
    da_label = xr.open_dataset(join(label_directory,f"labels_cc3d_year_{year}_ref_1975_2025.nc"),engine='netcdf4').label
    original_calendar = da_label.time.dt.calendar

    if original_calendar=='360_day':
        JJA_beg = 151 #151 is June 1st, 240 is August 30th
        JJA_end = 240
    else: # noleap calendar
        original_calendar = "noleap"
        JJA_beg = 152 #152 is June 1st, 243 is August 31st
        JJA_end = 243
    # Load mask
    da_mask = xr.open_dataset(join(other_data_path,'mask','mask_Europe_land_only_CORDEX_EUR11_ERA5.nc'),engine='netcdf4').mask
    # Load temperature data
    ds_temp = xr.open_dataset(temp_file,engine='netcdf4')
    if original_calendar!='360_day':
        ds_temp = ds_temp.convert_calendar("noleap")
    da_temp = ds_temp.tasmaxAdjust
    seasonal_cycle = xr.open_dataarray(join(label_directory,'seasonal_cycle_1975_2025.nc'),engine='netcdf4')
    da_threshold = xr.open_dataset(join(other_data_path,'threshold',f'distrib_threshold_95_{original_calendar}.nc'),engine='netcdf4').tasmax

    mask = (da_threshold.dayofyear>=JJA_beg) & (da_threshold.dayofyear<=JJA_end) # dayofyear ranges from 1 to 365 (or 360)
    da_threshold = da_threshold.sel(dayofyear=mask)
    seasonal_cycle = seasonal_cycle.sel(dayofyear=mask) # Keep only JJA values

    da_temp = da_temp.sel(time=(da_temp.time.dt.year==year)) # Keep only correct year
    da_temp = da_temp.sel(time=(da_temp.time.dt.season=='JJA')) # Keep only JJA
    da_temp = da_temp - seasonal_cycle.data # Compute anomaly
    da_temp = da_temp - da_threshold.data # Compute threshold exceedance

    da_temp = da_temp.where(da_mask.data==0)
    da_temp = da_temp.where(da_label.data!=0)

    matplotlib.use('Agg')
    nb_frames = len(da_label.time.data)

    max_val = max(np.ceil(np.abs(np.nanmin(da_temp.data))),np.ceil(np.abs(np.nanmax(da_temp.data))))

    #the_levels = np.arange(0,max_val,1)
    the_levels = np.arange(0,21,1)

    def make_figure():
        fig = plt.figure(figsize=(24,16))
        ax = plt.axes(projection=proj)
        return fig,ax

    fig,ax = make_figure()
    cax = plt.axes([0.35, 0.05, 0.35, 0.02])
    def draw(i):
        ax.clear()
        ax.set_extent([-28, 46, 35, 75])
        ax.set_title(f"{da_temp.time.data[i].year}-{da_temp.time.data[i].month}-{da_temp.time.data[i].day}",{'position':(0.5,-2),'fontsize':16})
        #ax.add_feature(cfeature.BORDERS)
        #ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.COASTLINE,linewidth=0.3)
        #ax.add_feature(cfeature.LAKES, alpha=0.5)
        #ax.add_feature(cfeature.RIVERS, alpha=0.5)
        CS1 = ax.contourf(da_label.lon.data,da_label.lat.data,da_temp.data[i,:,:],cmap='YlOrRd',transform=proj, levels=the_levels)
        plt.colorbar(CS1,cax=cax,orientation='horizontal')
        plt.title("Threshold exceedance (°C)",fontsize=14)
        return CS1

    def init():
        return draw(0)

    def update(i):
        return draw(i)

    model = label_directory.split('/')[-1]

    anim = animation.FuncAnimation(fig, update, init_func=init, frames=nb_frames, blit=False, interval=0.15, repeat=False)
    filename_movie = join(write_directory, 
                                f"animation_{year}_{model}.mp4")
    writervideo = animation.FFMpegWriter(fps=5)
    anim.save(filename_movie, writer=writervideo)
    ax.clear()
    plt.close()

def make_animation_selected_models(read_directory,write_directory,other_data_path):
    df_global_summers_results = pd.read_csv(join(write_directory,'df_global_summers_results.csv'),header=0,index_col=0)
    df_global_summers_results = df_global_summers_results[df_global_summers_results['model']!='ERA5']
    model_list_BC_results = np.unique(df_global_summers_results['model'])

    dict_hottest_years = {}

    for model in tqdm(model_list_BC_results):
        worst_year_idx = df_global_summers_results[df_global_summers_results['model']==model]['Relative accumulated area summer (%)'].idxmax()
        worst_year = df_global_summers_results.loc[worst_year_idx,'Year']
        temp_file_path = df_global_summers_results.loc[worst_year_idx,'temp_file_path']
        dict_hottest_years[model] = worst_year
        animation_hottest_summer(label_directory=join(read_directory,model),temp_file=temp_file_path,other_data_path=other_data_path,write_directory=write_directory,year=worst_year,proj=ccrs.PlateCarree())

    #dict_hottest_years = {'CLMcom_MOHC-HadGEM2-ES_rcp45_r1i1p1_CLMcom-CCLM4-8-17_SBCK-CDFt-ERA5-1976-2005_day_tasmaxAdjust_v20260512': 2091,
    #'CLMcom_MOHC-HadGEM2-ES_rcp85_r1i1p1_CLMcom-CCLM4-8-17_SBCK-CDFt-ERA5-1976-2005_day_tasmaxAdjust_v20260512': 2091,
    #'CLMcom_MPI-M-MPI-ESM-LR_rcp45_r1i1p1_CLMcom-CCLM4-8-17_SBCK-CDFt-ERA5-1976-2005_day_tasmaxAdjust_v20260512': 2062,
    #'CNRM_CNRM-CERFACS-CNRM-CM5_rcp85_r1i1p1_CNRM-ALADIN63_SBCK-CDFt-ERA5-1976-2005_day_tasmaxAdjust_v20260512': 2093,
    #'CNRM_MOHC-HadGEM2-ES_rcp85_r1i1p1_CNRM-ALADIN63_SBCK-CDFt-ERA5-1976-2005_day_tasmaxAdjust_v20260512': 2093,
    #'ICTP_MPI-M-MPI-ESM-LR_rcp85_r1i1p1_ICTP-RegCM4-6_SBCK-CDFt-ERA5-1976-2005_day_tasmaxAdjust_v20260512': 2088,
    #'KNMI_ICHEC-EC-EARTH_rcp45_r1i1p1_KNMI-RACMO22E_SBCK-CDFt-ERA5-1976-2005_day_tasmaxAdjust_v20260512': 2089,
    #'KNMI_ICHEC-EC-EARTH_rcp85_r1i1p1_KNMI-RACMO22E_SBCK-CDFt-ERA5-1976-2005_day_tasmaxAdjust_v20260512': 2096,
    #'SMHI_ICHEC-EC-EARTH_rcp85_r1i1p1_SMHI-RCA4_SBCK-CDFt-ERA5-1976-2005_day_tasmaxAdjust_v20260512': 2095}

    return

def movement_windrose_map(data,ax,write_directory='',var_col_arrows='Duration',var_col_arrows_unit='days',
                 cmap=cm.YlOrRd,legend = False,y_lim = False,savefig=False,
                 bins = 10):
    # HW directions
    x1 = np.array([eval(p)[1] for p in data['Centroid_p1']]) # List of first longitudes 
    x2 = np.array([eval(p)[1] for p in data['Centroid_p2']]) # List of second longitudes 
    y1 = np.array([eval(p)[0] for p in data['Centroid_p1']]) # List of first latitudes 
    y2 = np.array([eval(p)[0] for p in data['Centroid_p2']]) # List of second latitudes 

    #fig = plt.figure(figsize=(10,9))   #(21,5):横版
    #ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree()) 
    levels_ = np.arange(4,94,0.1)
    ticks_ = np.arange(4,94,8)
    ax.set_extent([-28, 46, 34, 75], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, edgecolor="dimgray",linewidths = 0.75)

    hc = ax.quiver(x1,y1,x2-x1,y2-y1,data[var_col_arrows], units='xy', cmap = cmap, width = 0.25, 
                    edgecolor="grey",linewidth=0.15,
                    headwidth=6, headlength=5, headaxislength=4.5,
                    transform=ccrs.PlateCarree(),zorder=300)
    cb = plt.colorbar(hc, boundaries=levels_, ticks=ticks_,
                        orientation="horizontal", fraction=0.05, 
                        pad=0.09, extendfrac='auto',
                        extend='both', extendrect=True)
    cb.mappable.set_clim(min(levels_), max(levels_))
    cb.set_label(label=f"{var_col_arrows} ({var_col_arrows_unit})", fontsize=16)
    cb.ax.tick_params(labelsize=16) 
    cb.remove()
    # set major and minor ticks
    ax.add_feature(cfeature.OCEAN)
    ax.minorticks_on()
    ax.set_xticks(np.arange(-25, 55, 10), crs=ccrs.PlateCarree())
    ax.set_xticklabels(np.arange(-25, 55, 10), fontsize=14)
    ax.xaxis.set_minor_locator(plt.MultipleLocator(30))
    ax.xaxis.set_major_formatter(LongitudeFormatter()) 

    ax.set_yticks(np.arange(35, 75, 10), crs=ccrs.PlateCarree())
    ax.set_yticklabels(np.arange(35, 75, 10), fontsize=14)
    ax.yaxis.set_minor_locator(plt.MultipleLocator(20))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(top=False,bottom=True,left=True, right=False)

    plt.style.use('fast')
    # Windrose
    height_deg = 12
    ax2 = inset_axes(
        ax,
        width="100%",  # size in % of bbox
        height="100%",  # size in % of bbox
        # specify the center lon and lat of the plot, and size in degree
        bbox_to_anchor=(
            -24,
            37,
            height_deg,
            height_deg,
        ),
        bbox_transform=ax.transData,
        axes_class=WindroseAxes,
    )

    geodesic = pyproj.Geod(ellps='WGS84')
    _angles, _, _Delta_distance  = geodesic.inv(np.vstack(x1), 
                                                np.vstack(y1),
                                                np.vstack(x2), 
                                                np.vstack(y2))
    
    _angles[_angles<0] = 360 + _angles[_angles<0]

    ax2.bar(direction=np.reshape(_angles,-1),
           var=np.reshape(_Delta_distance,-1),
           bins=bins,
           cmap=cmap, lw=0.000001,   
           nsector=16, normed=True, opening=1, 
           edgecolor='w',alpha = 1)

    ax2.set_xticklabels(['E', 'NE', 'N', 'NW',  'W', 'SW', 'S', 'SE'], fontsize = 14)
    if legend:
        ax2.set_legend(loc='lower right', bbox_to_anchor=(1.12, -0.12), ncol=2, 
                      fontsize = 20, labelspacing=0.1, columnspacing=0.8)
    
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda s,position:'{:.0f}%'.format(s)))
    
    if y_lim:
        ax2.set_yticks(np.arange(0, 25, step=3), fontsize = 10)
        ax2.set_ylim([0, 24])
    else:
        ax2.set_yticklabels("",fontsize = 10)
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda s,position:'{:.0f}%'.format(s)))

    plt.tight_layout()
    if savefig:
        plt.savefig(join(write_directory,'movement_map.png'))
        plt.savefig(join(write_directory,'movement_map.pdf'),dpi=1200)

    return ax

def plot_4_movement_maps(write_directory):
    df_htws_BC_results = pd.read_csv(join(write_directory,'df_htws_BC_results.csv'),header=0,index_col=0)
    df_ERA5 = df_htws_BC_results[df_htws_BC_results['model']=='ERA5']
    df_htws_BC_results = df_htws_BC_results[df_htws_BC_results['model']!='ERA5']

    fig,axes = plt.subplots(2,2,figsize=(18,16),subplot_kw={'projection':ccrs.PlateCarree()})
    
    axes[0,0] = movement_windrose_map(write_directory="/home/user/These/cordex_htws_cc3d/Data/figs",data=df_ERA5,ax=axes[0,0])
    axes[0,0].set_title('ERA5 1975-2025',fontsize=16)
    axes[0,1] = movement_windrose_map(write_directory="/home/user/These/cordex_htws_cc3d/Data/figs",data=df_htws_BC_results[df_htws_BC_results['Period RWL 1']],ax=axes[0,1])
    axes[0,1].set_title('RWL 1 (+2.1°C)',fontsize=16)
    axes[1,0] = movement_windrose_map(write_directory="/home/user/These/cordex_htws_cc3d/Data/figs",data=df_htws_BC_results[df_htws_BC_results['Period RWL 2']],ax=axes[1,0])
    axes[1,0].set_title('RWL 2 (+2.6°C)',fontsize=16)
    axes[1,1] = movement_windrose_map(write_directory="/home/user/These/cordex_htws_cc3d/Data/figs",data=df_htws_BC_results[df_htws_BC_results['Period RWL 3']],ax=axes[1,1])
    axes[1,1].set_title('RWL 3 (+4.0°C)',fontsize=16)
    
    levels_ = np.arange(4,94,0.1)
    ticks_ = np.arange(4,94,8)
    
    norm = clrs.Normalize(vmin=4, vmax=92)
    sm = plt.cm.ScalarMappable(cmap="YlOrRd",norm=norm)
    sm.set_array([])
    
    cb = fig.figure.colorbar(sm,ax=axes,boundaries=levels_, ticks=ticks_,
                            orientation="horizontal", fraction=0.05, 
                            pad=0.03, extendfrac='auto',
                            extend='both', extendrect=True)
    cb.set_label(label=f"Duration (days)", fontsize=16)
    cb.ax.tick_params(labelsize=16)
    
    plt.savefig(join(write_directory,'movement_4_panels.png'))
    plt.savefig(join(write_directory,'movement_4_panels.pdf'),dpi=1200)
    
    return

def compute_mk_trends(read_directory,other_data_path,start_year=1975,end_year=2099,split_year_population=2025,yearly_aggregation = False):
    df_htws = pd.read_csv(join(read_directory,'df_htws_BC_results.csv'),header=0,index_col=0)

    df_aerosols = pd.read_csv(join(other_data_path,'cordex_aerosols_info.csv'),header=0,index_col=0)

    model_list = np.unique(df_htws['model'])

    if start_year>2025:
        model_list = model_list[model_list!='ERA5']
    
    df_summers = pd.read_csv(join(read_directory,'df_global_summers_results.csv'),header=0,index_col=0)
    df_summers = df_summers[df_summers['model'].isin(model_list)]

    iterables = [
    ['Frequency','Nb hot days','Accumulated area summer','Relative accumulated area summer (%)','Mean duration','Max duration',
    'Mean spatial extent', 'Max spatial extent', 'Mean accumulated area', 'Max accumulated area','Mean speed','Max speed','Mean distance','Max distance',
    'Intensity','Spatial extent','Relative spatial extent (%)','Accumulated area',
    'Relative accumulated area (%)','Duration','Max','HWMId_sum',
    'Exposed_population_ssp1_all_period','HWMId_pop_ssp1_all_period',
    'Exposed_population_ssp2_all_period','HWMId_pop_ssp2_all_period','Exposed_population_ssp3_all_period','HWMId_pop_ssp3_all_period',
    'Exposed_population_ssp4_all_period','HWMId_pop_ssp4_all_period','Exposed_population_ssp5_all_period','HWMId_pop_ssp5_all_period',
    'Total_exposed_population_ssp1_all_period','Total_exposed_population_ssp2_all_period','Total_exposed_population_ssp3_all_period',
    'Total_exposed_population_ssp4_all_period','Total_exposed_population_ssp5_all_period','Distance','Speed'],
    ["p", "ss", "slope", "ucl", "lcl"]
    ]

    # Restric data to period of interest
    df_htws = df_htws[(df_htws['Year']>=start_year) & (df_htws['Year']<=end_year)]
    df_summers = df_summers[(df_summers['Year']>=start_year) & (df_summers['Year']<=end_year)]

    df_mk = pd.DataFrame(index=model_list,columns=pd.MultiIndex.from_product(iterables, names=["index", "mk"]),dtype=float)

    index_list = ['Frequency','Nb hot days','Accumulated area summer','Relative accumulated area summer (%)','Mean duration','Max duration',
    'Mean spatial extent', 'Max spatial extent', 'Mean accumulated area', 'Max accumulated area','Mean speed','Max speed','Mean distance','Max distance',
    'Intensity','Spatial extent','Relative spatial extent (%)','Accumulated area',
    'Relative accumulated area (%)','Duration','Max','HWMId_sum',
    'Exposed_population_ssp1_all_period','HWMId_pop_ssp1_all_period',
    'Exposed_population_ssp2_all_period','HWMId_pop_ssp2_all_period','Exposed_population_ssp3_all_period','HWMId_pop_ssp3_all_period',
    'Exposed_population_ssp4_all_period','HWMId_pop_ssp4_all_period','Exposed_population_ssp5_all_period','HWMId_pop_ssp5_all_period',
    'Total_exposed_population_ssp1_all_period','Total_exposed_population_ssp2_all_period','Total_exposed_population_ssp3_all_period',
    'Total_exposed_population_ssp4_all_period','Total_exposed_population_ssp5_all_period','Distance','Speed']
    # List of coefficients to reduce memory load, suited to the scale of each index
    index_memory_coeff = {'Frequency':1,'Nb hot days':1,'Accumulated area summer':1e6,'Relative accumulated area summer (%)':1,'Mean duration':1,'Max duration':1,
    'Mean spatial extent':1e3, 'Max spatial extent':1e3, 'Mean accumulated area':1e6, 'Max accumulated area':1e6,'Mean speed':10,'Max speed':10,'Mean distance':1e2,'Max distance':1e2,
    'Relative accumulated area (%)':1,'Intensity':1,'Spatial extent':1e3,'Accumulated area':1e6,'Relative spatial extent (%)':1,'Duration':1,'Max':1,'HWMId_sum':1e6,'Exposed_population_ssp1_all_period':1e6,'HWMId_pop_ssp1_all_period':1e6,
        'Exposed_population_ssp2_all_period':1e6,'HWMId_pop_ssp2_all_period':1e6,'Exposed_population_ssp3_all_period':1e6,'HWMId_pop_ssp3_all_period':1e6,
        'Exposed_population_ssp4_all_period':1e6,'HWMId_pop_ssp4_all_period':1e6,'Exposed_population_ssp5_all_period':1e6,'HWMId_pop_ssp5_all_period':1e6,
        'Total_exposed_population_ssp1_all_period':1e6,'Total_exposed_population_ssp2_all_period':1e6,'Total_exposed_population_ssp3_all_period':1e6,
    'Total_exposed_population_ssp4_all_period':1e6,'Total_exposed_population_ssp5_all_period':1e6,'Distance':1e2,'Speed':10}

    # Compute MK trend test for 'Frequency','Nb hot days','Accumulated area summer'
    count_fail=0
    res_dict = {'Frequency':0.9,'Nb hot days':0.9,'Accumulated area summer':0.01,'Relative accumulated area summer (%)':0.01,'Mean duration':0.01,'Max duration':0.9,
    'Mean spatial extent':0.01, 'Max spatial extent':0.01, 'Mean accumulated area':0.01, 'Max accumulated area':0.01,'Mean speed':0.01,'Max speed':0.01,'Mean distance':0.01,'Max distance':0.01}
    
    for index in ['Frequency','Nb hot days','Accumulated area summer','Relative accumulated area summer (%)','Mean duration','Max duration',
    'Mean spatial extent', 'Max spatial extent', 'Mean accumulated area', 'Max accumulated area','Mean speed','Max speed','Mean distance','Max distance']:
        print(index)
        for model in tqdm(df_mk.index):
            sub_df = df_summers[df_summers['model']==model]
            sub_df[index] = sub_df[index]/index_memory_coeff[index] # Divide to reduce memory load
            dates_array = np.array([datetime(i,1,1) for i in sub_df['Year'].values]) # Convert Year to proper datetime
            observations_array = np.array(sub_df[index]).astype(float)
            mk_result = mk.mk_temp_aggr(dates_array,observations_array,resolution=res_dict[index])
            mk_result = mk_result[list(mk_result.keys())[-1]] # Only take last dict
            df_mk.loc[model,(index,'p')] = mk_result['p']
            df_mk.loc[model,(index,'ss')] = mk_result['ss']
            df_mk.loc[model,(index,'slope')] = mk_result['slope']*index_memory_coeff[index] # Multiply to get correct value
            df_mk.loc[model,(index,'lcl')] = mk_result['lcl']*index_memory_coeff[index] # Multiply to get correct value
            df_mk.loc[model,(index,'ucl')] = mk_result['ucl']*index_memory_coeff[index] # Multiply to get correct value


    # Compute MK trend test for all other indices
    for index in index_list[14:]:
        print(index)
        for model in tqdm(df_mk.index):
            sub_df = df_htws[df_htws['model']==model]
            sub_df[index] = sub_df[index]/index_memory_coeff[index] # Divide to reduce memory load
            if yearly_aggregation:
                dates_array = np.array([datetime(i,1,1) for i in sub_df['Year'].values]) # Convert Year to proper datetime
            else:
                dates_array = np.array([datetime.strptime(i,'%Y-%m-%d %H:%M:%S') for i in sub_df['Global_centroid_date'].values]) # Convert Year to proper datetime
            observations_array = np.array(sub_df[index]).astype(float)
            if index=='Duration':
                mk_result = mk.mk_temp_aggr(dates_array,observations_array,resolution=0.9)
            else:
                mk_result = mk.mk_temp_aggr(dates_array,observations_array,resolution=0.01)
            mk_result = mk_result[list(mk_result.keys())[-1]] # Only take last dict
            df_mk.loc[model,(index,'p')] = mk_result['p']
            df_mk.loc[model,(index,'ss')] = mk_result['ss']
            df_mk.loc[model,(index,'slope')] = mk_result['slope']*index_memory_coeff[index] # Multiply to get correct value
            df_mk.loc[model,(index,'lcl')] = mk_result['lcl']*index_memory_coeff[index] # Multiply to get correct value
            df_mk.loc[model,(index,'ucl')] = mk_result['ucl']*index_memory_coeff[index] # Multiply to get correct value

    df_mk.to_csv(join(read_directory,f'df_mk_trends_{start_year}_{end_year}{'_year_agg'*yearly_aggregation}.csv'))

    return

def plot_RWL_figures(read_directory,write_directory,regional_warming_levels_list=[2.1,2.6,4.0,5.1], RWLs_to_plot = [0,1,2]):
    df_global_htws = pd.read_csv(join(write_directory,"df_global_htws.csv"))
    # Reference heatwaves
    reference_htws = pd.read_csv(join(read_directory,'ERA5','df_htws.csv'),header=0,index_col=0) # TODO Check this path
    reference_htws = reference_htws.loc[[26,91,191],:] # 26: 1987 Greece, 91: 2003 Europe, 120: 2010 Russia, 191: 2022 Europe
    reference_htws['label'] = reference_htws['Year']

    log_indices = ['Spatial extent','Accumulated area','HWMId_sum','Exposed_population_ghs','HWMId_pop_ghs','Exposed_population_ssp1','HWMId_pop_ssp1','Exposed_population_ssp2','HWMId_pop_ssp2',
    'Exposed_population_ssp3','HWMId_pop_ssp3','Exposed_population_ssp4','HWMId_pop_ssp4','Exposed_population_ssp5','HWMId_pop_ssp5']

    for index in log_indices :
        df_global_htws[index] = df_global_htws[index].apply(mylog)
        reference_htws[index] = reference_htws[index].apply(mylog)

    bounds_dict = {
        "Intensity":[0.5,4],
        'Spatial extent':[4,7.5],
        'Duration':[4,90],
        'Max':[0,20],
        'HWMId_sum':[1,7]
    }

    print("Plotting individual indices")
    for index in tqdm(['Intensity','Duration','Max','Spatial extent','Accumulated area','HWMId_sum']):
        # Cast objet type to numeric type
        df_global_htws[index] = df_global_htws[index].astype(float)
        # Plot figure
        plotter = PeriodDistributionPlotter()
        # Customize the visualization settings --> see documentation for all options
        plotter.update_config(
            kde_resolution=1000,  # Number of points to evaluate the KDE to reduce the computation time, default is 1000
        )

        plotter.plot(
            data=df_global_htws[pd.isnull(df_global_htws[index])==False],#.query("sdi == 'SPI' and aggregation == 3"),
            variable=index,
            periods_columns_labels={
                "Historical": "Historical",
                f"Period RWL {RWLs_to_plot[0]+1}": f"+{regional_warming_levels_list[RWLs_to_plot[0]]}°C",#"+1.5°C",f"Period RWL {RWLs_to_plot[0]+1}": f"+{regional_warming_levels_list[RWLs_to_plot[0]]}°C",#"+1.5°C",
                f"Period RWL {RWLs_to_plot[1]+1}": f"+{regional_warming_levels_list[RWLs_to_plot[1]]}°C",#"+2°C",
                f"Period RWL {RWLs_to_plot[2]+1}": f"+{regional_warming_levels_list[RWLs_to_plot[2]]}°C",#"+3°C",
            },
            # reference_events=None,
            reference_events=reference_htws,#.query("sdi == 'SPI' and aggregation == 3"),
            cut_kdes=False,  # Whether to cut the KDEs at the minimum / maximum values of the events at each period
            bounds=(
                bounds_dict[index][0],#1,
                bounds_dict[index][1],
                #* 1.1,
            ),
        )
        # Accessing the figure object to modify it further if needed
        fig = plotter.fig
        ax = plotter.ax
        fig.text(
                0.5,
                1,
                index,
                ha="center",
                va="center",
                fontsize="large",
                fontweight="bold",
            )

        if index in log_indices:
            labels = np.array(ax.get_xticks().tolist(), dtype=np.float64)
            new_labels = [r'$10^{%.0f}$' % (labels[i]) for i in range(len(labels))]
            ax.set_xticklabels(new_labels)
        
        fig.set_facecolor('white')
        ax.set_facecolor('white')
        # Saving the figure
        plotter.save(join(write_directory,f"distrib_RWL_{index}.pdf".replace(" ","_")))
        plotter.save(join(write_directory,f"distrib_RWL_{index}.jpg".replace(" ","_")))
        plotter.save(join(write_directory,f"distrib_RWL_{index}.png".replace(" ","_")))
        plt.close()

    # Plot all HWMId_pop_sspX on one figure and all Exposed_population_sspX on another figure
    print("Plotting indices for all SSPs")
    for ind in tqdm(['HWMId_pop','Exposed_population']):
        # 1. Initialize the figure and axes
        _ncols = 2  # Number of columns for the plot corresponding here to the number of indices
        _nrows = 3
        fig, axs = plt.subplots(
            nrows=_nrows*2,# + 1,  # Adding one row for the titles
            ncols=_ncols,# + 1,  # Adding one column for the y-axis labels
            figsize=(8.3 * 1.4, 11.7 * 0.9),
            width_ratios=[1] * _ncols,
            height_ratios=[0.1, 1] * _nrows,
            gridspec_kw={"hspace": 0.5, "wspace": 0.1},
        )

        for ssp in range(1,6):
            index = f"{ind}_ssp{ssp}"
            # Select the events to plot based on the chosen index (and ssp)
            _events_to_plot = df_global_htws[pd.isnull(df_global_htws[index])==False]
            _reference_events_to_plot = reference_htws
            row = 2*((ssp-1)//_ncols)+1
            col = (ssp-1)%_ncols
            ax = axs[row, col]  # +1 to skip the title row and y-axis label column
            plotter = PeriodDistributionPlotter(ax=ax, language="en")
            # Customize the visualization settings --> see documentation for all options
            plotter.update_config(
                kde_resolution=100,  # Number of points to evaluate the KDE to reduce the computation time
            )
            plotter.plot(
                data=_events_to_plot,
                variable=index,
                periods_columns_labels={
                    "Historical": "Historical",
                    f"Period RWL {RWLs_to_plot[0]+1}": f"+{regional_warming_levels_list[RWLs_to_plot[0]]}°C",#"+1.5°C",
                    f"Period RWL {RWLs_to_plot[1]+1}": f"+{regional_warming_levels_list[RWLs_to_plot[1]]}°C",#"+2°C",
                    f"Period RWL {RWLs_to_plot[2]+1}": f"+{regional_warming_levels_list[RWLs_to_plot[2]]}°C",#"+3°C",
                },
                reference_events=_reference_events_to_plot,
                cut_kdes=False,
                bounds=(
                    2.5, #if ind == "HWMId_pop" else df_global_htws[index].min(),
                    9 if ind == "HWMId_pop" else 7,#df_global_htws[index].max(),  # to limit the x-axis range (for better visualization)
                ),
            )
            labels = np.array(ax.get_xticks().tolist(), dtype=np.float64)
            new_labels = [r'$10^{%.0f}$' % (labels[i]) for i in range(len(labels))]
            ax.set_xticklabels(new_labels)
            # Format the titles
            axs[row-1, col].axis("off")
            axs[row-1, col].text(
                0.5,
                0.5,
                f"SSP{ssp}",
                ha="center",
                va="center",
                fontsize="large",
                fontweight="bold",
                rotation=0,
            )
        axs[-2, -1].axis("off") # Empty title of bottom-left cell
        axs[-1, -1].axis("off") # Bottom-left empty cell
        plt.savefig(join(write_directory,f"all_ssp_distrib_{ind}.pdf"),dpi=1200)
        plt.savefig(join(write_directory,f"all_ssp_distrib_{ind}.png"))
        plt.close()

    return

def plot_comparison_reanalysis_figures(read_directory,write_directory):
    dir_list = [item for item in listdir(read_directory) if isdir(join(read_directory,item))] # List all subdirectories
    dir_list = [_dir for _dir in dir_list if 'figs' not in _dir]
    dataframe_path_list = [join(read_directory,subdir,"df_htws.csv") for subdir in dir_list if exists(join(read_directory,subdir,"df_htws.csv"))] # Get all df_htws.csv files, containing heatwaves indices and data
    df_global_htws = pd.DataFrame(data=None,columns=['Year','Start Date','End Date','model','RWL_1','RWL_2','RWL_3','RWL_4','Intensity','Spatial extent','Accumulated area','Duration','Max', 'HWMId_sum',
    'Exposed_population_ghs','HWMId_pop_ghs','Exposed_population_ssp1','HWMId_pop_ssp1','Exposed_population_ssp2','HWMId_pop_ssp2',
    'Exposed_population_ssp3','HWMId_pop_ssp3','Exposed_population_ssp4','HWMId_pop_ssp4','Exposed_population_ssp5','HWMId_pop_ssp5',
    'GCM','RCM','simulation','version','ensemble','version_date','calendar','bias-adjusted','grid_mapping'])
    
    for df_path in dataframe_path_list:
        df_htws = pd.read_csv(df_path,header=0,parse_dates=["Start Date", "End Date"],date_format="%Y/%m/%d",usecols=lambda x: x!="Unnamed: 0")
        df_htws.insert(loc=3,column='model',value=df_path.split("/")[-2])
        df_global_htws = pd.concat([df_global_htws,df_htws],ignore_index=True)
    df_global_htws["Period RWL 1"] = False
    df_global_htws["Period RWL 2"] = False
    df_global_htws["Period RWL 3"] = False
    df_global_htws["Period RWL 4"] = False
    df_global_htws["Historical"] = False

    for idx in df_global_htws.index:
        for i in range(4):
            rwl = regional_warming_levels_list[i]
            df_global_htws.loc[idx,f"Period RWL {i+1}"] = (df_global_htws.loc[idx,"RWL_1"]==rwl) + (df_global_htws.loc[idx,"RWL_2"]==rwl) + (df_global_htws.loc[idx,"RWL_3"]==rwl) + (df_global_htws.loc[idx,"RWL_4"]==rwl)>0
        if df_global_htws.loc[idx,"model"]=='ERA5': #TODO Check this condition when data is ready
            df_global_htws.loc[idx,"Historical"] = True

    # Reference heatwaves
    reference_htws = pd.read_csv(join(read_directory,'ERA5','df_htws.csv'),header=0,index_col=0) # TODO Check this path
    reference_htws = reference_htws.loc[[26,91,191],:] # 26: 1987 Greece, 91: 2003 Europe, 120: 2010 Russia, 191: 2022 Europe
    reference_htws['label'] = reference_htws['Year']

    log_indices = ['Spatial extent','Accumulated area','HWMId_sum','Exposed_population_ghs','HWMId_pop_ghs','Exposed_population_ssp1','HWMId_pop_ssp1','Exposed_population_ssp2','HWMId_pop_ssp2',
    'Exposed_population_ssp3','HWMId_pop_ssp3','Exposed_population_ssp4','HWMId_pop_ssp4','Exposed_population_ssp5','HWMId_pop_ssp5']

    for index in log_indices :
        df_global_htws[index] = df_global_htws[index].apply(mylog)
        reference_htws[index] = reference_htws[index].apply(mylog)

    df_global_htws.to_csv(join(write_directory,"df_global_htws.csv"))

    bounds_dict = {
        "Intensity":[0.5,4],
        'Spatial extent':[4,7.5],
        'Duration':[4,90],
        'Max':[0,20],
        'HWMId_sum':[1,7]
    }

    print("Plotting individual indices")
    for index in tqdm(['Intensity','Duration','Max','Spatial extent','Accumulated area','HWMId_sum']):
        # Cast objet type to numeric type
        df_global_htws[index] = df_global_htws[index].astype(float)
        # Plot figure
        plotter = PeriodDistributionPlotter()
        # Customize the visualization settings --> see documentation for all options
        plotter.update_config(
            kde_resolution=1000,  # Number of points to evaluate the KDE to reduce the computation time, default is 1000
        )

        plotter.plot(
            data=df_global_htws[pd.isnull(df_global_htws[index])==False],#.query("sdi == 'SPI' and aggregation == 3"),
            variable=index,
            periods_columns_labels={
                "Historical": "Historical",
                f"Period RWL {RWLs_to_plot[0]+1}": f"+{regional_warming_levels_list[RWLs_to_plot[0]]}°C",#"+1.5°C",f"Period RWL {RWLs_to_plot[0]+1}": f"+{regional_warming_levels_list[RWLs_to_plot[0]]}°C",#"+1.5°C",
                f"Period RWL {RWLs_to_plot[1]+1}": f"+{regional_warming_levels_list[RWLs_to_plot[1]]}°C",#"+2°C",
                f"Period RWL {RWLs_to_plot[2]+1}": f"+{regional_warming_levels_list[RWLs_to_plot[2]]}°C",#"+3°C",
            },
            # reference_events=None,
            reference_events=reference_htws,#.query("sdi == 'SPI' and aggregation == 3"),
            cut_kdes=False,  # Whether to cut the KDEs at the minimum / maximum values of the events at each period
            bounds=(
                bounds_dict[index][0],#1,
                bounds_dict[index][1],
                #* 1.1,
            ),
        )
        # Accessing the figure object to modify it further if needed
        fig = plotter.fig
        ax = plotter.ax
        fig.text(
                0.5,
                1,
                index,
                ha="center",
                va="center",
                fontsize="large",
                fontweight="bold",
            )

        if index in log_indices:
            labels = np.array(ax.get_xticks().tolist(), dtype=np.float64)
            new_labels = [r'$10^{%.0f}$' % (labels[i]) for i in range(len(labels))]
            ax.set_xticklabels(new_labels)
        
        fig.set_facecolor('white')
        ax.set_facecolor('white')
        # Saving the figure
        plotter.save(join(write_directory,f"distrib_RWL_{index}.pdf".replace(" ","_")))
        plotter.save(join(write_directory,f"distrib_RWL_{index}.jpg".replace(" ","_")))
        plotter.save(join(write_directory,f"distrib_RWL_{index}.png".replace(" ","_")))
        plt.close()

    # Plot all HWMId_pop_sspX on one figure and all Exposed_population_sspX on another figure
    print("Plotting indices for all SSPs")
    for ind in tqdm(['HWMId_pop','Exposed_population']):
        # 1. Initialize the figure and axes
        _ncols = 2  # Number of columns for the plot corresponding here to the number of indices
        _nrows = 3
        fig, axs = plt.subplots(
            nrows=_nrows*2,# + 1,  # Adding one row for the titles
            ncols=_ncols,# + 1,  # Adding one column for the y-axis labels
            figsize=(8.3 * 1.4, 11.7 * 0.9),
            width_ratios=[1] * _ncols,
            height_ratios=[0.1, 1] * _nrows,
            gridspec_kw={"hspace": 0.5, "wspace": 0.1},
        )

        for ssp in range(1,6):
            index = f"{ind}_ssp{ssp}"
            # Select the events to plot based on the chosen index (and ssp)
            _events_to_plot = df_global_htws[pd.isnull(df_global_htws[index])==False]
            _reference_events_to_plot = reference_htws
            row = 2*((ssp-1)//_ncols)+1
            col = (ssp-1)%_ncols
            ax = axs[row, col]  # +1 to skip the title row and y-axis label column
            plotter = PeriodDistributionPlotter(ax=ax, language="en")
            # Customize the visualization settings --> see documentation for all options
            plotter.update_config(
                kde_resolution=100,  # Number of points to evaluate the KDE to reduce the computation time
            )
            plotter.plot(
                data=_events_to_plot,
                variable=index,
                periods_columns_labels={
                    "Historical": "Historical",
                    f"Period RWL {RWLs_to_plot[0]+1}": f"+{regional_warming_levels_list[RWLs_to_plot[0]]}°C",#"+1.5°C",
                    f"Period RWL {RWLs_to_plot[1]+1}": f"+{regional_warming_levels_list[RWLs_to_plot[1]]}°C",#"+2°C",
                    f"Period RWL {RWLs_to_plot[2]+1}": f"+{regional_warming_levels_list[RWLs_to_plot[2]]}°C",#"+3°C",
                },
                reference_events=_reference_events_to_plot,
                cut_kdes=False,
                bounds=(
                    2.5, #if ind == "HWMId_pop" else df_global_htws[index].min(),
                    9 if ind == "HWMId_pop" else 7,#df_global_htws[index].max(),  # to limit the x-axis range (for better visualization)
                ),
            )
            labels = np.array(ax.get_xticks().tolist(), dtype=np.float64)
            new_labels = [r'$10^{%.0f}$' % (labels[i]) for i in range(len(labels))]
            ax.set_xticklabels(new_labels)
            # Format the titles
            axs[row-1, col].axis("off")
            axs[row-1, col].text(
                0.5,
                0.5,
                f"SSP{ssp}",
                ha="center",
                va="center",
                fontsize="large",
                fontweight="bold",
                rotation=0,
            )
        axs[-2, -1].axis("off") # Empty title of bottom-left cell
        axs[-1, -1].axis("off") # Bottom-left empty cell
        plt.savefig(join(write_directory,f"all_ssp_distrib_{ind}.pdf"),dpi=1200)
        plt.savefig(join(write_directory,f"all_ssp_distrib_{ind}.png"))
        plt.close()

    return