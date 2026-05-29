import numpy as np
import xarray as xr
import netCDF4 as nc #load and write netcdf data
from datetime import date, datetime #create file history with creation date
from tqdm import tqdm #create a user-friendly feedback while script is running
from os import listdir
from os.path import isfile, join, isdir, exists
import re #Use RegEx 
import pandas as pd #handle dataframes
import cc3d #connected components patterns
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import json
import glob
from plots import *
import subprocess
import os
import matplotlib.ticker as mticker

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

    if 'x' in da.dims and 'y' in da.dims:
        da = da.chunk(chunks={'time':len(da.time),'x': 82,'y': 84})
    elif 'rlat' in da.dims and 'rlon' in da.dims:
        da = da.chunk(chunks={'time':len(da.time),'rlat': 82,'rlon': 84})
    elif 'lat' in da.dims and 'lon' in da.dims:
        da = da.chunk(chunks={'time':len(da.time),'lat': 36,'lon': 54})
    else:
        raise ValueError("Dimensions are not as expected. Should be either ('lat','lon'), ('rlat','lon'), or ('x','y').")
    # Since the files generally cover several years, have to select sub-period (it may not exactly match the boundaries of loaded files)
    mask = (da.time.dt.year>=start_year_ref) & (da.time.dt.year<=end_year_ref)
    da = da.sel(time=mask)
    del mask

    # Drop Feb 29
    original_calendar = da.time.dt.calendar
    if original_calendar != '360_day': 
        da = da.convert_calendar("noleap")
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
    threshold.to_netcdf(join(write_directory,f"distrib_threshold_{threshold_value}.nc"))

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
    if 'ERA5' not in read_directory_historical:#bias_adjusted==False:
        dust_threshold = int(dust_threshold * (0.25/resolution_CORDEX)**2) # resolution_CORDEX is given in °

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
        JJA_beg = 152 #152 is June 1st, 243 is August 31st
        JJA_end = 243

    if anomaly:
        seasonal_cycle = xr.open_dataarray(join(write_directory,f"seasonal_cycle_{start_year_ref}_{end_year_ref}.nc"), engine='netcdf4')
        # Keep only JJA values
        mask = (seasonal_cycle.dayofyear>=JJA_beg) & (seasonal_cycle.dayofyear<=JJA_end) # dayofyear ranges from 1 to 365 (or 360)
        seasonal_cycle = seasonal_cycle.sel(dayofyear=mask)

    if relative_threshold: # Load temperature threshold for reference period:
        threshold = xr.open_dataarray(join(write_directory,f"distrib_threshold_{threshold_value}.nc"), engine='netcdf4')
        # Keep only JJA values
        mask = (threshold.dayofyear>=JJA_beg) & (threshold.dayofyear<=JJA_end) 
        threshold = threshold.sel(dayofyear=mask)
    else: # If absolute threshold, only need a scalar, not a 3D array
        threshold = threshold_value

    N_labels = 0 # Count the numbers of patterns
    N_heatwaves = 0 # Count real number of heatwaves
    print("Computing cc3d.connected_components labels and dusting...")
    for file in tqdm(files_to_load):
        file_start_year = int(file[-20:-16])
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
            try:
                label.data = remove_sea_heatwaves(read_directory=join(other_data_path,"mask"),labels=label.data,grid_mapping=da.grid_mapping,connectivity=connectivity)
            except:
                label.data = remove_sea_heatwaves(read_directory=join(other_data_path,"mask"),labels=label.data,grid_mapping='ERA5',connectivity=connectivity)
            
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

def remove_sea_heatwaves(read_directory,labels,grid_mapping,resolution_CORDEX=0.11,land_area_fraction_threshold=0.5,connectivity=26,dust_threshold=775,bias_adjusted=False):
    '''
    '''
    # Set dust threshold to supress small amount of points. The threshold of 775 have been established with ERA5 0.25°, and is translated according to CORDEX resolution, hence the following calculation
    if grid_mapping!='ERA5':#bias_adjusted==False:
        dust_threshold = int(dust_threshold * (0.25/resolution_CORDEX)**2) # resolution_CORDEX is given in °

    land_sea_mask = xr.open_dataset(join(read_directory,f"mask_Europe_land_only_CORDEX_EUR11_{grid_mapping}.nc"),engine='netcdf4').mask # ERA5 land-sea mask

    # Remove sea points and non-European points
    labels = labels * (land_sea_mask.data==0) #mask is 0 for European countries, 1 elsewhere (sea and non-European countries)

    # Remove small heatwaves without using the cc3d dust function
    labels_list = np.unique(labels)
    labels_list = labels_list[np.where(labels_list!=0)] # Remove 0 which is not a heatwave label

    for lab in labels_list:
        if (labels==lab).sum() < dust_threshold:
            labels = labels*((labels!=lab)) # Remove every point where label is one of a small heatwave
    
    #labels = labels * cc3d.dust((labels>0),dust_threshold,connectivity=connectivity)
    
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
    try:
        grid_mapping = da.grid_mapping
    except:
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
    try:
        grid_mapping = xr.open_dataset(join(write_directory,f"labels_cc3d_year_{start_year}_ref_{start_year_ref}_{end_year_ref}.nc"), engine="netcdf4").label.grid_mapping
    except:
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
        JJA_beg = 152 #152 is June 1st, 243 is August 31st
        JJA_end = 243

    da_threshold = xr.open_dataarray(join(write_directory,f"distrib_threshold_{threshold_value}.nc"),engine='netcdf4')
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
    'Intensity','Spatial extent','Duration','Max','HWMId_sum',
    'Exposed_population_ghs','HWMId_pop_ghs','Exposed_population_ssp1','HWMId_pop_ssp1',
    'Exposed_population_ssp2','HWMId_pop_ssp2','Exposed_population_ssp3','HWMId_pop_ssp3',
    'Exposed_population_ssp4','HWMId_pop_ssp4','Exposed_population_ssp5','HWMId_pop_ssp5','GCM','RCM','simulation','version','ensemble','version_date','calendar','bias-adjusted','grid_mapping']) # Create DataFrame

    # Initialize variables used to find the correct temperature file to load for each year
    loaded_temp_file = None
    old_loaded_temp_file = None

    heatwaves_year = 0

    for year in tqdm(range(start_year,end_year+1)): # Iterate over the years
        # Load labels if there are heatwaves on given year
        try:
            ds_labels = xr.open_dataset(join(write_directory,f"labels_cc3d_year_{year}_ref_{start_year_ref}_{end_year_ref}.nc"), engine="netcdf4") # Load the corresponding nc file
            labels_exist = True
            heatwaves_year += 1
        except:
            labels_exist = False
        
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
            
            for label in labels_list: # Iterate over the heatwaves (one label = one heatwave).
                df_htws.loc[label,"Year"]=year
                # Find the corresponding RWL(s)
                rwl_count=0
                for rwl in RWL_dict:
                    if (RWL_dict[rwl] != None) and (year in range(RWL_dict[rwl]["start_year"],RWL_dict[rwl]["end_year"]+1)):
                        rwl_count += 1 # Find the correct column to fill
                        df_htws.loc[label,f"RWL_{rwl_count}"] = float(rwl) # Record the corresponding RWL
                da = da_labels
                da_bool_htw = da.where(da==label, drop=True).fillna(0)>0 # Select days and grid points for the heatwave of interest and convert to bool array
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
                df_htws.loc[label,'Duration'] = len(da_temp_htw.time)
                df_htws.loc[label,'Max'] = da_temp_htw.max().data
                df_htws.loc[label,'HWMId_sum'] = da_HWMId_htw.weighted(weights.fillna(0)).sum().data
                if year<=2030: #GHS-POP covers 1975-2030
                    da_pop = ds_ghs_pop.Band1
                    da_pop_htw = da_pop.sel(time=(da_pop.time.dt.year==year))
                    
                    # Workaround bug in da_pop_htw in some cases
                    pop_copy = (da_temp.isel(time=0)).copy()
                    pop_copy.data = da_pop_htw.data[0]
                    da_pop_htw = pop_copy

                    da_pop_htw = da_pop_htw.where(labels_bool_2D,drop=True)
                    df_htws.loc[label,f'Exposed_population_ghs'] = da_pop_htw.sum().data
                    df_htws.loc[label,f'HWMId_pop_ghs'] = (da_HWMId_htw*da_pop_htw.data).weighted(weights.fillna(0)).sum().data
                if year>=2020: # FPOP covers 2020-2100
                    for ssp in range(1,6):
                        da_pop = ssp_file_dict[f"ds_pop_ssp{ssp}"].Band1
                        da_pop_htw = da_pop.sel(time=(da_pop.time.dt.year==year))
                    
                        # Workaround bug in da_pop_htw in some cases
                        pop_copy = (da_temp.isel(time=0)).copy()
                        pop_copy.data = da_pop_htw.data[0]
                        da_pop_htw = pop_copy

                        da_pop_htw = da_pop_htw.where(labels_bool_2D,drop=True)
                        
                        df_htws.loc[label,f'Exposed_population_ssp{ssp}'] = da_pop_htw.sum().data
                        df_htws.loc[label,f'HWMId_pop_ssp{ssp}'] = (da_HWMId_htw*da_pop_htw.data).sum().data
                if year <= split_year:
                    df_htws['simulation'] = "historical"
                else:
                    df_htws['simulation'] = read_directory_rcp.split("/")[-7] #rcp45 or rcp85

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

def merge_heatwaves_dataframes(read_directory,write_directory,regional_warming_levels_list=[2.1,2.6,4.0,5.1]):
    dir_list = [item for item in listdir(read_directory) if isdir(join(read_directory,item))] # List all subdirectories
    dir_list = [_dir for _dir in dir_list if 'figs' not in _dir]
    dataframe_path_list = [join(read_directory,subdir,"df_htws.csv") for subdir in dir_list if exists(join(read_directory,subdir,"df_htws.csv"))] # Get all df_htws.csv files, containing heatwaves indices and data
    df_global_htws = pd.DataFrame(data=None,columns=['Year','Start Date','End Date','model','original_label','RWL_1','RWL_2','RWL_3','RWL_4','Intensity','Spatial extent','Duration','Max', 'HWMId_sum',
    'Exposed_population_ghs','HWMId_pop_ghs','Exposed_population_ssp1','HWMId_pop_ssp1','Exposed_population_ssp2','HWMId_pop_ssp2',
    'Exposed_population_ssp3','HWMId_pop_ssp3','Exposed_population_ssp4','HWMId_pop_ssp4','Exposed_population_ssp5','HWMId_pop_ssp5',
    'GCM','RCM','simulation','version','ensemble','version_date','calendar','bias-adjusted','grid_mapping'])
    
    for df_path in tqdm(dataframe_path_list):
        df_htws = pd.read_csv(df_path,index_col=0,header=0,parse_dates=["Start Date", "End Date"],date_format="%Y/%m/%d")
        df_htws.insert(loc=2,column='model',value=df_path.split("/")[-2])
        df_htws.insert(loc=3,column='original_label',value=df_htws.index)
        df_global_htws = pd.concat([df_global_htws,df_htws],ignore_index=True)
    df_global_htws["Period RWL 1"] = False
    df_global_htws["Period RWL 2"] = False
    df_global_htws["Period RWL 3"] = False
    df_global_htws["Period RWL 4"] = False
    df_global_htws["Historical"] = False

    for idx in tqdm(df_global_htws.index):
        for i in range(4):
            rwl = regional_warming_levels_list[i]
            df_global_htws.loc[idx,f"Period RWL {i+1}"] = (df_global_htws.loc[idx,"RWL_1"]==rwl) + (df_global_htws.loc[idx,"RWL_2"]==rwl) + (df_global_htws.loc[idx,"RWL_3"]==rwl) + (df_global_htws.loc[idx,"RWL_4"]==rwl)>0
        if df_global_htws.loc[idx,"model"]=='ERA5': #TODO Check this condition when data is ready
            df_global_htws.loc[idx,"Historical"] = True

    df_global_htws.to_csv(join(write_directory,"df_global_htws.csv"))
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
                        os.makedirs(join(write_directory,model),exist_ok=True)
                        subprocess.call(f"cdo -remapnn,{join(target_grid_directory,f'gridarea_CORDEX_EUR11_{mapping_target}.nc')} {file} {join(write_directory,model,file.split('/')[-1])}",shell=True)
        else: # i.e grid_mapping is mapping_target, just copy file
            mapping_model_list = np.unique(df_htws[df_htws['grid_mapping']==grid_mapping]['model'])
            for model in tqdm(mapping_model_list):
                print(model)
                labels_files = glob.glob(join(read_directory,model,'labels*.nc'))
                for file in labels_files:
                    if overwrite == True or exists(file) == False:
                        os.makedirs(join(write_directory,model),exist_ok=True)
                        subprocess.call(f"cp {file} {join(write_directory,model,file.split('/')[-1])}",shell=True)

def plot_4_panel_hot_days(read_directory,write_directory,start_year=1975,end_year=2099,need_to_compute_labels=False):
    """Plot hot days per location"""
    if need_to_compute_labels:
        files_list = np.sort(glob.glob(join(read_directory,'remapped_labels_for_figs','*','labels*.nc')))
        # Initialize arrays to hold data
        labels_days_all = xr.open_dataset(files_list[0],engine='netcdf4').label.copy().sum(dim='time')
        labels_days_all.data = np.zeros(np.shape(labels_days_all))
        labels_htws_all = labels_days_all.copy()
        labels_days_all_92 = labels_days_all.copy()
        labels_htws_all_92 = labels_days_all.copy()

        labels_days_raw = labels_days_all.copy()
        labels_htws_raw = labels_days_all.copy()
        labels_days_raw_92 = labels_days_all.copy()
        labels_htws_raw_92 = labels_days_all.copy()

        labels_days_adj = labels_days_all.copy()
        labels_htws_adj = labels_days_all.copy()
        labels_days_adj_92 = labels_days_all.copy()
        labels_htws_adj_92 = labels_days_all.copy()

        df_htws = pd.read_csv(join(write_directory,'df_global_htws.csv'),header=0,index_col=0)
        df_92_days = df_htws[df_htws['Duration']==92]

        # Iterate over files
        for file in tqdm(files_list):
            year = int(file[-21:-17])
            model = file.split("/")[-2]
            bias_adjusted = 'Adjust' in model
            if year >= start_year and year <= end_year and model != 'ERA5':
                sub_df = df_92_days[(df_92_days['model']==model) & (df_92_days['Year']==year)]
                file_labels = xr.open_dataset(file,engine='netcdf4').label
                bool_labels = (file_labels>0)
                labels_days_all.data = labels_days_all.data + bool_labels.sum(dim='time').data
                if bias_adjusted:
                    labels_days_adj.data = labels_days_adj.data + bool_labels.sum(dim='time').data
                else:
                    labels_days_raw.data = labels_days_raw.data + bool_labels.sum(dim='time').data
                labels_list = np.unique(file_labels.data)
                labels_list = labels_list[np.where(labels_list!=0)]
                for lab in labels_list:
                    bool_labels = np.max(file_labels==lab,axis=0)
                    labels_htws_all.data = labels_htws_all.data + bool_labels.data
                    if bias_adjusted:
                        labels_htws_adj.data = labels_htws_adj.data + bool_labels.data
                    else:
                        labels_htws_raw.data = labels_htws_raw.data + bool_labels.data
                    if lab in sub_df['original_label'].values:
                        labels_htws_all_92.data = labels_htws_all_92.data + bool_labels.data
                        labels_days_all_92.data = labels_days_all_92.data + np.sum(file_labels==lab,axis=0)
                        if bias_adjusted:
                            labels_htws_adj_92.data = labels_htws_adj_92.data + bool_labels.data
                            labels_days_adj_92.data = labels_days_adj_92.data + np.sum(file_labels==lab,axis=0)
                        else:
                            labels_htws_raw_92.data = labels_htws_raw_92.data + bool_labels.data
                            labels_days_raw_92.data = labels_days_raw_92.data + np.sum(file_labels==lab,axis=0)
        # Export label files to netcdf
        labels_days_all.to_dataset(name="label")
        labels_htws_all.to_dataset(name="label")
        labels_days_all_92.to_dataset(name="label")
        labels_htws_all_92.to_dataset(name="label")

        labels_days_all.to_netcdf(join(write_directory,"labels_days_all.nc"))
        labels_htws_all.to_netcdf(join(write_directory,"labels_htws_all.nc"))
        labels_days_all_92.to_netcdf(join(write_directory,"labels_days_all_92.nc"))
        labels_htws_all_92.to_netcdf(join(write_directory,"labels_htws_all_92.nc"))

        labels_days_raw.to_dataset(name="label")
        labels_htws_raw.to_dataset(name="label")
        labels_days_raw_92.to_dataset(name="label")
        labels_htws_raw_92.to_dataset(name="label")

        labels_days_raw.to_netcdf(join(write_directory,"labels_days_raw.nc"))
        labels_htws_raw.to_netcdf(join(write_directory,"labels_htws_raw.nc"))
        labels_days_raw_92.to_netcdf(join(write_directory,"labels_days_raw_92.nc"))
        labels_htws_raw_92.to_netcdf(join(write_directory,"labels_htws_raw_92.nc"))

        labels_days_adj.to_dataset(name="label")
        labels_htws_adj.to_dataset(name="label")
        labels_days_adj_92.to_dataset(name="label")
        labels_htws_adj_92.to_dataset(name="label")

        labels_days_adj.to_netcdf(join(write_directory,"labels_days_adj.nc"))
        labels_htws_adj.to_netcdf(join(write_directory,"labels_htws_adj.nc"))
        labels_days_adj_92.to_netcdf(join(write_directory,"labels_days_adj_92.nc"))
        labels_htws_adj_92.to_netcdf(join(write_directory,"labels_htws_adj_92.nc"))
    else:
        labels_days_all = xr.open_dataset(join(write_directory,"labels_days_all.nc"),engine='netcdf4').label
        labels_htws_all = xr.open_dataset(join(write_directory,"labels_htws_all.nc"),engine='netcdf4').label
        labels_days_all_92 = xr.open_dataset(join(write_directory,"labels_days_all_92.nc"),engine='netcdf4').label
        labels_htws_all_92 = xr.open_dataset(join(write_directory,"labels_htws_all_92.nc"),engine='netcdf4').label

        labels_days_raw = xr.open_dataset(join(write_directory,"labels_days_raw.nc"),engine='netcdf4').label
        labels_htws_raw = xr.open_dataset(join(write_directory,"labels_htws_raw.nc"),engine='netcdf4').label
        labels_days_raw_92 = xr.open_dataset(join(write_directory,"labels_days_raw_92.nc"),engine='netcdf4').label
        labels_htws_raw_92 = xr.open_dataset(join(write_directory,"labels_htws_raw_92.nc"),engine='netcdf4').label

        labels_days_adj = xr.open_dataset(join(write_directory,"labels_days_adj.nc"),engine='netcdf4').label
        labels_htws_adj = xr.open_dataset(join(write_directory,"labels_htws_adj.nc"),engine='netcdf4').label
        labels_days_adj_92 = xr.open_dataset(join(write_directory,"labels_days_adj_92.nc"),engine='netcdf4').label
        labels_htws_adj_92 = xr.open_dataset(join(write_directory,"labels_htws_adj_92.nc"),engine='netcdf4').label
    
    # Plot figure
    labels_list = [
        [[labels_days_all,labels_htws_all],[labels_days_all_92,labels_htws_all_92]],
        [[labels_days_adj,labels_htws_adj],[labels_days_adj_92,labels_htws_adj_92]],
        [[labels_days_raw,labels_htws_raw],[labels_days_raw_92,labels_htws_raw_92]]
    ]

    proj_pc = ccrs.PlateCarree()
    proj_rp = ccrs.RotatedPole(pole_latitude=39.25, pole_longitude=198)

    for count,fig_title in enumerate(['hot_days_4_panels_all_models','hot_days_4_panels_bias_adj_models','hot_days_4_panels_raw_models']):
        fig,axes = plt.subplots(
        2, 2, figsize=(14,10), sharex=True, sharey=True,
        subplot_kw={'projection': proj_pc, "aspect": 1},
        gridspec_kw = {'wspace':0.02, 'hspace':0.07},
    )

        titles_list = [
            ['Hot days','Heatwaves occurrences'],
            ['Hot days in 92-day heatwaves','92-day heatwaves occurrences']
            ]
        
        for i in [0,1]:
            for j in [0,1]:
                ax = axes[i][j]
                labels = labels_list[count][i][j]
                labels = labels.where(labels>0)

                img = labels.plot(cmap='YlOrRd',ax=ax,levels=7,add_labels=False,transform=proj_rp)
                ax.set_extent([-30, 67, 30, 75], crs=proj_pc)
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
        print(f"Saving {fig_title}.pdf ...")
        plt.savefig(join(write_directory,f'{fig_title}.pdf'),dpi=1200)
        print(f"Saved.")
    return

def plot_RWL_figures(read_directory,write_directory,regional_warming_levels_list=[2.1,2.6,4.0,5.1], RWLs_to_plot = [0,1,2]):
    df_global_htws = pd.read_csv(join(write_directory,"df_global_htws.csv"))
    # Reference heatwaves
    reference_htws = pd.read_csv(join(read_directory,'ERA5','df_htws.csv'),header=0,index_col=0) # TODO Check this path
    reference_htws = reference_htws.loc[[26,91,191],:] # 26: 1987 Greece, 91: 2003 Europe, 120: 2010 Russia, 191: 2022 Europe
    reference_htws['label'] = reference_htws['Year']

    log_indices = ['Spatial extent','HWMId_sum','Exposed_population_ghs','HWMId_pop_ghs','Exposed_population_ssp1','HWMId_pop_ssp1','Exposed_population_ssp2','HWMId_pop_ssp2',
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
    for index in tqdm(['Intensity','Duration','Max','Spatial extent','HWMId_sum']):
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
    df_global_htws = pd.DataFrame(data=None,columns=['Year','Start Date','End Date','model','RWL_1','RWL_2','RWL_3','RWL_4','Intensity','Spatial extent','Duration','Max', 'HWMId_sum',
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

    log_indices = ['Spatial extent','HWMId_sum','Exposed_population_ghs','HWMId_pop_ghs','Exposed_population_ssp1','HWMId_pop_ssp1','Exposed_population_ssp2','HWMId_pop_ssp2',
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
    for index in tqdm(['Intensity','Duration','Max','Spatial extent','HWMId_sum']):
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

def plot_hot_days_locations(read_directory,write_directory):
    #dir_list = [item for item in listdir(read_directory) if isdir(join(read_directory,item))] # List all subdirectories
    #dir_list = [_dir for _dir in dir_list if 'figs' not in _dir]
    #dataframe_path_list = [join(read_directory,subdir,"df_htws.csv") for subdir in dir_list if exists(join(read_directory,subdir,"df_htws.csv"))] # Get all df_htws.csv files, containing heatwaves indices and data
    labels_file_list = np.sort(glob.glob("*/labels*.nc"))