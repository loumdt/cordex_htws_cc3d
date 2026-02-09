import numpy as np
import numpy.ma as ma #use masked array
import xarray as xr
import netCDF4 as nc #load and write netcdf data
from datetime import date, timedelta, datetime #create file history with creation date
from tqdm import tqdm #create a user-friendly feedback while script is running
from os import listdir
from os.path import isfile, join
import re #Use RegEx 
import pandas as pd #handle dataframes
import cc3d #connected components patterns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as clrs
from scipy import stats
from scipy import signal
import json

# Split year is the last year of the historical period. In CORDEX runs based on CMIP5 experiment, it is 2005.
# Have to change value for a different version of CORDEX. 
split_year = 2005

def find_correct_year_file(read_directory,start_year,interval) :
    files_list = [f for f in listdir(read_directory) if isfile(join(read_directory, f))]
    pattern = re.compile(f"{start_year}0101-{start_year+interval-1}123[0-1].nc$")
    counter = 1
    for file in files_list :
        match = pattern.search(file)
        if match is not None :
            break
        elif counter==len(files_list) :
            raise FileNotFoundError(f"File not for found year {start_year} and interval {interval} in directory {read_directory}")
        counter+=1
    return file

def check_time_consistency(read_directory,interval=5,start_year=1951,end_year=split_year) :
    """Checks that the time intervals are consistent with pre-defined parameters. 
    Returns True if files present in read_directory are consistent, False otherwise"""
    if (end_year-start_year+1)%interval != 0 :
        raise ValueError(f"Incorrect input. Inconsistency between start_year ({start_year}), end_year ({end_year}), and interval ({interval}).")
    files_list = [f for f in listdir(read_directory) if isfile(join(read_directory, f))]
    for year in range(start_year,end_year,interval) :
        pattern = re.compile(f"{year}0101-{year+interval-1}123[0-1].nc$")
        flag = np.array([not(pattern.search(file) is None) for file in files_list]).any() # flag is True if there is a match for one of the files, False otherwise
        if not flag :
            break
    return flag

def compute_climatology_smooth(read_directory_historical,read_directory_rcp,write_directory,start_year_ref=1951,end_year_ref=2020,interval=5,temp_variable='tas',smooth_span=15) :
    '''This function computes a climatology for each calendar day of the year. The seasonal cycle is then smoothed with a 31-day window. 
    By default, the climatology is computed over 1951-2020.
    This function can be used with several models and variables.'''

    # Create list of years of the beginning of each file
    year_list = range(start_year_ref,end_year_ref,interval)

    # Load .nc files in a dictionary
    # Not using xr.open_mfdataset() to have better performances (the entire function is ~6x slower if using use xr.open_mfdataset() according to test)
    ds_dict = {}
    for year in year_list :
        if year<=split_year : # Before split_year, historical run
            ds_dict[year] = xr.open_dataset(join(read_directory_historical,find_correct_year_file(read_directory_historical,year,interval)), engine="netcdf4") 
        else : # After split_year, RCP (4.5 or 8.5)
            ds_dict[year] = xr.open_dataset(join(read_directory_rcp,find_correct_year_file(read_directory_rcp,year,interval)), engine="netcdf4") 

   # Initialize output data array with the first file
    da = getattr(ds_dict[start_year_ref], temp_variable)
    # Drop Feb 29
    da = da.convert_calendar("noleap")
    # Group using dayofyear and sum to compute mean at the end
    climatology = da.groupby(da.time.dt.dayofyear).sum()

    # Iterate over files, except first one which has already been used in initialization
    for year in tqdm(year_list[1:]) : 
        da = getattr(ds_dict[year], temp_variable)
        da = da.convert_calendar("noleap")
        da = da.groupby(da.time.dt.dayofyear).sum()
        climatology += da
    
    # Divide by the number of years to compute mean
    climatology /= end_year_ref - start_year_ref +1 

    # Smoothing
    # Handle easily first and last day of year by taking thrice the entire dataset and working on the middle one (avoiding border effects)
    extended_temp=np.zeros((365*3,np.shape(climatology.data)[1],np.shape(climatology.data)[2]))
    extended_temp[0:365,:,:]=climatology.data[:,:,:]
    extended_temp[365:730,:,:]=climatology.data[:,:,:]
    extended_temp[730:,:,:]=climatology.data[:,:,:]

    print("Smoothing...")
    #For each day d, compute the mean of the values between day d-smooth_span and d+smooth_span to have a physically realistic seasonal cycle
    for i in tqdm(range(365,730)):
        val_table=np.array(np.zeros((2*smooth_span+1,np.shape(climatology.data)[1],np.shape(climatology.data)[2])))
        for j in range(-smooth_span,smooth_span+1,1):
            val_table[j]=extended_temp[i+j,:,:]
        climatology.data[i-365,:,:] = np.nanmean(val_table[:],axis=0)

    # Export data to netcdf file
    climatology.to_netcdf(join(write_directory,f"{temp_variable}_climatology_{start_year_ref}_{end_year_ref}.nc"))

    # Clear resources
    da.close()
    climatology.close()
    return

def compute_distrib_percentile(read_directory_historical,read_directory_rcp,write_directory,start_year_ref=1951,end_year_ref=2020,interval=5,temp_variable='tas',threshold_value=95,distrib_window_size=15,anomaly=True) :
    '''This function computes, for every calendar day, the n-th (n is the threshold_value, default 95) percentile of the corresponding distribution of daily variable. 
    By default, the distribution is computed over 1951-2020.'''

    if distrib_window_size%2==0:
        raise ValueError('distrib_window_size is even. It has to be odd so the window can be centered on the computed day.')

    # Create list of years of the beginning of each file
    year_list = range(start_year_ref,end_year_ref,interval)

    # Load .nc files in a dictionary
    ds_dict = {}
    for year in year_list :
        pattern = re.compile(f"{year}0101-{year+interval-1}123[0-1].nc$") # Choose file based on start year and interval
        if year<=split_year : # Before split_year, historical run
            ds_dict[year] = xr.open_dataset(join(read_directory_historical,find_correct_year_file(read_directory_historical,year,interval)), engine="netcdf4") 
        else : # After split_year, RCP (4.5 or 8.5)
            ds_dict[year] = xr.open_dataset(join(read_directory_rcp,find_correct_year_file(read_directory_rcp,year,interval)), engine="netcdf4")

    # Load climatology file to create output data structure and compute anomaly
    climatology = xr.open_dataarray(join(write_directory,f"{temp_variable}_climatology_{start_year_ref}_{end_year_ref}.nc"), engine='netcdf4')
    # Create threshold table by copying climatology table, values will be updated later
    threshold = climatology.copy()

    # Initialize data array with the first file
    da = getattr(ds_dict[start_year_ref], temp_variable) # Iterate over files, except first one which has already been used in initialization
    # Iterate over files, except first one which has already been used in initialization
    for year in year_list[1:] : 
        da = xr.concat(objs=[da,getattr(ds_dict[year], temp_variable)], dim="time")
    # Drop 29 Feb
    da = da.convert_calendar("noleap")
    if anomaly :
        # Compute anomaly
        for year in tqdm(range(len(da.time)//365)) :
            da[year*365:(year+1)*365,:,:] = da[year*365:(year+1)*365,:,:] - climatology.data # Compute anomaly
    else :
        climatology.close()
    
    for day in tqdm(range(1,366)) : # Calendar days ranging from 1 to 365 (no leap years)
        if day - (distrib_window_size//2) <= 0  : # day is at the beginning of January, window overlapping with December
            distrib_start_bound = 365 + day - (distrib_window_size//2) # start bound of the distribution window
            distrib_end_bound = day + (distrib_window_size//2) # end bound of the distribution window
            mask = (da.time.dt.dayofyear >= distrib_start_bound) + (da.time.dt.dayofyear <= distrib_end_bound) # Create a boolean mask for days between distrib_start_bound and distrib_end_bound of the year
        elif day + (distrib_window_size//2) > 365 : # day is at the end of December , window overlapping with January
            distrib_start_bound = day - (distrib_window_size//2)
            distrib_end_bound = day + (distrib_window_size//2) - 365
            mask = (da.time.dt.dayofyear >= distrib_start_bound) + (da.time.dt.dayofyear <= distrib_end_bound) # Create a boolean mask for days between distrib_start_bound and distrib_end_bound of the year
        else :
            distrib_start_bound = day - (distrib_window_size//2)
            distrib_end_bound = day + (distrib_window_size//2)
            mask = (da.time.dt.dayofyear >= distrib_start_bound) & (da.time.dt.dayofyear <= distrib_end_bound)
        # Apply the mask to select the corresponding days and compute percentile, take day-1 because day ranges from 1 to 365 but python indexing ranges from 0 to 364
        threshold.data[day-1,:,:] = np.percentile(da.sel(time=mask).data,threshold_value,axis=0)

    # Export data to netcdf file
    threshold.to_netcdf(join(write_directory,f"{temp_variable}_distrib_{'ano_'*anomaly}{start_year_ref}_{end_year_ref}_threshold_{threshold_value}_window_{distrib_window_size}d.nc"))

    # Clear resources
    da.close()
    threshold.close()
    if anomaly :
        climatology.close()
    return

def cc3d_scan_heatwaves(read_directory_historical,read_directory_rcp,write_directory,start_year=1951,end_year=2100,start_year_ref=1951,end_year_ref=2020,interval=5,temp_variable='tas',threshold_value=95,relative_threshold=True,distrib_window_size=15,anomaly=True,nb_days=4,resolution_CORDEX=0.11) :
    '''This function carries out a cc3d scan (https://pypi.org/project/connected-components-3d/) to detect heatwaves in the meteorological database (default ERA5, t2m, tg).
    The heatwaves point are labeled with a number corresponding to a heatwave identifier.
    Otherwise, values are set to -9999.'''
    # Set dust threshold to supress small amount of points. The threshold of 775 have been established with ERA5 0.25°, and is translated according to CORDEX resolution, hence the following calculation
    dust_threshold = int(775 * (0.25/resolution_CORDEX)**2) # resolution_CORDEX is given in °

    # Create list of years of the beginning of each file
    year_list = range(start_year,end_year,interval)

    # Load .nc files in a dictionary
    ds_dict = {}
    for year in year_list :
        pattern = re.compile(f"{year}0101-{year+interval-1}123[0-1].nc$") # Choose file based on start year and interval
        if year<=split_year : # Before split_year, historical run
            ds_dict[year] = xr.open_dataset(join(read_directory_historical,find_correct_year_file(read_directory_historical,year,interval)), engine="netcdf4") 
        else : # After split_year, RCP (4.5 or 8.5)
            ds_dict[year] = xr.open_dataset(join(read_directory_rcp,find_correct_year_file(read_directory_rcp,year,interval)), engine="netcdf4")

    # Initialize data array with the first file
     # Iterate over files, except first one which has already been used in initialization
    # Iterate over files, except first one which has already been used in initialization
    if anomaly :
        climatology = xr.open_dataarray(join(write_directory,f"{temp_variable}_climatology_{start_year_ref}_{end_year_ref}.nc"), engine='netcdf4')
        # Keep only JJA values
        mask = (climatology.dayofyear>=152) & (climatology.dayofyear<=243) # dayofyear ranges from 1 to 365 ; 152 is June 1st, 243 is August 31st
        climatology = climatology.sel(dayofyear=mask)

    if relative_threshold : # Load temperature threshold for reference period :
        threshold = xr.open_dataarray(join(write_directory,f"{temp_variable}_distrib_{'ano_'*anomaly}{start_year_ref}_{end_year_ref}_threshold_{threshold_value}_window_{distrib_window_size}d.nc"), engine='netcdf4')
        # Keep only JJA values
        mask = (threshold.dayofyear>=152) & (threshold.dayofyear<=243) 
        threshold = threshold.sel(dayofyear=mask)
    else : # If absolute threshold, only need a scalar, not a 3D array
        threshold = threshold_value

    N_labels = 0 # Count the numbers of patterns
    print("Computing cc3d.connected_components labels and dusting...")
    for year_file in tqdm(year_list[1:]) :
        da = getattr(ds_dict[year_file], temp_variable)
        # Drop 29 Feb and correct day of year
        da = da.convert_calendar("noleap")
        # Keep only JJA values
        mask = (da.time.dt.season=='JJA')
        da = da.sel(time=mask)

        for year in range(interval) :# Iterate over the years
            da_year = da[year*92:(year+1)*92,:,:]# Select data for the given year
            if anomaly : # Substract climatology to compute anomaly
                da_year = da_year - climatology.data
            da_year = da_year*(da_year>threshold.data)-9999*(da_year<=threshold.data) # Set to -9999 the values that do not exceed threshold
            stack_temp = -9999*np.ones((92,np.shape(da_year)[1],np.shape(da_year)[2])) # Create a 3D array that will hold the temperature values when and where there are heatwaves
            stack_where = np.zeros((np.shape(da_year)[1],np.shape(da_year)[2])) # Create a 2D array that holds the number of consecutive hot days for each location, computed for each day
            for day in range(92):
                stack_where[:,:] = stack_where[:,:] + np.ones((np.shape(da_year)[1],np.shape(da_year)[2]))*(da_year[day,:,:]!=-9999) # Add one day to each potential heatwave location
                stack_where[:,:] = stack_where[:,:]*(da_year[day,:,:]!=-9999) # When not adding a day, have to set back the duration to zero
                if day>=nb_days-1 :
                    stack_temp[day-(nb_days-1):day+1,:,:] = stack_temp[day-(nb_days-1):day+1,:,:]*(stack_temp[day-(nb_days-1):day+1,:,:]!=-9999)+da_year[day-(nb_days-1):day+1,:,:]*((stack_where>=nb_days)*stack_temp[day-(nb_days-1):day+1,:,:]==-9999)+(-9999*((stack_where<nb_days)*(stack_temp[day-(nb_days-1):day+1,:,:]==-9999))) #record the last four days for the corresponding scanning window, add new consecutive hot days and set not hot days to -9999

            # Compute connected components for the remaining values of stack_temp
            connectivity = 26 # only 4,8 (2D) and 26, 18, and 6 (3D) are allowed
            labels_in = cc3d.dust((stack_temp!=-9999),dust_threshold)
            labels_out, N_added = cc3d.connected_components(labels_in, connectivity=connectivity,return_N=True) # Return the table of labels and the number of added patterns
            # Initialize output array 
            label = da_year.copy()
            # Record labels and add N_labels offset where labels are nonzero
            label.data = labels_out + N_labels*(labels_out>0)

            #Remove sea heatwaves
            label.data = remove_sea_heatwaves(read_directory="/data/tmandonnet/CORDEX/sftlf/",labels=label.data,grid_mapping=da.grid_mapping)
            #label.data = remove_sea_heatwaves(read_directory="/home/user/These/cordex_htws_cc3d/Data/sftlf/",labels=label.data)

            # Set DataArray to Dataset to set variable name
            label = label.to_dataset(name="label")
            # Save to netCDF 
            label.to_netcdf(join(write_directory,f"{temp_variable}_labels_{'ano_'*anomaly}year_{year_file+year}_ref_{start_year_ref}_{end_year_ref}_threshold_{threshold_value}_{nb_days}d_window_{distrib_window_size}d.nc"))

            # Update N_labels
            N_labels += N_added

        # Clear resources
        ds_dict[year_file].close()
        label.close()
        threshold.close()
        if anomaly :
            climatology.close()
    print(N_labels,"heatwaves detected")
    return

def remove_sea_heatwaves(read_directory,labels,grid_mapping,resolution_CORDEX=0.11,land_area_fraction_threshold=0.5) :
    '''
    '''
    # Set dust threshold to supress small amount of points. The threshold of 775 have been established with ERA5 0.25°, and is translated according to CORDEX resolution, hence the following calculation
    dust_threshold = int(775 * (0.25/resolution_CORDEX)**2) # resolution_CORDEX is given in °

    land_area_fraction = xr.open_dataset(join(read_directory,f"CORDEX_EUR-11_land_area_fraction_{grid_mapping}.nc"),engine='netcdf4').sftlf
    if np.max(land_area_fraction)==100 : # Have to check if land_area_fraction ranges from 0 to 100% or from 0 to 1
        land_area_fraction_threshold /= 100

    # Remove sea points
    labels = labels * (land_area_fraction.data>=land_area_fraction_threshold)

    # Remove small heatwaves with the cc3d dust function
    connectivity = 26 # only 4,8 (2D) and 26, 18, and 6 (3D) are allowed
    labels = labels * cc3d.dust((labels>0),dust_threshold)
    return labels

def compute_warming_level(read_directory_historical,read_directory_rcp,write_directory,pre_industrial_avg,start_year=1951,end_year=2100,interval=5,running_mean_window_size=20,warming_levels_list=[1.0,1.5,2.0,2.5,3.0,3.5,4,4.5,5]) :

    if running_mean_window_size % 2 != 0:
        raise ValueError(f"running_mean_window_size must be an even integer, found {running_mean_window_size}")

    # Create list of years of the beginning of each file
    year_list = range(start_year,end_year,interval)

    #Create list of files to load
    correct_files_list=[""]*len(year_list)
    for i in len(year_list) :
        if year<=split_year : # Before split_year, historical run
            correct_files_list[i] = join(read_directory_historical,find_correct_year_file(read_directory_historical,year_list[i],interval))
        else : # After split_year, RCP (4.5 or 8.5)
            correct_files_list[i] = join(read_directory_rcp,find_correct_year_file(read_directory_rcp,year_list[i],interval))

    # Load multi-file dataset
    ds = xr.open_mfdataset(correct_files_list, engine='netcdf4')
    # Variable is necessarily 'tas' for the computation of warming level
    da = ds.tas
    da = da.groupby(da.time.dt.year).mean() # Compute annual mean
    da = da.rolling(time=running_mean_window_size, center=True).mean() # Compute running mean
    # Compute latitude-weighted mean
    try : # Have to take into account the fact that grid_mapping is not always the same in CORDEX outputs
        weights = np.cos(np.deg2rad(da.rlat))
    except :
        weights = np.cos(np.deg2rad(da.lat))
    weights.name = "weights"
    da_weighted = da.weighted(weights)
    try : # Have to take into account the fact that grid_mapping is not always the same in CORDEX outputs
        da_weighted = da_weighted.mean(("rlat","rlon"))
    except :
        da_weighted = da_weighted.mean(("lat","lon"))

    warming_levels_dict = {}
    for level in warming_levels_list :
        # Find years warmer than level and get the first matching year
        bool_array = da_weighted - warming_level > 0.0
        if not bool_array.any() : # If the corresponding level has not been reached by the model
            warming_levels_dict[level] = None
        central_year = int(bool_array.idxmax())
        warming_levels_dict[level] = {"start_year" : int(central_year - running_mean_window_size/2), "end_year" : int(central_year + (running_mean_window_size/2 - 1))}
    
    with open(join(write_directory,'warming_levels.json'), 'w') as fp:
        json.dump(warming_levels_dict, fp)

    return