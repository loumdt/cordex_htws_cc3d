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
import glob
import dask
from dask.distributed import Client

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
            files_to_load.append(str(file))
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
    #client = Client(n_workers=20, threads_per_worker=2, memory_limit='60GB')
    ds = xr.open_mfdataset(files_to_load, engine='netcdf4',data_vars='all',chunks={'time': 1461})
    da = getattr(ds, temp_variable)
    try:
        da = da.chunk(chunks={'time':len(da.time),'rlat': 82,'rlon': 84})
    except:
        da = da.chunk(chunks={'time':len(da.time),'x': 82,'y': 84})
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
    #client = Client(n_workers=20, threads_per_worker=2, memory_limit='60GB')
    ds = xr.open_mfdataset(files_to_load, engine='netcdf4',data_vars='all',chunks={'time': 1461})#,parallel=True)
    da = getattr(ds, temp_variable)
    try:
        da = da.chunk(chunks={'time':len(da.time),'rlat': 82,'rlon': 84})
    except:
        da = da.chunk(chunks={'time':len(da.time),'x': 82,'y': 84})
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

def cc3d_scan_heatwaves(read_directory_historical,read_directory_rcp,write_directory,other_data_path,start_year=1975,end_year=2099,start_year_ref=1975,end_year_ref=2025,temp_variable='tasmax',threshold_value=95,relative_threshold=True,distrib_window_size=15,anomaly=True,nb_days=4,resolution_CORDEX=0.11,connectivity=26,dust_threshold=775):
    '''This function carries out a cc3d scan (https://pypi.org/project/connected-components-3d/) to detect heatwaves in the meteorological database (default ERA5, t2m, tg).
    The heatwaves point are labeled with a number corresponding to a heatwave identifier.
    Otherwise, values are set to -9999.'''
    # Set dust threshold to supress small amount of points. The threshold of 775 have been established empirically with ERA5 0.25° (see Mandonnet et al. 2026 https://hal.science/hal-05495839v1), and is translated according to CORDEX resolution, hence the following calculation
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
    else:
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
            labels_out, N_added = cc3d.connected_components(labels_in, connectivity=connectivity,return_N=True) # Return the table of labels and the number of added patterns
            # Initialize output array 
            label = da_year.copy()
            # Record labels and add N_labels offset where labels are nonzero
            label.data = labels_out + N_labels*(labels_out>0)

            #Remove sea heatwaves
            label.data = remove_sea_heatwaves(read_directory=join(other_data_path,"sftlf"),labels=label.data,grid_mapping=da.grid_mapping,connectivity=connectivity)

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
    print(N_labels,"heatwaves detected")
    return

def remove_sea_heatwaves(read_directory,labels,grid_mapping,resolution_CORDEX=0.11,land_area_fraction_threshold=0.5,connectivity=26,dust_threshold=775):
    '''
    '''
    # Set dust threshold to supress small amount of points. The threshold of 775 have been established with ERA5 0.25°, and is translated according to CORDEX resolution, hence the following calculation
    dust_threshold = int(dust_threshold * (0.25/resolution_CORDEX)**2) # resolution_CORDEX is given in °

    land_area_fraction = xr.open_dataset(join(read_directory,f"CORDEX_EUR-11_land_area_fraction_{grid_mapping}.nc"),engine='netcdf4').sftlf
    if np.max(land_area_fraction)==100: # Have to check if land_area_fraction ranges from 0 to 100 or from 0 to 1
        land_area_fraction_threshold *= 100

    # Remove sea points
    labels = labels * (land_area_fraction.data>=land_area_fraction_threshold)

    # Remove small heatwaves with the cc3d dust function
     # only 4,8 (2D) and 26, 18, and 6 (3D) are allowed
    labels = labels * cc3d.dust((labels>0),dust_threshold,connectivity=connectivity)
    return labels

def compute_regional_warming_levels(read_directory_historical,read_directory_rcp,write_directory,start_year=1975,end_year=2099,start_year_ref=1986,end_year_ref=2005,temp_variable='tasmax',ref_period_offset=0.72,running_mean_window_size=20,regional_warming_levels_list=[2.1,2.6,4.0,5.1]):
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
        tas_file = files_to_load[i].replace(temp_variable,'tas')
        files_to_load[i] = tas_file

    # Load multi-file dataset
    #client = Client(n_workers=20, threads_per_worker=2, memory_limit='60GB')
    ds = xr.open_mfdataset(files_to_load, engine='netcdf4',chunks={'time': 1461},data_vars='all')
    # Variable is necessarily 'tas' for the computation of warming levels
    da = ds.tas
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
    
    # Compute latitude-weighted mean to obtain 1D-array of annual mean
    weights = np.cos(np.deg2rad(da.lat))
    weights.name = "weights"
    da_weighted = da.weighted(weights)
    da_ref = da_ref.weighted(weights)
    
    try: # Have to take into account the fact that grid_mapping is not always the same in CORDEX outputs
        da_weighted = da_weighted.mean(("rlat","rlon")) # 1D-array of annual values
        da_ref = da_ref.mean(("rlat","rlon")) # one single value
    except:
        da_weighted = da_weighted.mean(("x","y")) # 1D-array of annual values
        da_ref = da_ref.mean(("x","y")) # one single value
        
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
    #client = Client(n_workers=20, threads_per_worker=2, memory_limit='60GB')
    ds = xr.open_mfdataset(files_to_load, engine='netcdf4',data_vars='all')#,parallel=True)
    da = getattr(ds,temp_variable)
    original_calendar = da.time.dt.calendar
    try:
        da = da.chunk(chunks={'time':len(da.time),'rlat': 82,'rlon': 84})
    except:
        da = da.chunk(chunks={'time':len(da.time),'x': 82,'y': 84})
    
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
    grid_mapping = xr.open_dataset(join(write_directory,f"labels_cc3d_year_{start_year}_ref_{start_year_ref}_{end_year_ref}.nc"), engine="netcdf4").label.grid_mapping    

    # Create list of temperature files to load
    if end_year <= split_year: # If all files are in historical directory, ignore rcp directory
        files_to_load = create_files_list_to_load(read_directory_historical=read_directory_historical,read_directory_rcp=None,start_year=start_year,end_year=end_year)
    elif start_year > split_year: # If all files are in rcp directory, ignore historical directory
        files_to_load = create_files_list_to_load(read_directory_historical=None,read_directory_rcp=read_directory_rcp,start_year=start_year,end_year=end_year)
    else:
        files_to_load = create_files_list_to_load(read_directory_historical,read_directory_rcp,start_year,end_year)

    original_calendar = getattr(xr.open_dataset(files_to_load[0], engine="netcdf4"),temp_variable).time.dt.calendar
    da_threshold = xr.open_dataarray(join(write_directory,f"distrib_threshold_{threshold_value}.nc"),engine='netcdf4')
    if original_calendar=='360_day':
        JJA_beg = 151 #150 is June 1st, 240 is August 30th
        JJA_end = 240
    else:
        JJA_beg = 152 #151 is June 1st, 243 is August 31st
        JJA_end = 243
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
    'Exposed_population_ssp4','HWMId_pop_ssp4','Exposed_population_ssp5','HWMId_pop_ssp5','GCM','RCM','simulation','version','ensemble']) # Create DataFrame

    # Initialize variables used to find the correct temperature file to load for each year
    loaded_temp_file = None
    old_loaded_temp_file = None

    # Compute weights for latitude-weighted mean
    weights = np.cos(np.deg2rad(da_HWMId.lat))
    weights.name = "weights"

    for year in tqdm(range(start_year,end_year+1)): # Iterate over the years
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
        
        # Load labels
        ds_labels = xr.open_dataset(join(write_directory,f"labels_cc3d_year_{year}_ref_{start_year_ref}_{end_year_ref}.nc"), engine="netcdf4") # Load the corresponding nc file
        da_labels = ds_labels.label

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
            da_HWMId_htw = da_HWMId_year.where(da==label, drop=True)

            df_htws.loc[label,'Year'] = year
            df_htws.loc[label,'Start Date'] = da_temp_htw.time.data[0]
            df_htws.loc[label,'End Date'] = da_temp_htw.time.data[-1]

            labels_bool_2D = np.max(da==label,axis=0) # Squeeze heatwave labels on a boolean 2D-map to see maximum spatial extension
        
            df_htws.loc[label,'Intensity'] = da_temp_htw.weighted(weights).mean().data
            df_htws.loc[label,'Spatial extent'] = (da_cell_area*labels_bool_2D).sum().data
            df_htws.loc[label,'Duration'] = len(da_temp_htw.time)
            df_htws.loc[label,'Max'] = da_temp_htw.max().data
            df_htws.loc[label,'HWMId_sum'] = da_HWMId_htw.weighted(weights).sum().data
            if year<=2030: #GHS-POP covers 1975-2030
                da_pop = ds_ghs_pop.Band1*1000 # Data is originally in thousands of people
                da_pop_htw = da_pop.sel(time=(da_pop.time.dt.year==year))
                da_pop_density_htw = da_pop_htw/da_cell_area # Population density in thousands of person/km²
                da_pop_htw = da_pop_htw.where(labels_bool_2D,drop=True)
                da_pop_density_htw = da_pop_density_htw.where(labels_bool_2D,drop=True)
                df_htws.loc[label,f'Exposed_population_ghs'] = da_pop_htw.sum().data/1e3 # Compute population in thousands to avoid later memory issues with bootstrap and MK test 
                df_htws.loc[label,f'HWMId_pop_ghs'] = (da_HWMId_htw*da_pop_density_htw.data).weighted(weights).sum().data
            if year>=2020: # FPOP covers 2020-2100
                for ssp in range(1,6):
                    da_pop = ssp_file_dict[f"ds_pop_ssp{ssp}"].Band1*1000 # Data is originally in thousands of people
                    da_pop_htw = da_pop.sel(time=(da_pop.time.dt.year==year))
                    da_pop_density_htw = da_pop_htw/da_cell_area # Population density in thousands of person/km²
                    da_pop_htw = da_pop_htw.where(labels_bool_2D,drop=True)
                    da_pop_density_htw = da_pop_density_htw.where(labels_bool_2D,drop=True)
                    df_htws.loc[label,f'Exposed_population_ssp{ssp}'] = da_pop_htw.sum().data/1e3 # Compute population in thousands to avoid later memory issues with bootstrap and MK test 
                    df_htws.loc[label,f'HWMId_pop_ssp{ssp}'] = (da_HWMId_htw*da_pop_density_htw.data).weighted(weights).sum().data
            if year <= split_year:
                df_htws['simulation'] = read_directory_historical.split("/")[-7]
            else:
                df_htws['simulation'] = read_directory_rcp.split("/")[-7]
    
    df_htws['GCM'] = read_directory_historical.split("/")[-8]
    df_htws['RCM'] = read_directory_historical.split("/")[-5]
    df_htws['version'] = read_directory_historical.split("/")[-4]
    df_htws['ensemble'] = read_directory_historical.split("/")[-6]
    df_htws['version_date'] = read_directory_historical.split("/")[-1]
    
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

def plot_RWL_figures(read_directory,regional_warming_levels_list=[2.1,2.6,4.0,5.1]):
    dir_list = [item for item in os.listdir(read_directory) if os.path.isdir(os.path.join(read_directory,item))] # List all subdirectories
    dataframe_path_list = [join(subdir,"df_htws.csv") for subdir in dir_list] # Get all df_htws.csv files, containing heatwaves indices and data
    array_dict = {reanalyses_array: []} # Dictionary to hold heatwaves indices for each RWL
    df_global_htws = pd.DataFrame(data=None,columns=['Year','Start Date','End Date','model','RWL_1','RWL_2','RWL_3','RWL_4','Intensity','Duration','Max','Spatial extent',
    'Exposed_population_ghs','HWMId_pop_ghs','Exposed_population_ssp1','HWMId_pop_ssp1','Exposed_population_ssp2','HWMId_pop_ssp2'
    'Exposed_population_ssp3','HWMId_pop_ssp3','Exposed_population_ssp4','HWMId_pop_ssp4','Exposed_population_ssp5','HWMId_pop_ssp5','GCM','RCM','simulation','version','ensemble'])
    
    for df_path in dataframe_path_list:
        df_htws = pd.read_csv(join(write_directory,"df_htws.csv"),header=0,parse_dates=["Start Date", "End Date"],date_format="%Y/%m/%d",usecols=lambda x: x!="Unnamed: 0")
        df_htws.insert(loc=3,column='model',value=df_path.split("/")[-1])
        df_global_htws = pd.concat(df_global_htws,df_htws)
    df_global_htws["Period RWL 1"] = False
    df_global_htws["Period RWL 2"] = False
    df_global_htws["Period RWL 3"] = False
    df_global_htws["Historical"] = False

    for idx in df_global_htws.index:
        for i in range(3):
            rwl = regional_warming_levels_list[i]
            df_global_htws.loc[idx,f"Period RWL {i+1}"] = (df_global_htws.loc[idx,"RWL_1"]==rwl) + (df_global_htws.loc[idx,"RWL_2"]==rwl) + (df_global_htws.loc[idx,"RWL_3"]==rwl) + (df_global_htws.loc[idx,"RWL_4"]==rwl)>0
        if df_global_htws.loc[idx,"model"]=='ERA5': #TODO Check this condition when data is ready
            df_global_htws.loc[idx,"Historical"] = True

    # Reference heatwaves
    reference_htws = pd.read_csv("Data/df_htws_ERA5.csv",header=0,index_col=0) # TODO Check this path
    reference_htws = reference_htws.loc[[159,226,380],:] # 2003, 2010, 2022
    reference_htws['label'] = reference_htws['Year']

    htw_index_list = ['Intensity','Duration','Max','Spatial extent',
    'Exposed_population_ghs','HWMId_pop_ghs','Exposed_population_ssp1','HWMId_pop_ssp1','Exposed_population_ssp2','HWMId_pop_ssp2'
    'Exposed_population_ssp3','HWMId_pop_ssp3','Exposed_population_ssp4','HWMId_pop_ssp4','Exposed_population_ssp5','HWMId_pop_ssp5']

    for htw_index in htw_index_list: # Cast objet type to numeric type
        df_global_htws[index] = df_global_htws[index].astype(float)
        # Plot figure
        plotter = PeriodDistributionPlotter()

        # Customize the visualization settings --> see documentation for all options
        plotter.update_config(
            kde_resolution=100,  # Number of points to evaluate the KDE to reduce the computation time
        )

        plotter.plot(
            data=df_global_htws,
            variable=htw_index,
            periods_columns_labels={
                "Historical": "Historical", # TODO Check Historical hetawaves
                "Period RWL 1": "+1.5°C",
                "Period RWL 2": "+2°C",
                "Period RWL 3": "+3°C",
            },
            # reference_events=None,
            reference_events=reference_htws,# TODO Check reference heatwaves
            cut_kdes=False,  # Whether to cut the KDEs at the minimum / maximum values of the events at each period
            #bounds=(
            #    1,
            #    reference_htws["Intensity"].max()
            #    * 1.1,
            #),
        )

        # Saving the figure
        plotter.save(join(read_directory,f"single_plot.pdf"))

        # Accessing the figure object to modify it further if needed
        fig = plotter.fig
        ax = plotter.ax
    return