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

def create_files_list_to_load(read_directory_historical=None,read_directory_rcp=None,start_year=1971,end_year=split_year):
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
    if start_year < first_file_start_year :
        raise ValueError(f"Start year {start_year} is before beginning of first file {first_file_start_year}.")
    last_file_end_year = int(files_to_load[-1][-11:-7])
    if end_year > last_file_end_year :
        raise ValueError(f"End year {end_year} is after ending of last file {last_file_end_year}.")

    for i in range(1,len(files_to_load)):
        previous_file_end_year = int(files_to_load[i-1][-11:-7])
        file_start_year = int(files_to_load[i][-20:-16])
        if file_start_year != previous_file_end_year+1 :
            raise ValueError(f"Time is not consistent in files_to_load. Missing year(s) between {previous_file_end_year} and {file_start_year}.")
    print(f"Time is consistent between start year {start_year} and end year {end_year}.")
    return(files_to_load)

def compute_climatology_smooth(read_directory_historical,read_directory_rcp,write_directory,start_year_ref=1971,end_year_ref=2025,temp_variable='tasmax',smooth_span=15) :
    '''This function computes a climatology for each calendar day of the year. The seasonal cycle is then smoothed with a 31-day window. 
    By default, the climatology is computed over 1971-2025.
    This function can be used with several models and variables.'''

    #Create list of files to load
    if end_year_ref <= split_year:
        files_to_load = create_files_list_to_load(read_directory_historical=read_directory_historical,read_directory_rcp=None,start_year=start_year_ref,end_year=end_year_ref)
    else:
        files_to_load = create_files_list_to_load(read_directory_historical,read_directory_rcp,start_year_ref,end_year_ref)

    # Load multi-file dataset
    client = Client(n_workers=20, threads_per_worker=2, memory_limit='60GB')
    ds = xr.open_mfdataset(files_to_load, engine='netcdf4', chunks={'time': 1461},data_vars='all',parallel=True)
    da = getattr(ds, temp_variable)
    # Since the files generally cover several years, have to select sub-period (it may not exactly match the boundaries of loaded files)
    mask = (da.time.dt.year>=start_year_ref) & (da.time.dt.year<=end_year_ref)
    da = da.sel(time=mask)
    del mask

    # Drop Feb 29
    da = da.convert_calendar("noleap")
    # Group using dayofyear and sum to compute mean at the end
    climatology = da.groupby(da.time.dt.dayofyear).mean(dim="time")

    # Smoothing
    # Handle easily first and last day of year by taking thrice the entire dataset and working on the middle one (avoiding border effects)
    extended_temp=np.zeros((365*3,np.shape(climatology.data)[1],np.shape(climatology.data)[2]))
    extended_temp[0:365,:,:]=climatology.data[:,:,:]
    extended_temp[365:730,:,:]=climatology.data[:,:,:]
    extended_temp[730:,:,:]=climatology.data[:,:,:]

    print("Smoothing...")
    #For each day d, compute the mean of the values between day d-smooth_span and d+smooth_span to have a physically realistic seasonal cycle
    for i in tqdm(range(365,730)):
        climatology.data[i-365,:,:] = np.nanmean(extended_temp[i-smooth_span:i+smooth_span+1,:,:],axis=0)

    # Export data to netcdf file
    climatology.to_netcdf(join(write_directory,f"{temp_variable}_climatology_{start_year_ref}_{end_year_ref}.nc"))

    # Clear resources
    da.close()
    climatology.close()
    ds.close()
    return

def compute_distrib_percentile(read_directory_historical,read_directory_rcp,write_directory,start_year_ref=1971,end_year_ref=2025,temp_variable='tasmax',threshold_value=95,distrib_window_size=15,anomaly=False) :
    '''This function computes, for every calendar day, the n-th (n is the threshold_value, default 95) percentile of the corresponding distribution of daily variable. 
    By default, the distribution is computed over 1971-2025.'''

    if distrib_window_size%2==0:
        raise ValueError('distrib_window_size is even. It has to be odd so the window can be centered on the computed day.')

    #Create list of files to load
    if end_year_ref <= split_year: # If all files are in historical directory, ignore rcp directory
        files_to_load = create_files_list_to_load(read_directory_historical=read_directory_historical,read_directory_rcp=None,start_year=start_year_ref,end_year=end_year_ref)
    else:
        files_to_load = create_files_list_to_load(read_directory_historical,read_directory_rcp,start_year_ref,end_year_ref)

    # Load multi-file dataset
    client = Client(n_workers=20, threads_per_worker=2, memory_limit='60GB')
    ds = xr.open_mfdataset(files_to_load, engine='netcdf4', chunks={'time': 1461},data_vars='all',parallel=True)
    da = getattr(ds, temp_variable)
    # Since files generally cover several years, have to select sub-period (it may not exactly match the boundaries of loaded files)
    mask = (da.time.dt.year>=start_year_ref) & (da.time.dt.year<=end_year_ref)
    da = da.sel(time=mask)
    del mask
    
    # Load climatology file to create output data structure and compute anomaly
    climatology = xr.open_dataarray(join(write_directory,f"{temp_variable}_climatology_{start_year_ref}_{end_year_ref}.nc"), engine='netcdf4')

    # Drop 29 Feb
    da = da.convert_calendar("noleap")

    # Create threshold table by copying climatology table, values will be updated later
    threshold = climatology.copy()

    if anomaly :
        for year in tqdm(range(len(da.time)//365)) : # Iterate over the number of years
            da[year*365:(year+1)*365,:,:] = da[year*365:(year+1)*365,:,:] - climatology.data # Compute anomaly
    else :
        climatology.close()
    
    for day in tqdm(range(1,366)) : # Calendar days ranging from 1 to 365 (no leap years)
        if day - (distrib_window_size//2) <= 0  : # day is at the beginning of January, window overlapping with December
            distrib_start_bound = 365 + day - (distrib_window_size//2) # start bound of the distribution window
            distrib_end_bound = day + (distrib_window_size//2) # end bound of the distribution window
            mask = (da.time.dt.dayofyear >= distrib_start_bound) + (da.time.dt.dayofyear <= distrib_end_bound) # Create a boolean mask for days between distrib_start_bound and distrib_end_bound
        elif day + (distrib_window_size//2) > 365 : # day is at the end of December , window overlapping with January
            distrib_start_bound = day - (distrib_window_size//2)
            distrib_end_bound = day + (distrib_window_size//2) - 365
            mask = (da.time.dt.dayofyear >= distrib_start_bound) + (da.time.dt.dayofyear <= distrib_end_bound) # Create a boolean mask for days between distrib_start_bound and distrib_end_bound
        else :
            distrib_start_bound = day - (distrib_window_size//2)
            distrib_end_bound = day + (distrib_window_size//2)
            mask = (da.time.dt.dayofyear >= distrib_start_bound) & (da.time.dt.dayofyear <= distrib_end_bound) # Create a boolean mask for days between distrib_start_bound and distrib_end_bound
        # Apply the mask to select the corresponding days and compute percentile, take day-1 because day ranges from 1 to 365 but python indexing ranges from 0 to 364
        threshold.data[day-1,:,:] = np.percentile(da.sel(time=mask).data,threshold_value,axis=0)

    # Export data to netcdf file
    threshold.to_netcdf(join(write_directory,f"{temp_variable}_distrib_{'ano_'*anomaly}{start_year_ref}_{end_year_ref}_threshold_{threshold_value}_window_{distrib_window_size}d.nc"))

    # Clear resources
    da.close()
    threshold.close()
    if anomaly :
        climatology.close()
    ds.close()
    return

def cc3d_scan_heatwaves(read_directory_historical,read_directory_rcp,write_directory,start_year=1971,end_year=2099,start_year_ref=1971,end_year_ref=2025,temp_variable='tasmax',threshold_value=95,relative_threshold=True,distrib_window_size=15,anomaly=False,nb_days=4,resolution_CORDEX=0.11) :
    '''This function carries out a cc3d scan (https://pypi.org/project/connected-components-3d/) to detect heatwaves in the meteorological database (default ERA5, t2m, tg).
    The heatwaves point are labeled with a number corresponding to a heatwave identifier.
    Otherwise, values are set to -9999.'''
    # Set dust threshold to supress small amount of points. The threshold of 775 have been established empirically with ERA5 0.25° (see Mandonnet et al. 2026 https://hal.science/hal-05495839v1), and is translated according to CORDEX resolution, hence the following calculation
    dust_threshold = int(775 * (0.25/resolution_CORDEX)**2) # resolution_CORDEX is given in °
    
    # Create list of files to load
    if end_year <= split_year: # If all files are in historical directory, ignore rcp directory
        files_to_load = create_files_list_to_load(read_directory_historical=read_directory_historical,read_directory_rcp=None,start_year=start_year,end_year=end_year)
    elif start_year > split_year: # If all files are in rcp directory, ignore historical directory
        files_to_load = create_files_list_to_load(read_directory_historical=None,read_directory_rcp=read_directory_rcp,start_year=start_year,end_year=end_year)
    else:
        files_to_load = create_files_list_to_load(read_directory_historical,read_directory_rcp,start_year,end_year)

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
    for file in tqdm(len(files_to_load)):
        file_start_year = int(file[-20:-16])
        ds = xr.open_dataset(file,engine='netcdf4')
        da = getattr(ds, temp_variable)
        # Drop 29 Feb and correct day of year
        da = da.convert_calendar("noleap")
        # Keep only JJA values
        mask = (da.time.dt.season=='JJA')
        da = da.sel(time=mask)

        for year in range(len(da.time)//365) :# Iterate over the years, number of years depends on the file
            da_year = da[year*92:(year+1)*92,:,:]# Select data for the given year
            if anomaly : # Substract climatology to compute anomaly
                da_year = da_year - climatology.data
            da_year = da_year*(da_year>threshold.data)-9999*(da_year<=threshold.data) # Set to -9999 the values that do not exceed threshold
            stack_temp = -9999*np.ones((92,np.shape(da_year)[1],np.shape(da_year)[2])) # Create a 3D array that will hold the temperature values when and where there are heatwaves
            stack_where = np.zeros((np.shape(da_year)[1],np.shape(da_year)[2])) # Create a 2D array that holds the number of consecutive hot days for each location, computed for each day
            for day in range(92):
                stack_where[:,:] = stack_where[:,:] + np.ones((np.shape(da_year)[1],np.shape(da_year)[2]))*(da_year[day,:,:]!=-9999) # Add one day to each potential heatwave location
                stack_where[:,:] = stack_where[:,:]*(da_year[day,:,:]!=-9999) # When not adding a day, have to set back the duration to zero
                if day>=nb_days-1 : #record the last four days for the corresponding scanning window, add new consecutive hot days and set not hot days to -9999
                    stack_temp[day-(nb_days-1):day+1,:,:] = stack_temp[day-(nb_days-1):day+1,:,:]*(stack_temp[day-(nb_days-1):day+1,:,:]!=-9999)+da_year[day-(nb_days-1):day+1,:,:]*((stack_where>=nb_days)*stack_temp[day-(nb_days-1):day+1,:,:]==-9999)+(-9999*((stack_where<nb_days)*(stack_temp[day-(nb_days-1):day+1,:,:]==-9999))) 

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
            label.to_netcdf(join(write_directory,f"{temp_variable}_labels_{'ano_'*anomaly}year_{file_start_year+year}_ref_{start_year_ref}_{end_year_ref}_threshold_{threshold_value}_{nb_days}d_window_{distrib_window_size}d.nc"))

            # Update N_labels
            N_labels += N_added

        # Clear resources
        da.close()
        ds.close()
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

def compute_regional_warming_levels(read_directory_historical,read_directory_rcp,write_directory,start_year=1971,end_year=2099,start_year_ref=1986,end_year_ref=2005,temp_variable='tasmax',ref_period_offset=0.72,running_mean_window_size=20,regional_warming_levels_list=[2.1,2.6,4.0,5.1]) :
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
    client = Client(n_workers=20, threads_per_worker=2, memory_limit='60GB')
    ds = xr.open_mfdataset(files_to_load, engine='netcdf4', chunks={'time': 1461},data_vars='all',parallel=True)
    # Variable is necessarily 'tas' for the computation of warming levels
    da = ds.tas
    # Since files generally cover several years, have to select sub-period (it may not exactly match the boundaries of loaded files)
    mask = (da.time.dt.year>=start_year_ref) & (da.time.dt.year<=end_year_ref)
    da = da.sel(time=mask)
    del mask
    da = da.groupby(da.time.dt.year).mean() # Compute annual mean

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
    
    try : # Have to take into account the fact that grid_mapping is not always the same in CORDEX outputs
        da_weighted = da_weighted.mean(("rlat","rlon")) # 1D-array of annual values
        da_ref = da_ref.mean(("rlat","rlon")) # one single value
    except :
        da_weighted = da_weighted.mean(("x","y")) # 1D-array of annual values
        da_ref = da_ref.mean(("x","y")) # one single value
        
    da_weighted = da_weighted - da_ref # 1D-array of annual values reprenting the european warming since the reference period (default 1986-2005)
    
    # Create dictionary to hold the data of
    warming_levels_dict = {}
    for level in regional_warming_levels_list :
        level_diff = level - ref_period_offset # Since reference period is not 1850-1900, have to introduce offset (default is 0.61°C for 1986-2005 reference period)
        # Find years warmer than level and get the first matching year
        bool_array = (da_weighted - level_diff > 0)
        if not bool_array.any() : # If the corresponding level has not been reached by the model
            warming_levels_dict[level] = None
        else :
            central_year = int(bool_array.idxmax())
            warming_levels_dict[level] = {"start_year" : int(central_year - running_mean_window_size/2), "end_year" : int(central_year + (running_mean_window_size/2 - 1))}
    
    with open(join(write_directory,'regional_warming_levels.json'), 'w') as fp:
        json.dump(warming_levels_dict, fp)

    return

#%%
def compute_Russo_HWMId(read_directory_historical,read_directory_rcp,write_directory,start_year=1971,end_year=2099,temp_variable='tasmax',distrib_window_size=15,anomaly=False) :
    """Compute the pseudo_HWMId index map.
    Based on HWMId defined by Russo et al (2015, https://dx.doi.org/10.1088/1748-9326/10/12/124003 )."""

    # Create list of files to load
    if end_year <= split_year: # If all files are in historical directory, ignore rcp directory
        files_to_load = create_files_list_to_load(read_directory_historical=read_directory_historical,read_directory_rcp=None,start_year=start_year,end_year=end_year)
    elif start_year > split_year: # If all files are in rcp directory, ignore historical directory
        files_to_load = create_files_list_to_load(read_directory_historical=None,read_directory_rcp=read_directory_rcp,start_year=start_year,end_year=end_year)
    else:
        files_to_load = create_files_list_to_load(read_directory_historical,read_directory_rcp,start_year,end_year)

    # Load multi-file dataset
    client = Client(n_workers=20, threads_per_worker=2, memory_limit='60GB')
    ds = xr.open_mfdataset(files_to_load, engine='netcdf4', chunks={'time': 1461},data_vars='all',parallel=True)
    # Drop Feb 29 and only keep JJA days and years of interest
    ds = ds.convert_calendar("noleap")
    mask = (ds.time.dt.year>=start_year_ref) & (ds.time.dt.year<=end_year_ref)
    ds = ds.sel(time=mask)
    ds = ds.sel(time=(da.time.dt.season=='JJA'))
    ds = ds.chunk(chunks={'time': 368})
    da = getattr(ds,temp_variable)
    
    if anomaly :
        climatology = xr.open_dataarray(join(write_directory,f"{temp_variable}_climatology_{start_year_ref}_{end_year_ref}.nc"), engine='netcdf4')
        # Keep only JJA values
        mask = (climatology.dayofyear>=152) & (climatology.dayofyear<=243) # dayofyear ranges from 1 to 365 ; 152 is June 1st, 243 is August 31st
        climatology = climatology.sel(dayofyear=mask)
        da = da - climatology
    del mask
    
    # Compute values needed for HWMId
    temp_25p = np.percentile(da.groupby(da.time.dt.year).max(), 25, axis=0)
    temp_75p = np.percentile(da.groupby(da.time.dt.year).max(), 75, axis=0)

    # Copy da to create output data structure
    Russo_HWMId = da.copy()

    # Compute HWMId
    Russo_HWMId.data = np.maximum((da - temp_25p)/(temp_75p - temp_25p), 0)

    Russo_HWMId = Russo_HWMId.to_dataset(name="HWMId")
    Russo_HWMId.to_netcdf(join(write_directory,f"Russo_HWMId.nc")) # Save to netCDF 

    Russo_HWMId.close()
    da.close()
    ds.close()
    if anomaly :
        climatology.close()
    return

def create_heatwaves_indices_database(read_directory_historical,read_directory_rcp,write_directory,start_year=1971,end_year=2099,start_year_ref=1971,end_year_ref=2025,temp_variable='tasmax',threshold_value=95,relative_threshold=True,distrib_window_size=15,anomaly=False,nb_days=4,resolution_CORDEX=0.11):
    '''This function is used to create the dataset of the indices of the detected heatwaves. The set of detected heatwaves depends on all the parameters.'''

    # Account for the fact that RWL can overlap. Create 4 RWL columns to avoid edge cases but RWL_3 and RWL_4 should remain empty
    RWL_col_dict = {1:"RWL_1",2:"RWL_2",3:"RWL_3",4:"RWL_4"}

    # Create list of computed years (only the years that match a RWL period)
    RWL_years = []
    for k in RWL_dict.keys() :
        if RWL_dict[k] is not None :
            RWL_years += list(range(RWL_dict[k]["start_year"],RWL_dict[k]["end_year"]))
    RWL_years = np.unique(RWL_years)
    df_heatwaves = pd.DataFrame(data=None,columns=["Year","RWL_1","RWL_2","RWL_3","RWL_4","Mean","Max","Duration","Spatial Extent","Temp sum", "Pseudo_HWMId"]) # Create DataFrame
    
    # TO DO: Must population temperature file !!!!!!!!!!
    
    for year in RWL_years : # Iterate over the years that match a RWL period
        ds_labels = xr.open_dataset(join(write_directory,f"{temp_variable}_labels_{'ano_'*anomaly}year_{year}_ref_{start_year_ref}_{end_year_ref}_threshold_{threshold_value}_{nb_days}d_window_{distrib_window_size}d.nc"), engine="netcdf4") # Load the corresponding nc file
        da_labels = ds_labels.label

        # TO DO: Must load temperature file !!!!!!!!!!
        
        

        for label in np.unique(da_labels)[1:] : # Iterate over the heatwaves (one label = one heatwave). Skip the first item which is always 0, not a heatwave
            df_heatwaves.loc[label,"Year"]=Year
            # Find the corresponding RWL(s)
            rwl_count=0
            for rwl in RWL_dict :
                if year in range(RWL_dict[rwl]["start_year"],RWL_dict[rwl]["end_year"]) :
                    rwl_count += 1 # Find the correct column to fill
                    df_heatwaves.loc[label,RWL_col_dict[rwl_count]] = float(rwl) # Record the corresponding RWL
            da = da_labels
            
            # TO DO: df_heatwaves.loc[label,"Mean"] = ...
            # ...
                
    f_land_sea_mask = nc.Dataset(os.path.join(datadir,database,"Mask",f"Mask_Europe_land_only_{database}_{resolution}deg.nc"),mode='r')
    land_sea_mask = f_land_sea_mask.variables['mask'][:]

    f_temp = nc.Dataset(os.path.join(datadir,database,datavar,f"{database}_{datavar}_{daily_var}_{name_dict_anomaly[anomaly]}_JJA_{year_beg}_{year_end}_climatology_{year_beg_climatology}_{year_end_climatology}_{distrib_window_size}days.nc"),mode='r')
    f_Russo_HWMId = nc.Dataset(os.path.join(datadir,database,datavar,f"Russo_HWMId_{database}_{datavar}_{daily_var}_{name_dict_anomaly[anomaly]}_{year_beg_climatology}_{year_end_climatology}_{distrib_window_size}days.nc.nc"),mode='r')#path to the output netCDF file
    f_gdp_cap = nc.Dataset(os.path.join(datadir,database,"Socio_eco_maps",f"GDP_cap_{database}_Europe_{resolution}deg.nc"),mode='r')#path to the output netCDF file
    
    f_pop_GHS_1975 = nc.Dataset(os.path.join(datadir,"Pop","GHS_POP",f"GHS_POP_1975_{database}_grid_Europe.nc"))
    f_pop_GHS_1980 = nc.Dataset(os.path.join(datadir,"Pop","GHS_POP",f"GHS_POP_1980_{database}_grid_Europe.nc"))
    f_pop_GHS_1985 = nc.Dataset(os.path.join(datadir,"Pop","GHS_POP",f"GHS_POP_1985_{database}_grid_Europe.nc"))
    f_pop_GHS_1990 = nc.Dataset(os.path.join(datadir,"Pop","GHS_POP",f"GHS_POP_1990_{database}_grid_Europe.nc"))
    f_pop_GHS_1995 = nc.Dataset(os.path.join(datadir,"Pop","GHS_POP",f"GHS_POP_1995_{database}_grid_Europe.nc"))
    f_pop_GHS_2000 = nc.Dataset(os.path.join(datadir,"Pop","GHS_POP",f"GHS_POP_2000_{database}_grid_Europe.nc"))
    f_pop_GHS_2005 = nc.Dataset(os.path.join(datadir,"Pop","GHS_POP",f"GHS_POP_2005_{database}_grid_Europe.nc"))
    f_pop_GHS_2010 = nc.Dataset(os.path.join(datadir,"Pop","GHS_POP",f"GHS_POP_2010_{database}_grid_Europe.nc"))
    f_pop_GHS_2015 = nc.Dataset(os.path.join(datadir,"Pop","GHS_POP",f"GHS_POP_2015_{database}_grid_Europe.nc"))
    f_pop_GHS_2020 = nc.Dataset(os.path.join(datadir,"Pop","GHS_POP",f"GHS_POP_2020_{database}_grid_Europe.nc"))
    
    #LOAD POPULATION FILES
    #Redirect the different years towards the correct (nearest in time) population data file :

    output_dir = os.path.join("Output",database,f"{datavar}_{daily_var}",
                            f"{database}_{datavar}_{daily_var}_{name_dict_anomaly[anomaly]}_JJA_{nb_days}days_before_scan_{year_beg}_{year_end}_{threshold_value}{name_dict_threshold[relative_threshold]}_{distrib_window_size}days_window_climatology_{year_beg_climatology}_{year_end_climatology}")
    df_htw = pd.read_excel(os.path.join(output_dir,f"df_htws_V0_detected_{database}_{datavar}_{daily_var}_{name_dict_anomaly[anomaly]}_JJA_{nb_days}days_before_scan_{year_beg}_{year_end}_{threshold_value}{name_dict_threshold[relative_threshold]}_{distrib_window_size}days_window_climatology_{year_beg_climatology}_{year_end_climatology}.xlsx"),header=0,index_col=0)
    #--------------------------

    htw_criteria = ['Global_mean','Spatial_extent','Duration','Max','Max_spatial','Temp_sum','Pseudo_HWMId','Total_affected_pop','Global_mean_pop','Duration_pop','Max_pop','Max_spatial_pop',
    'Spatial_extent_pop','Temp_sum_pop','Pseudo_HWMId_pop','Multi_index_temp','Multi_index_HWMId','Temp_sum_pop_NL','Pseudo_HWMId_pop_NL','Multi_index_temp_NL','Multi_index_HWMId_NL']

    for htw_charac in htw_criteria:
        df_htw[htw_charac] = None

    res_lat = np.abs(np.mean(lat_in[1:]-lat_in[:-1])) #latitude resolution in degrees
    res_lon = np.abs(np.mean(lon_in[1:]-lon_in[:-1])) #longitude resolution in degrees

    cell_area = np.array([6371**2*np.cos(np.pi*lat_in/180)*res_lat*np.pi/180*res_lon*np.pi/180]*len(lon_in)).T # the area in km² of each cell, depending on the latitude
    cell_area_3d = np.array([cell_area]*92)
    cell_area_3d_ratio = cell_area_3d/(6371**2*res_lat*np.pi/180*res_lon*np.pi/180) # each cell area as a percentage of the maximum possible cell area (obtained with lat=0°) in order to correctly weigh each cell when carrying out average

    gdp_time = f_gdp_cap.variables['time'][:]

    for htw_id in tqdm(df_htw.index.values[:]) : #list of heatwaves detected in the meteo database
        if htw_id not in not_computed_htw :
            df_htw.loc[htw_id,'Computed_heatwave']=True
            new_computed_htw = [htw_id]
            #Compute meteo indices
            year = df_htw.loc[htw_id,'Year']
            pop_unique = pop0*(np.nanmean(pop,axis=0)>0) #population density set to zero for points that are not affected by the considered heatwave(s) and "flattened" into a 2D array
            area_unique = cell_area*(pop_unique>0) #cell area set to zero for points that are not affected by the considered heatwave(s)
            duration = len(np.unique(np.where((data_label == vals[:, None, None, None])[0].data)[0]))
            #mean temperature anomaly over every point recorded as a part of the heatwave
            masked_temp = ma.masked_where(table_temp==0,table_temp)
            df_htw.loc[htw_id,'Global_mean'] = np.nanmean(table_temp*cell_area_3d_ratio)
            df_htw.loc[htw_id,'Global_mean'] = np.nanmean(table_temp*cell_area_3d_ratio)
            #area of the considered heatwave in km²
            df_htw.loc[htw_id,'Spatial_extent'] = np.nansum(area_unique)
            #duration in days
            df_htw.loc[htw_id,'Duration'] = duration
            #Sum of the normalized cell area multiplied by the temperature anomaly of every point recorded as a part of the heatwave
            df_htw.loc[htw_id,'Temp_sum'] = np.nansum(table_temp*cell_area_3d_ratio)
            #Sum of HWMId index over the heatwave (time and space), multiplied by the normalized cell area
            df_htw.loc[htw_id,'Pseudo_HWMId'] = np.nansum(cell_area_3d_ratio*table_HWMId)
            #maximum of the temperature anomaly of the heatwave, multiplied by the cumulative normalized area
            df_htw.loc[htw_id,'Max_spatial'] = np.max(table_temp)*np.nansum(area_unique) 
            

    #Save dataframe 
    df_htw.to_excel(os.path.join(output_dir,f"df_htws_detected{'_count_all_impacts'*count_all_impacts}_flex_time_{flex_time_span}days.csv"))
    #close netCDF files
    f_label.close()
    f_Russo_HWMId.close()
    f_temp.close()
    f_pop_GHS_1975.close()
    f_pop_GHS_1980.close()
    f_pop_GHS_1985.close()
    f_pop_GHS_1990.close()
    f_pop_GHS_1995.close()
    f_pop_GHS_2000.close()
    f_pop_GHS_2005.close()
    f_pop_GHS_2010.close()
    f_pop_GHS_2015.close()
    f_pop_GHS_2020.close()
    return