#%% 
import numpy as np
import sys
from tqdm import tqdm # Create a user-friendly feedback for loops while script is running
from os.path import join
from pathlib import Path
from utils import *
import time
start_time = time.time()
#%% Read variables and directories
# Collect details and find read folder of temperature variable
combination = ('CNRM-CERFACS-CNRM-CM5', 'GERICS-REMO2015', 'rcp85', 'v1')
read_directory_historical = "/bdd/CORDEX/output/EUR-11/GERICS/CNRM-CERFACS-CNRM-CM5/historical/r1i1p1/GERICS-REMO2015/v1/day/tas/v20170208/"
read_directory_rcp = "/bdd/CORDEX/output/EUR-11/GERICS/CNRM-CERFACS-CNRM-CM5/rcp85/r1i1p1/GERICS-REMO2015/v1/day/tas/v20170208/"

path_to_remove = "/bdd/CORDEX/output/EUR-11/"
write_directory = join("/scratchu/tmandonnet/CORDEX",read_directory_rcp.replace(path_to_remove,"").replace("/","_"))
Path(write_directory).mkdir(parents=True, exist_ok=True) 

start_year = 1971
end_year = 2100
start_year_ref = 1971
end_year_ref = 2005
interval = 5
temp_variable = 'tas'
smooth_span = 15
threshold_value = 95
distrib_window_size = 15
anomaly = True
relative_threshold = True
nb_days = 4


def compute_regional_warming_levels(read_directory_historical,read_directory_rcp,write_directory,start_year=1971,end_year=2100,start_year_ref=1976,end_year_ref=2005,ref_period_offset=0.61,interval=5,running_mean_window_size=20,regional_warming_levels_list=[2.1,2.6,4.0,5.1]) :

    if running_mean_window_size % 2 != 0:
        raise ValueError(f"running_mean_window_size must be an even integer, found {running_mean_window_size}")

    # Create list of years of the beginning of each file
    year_list = range(start_year,end_year,interval)

    #Create list of files to load
    correct_files_list=[""]*len(year_list)
    for i in range(len(year_list)) :
        year = year_list[i]
        if year<=split_year : # Before split_year, historical run
            correct_files_list[i] = join(read_directory_historical,find_correct_year_file(read_directory_historical,year_list[i],interval))
        else : # After split_year, RCP (4.5 or 8.5)
            correct_files_list[i] = join(read_directory_rcp,find_correct_year_file(read_directory_rcp,year_list[i],interval))

    # Load .nc files in a dictionary
#    ds_dict = {}
#    for year in year_list :
#        pattern = re.compile(f"{year}0101-{year+interval-1}123[0-1].nc$") # Choose file based on start year and interval
#        if year<=split_year : # Before split_year, historical run
#            ds_dict[year] = xr.open_dataset(join(read_directory_historical,find_correct_year_file(read_directory_historical,year,interval)), engine="netcdf4") 
#        else : # After split_year, RCP (4.5 or 8.5)
#            ds_dict[year] = xr.open_dataset(join(read_directory_rcp,find_correct_year_file(read_directory_rcp,year,interval)), engine="netcdf4")

    # Initialize data array with the first file
#    da = ds_dict[start_year].tas # Iterate over files, except first one which has already been used in initialization; variable is necessarily 'tas' for the computation of warming level
    # Iterate over files, except first one which has already been used in initialization
    #for year in year_list[1:] : 
    #    da = xr.concat(objs=[da,ds_dict[year].tas], dim="time")
    # Drop 29 Feb
    #da = da.convert_calendar("noleap")

    # Load multi-file dataset
    ds = xr.open_mfdataset(correct_files_list, engine='netcdf4', chunks={'time': 1461})
    da = ds.tas
    da = da.groupby(da.time.dt.year).mean()

    # Create reference period average
    mask = (da.year >= start_year_ref)&(da.year <= end_year_ref)
    da_ref = da.sel(year=mask)
    da_ref = da_ref.mean(dim="year")

    print("rolling mean")
    da = da.rolling(time=running_mean_window_size, center=True).mean() # Compute running mean
    # Compute latitude-weighted mean
    try : # Have to take into account the fact that grid_mapping is not always the same in CORDEX outputs
        weights = np.cos(np.deg2rad(da.rlat))
    except :
        weights = np.cos(np.deg2rad(da.lat))
    weights.name = "weights"
    da_weighted = da.weighted(weights)
    da_ref = da_ref.weighted(weights)
    try : # Have to take into account the fact that grid_mapping is not always the same in CORDEX outputs
        da_weighted = da_weighted.mean(("rlat","rlon"))
        da_ref = da_ref.mean(("rlat","rlon"))
    except :
        da_weighted = da_weighted.mean(("lat","lon"))
        da_ref = da_ref.mean(("lat","lon"))

    warming_levels_dict = {}
    for level in regional_warming_levels_list :
        level = level - ref_period_offset # Since reference period is not 1850-1900, have to introduce offset (default is 0.61°C for 1976-2005 reference period)
        # Find years warmer than level and get the first matching year
        bool_array = (da_weighted - level > 0)
        if not bool_array.any() : # If the corresponding level has not been reached by the model
            warming_levels_dict[level] = None
        central_year = int(bool_array.idxmax())
        warming_levels_dict[level] = {"start_year" : int(central_year - running_mean_window_size/2), "end_year" : int(central_year + (running_mean_window_size/2 - 1))}
    
    with open(join(write_directory,'regional_warming_levels.json'), 'w') as fp:
        json.dump(warming_levels_dict, fp)

    return warming_levels_dict

dict_wl = compute_regional_warming_levels(read_directory_historical=read_directory_historical,read_directory_rcp=read_directory_rcp,write_directory=write_directory,start_year=start_year,end_year=end_year)

print(dict_wl)