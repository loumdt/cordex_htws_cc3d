#%% 
import numpy as np
import sys
from tqdm import tqdm # Create a user-friendly feedback for loops while script is running
import os
from os.path import join
from utils_copy import *
import time
start_time = time.time()
#%% Read variables and directories
# Collect details and find read folder of temperature variable
combination = tuple((sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]))
read_directory_historical = sys.argv[5]
read_directory_rcp = sys.argv[6]

start_year = 1971
end_year = 2100
start_year_ref = 1971
end_year_ref = 2020
interval = 5
temp_variable = 'tas'
smooth_span = 15
threshold_value = 95
distrib_window_size = 15
anomaly = True
relative_threshold = True
nb_days = 4
#%%
#read_directory_historical = "/bdd/CORDEX/output/EUR-11/GERICS/CNRM-CERFACS-CNRM-CM5/historical/r1i1p1/GERICS-REMO2015/v2/day/tas/latest"
#read_directory_rcp = "/bdd/CORDEX/output/EUR-11/GERICS/CNRM-CERFACS-CNRM-CM5/rcp85/r1i1p1/GERICS-REMO2015/v2/day/tas/latest"
path_to_remove = "/bdd/CORDEX/output/EUR-11/"
write_directory = join("/scratchu/tmandonnet/CORDEX",read_directory_rcp.replace(path_to_remove,"").replace("/","_"))
os.makedirs(write_directory,exist_ok=True)
#%%
print('GCM, RCM, RCP, version :',combination)
print('read_directory_historical :',read_directory_historical)
print('read_directory_rcp :',read_directory_rcp)
print('temp_variable :',temp_variable)
print('start_year_ref :',start_year_ref)
print('end_year_ref :',end_year_ref)

#%% Check time consistency
print("Running check_time_consistency for historical directory...")
if check_time_consistency(read_directory_historical,interval,start_year=1971,end_year=2005) :
    print(f"{read_directory_historical} is time-consistent.")
else :
    raise ValueError(f"{read_directory_historical} is not time-consistent for start_year {start_year_ref}, end_year {split_year} and interval {interval}.")

print("Running check_time_consistency for RCP directory...")
if check_time_consistency(read_directory_rcp,interval,start_year=2006,end_year=2100) :
    print(f"{read_directory_rcp} is time-consistent.")
else :
    raise ValueError(f"{read_directory_rcp} is not time-consistent for start_year {split_year+1}, end_year {end_year} and interval {interval}.")


#%% Compute climatology smooth
print("--- %.2f seconds ---" % (time.time() - start_time))
print("Running compute_climatology_smooth...")
compute_climatology_smooth(read_directory_historical,read_directory_rcp,write_directory,start_year_ref=start_year_ref,end_year_ref=end_year_ref,interval=interval,temp_variable=temp_variable,smooth_span=smooth_span)
print("Done.")
print("--- %.2f seconds ---" % (time.time() - start_time))
#%% Compute distribution of reference period and threshold
print("Running compute_distrib_percentile...")
compute_distrib_percentile(read_directory_historical,read_directory_rcp,write_directory,start_year_ref=start_year_ref,end_year_ref=end_year_ref,interval=interval,temp_variable=temp_variable,threshold_value=threshold_value,distrib_window_size=distrib_window_size,anomaly=anomaly)
print("Done.")
print("--- %.2f seconds ---" % (time.time() - start_time))
#%% Compute heatwaves CC3D scan
print("Running cc3d_scan_heatwaves...")
cc3d_scan_heatwaves(read_directory_historical,read_directory_rcp,write_directory,start_year=start_year,end_year=end_year,start_year_ref=start_year_ref,end_year_ref=2020,interval=5,temp_variable='tas',threshold_value=threshold_value,relative_threshold=True,distrib_window_size=distrib_window_size,anomaly=anomaly,nb_days=nb_days)
print("Done.")
print("--- %.2f seconds ---" % (time.time() - start_time))
#%% 
#%% Compute distribution of reference period and threshold for HWMId computation
print("Running compute_distrib_percentile for 25th percentile...")
compute_distrib_percentile(read_directory_historical,read_directory_rcp,write_directory,start_year_ref=start_year_ref,end_year_ref=end_year_ref,interval=interval,temp_variable=temp_variable,threshold_value=25,distrib_window_size=distrib_window_size,anomaly=anomaly)
print("Done.")
print("--- %.2f seconds ---" % (time.time() - start_time))
#%% Compute distribution of reference period and threshold for HWMId computation
print("Running compute_distrib_percentile for 75th percentile ...")
compute_distrib_percentile(read_directory_historical,read_directory_rcp,write_directory,start_year_ref=start_year_ref,end_year_ref=end_year_ref,interval=interval,temp_variable=temp_variable,threshold_value=75,distrib_window_size=distrib_window_size,anomaly=anomaly)
print("Done.")
print("--- %.2f seconds ---" % (time.time() - start_time))
#%% Compute Regional Warming Level periods
print("Running compute_distrib_percentile for 75th percentile ...")
compute_regional_warming_levels(read_directory_historical,read_directory_rcp,write_directory,start_year=start_year,end_year=end_year,start_year_ref=1986,end_year_ref=2005,ref_period_offset=0.72,interval=interval,running_mean_window_size=20,regional_warming_levels_list=[2.1,2.6,4.0,5.1])
print("Done.")
print("--- %.2f seconds ---" % (time.time() - start_time))