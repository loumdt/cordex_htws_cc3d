#%% 
import numpy as np
import sys
from tqdm import tqdm # Create a user-friendly feedback for loops while script is running
import os
from os.path import join,exists
from utils import *
import time
if __name__ == "__main__":
    #%% Read variables and directories
    start_time = time.time()
    # Collect details and find read folder of temperature variable
    #combination = tuple((sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]))
    combination = tuple(('CNRM-CERFACS-CNRM-CM5','GERICS-REMO2015','rcp85','v1'))
    read_directory_historical = "/home/user/These/cordex_htws_cc3d/Data/historical"#sys.argv[5]
    read_directory_rcp = "/home/user/These/cordex_htws_cc3d/Data/rcp85"#sys.argv[6]
    #read_directory_historical = sys.argv[5]
    #read_directory_rcp = #sys.argv[6]

    bias_adjusted = False

    start_year = 1975
    end_year = 2099
    start_year_ref = 1975
    end_year_ref = 2025
    temp_variable = 'tasmax'+'Adjust'*bias_adjusted
    threshold_value = 95
    distrib_window_size = 15
    anomaly = True
    relative_threshold = True
    nb_days = 4
    connectivity = 26

    #%%
    #read_directory_historical = "/bdd/CORDEX/output/EUR-11/GERICS/CNRM-CERFACS-CNRM-CM5/historical/r1i1p1/GERICS-REMO2015/v2/day/tas/latest"
    read_directory_historical = "/home/user/These/cordex_htws_cc3d/Data/ERA5/tasmax/before_2005"
    read_directory_rcp = "/home/user/These/cordex_htws_cc3d/Data/ERA5/tasmax/after_2005"
    #read_directory_rcp = "/bdd/CORDEX/output/EUR-11/GERICS/CNRM-CERFACS-CNRM-CM5/rcp85/r1i1p1/GERICS-REMO2015/v2/day/tas/latest"
    #path_to_remove = "/bdd/CORDEX/output/EUR-11/"
    #write_directory = join("/scratchu/tmandonnet/CORDEX",read_directory_rcp.replace(path_to_remove,"").replace("/","_"))

    write_directory = "/home/user/These/cordex_htws_cc3d/Data/output"
    pop_data_path = "/home/user/These/cordex_htws_cc3d/Data"
    other_data_path = "/home/user/These/cordex_htws_cc3d/Data"
    #pop_data_path = "/scratchu/tmandonnet"
    #other_data_path = "/data/tmandonnet/CORDEX"

    os.makedirs(write_directory,exist_ok=True)
    #%%
    print('GCM, RCM, RCP, version :',combination)
    print('read_directory_historical :',read_directory_historical)
    print('read_directory_rcp :',read_directory_rcp)
    print("temp_variable:", temp_variable)
    print("nb_days:", nb_days)
    print("threshold_value:", threshold_value)
    print("connectivity:", connectivity)
    print(f"study period: {start_year}-{end_year}")
    print(f"baseline period: {start_year_ref}-{end_year_ref}")

    overwrite_files=False #If True, overwrite output files that already exists (may be relevant in case of code or data update)
    # if overwrite_file is True or if output file does not exist : call function ; else pass
    #%% Compute climatology smooth
    if (overwrite_files or exists(join(write_directory,f"seasonal_cycle_{start_year_ref}_{end_year_ref}.nc"))==False):
        print("--- %.2f seconds ---" % (time.time() - start_time))
        print("Running compute_seasonal_cycle...")
        compute_seasonal_cycle(read_directory_historical,read_directory_rcp,write_directory,start_year_ref=start_year_ref,end_year_ref=end_year_ref,temp_variable=temp_variable)
        print("Done.")
    #%% Compute distribution of reference period and threshold
    if (overwrite_files or exists(join(write_directory,f"distrib_threshold_{threshold_value}.nc"))==False) and relative_threshold==True: # Only used for relative thresholds
        print("--- %.2f seconds ---" % (time.time() - start_time))
        print("Running compute_distrib_percentile...")
        compute_distrib_percentile(read_directory_historical,read_directory_rcp,write_directory,start_year_ref=start_year_ref,end_year_ref=end_year_ref,temp_variable=temp_variable,threshold_value=threshold_value,distrib_window_size=distrib_window_size,anomaly=anomaly)
        print("Done.")
    #%% Compute heatwaves CC3D scan
    if overwrite_files or exists(join(write_directory,f"labels_cc3d_year_{end_year}_ref_{start_year_ref}_{end_year_ref}.nc"))==False:
        print("--- %.2f seconds ---" % (time.time() - start_time))
        print("Running cc3d_scan_heatwaves...")
        cc3d_scan_heatwaves(read_directory_historical,read_directory_rcp,write_directory,other_data_path,start_year=start_year,end_year=end_year,start_year_ref=start_year_ref,end_year_ref=end_year_ref,temp_variable=temp_variable,threshold_value=threshold_value,relative_threshold=True,distrib_window_size=distrib_window_size,anomaly=anomaly,nb_days=nb_days)
        print("Done.")
    #%% Compute Regional Warming Level periods
    if overwrite_files or exists(join(write_directory,"regional_warming_levels.json"))==False:
        print("--- %.2f seconds ---" % (time.time() - start_time))
        print("Running compute_regional_warming_levels ...")
        compute_regional_warming_levels(read_directory_historical,read_directory_rcp,write_directory,start_year=start_year,end_year=end_year,start_year_ref=1986,end_year_ref=2005,temp_variable=temp_variable,ref_period_offset=0.72,running_mean_window_size=20,regional_warming_levels_list=[2.1,2.6,4.0,5.1],bias_adjusted=bias_adjusted)
        print("Done.")
    #%% Compute HWMId index
    if overwrite_files or exists(join(write_directory,"Russo_HWMId.nc"))==False:
        print("--- %.2f seconds ---" % (time.time() - start_time))
        print("Running compute_Russo_HWMId ...")
        compute_Russo_HWMId(read_directory_historical,read_directory_rcp,write_directory,start_year=start_year,end_year=end_year,start_year_ref=start_year_ref,end_year_ref=end_year_ref,temp_variable=temp_variable)
        print("Done.")
    #%% Compute Heatwaves indices database
    if overwrite_files or exists(join(write_directory,"df_htws.csv"))==False:
        print("--- %.2f seconds ---" % (time.time() - start_time))
        print("Running create_heatwaves_indices_database ...")
        #create_heatwaves_indices_database(read_directory_historical,read_directory_rcp,write_directory,pop_data_path,other_data_path,start_year=start_year,end_year=end_year,start_year_ref=start_year_ref,end_year_ref=end_year_ref,temp_variable=temp_variable,threshold_value=threshold_value,anomaly=True)
        #print("Done.")
    print("--- %.2f seconds ---" % (time.time() - start_time))