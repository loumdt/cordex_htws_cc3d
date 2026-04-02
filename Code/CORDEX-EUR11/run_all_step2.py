#%% 
import numpy as np
import sys
from tqdm import tqdm # Create a user-friendly feedback for loops while script is running
import os
from os.path import join
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

    start_year = 1975
    end_year = 2099
    start_year_ref = 1975
    end_year_ref = 2025
    temp_variable = 'tasmax'
    threshold_value = 95
    distrib_window_size = 15
    anomaly = True
    relative_threshold = True
    nb_days = 4
    connectivity = 26
    #%%
    #read_directory_historical = "/bdd/CORDEX/output/EUR-11/GERICS/CNRM-CERFACS-CNRM-CM5/historical/r1i1p1/GERICS-REMO2015/v2/day/tas/latest"
    #read_directory_rcp = "/bdd/CORDEX/output/EUR-11/GERICS/CNRM-CERFACS-CNRM-CM5/rcp85/r1i1p1/GERICS-REMO2015/v2/day/tas/latest"
    #path_to_remove = "/bdd/CORDEX/output/EUR-11/"
    #write_directory = join("/scratchu/tmandonnet/CORDEX",f"figs_{temp_variable}_period_{start_year}_{end_year}_ref_{start_year_ref}_{end_year_ref}_threshold_{threshold_value}_{nb_days}days_connec_{connectivity}{'_ano'*anomaly}")

    write_directory = "/home/user/These/cordex_htws_cc3d/Data/output"
    pop_data_path = "/home/user/These/cordex_htws_cc3d/Data"
    other_data_path = "/home/user/These/cordex_htws_cc3d/Data"
    #pop_data_path = "/scratchu/tmandonnet"
    #other_data_path = "/data/tmandonnet/CORDEX"

    os.makedirs(write_directory,exist_ok=True)
    #%%
    print('temp_variable :',temp_variable)
    print('start_year_ref :',start_year_ref)
    print('end_year_ref :',end_year_ref)

    #%% Compute climatology smooth
    print("--- %.2f seconds ---" % (time.time() - start_time))
    print("Running compute_seasonal_cycle...")
    #compute_seasonal_cycle(read_directory_historical,read_directory_rcp,write_directory,start_year_ref=start_year_ref,end_year_ref=end_year_ref,temp_variable=temp_variable)
    print("Done.")
    print("--- %.2f seconds ---" % (time.time() - start_time))
    #%% Compute distribution of reference period and threshold
    print("Running compute_distrib_percentile...")
    #compute_distrib_percentile(read_directory_historical,read_directory_rcp,write_directory,start_year_ref=start_year_ref,end_year_ref=end_year_ref,temp_variable=temp_variable,threshold_value=threshold_value,distrib_window_size=distrib_window_size,anomaly=anomaly)
    print("Done.")
    print("--- %.2f seconds ---" % (time.time() - start_time))
    #%% Compute heatwaves CC3D scan
    print("Running cc3d_scan_heatwaves...")
    cc3d_scan_heatwaves(read_directory_historical,read_directory_rcp,write_directory,other_data_path,start_year=start_year,end_year=end_year,start_year_ref=start_year_ref,end_year_ref=end_year_ref,temp_variable=temp_variable,threshold_value=threshold_value,relative_threshold=True,distrib_window_size=distrib_window_size,anomaly=anomaly,nb_days=nb_days)
    print("Done.")
    print("--- %.2f seconds ---" % (time.time() - start_time))
    #%% Compute Regional Warming Level periods
    print("Running compute_regional_warming_levels ...")
    #compute_regional_warming_levels(read_directory_historical,read_directory_rcp,write_directory,start_year=start_year,end_year=end_year,start_year_ref=1986,end_year_ref=2005,ref_period_offset=0.72,running_mean_window_size=20,regional_warming_levels_list=[2.1,2.6,4.0,5.1])
    print("Done.")
    print("--- %.2f seconds ---" % (time.time() - start_time))
    #%% Compute HWMId index
    print("Running compute_Russo_HWMId ...")
    #compute_Russo_HWMId(read_directory_historical,read_directory_rcp,write_directory,start_year=start_year,end_year=end_year,start_year_ref=start_year_ref,end_year_ref=end_year_ref,temp_variable=temp_variable)
    print("Done.")
    print("--- %.2f seconds ---" % (time.time() - start_time))
    #%% Compute Heatwaves indices database
    print("Running create_heatwaves_indices_database ...")
    create_heatwaves_indices_database(read_directory_historical,read_directory_rcp,write_directory,other_data_path,start_year=start_year,end_year=end_year,start_year_ref=start_year_ref,end_year_ref=end_year_ref,temp_variable=temp_variable,threshold_value=threshold_value,anomaly=True)
    print("Done.")
    print("--- %.2f seconds ---" % (time.time() - start_time))