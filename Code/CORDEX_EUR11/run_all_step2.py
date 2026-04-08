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
    #read_directory = "/scratchu/tmandonnet/CORDEX"
    #write_directory = join("/scratchu/tmandonnet/CORDEX",f"figs_{temp_variable}_period_{start_year}_{end_year}_ref_{start_year_ref}_{end_year_ref}_threshold_{threshold_value}_{nb_days}days_connec_{connectivity}{'_ano'*anomaly}")

    read_directory = "/home/user/These/cordex_htws_cc3d/Data"
    write_directory = "/home/user/These/cordex_htws_cc3d/Data/output"
    pop_data_path = "/home/user/These/cordex_htws_cc3d/Data"
    other_data_path = "/home/user/These/cordex_htws_cc3d/Data"
    #pop_data_path = "/scratchu/tmandonnet"
    #other_data_path = "/data/tmandonnet/CORDEX"

    os.makedirs(write_directory,exist_ok=True)
    #%%
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
        print("Running plot_figures...")
        plot_figures(read_directory,write_directory,start_year_ref=start_year_ref,end_year_ref=end_year_ref,temp_variable=temp_variable)
        print("Done.")
    #%% Compute Heatwaves indices database
    if overwrite_files or exists(join(write_directory,"df_htws.csv"))==False:
        print("--- %.2f seconds ---" % (time.time() - start_time))
        print("Running create_heatwaves_indices_database ...")
        create_heatwaves_indices_database(read_directory_historical,read_directory_rcp,write_directory,pop_data_path,other_data_path,start_year=start_year,end_year=end_year,start_year_ref=start_year_ref,end_year_ref=end_year_ref,temp_variable=temp_variable,threshold_value=threshold_value,anomaly=True)
        print("Done.")
    print("--- %.2f seconds ---" % (time.time() - start_time))