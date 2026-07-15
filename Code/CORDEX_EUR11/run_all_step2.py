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

    regional_warming_levels_list=[2.1,2.6,4.0,5.1]
    #%%
    read_directory = "/scratchu/tmandonnet/CORDEX"
    write_directory = join("/scratchu/tmandonnet/CORDEX",f"figs_{temp_variable}_period_{start_year}_{end_year}_ref_{start_year_ref}_{end_year_ref}_threshold_{threshold_value}_{nb_days}days_connec_{connectivity}{'_ano'*anomaly}")

    #read_directory = "/home/user/These/cordex_htws_cc3d/Data"
    #write_directory = "/home/user/These/cordex_htws_cc3d/Data/output"
    #pop_data_path = "/home/user/These/cordex_htws_cc3d/Data"
    #other_data_path = "/home/user/These/cordex_htws_cc3d/Data"
    pop_data_path = "/scratchu/tmandonnet"
    other_data_path = "/data/tmandonnet/CORDEX"

    os.makedirs(write_directory,exist_ok=True)
    #%%
    print("temp_variable:", temp_variable)
    print("nb_days:", nb_days)
    print("threshold_value:", threshold_value)
    print("connectivity:", connectivity)
    print(f"study period: {start_year}-{end_year}")
    print(f"baseline period: {start_year_ref}-{end_year_ref}")

    overwrite_files=True #If True, overwrite output files that already exists (may be relevant in case of code or data update)
    # if overwrite_file is True or if output file does not exist : call function ; else pass
    if (overwrite_files or exists(join(write_directory,"df_global_htws.csv"))==False):
        print("--- %.0f seconds ---" % (time.time() - start_time))
        print("Running merge_heatwaves_dataframes...")
        #merge_heatwaves_dataframes(read_directory,write_directory,start_year_ref=start_year_ref,end_year_ref=end_year_ref,regional_warming_levels_list=regional_warming_levels_list)
        print("Done.")
    if overwrite_files:
        print("--- %.0f seconds ---" % (time.time() - start_time))
        print("Running remap_labels_for_comparison...")
        #remap_labels_for_comparison(read_directory=read_directory,write_directory=join(read_directory,'remapped_labels_for_figs'),target_grid_directory='/data/tmandonnet/CORDEX/cellarea',mapping_target='rotated_pole',overwrite=overwrite_files)
        print("Done.")
    if overwrite_files or exists(join(write_directory,'hot_days_4_panels_raw_models.pdf'))==False:
        print("--- %.0f seconds ---" % (time.time() - start_time))
        print("Running plot_4_panel_hot_days...")
        #plot_4_panel_hot_days_RWL(read_directory=read_directory,write_directory=write_directory,start_year=2026,end_year=end_year,need_to_compute_labels=True)
        print("Done.")
    if overwrite_files or exists(join(write_directory,'movement_4_panels.pdf'))==False:
        print("--- %.0f seconds ---" % (time.time() - start_time))
        print("Running plot_4_movement_maps...")
        #plot_4_movement_maps(write_directory=write_directory)
        print("Done.")
    if overwrite_files or exists(join(write_directory,f'trends_nb_mean_rest_hot_days_{end_year_ref+1}_{end_year}.pdf'))==False:
        print("--- %.0f seconds ---" % (time.time() - start_time))
        print("Running plot_grid_point_trends...")
        plot_grid_point_trends(read_directory,write_directory,start_year=end_year_ref+1,end_year=end_year)
        print("Done.")
    if overwrite_files or exists(join(write_directory,'hot_days_4_panels_raw_models.pdf'))==False:
        print("--- %.0f seconds ---" % (time.time() - start_time))
        print("Running make_animation_selected_models...")
        #make_animation_selected_models(read_directory=read_directory,write_directory=write_directory,other_data_path=other_data_path)
        print("Done.")
    if overwrite_files or exists(join(write_directory,"df_mk_trends_1975_2025.csv"))==False:
        print("--- %.0f seconds ---" % (time.time() - start_time))
        print("Running compute_mk_trends...")
        #compute_mk_trends(read_directory=write_directory,other_data_path=other_data_path,start_year=1975,end_year=2025,split_year_population=2025,yearly_aggregation=False)
        print("Done.")
    if overwrite_files or exists(join(write_directory,"df_mk_trends_2026_2099.csv"))==False:
        print("--- %.0f seconds ---" % (time.time() - start_time))
        print("Running compute_mk_trends...")
        #compute_mk_trends(read_directory=write_directory,other_data_path=other_data_path,start_year=2026,end_year=2099,split_year_population=2025,yearly_aggregation=False)
        print("Done.")
    if overwrite_files or exists(join(write_directory,"df_mk_trends_1975_2025_year_agg.csv"))==False:
        print("--- %.0f seconds ---" % (time.time() - start_time))
        print("Running compute_mk_trends...")
        #compute_mk_trends(read_directory=write_directory,other_data_path=other_data_path,start_year=1975,end_year=2025,split_year_population=2025,yearly_aggregation=True)
        print("Done.")
    if overwrite_files or exists(join(write_directory,"df_mk_trends_2026_2099_year_agg.csv"))==False:
        print("--- %.0f seconds ---" % (time.time() - start_time))
        print("Running compute_mk_trends...")
        #compute_mk_trends(read_directory=write_directory,other_data_path=other_data_path,start_year=2026,end_year=2099,split_year_population=2025,yearly_aggregation=True)
        print("Done.")
    if (overwrite_files or exists(join(write_directory,"all_ssp_distrib_Exposed_population.pdf"))==False):
        print("--- %.0f seconds ---" % (time.time() - start_time))
        print("Running plot_RWL_figures...")
        #plot_RWL_figures(read_directory,write_directory,regional_warming_levels_list=regional_warming_levels_list,RWLs_to_plot=[0,1,2])
        print("Done.")
    if overwrite_files or exists(join(write_directory,"FIGNAME.pdf"))==False:
        print("--- %.0f seconds ---" % (time.time() - start_time))
        print("Running plot_comparison_reanalysis_figures...")
        #plot_comparison_reanalysis_figures(read_directory,write_directory)
        print("Done.")
    print("--- %.0f hours and %.0f minutes ---" % ((time.time() - start_time)//3600 , (time.time() - start_time)%3600//60))