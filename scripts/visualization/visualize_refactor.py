# refactor the visualization code
# editor: Haedo Cho
# last updated: Mar 24 2025

import pandas as pd
from pathlib import Path
import yaml
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from datetime import date
import seaborn as sns

import sys
from pathlib import Path
from utils_visulization import readMotionFile, read_sto_file

from utils.plotting import extract_muscle_activations, plot_res_assist_forces_grf, plot_residual_forces_trans, plot_residual_torques_rot
from utils.plotting import plot_residual_all, plot_residual_torques_leg, plot_coordactuator_torques_arm, plot_coordactuator_torques_back, plot_residual_forces_torques_all
from utils.plotting import plot_residual_torques_leg_right, plot_residual_torques_leg_left
from utils.plotting import plot_muscles_group_activations, plot_muscles_activations, plot_muscles_group_activation_differences, plot_muscles_group_activations2, plot_muscles_group_activation_differences2, plot_muscles_group_activations_mean2
from utils.plotting import plot_GRF_osim, plot_res_assist_forces_grf_osim, plot_GRF_osim2, plot_GRF_osim3, plot_GRF_osim4, plot_GRF_osim2_averaged
from utils.plotting import plot_residual_forces_torques_all2, format_muscle_name
from utils.data_preprocessing import smooth_dataframe, interpolate_dataframe
from utils.data_preprocessing import butterworth_low_pass_df

show_plot = True 

# Session paramaters
subject = "CG" # CG / ZK
task = "sts" # sts / reaching / stairs
assistance = "no_arm" # no_arm / assistive_arm_off / assistive_arm_on
trial_ID = "1"
#assistive_force = None  # 700N or None ?? or 100N??
model_type = "simple" # simple / full
date = "20240503"

# Opencap kinematics data path
trial_name = f"{task}_{assistance}_{trial_ID}" # sts_no_arm_1
session = Path(f"./data/open_cap_data/{date}_opencap_{task}_{subject}") 

kinematics_path = str(session / "OpenSimData" / "Kinematics" / f"{trial_name}.mot")

# OpenSim Moco Solution path
solutions_path = Path(f"./data/moco_solution/{trial_name}")

assist_false = True
assist_true = False
mocap_grfs_added = False
ground_forces_estimated_false = True #put back to True for 2024-07-30 16-28
ground_forces_estimated_true = False
osim_grfs_added = False
assist_false_pelvis_assist = True

if assist_false:
    date_simulation="2024-08-20"
    time_simulation="17-07"
    solution_assist_false = solutions_path / f"{subject}_{model_type}_{task}_{assistance}_{trial_ID}_assistance_none_{date_simulation}_{time_simulation}.sto"
    if ground_forces_estimated_false: 
        grf_osim_feet_false = solutions_path / f"{subject}_{model_type}_{task}_{assistance}_{trial_ID}_assistance_none_{date_simulation}_{time_simulation}_grfs_osim_feet.sto" 
        grf_osim_chair_false = solutions_path / f"{subject}_{model_type}_{task}_{assistance}_{trial_ID}_assistance_none_{date_simulation}_{time_simulation}_grfs_osim_chair.sto" 

if assist_true:
    date_simulation2="2024-08-20"
    time_simulation2="17-07"
    force_value="700"
    solution_assist_true = solutions_path / f"{subject}_{model_type}_{task}_{assistance}_{trial_ID}_assistance_{force_value}_{date_simulation2}_{time_simulation2}.sto"
    if ground_forces_estimated_true: 
        grf_osim_feet_true = solutions_path / f"{subject}_{model_type}_{task}_{assistance}_{trial_ID}_assistance_{force_value}_{date_simulation2}_{time_simulation2}_grfs_osim_feet.sto" 
        grf_osim_chair_true = solutions_path / f"{subject}_{model_type}_{task}_{assistance}_{trial_ID}_assistance_{force_value}_{date_simulation2}_{time_simulation2}_grfs_osim_chair.sto"
    if osim_grfs_added:
        grf_osim_precomputed_feet_true = solutions_path / f"{subject}_{model_type}_{task}_{assistance}_{trial_ID}_assistance_none_{date_simulation}_{time_simulation}_grfs_osim_feet_new_version.sto" 
        grf_osim_precomputed_chair_true = solutions_path / f"{subject}_{model_type}_{task}_{assistance}_{trial_ID}_assistance_none_{date_simulation}_{time_simulation}_grfs_osim_chair_new_version.sto" 


if assist_false_pelvis_assist:
    # Tuned Unassisted
    date_simulation="2024-07-30"
    time_simulation="16-28"
    # Untuned Unassisted 
    #date_simulation="2024-07-23"
    #time_simulation="17-25"
    solution_assist_false_before = solutions_path / f"{subject}_{model_type}_{task}_{assistance}_{trial_ID}_assistance_none_{date_simulation}_{time_simulation}.sto"
    if ground_forces_estimated_false: 
        grf_osim_feet_false_before = solutions_path / f"{subject}_{model_type}_{task}_{assistance}_{trial_ID}_assistance_none_{date_simulation}_{time_simulation}_grfs_osim_feet.sto" 
        grf_osim_chair_false_before = solutions_path / f"{subject}_{model_type}_{task}_{assistance}_{trial_ID}_assistance_none_{date_simulation}_{time_simulation}_grfs_osim_chair.sto" 

#Store solutions sto files into dataframes

if assist_false:
    df_assist_false = pd.read_csv(solution_assist_false, delimiter="\t", skiprows=18)
    config_path_false = solutions_path / f"{solution_assist_false.stem}.yaml"
    with open(config_path_false, 'r') as f:
        config_file_false = yaml.load(f, Loader=yaml.FullLoader)
    print(yaml.dump(config_file_false, indent=4, sort_keys=False))

if assist_true:
    df_assist_true = pd.read_csv(solution_assist_true, delimiter="\t", skiprows=18)
    config_path_true = solutions_path / f"{solution_assist_true.stem}.yaml"
    with open(config_path_true, 'r') as f:
        config_file_true = yaml.load(f, Loader=yaml.FullLoader)
    print(yaml.dump(config_file_true, indent=4, sort_keys=False))

#Store grfs files into dataframes
if mocap_grfs_added:
    ground_forces = pd.read_csv(grfs, delimiter="\t", skiprows=6)

if assist_false and ground_forces_estimated_false:
    df_grf_osim_feet_false = read_sto_file(grf_osim_feet_false)
    df_grf_osim_chair_false = read_sto_file(grf_osim_chair_false)

if assist_true and ground_forces_estimated_true:
    df_grf_osim_feet_true = read_sto_file(grf_osim_feet_true)
    df_grf_osim_chair_true = read_sto_file(grf_osim_chair_true)

if assist_true and osim_grfs_added:
    grf_osim_precomputed_feet_true = config_file_true["grf_osim_feet_precomputed_path"]
    grf_osim_precomputed_chair_true = config_file_true["grf_osim_chair_precomputed_path"]
    df_grf_osim_precomputed_feet_true = read_sto_file(grf_osim_precomputed_feet_true)
    df_grf_osim_precomputed_chair_true = read_sto_file(grf_osim_precomputed_chair_true)
    """
    if (config_file_true["grf_osim_feet_precomputed_path"] == str(grf_osim_feet_false)) and (config_file_true["grf_osim_chair_precomputed_path"] == str(grf_osim_chair_false)):
        print("OpenSim GRFs loaded from the assist_false simulation")
        grf_osim_precomputed_feet_true = config_file_true["grf_osim_feet_precomputed_path"]
        grf_osim_precomputed_chair_true = config_file_true["grf_osim_chair_precomputed_path"]
        df_grf_osim_precomputed_feet_true = read_sto_file(grf_osim_precomputed_feet_true)
        df_grf_osim_precomputed_chair_true = read_sto_file(grf_osim_precomputed_chair_true)
    else:
        print("Assist false simulation different from grf_osim_precomputed simulation")
        print(grf_osim_precomputed_feet_true)
        print(config_file_true["grf_osim_feet_precomputed_path"])
        print(grf_osim_precomputed_chair_true)
        print(config_file_true["grf_osim_chair_precomputed_path"])
        df_assist_false = None
    """

if assist_false_pelvis_assist:
    df_assist_false_before = pd.read_csv(solution_assist_false_before, delimiter="\t", skiprows=18)
    config_path_false_before = solutions_path / f"{solution_assist_false_before.stem}.yaml"
    with open(config_path_false_before, 'r') as f:
        config_file_false_before = yaml.load(f, Loader=yaml.FullLoader)
    print(yaml.dump(config_file_false_before, indent=4, sort_keys=False))

    if ground_forces_estimated_false:
        df_grf_osim_feet_false_before = read_sto_file(grf_osim_feet_false_before)
        df_grf_osim_chair_false_before = read_sto_file(grf_osim_chair_false_before)

    config_file_false_before["assist_with_reserve_pelvis_txy"]=False

if assist_false and assist_false_pelvis_assist:
    config_file_false["assistive_force"]=True
    config_file_false_before["assistive_force"]=False
    t0 = 4
    tf = 5.8
    df_assist_false_before = df_assist_false_before.loc[(df_assist_false_before.time >= t0) & (df_assist_false_before.time <= tf), :]
    if ground_forces_estimated_false:
        df_grf_osim_feet_false_before = df_grf_osim_feet_false_before.loc[(df_grf_osim_feet_false_before.time >= t0) & (df_grf_osim_feet_false_before.time <= tf), :]
        df_grf_osim_chair_false_before = df_grf_osim_chair_false_before.loc[(df_grf_osim_chair_false_before.time >= t0) & (df_grf_osim_chair_false_before.time <= tf), :]
        
    df_assist_false = df_assist_false.loc[(df_assist_false.time >= t0) & (df_assist_false.time <= tf), :]
    if ground_forces_estimated_false:
        df_grf_osim_feet_false = df_grf_osim_feet_false.loc[(df_grf_osim_feet_false.time >= t0) & (df_grf_osim_feet_false.time <= tf), :]
        df_grf_osim_chair_false = df_grf_osim_chair_false.loc[(df_grf_osim_chair_false.time >= t0) & (df_grf_osim_chair_false.time <= tf), :]

if assist_true:
    print(df_assist_true)

if assist_false and assist_false_pelvis_assist:
    motion_percentage = (df_assist_false.index - df_assist_false.index.min())/(df_assist_false.index.max() - df_assist_false.index.min()) * 100
    print(len(motion_percentage))

    df_assist_false_before['percentage'] = motion_percentage
    if ground_forces_estimated_false:
        df_grf_osim_feet_false_before['percentage'] = motion_percentage
        df_grf_osim_chair_false_before['percentage'] = motion_percentage  
    df_assist_false['percentage'] = motion_percentage
    if ground_forces_estimated_false:
        df_grf_osim_feet_false['percentage'] = motion_percentage
        df_grf_osim_chair_false['percentage'] = motion_percentage

# Synthesis plot
if show_plot:
    if assist_false:
        plot_residual_forces_torques_all(df=df_assist_false, config_file=config_file_false, filter_key_infos=False)
        plot_residual_forces_torques_all(df=df_assist_false, config_file=config_file_false, filter_key_infos=True, alpha=0.3)
    if assist_true:
        plot_residual_forces_torques_all(df=df_assist_true, config_file=config_file_true, filter_key_infos=False)
        plot_residual_forces_torques_all(df=df_assist_true, config_file=config_file_true, filter_key_infos=True, alpha=0.3)
    if assist_false_pelvis_assist:
        plot_residual_forces_torques_all(df=df_assist_false_before, config_file=config_file_false_before, filter_key_infos=False)
        plot_residual_forces_torques_all(df=df_assist_false_before, config_file=config_file_false_before, filter_key_infos=True, alpha=0.3)

cutoff = 6  # This value used to be 30Hz but seems too high as sampling frequency is 40Hz
order = 4

df_assist_false_before_filtered = butterworth_low_pass_df(df=df_assist_false_before, order=4, cutoff=cutoff)
df_assist_false_filtered = butterworth_low_pass_df(df=df_assist_false, order=4, cutoff=cutoff)

if show_plot:
    plot_residual_forces_torques_all2(df=df_assist_false_before_filtered, config_file=config_file_false_before, save=True, title="residual_actuators_unassisted", filter_key_infos=True, alpha=0.7)
    plot_residual_forces_torques_all2(df=df_assist_false_filtered, config_file=config_file_false, save=True, title="residual_actuators_assisted", filter_key_infos=True, alpha=0.7)

muscles_set1 = ["recfem_r", "recfem_l", "vasmed_r", "vasmed_l", "vaslat_r", "vaslat_l"]
#muscles_set1 = ["recfem_r", "recfem_l", "vasmed_r", "vasmed_l"]
muscles_set2 = ["psoas_r", "psoas_l"]
#muscles_set2 = []
muscles_set3 = ["bflh_r", "bflh_l", 
                "bfsh_r", "bfsh_l", 
                "glmax1_r", "glmax1_l", 
                "glmax2_r", "glmax2_l",
                "glmax3_r", "glmax3_l",
                "glmed1_r", "glmed1_l",
                "glmed2_r", "glmed2_l",
                "glmed3_r", "glmed3_l"]
#muscles_set3 = ["bflh_r", "bfsh_r", "glmax1_r"]
muscles_set4 = ["soleus_r", "soleus_l", "gasmed_r","gasmed_l", "tibant_r", "tibant_l"]

if assist_false:
    simple_df_false_activations = extract_muscle_activations(df_assist_false, apply_filter=False)
    simple_df_false_activations.set_index(df_assist_false.time, inplace=True)

if assist_true:
    if assist_false:
        simple_df_true_activations = extract_muscle_activations(df_assist_true, apply_filter=False)[simple_df_false_activations.columns]
    else:
        simple_df_true_activations = extract_muscle_activations(df_assist_true, apply_filter=False)
    simple_df_true_activations.set_index(df_assist_true.time, inplace=True)

if assist_false_pelvis_assist:
    simple_df_false_activations_before = extract_muscle_activations(df_assist_false_before, apply_filter=False)
    simple_df_false_activations_before.set_index(df_assist_false_before.time, inplace=True)

if show_plot:
    if assist_false_pelvis_assist:
        plot_muscles_group_activations(simple_df_false_activations_before, muscles_set1, muscles_set2, muscles_set3, muscles_set4, config_file_false_before)
        
    if assist_false:
        plot_muscles_group_activations(simple_df_false_activations, muscles_set1, muscles_set2, muscles_set3, muscles_set4, config_file_false)

    if assist_true:
        plot_muscles_group_activations(simple_df_true_activations, muscles_set1, muscles_set2, muscles_set3, muscles_set4, config_file_true)

if show_plot:
    if assist_false_pelvis_assist:
        cutoff = 20  # Adjust this based on your needs (e.g., 20 Hz cutoff)
        simple_df_false_activations_before_no_filtering = simple_df_false_activations_before.copy()
        
        ## need to fix this
        # simple_df_false_activations_before = butterworth_low_pass(df=simple_df_false_activations_before, cutoff=cutoff)

        #plot_muscles_group_activations2(simple_df_false_activations_before_no_filtering, muscles_set1, muscles_set2, muscles_set3, muscles_set4, config_file_false_before, motion_percentage)
        plot_muscles_group_activations2(simple_df_false_activations_before, muscles_set1, muscles_set2, muscles_set3, muscles_set4, config_file_false_before, motion_percentage)
    if assist_false:
        simple_df_false_activations_no_filtering = simple_df_false_activations.copy()
        # simple_df_false_activations = butterworth_low_pass(df=simple_df_false_activations, cutoff=cutoff)
        #plot_muscles_group_activations2(simple_df_false_activations_no_filtering, muscles_set1, muscles_set2, muscles_set3, muscles_set4, config_file_false, motion_percentage)
        plot_muscles_group_activations2(simple_df_false_activations, muscles_set1, muscles_set2, muscles_set3, muscles_set4, config_file_false, motion_percentage)

    if assist_true:
        plot_muscles_group_activations2(simple_df_true_activations, muscles_set1, muscles_set2, muscles_set3, muscles_set4, config_file_true, motion_percentage)

if show_plot:
    plot_muscles_group_activations_mean2(simple_df_false_activations_before, muscles_set1, muscles_set2, muscles_set3, muscles_set4, config_file_false_before, motion_percentage, save=False, title="muscle_activations_unassisted_mean")
    plot_muscles_group_activations_mean2(simple_df_false_activations, muscles_set1, muscles_set2, muscles_set3, muscles_set4, config_file_false, motion_percentage, save=False, title="muscle_activations_assisted_mean")

## Residuals and coordinate actuators 

if show_plot:
    if assist_false:
        plot_residual_forces_trans(df=df_assist_false, config_file=config_file_false)
    if assist_true:
        plot_residual_forces_trans(df=df_assist_true, config_file=config_file_true)

    if assist_false:
        plot_residual_torques_rot(df=df_assist_false, config_file=config_file_false)
    if assist_true:
        plot_residual_torques_rot(df=df_assist_true, config_file=config_file_true)

if show_plot:
    if assist_false:
        #plot_residual_all(df=df_assist_false, config_file=config_file_false)
        plot_residual_torques_leg(df=df_assist_false, config_file=config_file_false)
        plot_residual_torques_leg_right(df=df_assist_false, config_file=config_file_false)
        plot_residual_torques_leg_left(df=df_assist_false, config_file=config_file_false)
        
        plot_coordactuator_torques_arm(df=df_assist_false, config_file=config_file_false)
        plot_coordactuator_torques_back(df=df_assist_false, config_file=config_file_false)

    if assist_true:
        #plot_residual_all(df=df_assist_true, config_file=config_file_true)
        plot_residual_torques_leg(df=df_assist_true, config_file=config_file_true)
        plot_coordactuator_torques_arm(df=df_assist_true, config_file=config_file_true)
        plot_coordactuator_torques_back(df=df_assist_true, config_file=config_file_true)

# Assistive Force
dfs = {}

if mocap_grfs_added:
    if assist_false:
        ground_forces = ground_forces[(ground_forces["time"] >= config_file_false["t_0"]) & (ground_forces["time"] <= config_file_false["t_f"])]
    if assist_true:
        ground_forces = ground_forces[(ground_forces["time"] >= config_file_true["t_0"]) & (ground_forces["time"] <= config_file_true["t_f"])]

    if assist_false:
        dfs["assist_false"] = df_assist_false
    if assist_true:
        dfs["assist_true"] = df_assist_true
    dfs["ground_forces"] = ground_forces

if ground_forces_estimated_false:
    if assist_false:
        dfs["assist_false"] = df_assist_false
        dfs["grf_osim_feet_false"] = df_grf_osim_feet_false
        dfs["grf_osim_chair_false"] = df_grf_osim_chair_false
    if assist_false_pelvis_assist:
        dfs["assist_false_before"] = df_assist_false_before
        dfs["grf_osim_feet_false_before"] = df_grf_osim_feet_false_before
        dfs["grf_osim_chair_false_before"] = df_grf_osim_chair_false_before
    

if ground_forces_estimated_true:
    dfs["assist_true"] = df_assist_true
    dfs["grf_osim_feet_true"] = df_grf_osim_feet_true
    dfs["grf_osim_chair_true"] = df_grf_osim_chair_true

if osim_grfs_added:
    if assist_false:
        dfs["assist_false"] = df_assist_false
    if assist_true:
        dfs["assist_true"] = df_assist_true
        dfs["grf_osim_feet_precomputed_true"] = df_grf_osim_precomputed_feet_true
        dfs["grf_osim_chair_precomputed_true"] = df_grf_osim_precomputed_chair_true

if show_plot:
    if mocap_grfs_added and assist_true:
        plot_res_assist_forces_grf(time=df_assist_true.time, dataframes=dfs, figsize=(8,5), config=config_file_true)
    if ground_forces_estimated_false and assist_false:
        config_file_false['assistive_force']=False
        plot_res_assist_forces_grf_osim(time=df_assist_false.time, dataframes=dfs, figsize=(8,5), config=config_file_false)
        config_file_false['assistive_force']=True
    if ground_forces_estimated_true and assist_true:
        plot_res_assist_forces_grf_osim(time=df_assist_true.time, dataframes=dfs, figsize=(8,5), config=config_file_true)
    if osim_grfs_added and assist_true:
        plot_res_assist_forces_grf_osim(time=df_assist_true.time, dataframes=dfs, figsize=(8,5), config=config_file_true)

    if assist_true and osim_grfs_added:
        grf_chair_data_path = config_file_true["grf_osim_chair_precomputed_path"]
        df_grf_osim_chair = read_sto_file(grf_chair_data_path)
        df_grf_osim_chair.fillna(value=0, inplace=True)
        df_grf_osim_chair

if show_plot:
    # Plot muscles activation
    if assist_false:
        plot_muscles_activations(simple_df_false_activations, config_file_false)

    if assist_true:
        plot_muscles_activations(simple_df_true_activations, config_file_true)

if show_plot:
    if assist_false:
        plot_muscles_activations(simple_df_false_activations[muscles_set1], config_file_false)
        plot_muscles_activations(simple_df_false_activations[muscles_set2], config_file_false)
        plot_muscles_activations(simple_df_false_activations[muscles_set3], config_file_false)
        plot_muscles_activations(simple_df_false_activations[muscles_set4], config_file_false)
        
    if assist_true:
        plot_muscles_activations(simple_df_true_activations[muscles_set1], config_file_true)
        plot_muscles_activations(simple_df_true_activations[muscles_set2], config_file_true)
        plot_muscles_activations(simple_df_true_activations[muscles_set3], config_file_true)
        plot_muscles_activations(simple_df_true_activations[muscles_set4], config_file_true)

# Plot GRFs
if assist_false and ground_forces_estimated_false:
    df_grf_osim_feet_false.head()

if assist_false and ground_forces_estimated_false:
    df_grf_osim_chair_false.head()

if assist_true and ground_forces_estimated_true:
    df_grf_osim_feet_true.head()

if assist_true and ground_forces_estimated_true:
    df_grf_osim_chair_true.head()

if show_plot:
    if assist_false == True and ground_forces_estimated_false:
        plot_GRF_osim(df_grf_osim_feet_false, config_file_false, f"Feet Ground Reaction Forces and Torques Analysis ({'assisted' if config_file_false['assistive_force'] else 'unassisted'})")
        plot_GRF_osim(df_grf_osim_chair_false, config_file_false, f"Chair Ground Reaction Forces and Torques Analysis ({'assisted' if config_file_false['assistive_force'] else 'unassisted'})")

    if assist_true == True and ground_forces_estimated_true:
        plot_GRF_osim(df_grf_osim_feet_true, config_file_true, f"Feet Ground Reaction Forces and Torques Analysis ({'assisted' if config_file_true['assistive_force'] else 'unassisted'})")
        plot_GRF_osim(df_grf_osim_chair_true, config_file_true, f"Chair Ground Reaction Forces and Torques Analysis ({'assisted' if config_file_true['assistive_force'] else 'unassisted'})")

    if assist_true == True and osim_grfs_added:
        plot_GRF_osim(df_grf_osim_precomputed_feet_true, config_file_true, f"Feet Ground Reaction Forces and Torques Analysis ({'assisted' if config_file_true['assistive_force'] else 'unassisted'})")
        plot_GRF_osim(df_grf_osim_precomputed_chair_true, config_file_true, f"Chair Ground Reaction Forces and Torques Analysis ({'assisted' if config_file_true['assistive_force'] else 'unassisted'})")

    if assist_false == True and ground_forces_estimated_false:
        plot_GRF_osim2(df_grf_osim_feet_false, df_grf_osim_chair_false, config_file_false, f"Ground Reaction Forces and Torques Analysis ({'assisted' if config_file_false['assistive_force'] else 'unassisted'})")
        
    if assist_true == True and ground_forces_estimated_true:
        plot_GRF_osim2(df_grf_osim_feet_true, df_grf_osim_chair_true, config_file_true, f"Ground Reaction Forces and Torques Analysis ({'assisted' if config_file_true['assistive_force'] else 'unassisted'})")
        

# ------ The code is working up to this point -----
