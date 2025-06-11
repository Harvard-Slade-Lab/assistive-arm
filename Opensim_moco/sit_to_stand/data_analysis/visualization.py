import os
import json
import pandas as pd
import matplotlib.pyplot as plt

from utils_visualization import (
    read_sto_file,
    read_save_sto_metadata,
    apply_low_pass,
    plot_pelvis_kinematics,
    plot_joint_kinematics,
    plot_assistive_force,
    parameterize_assistive_force,
    plot_residual_force_torque_pelvis,
    plot_residual_force_torque_others,
    plot_muscle_activation,
    plot_muscle_activation_grid,
    plot_GRF_osim,
    plot_GRF_osim_v2,
    compare_muscle_activation_grid
)

# ========== Helper Functions ==========
def add_time_percentage(df):
    """Add a time percentage [0-100%] column."""
    total_time = df['time'].max() - df['time'].min()
    df['time_perc'] = ((df['time'] - df['time'].min()) / total_time) * 100
    return df

def load_sim_result(sim_path):
    for cur_file in os.listdir(sim_path):
        cur_file_path = os.path.join(sim_path, cur_file)
        if cur_file.endswith('.sto'):
            if 'grf' in cur_file:
                if 'floor' in cur_file:
                    df_grf_floor = read_sto_file(cur_file_path)
                elif 'chair' in cur_file:
                    df_grf_chair = read_sto_file(cur_file_path)
            else:
                df_sim_sol = read_sto_file(cur_file_path)
                read_save_sto_metadata(sto_path= cur_file_path, save_path = sim_path)

        elif cur_file.endswith('.json'):
            with open(cur_file_path, 'r') as file:
                sim_params = json.load(file)

    # ========== Process simulation results ==========
    df_sim_sol = add_time_percentage(df_sim_sol)

    # ========== Low-pass filter specs ==========
    lpf_order = 4
    lpf_cutoff = 6  # Hz

    # ========== Apply Low-pass Filter ==========
    df_sim_sol = apply_low_pass(df=df_sim_sol, cutoff=lpf_cutoff, order=lpf_order)

    return df_sim_sol

# load data

unassisted_path = './S002_sit_to_stand_4_20250605_113054_unassisted_converged'
assised_path = './S002_sit_to_stand_4_20250605_135908_assisted_converged'
result_path = './'
fig_title = 'S002_sit_to_stand_simulation'

unassisted_sol = load_sim_result(unassisted_path)
assisted_sol = load_sim_result(assised_path)


compare_muscle_activation_grid(df_unassisted = unassisted_sol, df_assisted = assisted_sol, fig_title = fig_title, fontsize = 8, figsize = (8, 4), fig_path = result_path)

