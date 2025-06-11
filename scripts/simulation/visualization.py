import os
import json
import matplotlib.pyplot as plt

from utils_visulization import (
    read_sto_file,
    apply_low_pass,
    plot_pelvis_kinematics,
    plot_joint_kinematics,
    plot_residual_force_torque,
    plot_muscle_activation,
    plot_GRF_osim
)

# ========== Session parameters ==========
date = "20250428_104917"
subj = "S001"
task = "sit-to-stand_4"

# ========== Low-pass filter specs ==========
lpf_order = 4
lpf_cutoff = 6  # Hz

# ========== Flags for visualization ==========
plot_kinematics_flag = False
plot_residual_force_torque_flag = False
plot_muscle_activation_flag = False
plot_osim_grf_flag = True

# ========== Paths ==========
sim_path = f'./sim_result/{subj}_{task}_{date}'
kinematics_path = f'./data/OpenCapData/{subj}/OpenSimData/Kinematics/{task}.mot'

# ========== Helper Functions ==========
def add_time_percentage(df):
    """Add a time percentage [0-100%] column."""
    total_time = df['time'].max() - df['time'].min()
    df['time_perc'] = ((df['time'] - df['time'].min()) / total_time) * 100
    return df

# ========== Load Simulation Results ==========
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

    elif cur_file.endswith('.json'):
        with open(cur_file_path, 'r') as file:
            sim_params = json.load(file)

# ========== Process simulation results ==========
df_sim_sol = add_time_percentage(df_sim_sol)
df_grf_floor = add_time_percentage(df_grf_floor)
df_grf_chair = add_time_percentage(df_grf_chair)

# ========== Load and Process Kinematics ==========
df_kinematics = read_sto_file(kinematics_path)
df_kinematics = df_kinematics[
    (df_kinematics['time'] >= sim_params['t0']) &
    (df_kinematics['time'] <= sim_params['tf'])
]  # Crop to simulation time
df_kinematics = add_time_percentage(df_kinematics)

# ========== Apply Low-pass Filter ==========
df_sim_sol = apply_low_pass(df=df_sim_sol, cutoff=lpf_cutoff, order=lpf_order)
df_kinematics = apply_low_pass(df=df_kinematics, cutoff=lpf_cutoff, order=lpf_order)
df_grf_floor = apply_low_pass(df=df_grf_floor, cutoff=lpf_cutoff, order=lpf_order)
df_grf_chair = apply_low_pass(df=df_grf_chair, cutoff=lpf_cutoff, order=lpf_order)

# ========== Visualization ==========
if plot_kinematics_flag:
    # Pelvis
    plot_pelvis_kinematics(df_sim=df_sim_sol, df_ref=df_kinematics, fontsize=8, figsize=(8, 2), fig_path=sim_path)
    # Left and Right Joint angles
    for side in ['l', 'r']:
        plot_joint_kinematics(df_sim=df_sim_sol, df_ref=df_kinematics, fontsize=8, figsize=(8, 2), side=side, fig_path=sim_path)

if plot_residual_force_torque_flag:
    plot_residual_force_torque(df_sim=df_sim_sol, fontsize=8, figsize=(12, 5), fig_path=sim_path)

if plot_muscle_activation_flag:
    plot_muscle_activation(df_sim=df_sim_sol, fontsize=8, figsize=(12, 5), fig_path=sim_path)

if plot_osim_grf_flag:
    plot_GRF_osim(df_sim_floor=df_grf_floor, df_sim_chair=df_grf_chair, fontsize=8, figsize=(12, 5), fig_path=sim_path)
    