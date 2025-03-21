import pandas as pd
import numpy as np
from typing import List
from pathlib import Path
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from scipy.interpolate import griddata
import matplotlib.colors as colors
from matplotlib.lines import Line2D
import seaborn as sns

def extract_reserve_columns(df: pd.DataFrame):
    reserve_actuators = [column for column in df.columns if "reserve" in column]
    df_reserve = df[reserve_actuators]
    df_reserve.columns = [column.split("/")[2] for column in df_reserve.columns]

    return df_reserve

# Plot opencap knee coordinates VS mocap knee coordinates
def plot_knee_coordinates(opencap_markers: pd.DataFrame, mocap_markers: pd.DataFrame, subject: str, trial: str, output_path: Path):
    fig, axs = plt.subplots(3, sharex=True, figsize=(10, 5))
    # Set fig title
    fig.suptitle(f"{subject} {trial}")

    for i, coord in zip(range(3), ["X", "Y", "Z"]):
        axs[i].plot(opencap_markers.Time.t, opencap_markers.LKnee[coord], label="opencap")
        axs[i].plot(mocap_markers.Time.t, mocap_markers.Knee[coord], label="mocap")
        axs[i].legend()
        axs[i].set_ylabel(coord)
        axs[i].set_xlabel("Time [s]")
        axs[i].grid()

    fig.savefig(output_path, dpi=300)

def plot_mocap_forces(opencap_markers: pd.DataFrame, mocap_markers: pd.DataFrame, mocap_forces: pd.DataFrame, force_plates: dict, motion_beginning: int=None, output_path: Path=None):
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 10))

    
    # Add 339 NaN values to the left knee values
    # Top subplot with the mocap left knee and opencap left knee
    axs[0].plot(mocap_markers.Time.t, mocap_markers.Knee.Y, label="MoCap Left Knee")
    axs[0].plot(opencap_markers.Time.t, opencap_markers.LKnee.Y, label="OpenCap Left Knee")
    axs[0].plot(opencap_markers.Time.t, opencap_markers["L.PSIS_study"].Y, label="OpenCap L.PSIS")
    axs[0].set_title('Marker position over time')
    axs[0].set_ylabel('Position (X)')
    axs[0].legend()

    if motion_beginning:
        ind_begin = mocap_forces.time.iloc[motion_beginning]
        axs[0].axvline(x=ind_begin, color='grey', linestyle='--')

    for i, coord in enumerate(["y"]): # ["x", "y", "z"]
        axs[i+1].plot(mocap_forces.time, mocap_forces[f"ground_force_l_v{coord}"], label="Left ForcePlate")
        axs[i+1].plot(mocap_forces.time, mocap_forces[f"ground_force_r_v{coord}"], label="Right ForcePlate")
        if motion_beginning:
            ind_begin = mocap_forces.time.iloc[motion_beginning]
            axs[0].axvline(x=ind_begin, color='grey', linestyle='--')
            axs[i+1].axvline(x=ind_begin, color='grey', linestyle='--')
        if "chair" in force_plates.keys():
            axs[i+1].plot(mocap_forces.time, mocap_forces[f"ground_force_chair_v{coord}"], label="Chair ForcePlate")
        axs[i+1].set_title(coord)
        axs[i+1].set_ylabel('Force (N)')
        axs[i+1].legend()
    axs[-1].set_xlabel('Time (s)')

    if output_path:
        fig.savefig(output_path, dpi=300, format='svg')

    plt.show()

def extract_muscle_activations(df: pd.DataFrame, apply_filter: bool=True) -> pd.DataFrame:
    activations = []

    for column in df.columns:
        if "activation" in column:
            activations.append(column)
    
    df_activations = df[activations]
    df_activations.columns = [column.split("/")[2] for column in df_activations.columns]

    if apply_filter:
        df_activations = df_activations.loc[:, abs(df_activations.std(axis=0)) > 0.01]

    return df_activations

def plot_res_assist_forces_grf(time: pd.Series, dataframes: List[pd.DataFrame], config: dict, figsize: tuple=(12, 5), output_path: Path=None):
    coords = ['x', 'y']
    
    assist_true = dataframes["assist_true"]
    assist_false = dataframes["assist_false"]
    grf = dataframes["ground_forces"]

    fig, axs = plt.subplots(len(coords), figsize=figsize, sharex=True)

    fig.suptitle('Residual, ground and assistive forces')

    for i, coord in enumerate(coords):
        axs[i].plot()

        # Pelvis T_coord
        axs[i].plot(assist_false.time, assist_false[f'/forceset/reserve_jointset_ground_pelvis_pelvis_t{coord}']*config["actuator_magnitude"], label=f'Residual {coord.upper()} (unassisted)')
        axs[i].plot(assist_true.time, assist_true[f'/forceset/reserve_jointset_ground_pelvis_pelvis_t{coord}']*config["actuator_magnitude"], label=f'Residual {coord.upper()} (assisted)')
        axs[i].plot(assist_true.time, assist_true[f"/forceset/assistive_force_{coord}"]*config["assistive_force"], label=f"Assistive Force {coord.upper()}")
        axs[i].plot(grf.time, grf[f'ground_force_l_v{coord}'], label=f'Ground Force left foot {coord.upper()}')
        axs[i].plot(grf.time, grf[f'ground_force_chair_v{coord}'], label=f'Ground Force chair {coord.upper()}')
        axs[i].set_title(coord.upper())
        axs[i].grid()
        axs[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    for ax in axs.flat:
        ax.set(ylabel='Force (N)')
    if output_path:
        fig.savefig(output_path, dpi=300)
    plt.show()

def plot_res_assist_forces_grf_osim(time: pd.Series, dataframes: List[pd.DataFrame], config: dict, figsize: tuple=(12, 5), output_path: Path=None):
    coords = ['x', 'y']
    
    if config["assistive_force"]:
        assist = dataframes["assist_true"]
        if config['add_osim_GRF_online']:
            grf_feet = dataframes["grf_osim_feet_true"]
            grf_chair = dataframes["grf_osim_chair_true"]
        if config['add_osim_GRF_precomputed']:
            grf_feet = dataframes["grf_osim_feet_precomputed_true"]
            grf_chair = dataframes["grf_osim_chair_precomputed_true"]
    else:
        assist = dataframes["assist_false"]
        if config['add_osim_GRF_online']:
            grf_feet = dataframes["grf_osim_feet_false"]
            grf_chair = dataframes["grf_osim_chair_false"]
    

    fig, axs = plt.subplots(len(coords), figsize=figsize, sharex=True)
    fig.suptitle(f"Residual, ground and assistive forces ({'assisted' if config['assistive_force'] else 'unassisted'})")

    for i, coord in enumerate(coords):
        axs[i].plot()

        # Pelvis T_coord
        #axs[i].plot(assist_false.time, assist_false[f'/forceset/reserve_jointset_ground_pelvis_pelvis_t{coord}']*config["actuator_magnitude"], label=f'Residual {coord.upper()} (unassisted)')
        #axs[i].plot(assist.time, assist[f'/forceset/reserve_jointset_ground_pelvis_pelvis_t{coord}']*config["actuator_magnitude"], label=f'Residual {coord.upper()} ({'assisted' if config['assistive_force'] else 'unassisted'})')
        if config["assist_with_reserve_pelvis_txy"]:
            axs[i].plot(assist.time, assist[f'/forceset/reserve_jointset_ground_pelvis_pelvis_t{coord}']*config["reserve_pelvis_opt_value"], label=f"Residual {coord.upper()} ({'assisted' if config['assistive_force'] else 'unassisted'})")
            print("Pelvis residual x/y are the assistive force with magnitude: ", config['reserve_pelvis_opt_value'])
        else:
            axs[i].plot(assist.time, assist[f'/forceset/reserve_jointset_ground_pelvis_pelvis_t{coord}']*config["actuator_magnitude"], label=f"Residual {coord.upper()} ({'assisted' if config['assistive_force'] else 'unassisted'})")
        axs[i].plot(grf_feet.time, grf_feet[f'ground_force_l_v{coord}'], label=f'GRF left foot {coord.upper()}')
        axs[i].plot(grf_feet.time, grf_feet[f'ground_force_r_v{coord}'], label=f'GRF right foot {coord.upper()}')
        axs[i].plot(grf_chair.time, grf_chair[f'ground_force_l_v{coord}'], label=f'GRF left chair {coord.upper()}')
        axs[i].plot(grf_chair.time, grf_chair[f'ground_force_r_v{coord}'], label=f'GRF right chair {coord.upper()}')
        if config["assistive_force"]:
            #if coord == 'y':
            axs[i].plot(assist.time, assist[f"/forceset/assistive_force_{coord}"]*config["assistive_force"], label=f"Assistive Force {coord.upper()}")
        if config["assist_with_reserve_pelvis_txy"]:
            axs[i].plot(assist.time, assist[f'/forceset/reserve_jointset_ground_pelvis_pelvis_t{coord}']*config["reserve_pelvis_opt_value"]+grf_feet[f'ground_force_l_v{coord}']+grf_feet[f'ground_force_r_v{coord}']+grf_chair[f'ground_force_l_v{coord}']+grf_chair[f'ground_force_r_v{coord}'], label="Total")
        elif config["assistive_force"]:
        #if config["assistive_force"]:
            #if coord == 'y':
            axs[i].plot(assist.time, assist[f'/forceset/reserve_jointset_ground_pelvis_pelvis_t{coord}']*config["actuator_magnitude"]+grf_feet[f'ground_force_l_v{coord}']+grf_feet[f'ground_force_r_v{coord}']+grf_chair[f'ground_force_l_v{coord}']+grf_chair[f'ground_force_r_v{coord}']+assist[f"/forceset/assistive_force_{coord}"]*config["assistive_force"], label="Total")
        else:
            axs[i].plot(assist.time, assist[f'/forceset/reserve_jointset_ground_pelvis_pelvis_t{coord}']*config["actuator_magnitude"]+grf_feet[f'ground_force_l_v{coord}']+grf_feet[f'ground_force_r_v{coord}']+grf_chair[f'ground_force_l_v{coord}']+grf_chair[f'ground_force_r_v{coord}'], label="Total")
        axs[i].set_title(coord.upper())
        axs[i].grid()
        axs[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    for ax in axs.flat:
        ax.set(ylabel='Force (N)')
    if output_path:
        fig.savefig(output_path, dpi=300)
    plt.show()

def plot_residual_forces_trans(df: pd.DataFrame, config_file: dict):
    tx = '/forceset/reserve_jointset_ground_pelvis_pelvis_tx'
    ty = '/forceset/reserve_jointset_ground_pelvis_pelvis_ty'
    tz = '/forceset/reserve_jointset_ground_pelvis_pelvis_tz'

    plt.plot(df.time, df[tx]*config_file["actuator_magnitude"], label=f'Pelvis tX')
    plt.plot(df.time, df[ty]*config_file["actuator_magnitude"], label=f'Pelvis tY')
    plt.plot(df.time, df[tz]*config_file["actuator_magnitude"], label=f'Pelvis tZ')
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.title(f"Residual forces ({'assisted' if config_file['assistive_force'] else 'unassisted'})")
    plt.legend()
    plt.show()


def plot_residual_torques_rot(df: pd.DataFrame, config_file: dict):
    tilt = '/forceset/reserve_jointset_ground_pelvis_pelvis_tilt'
    list = '/forceset/reserve_jointset_ground_pelvis_pelvis_list'
    rotation = '/forceset/reserve_jointset_ground_pelvis_pelvis_rotation'

    plt.plot(df.time, df[tilt]*config_file["actuator_magnitude"], label=f'Pelvis tilt')
    plt.plot(df.time, df[list]*config_file["actuator_magnitude"], label=f'Pelvis list')
    plt.plot(df.time, df[rotation]*config_file["actuator_magnitude"], label=f'Pelvis rotation')
    plt.xlabel("Time [s]")
    plt.ylabel("Torque [Nm]")
    plt.title(f"Residual torques ({'assisted' if config_file['assistive_force'] else 'unassisted'})")
    plt.legend()
    plt.show()
    
def plot_residual_all(df: pd.DataFrame, config_file: dict):
    tx = '/forceset/reserve_jointset_ground_pelvis_pelvis_tx'
    ty = '/forceset/reserve_jointset_ground_pelvis_pelvis_ty'
    tz = '/forceset/reserve_jointset_ground_pelvis_pelvis_tz'
    
    tilt = '/forceset/reserve_jointset_ground_pelvis_pelvis_tilt'
    list = '/forceset/reserve_jointset_ground_pelvis_pelvis_list'
    rotation = '/forceset/reserve_jointset_ground_pelvis_pelvis_rotation'

    hip_flexion_r = '/forceset/reserve_jointset_hip_r_hip_flexion_r'
    hip_adduction_r = '/forceset/reserve_jointset_hip_r_hip_adduction_r'
    hip_rotation_r  = '/forceset/reserve_jointset_hip_r_hip_rotation_r'
    knee_angle_r = '/forceset/reserve_jointset_walker_knee_r_knee_angle_r'
    ankle_angle_r = '/forceset/reserve_jointset_ankle_r_ankle_angle_r'
    subtalar_angle_r = '/forceset/reserve_jointset_subtalar_r_subtalar_angle_r'
    mtp_angle_r = '/forceset/reserve_jointset_mtp_r_mtp_angle_r'
    
    hip_flexion_l = '/forceset/reserve_jointset_hip_l_hip_flexion_l'
    hip_adduction_l = '/forceset/reserve_jointset_hip_l_hip_adduction_l'
    hip_rotation_l = '/forceset/reserve_jointset_hip_l_hip_rotation_l'
    knee_angle_l = '/forceset/reserve_jointset_walker_knee_l_knee_angle_l'
    ankle_angle_l = '/forceset/reserve_jointset_ankle_l_ankle_angle_l'
    subtalar_angle_l = '/forceset/reserve_jointset_subtalar_l_subtalar_angle_l'
    mtp_angle_l = '/forceset/reserve_jointset_mtp_l_mtp_angle_l'
    
    """
    lumbar_extension = '/forceset/reserve_jointset_back_lumbar_extension'
    lumbar_bending = '/forceset/reserve_jointset_back_lumbar_bending'
    lumbar_rotation = '/forceset/reserve_jointset_back_lumbar_rotation'
    
    arm_flex_r = '/forceset/reserve_jointset_acromial_r_arm_flex_r'
    arm_add_r = '/forceset/reserve_jointset_acromial_r_arm_add_r'
    arm_rot_r = '/forceset/reserve_jointset_acromial_r_arm_rot_r'
    elbow_flex_r = '/forceset/reserve_jointset_elbow_r_elbow_flex_r'
    pro_sup_r = '/forceset/reserve_jointset_radioulnar_r_pro_sup_r'
    
    arm_flex_l = '/forceset/reserve_jointset_acromial_l_arm_flex_l'
    arm_add_l = '/forceset/reserve_jointset_acromial_l_arm_add_l'
    arm_rot_l = '/forceset/reserve_jointset_acromial_l_arm_rot_l'
    elbow_flex_l = '/forceset/reserve_jointset_elbow_l_elbow_flex_l'
    pro_sup_l = '/forceset/reserve_jointset_radioulnar_l_pro_sup_l'
    """

    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax2 = ax1.twinx()

    # Plot forces on the first y-axis (left)
    ax1.plot(df.time, df[tx]*config_file["actuator_magnitude"], label='Pelvis x', color='r')
    ax1.plot(df.time, df[ty]*config_file["actuator_magnitude"], label='Pelvis y', color='g')
    ax1.plot(df.time, df[tz]*config_file["actuator_magnitude"], label='Pelvis z', color='b')
    
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Force [N]", color='blue')
    ax1.tick_params(axis='y', labelcolor='b')

    # Plot torques on the second y-axis (right)
    ax2.plot(df.time, df[tilt]*config_file["actuator_magnitude"], label='Pelvis tilt')
    ax2.plot(df.time, df[list]*config_file["actuator_magnitude"], label='Pelvis list')
    ax2.plot(df.time, df[rotation]*config_file["actuator_magnitude"], label='Pelvis rotation')

    ax2.plot(df.time, df[hip_flexion_r]*config_file["actuator_magnitude"], label='hip_flexion_r')
    ax2.plot(df.time, df[hip_adduction_r]*config_file["actuator_magnitude"], label='hip_adduction_r')
    ax2.plot(df.time, df[hip_rotation_r]*config_file["actuator_magnitude"], label=f'hip_rotation_r')
    ax2.plot(df.time, df[knee_angle_r]*config_file["actuator_magnitude"], label=f'knee_angle_r')
    ax2.plot(df.time, df[ankle_angle_r]*config_file["actuator_magnitude"], label=f'ankle_angle_r')
    ax2.plot(df.time, df[subtalar_angle_r]*config_file["actuator_magnitude"], label=f'subtalar_angle_r')
    ax2.plot(df.time, df[mtp_angle_r]*config_file["actuator_magnitude"], label=f'mtp_angle_r')

    ax2.plot(df.time, df[hip_flexion_l]*config_file["actuator_magnitude"], label=f'hip_flexion_l')
    ax2.plot(df.time, df[hip_adduction_l]*config_file["actuator_magnitude"], label=f'hip_adduction_l')
    ax2.plot(df.time, df[hip_rotation_l]*config_file["actuator_magnitude"], label=f'hip_rotation_l')
    ax2.plot(df.time, df[knee_angle_l]*config_file["actuator_magnitude"], label=f'knee_angle_l')
    ax2.plot(df.time, df[ankle_angle_l]*config_file["actuator_magnitude"], label=f'ankle_angle_l')
    ax2.plot(df.time, df[subtalar_angle_l]*config_file["actuator_magnitude"], label=f'subtalar_angle_l')
    ax2.plot(df.time, df[mtp_angle_l]*config_file["actuator_magnitude"], label=f'mtp_angle_l')

    """
    ax2.plot(df.time, df[lumbar_extension]*config_file["actuator_magnitude"], label=f'lumbar_extension')
    ax2.plot(df.time, df[lumbar_bending]*config_file["actuator_magnitude"], label=f'lumbar_bending')
    ax2.plot(df.time, df[lumbar_rotation]*config_file["actuator_magnitude"], label=f'lumbar_rotation')

    ax2.plot(df.time, df[arm_flex_r]*config_file["actuator_magnitude"], label=f'arm_flex_r')
    ax2.plot(df.time, df[arm_add_r]*config_file["actuator_magnitude"], label=f'arm_add_r')
    ax2.plot(df.time, df[arm_rot_r]*config_file["actuator_magnitude"], label=f'arm_rot_r')
    ax2.plot(df.time, df[elbow_flex_r]*config_file["actuator_magnitude"], label=f'elbow_flex_r')
    ax2.plot(df.time, df[pro_sup_r]*config_file["actuator_magnitude"], label=f'pro_sup_r')

    ax2.plot(df.time, df[arm_flex_l]*config_file["actuator_magnitude"], label=f'arm_flex_l')
    ax2.plot(df.time, df[arm_add_l]*config_file["actuator_magnitude"], label=f'arm_add_l')
    ax2.plot(df.time, df[arm_rot_l]*config_file["actuator_magnitude"], label=f'arm_rot_l')
    ax2.plot(df.time, df[elbow_flex_l]*config_file["actuator_magnitude"], label=f'elbow_flex_l')
    ax2.plot(df.time, df[pro_sup_l]*config_file["actuator_magnitude"], label=f'pro_sup_l')
    """

    ax2.set_ylabel("Torque [Nm]", color='red')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title(f"Residual forces and torques ({'assisted' if config_file['assistive_force'] else 'unassisted'})")

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # Move legend outside the plot
    fig.legend(lines1 + lines2, labels1 + labels2, loc='center left', bbox_to_anchor=(0.85, 0.5))

    # Adjust layout to make room for the legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)  # Adjust this value as needed

    plt.show()

def plot_residual_torques_leg(df: pd.DataFrame, config_file: dict):
    
    hip_flexion_r = '/forceset/reserve_jointset_hip_r_hip_flexion_r'
    hip_adduction_r = '/forceset/reserve_jointset_hip_r_hip_adduction_r'
    hip_rotation_r  = '/forceset/reserve_jointset_hip_r_hip_rotation_r'
    knee_angle_r = '/forceset/reserve_jointset_walker_knee_r_knee_angle_r'
    ankle_angle_r = '/forceset/reserve_jointset_ankle_r_ankle_angle_r'
    subtalar_angle_r = '/forceset/reserve_jointset_subtalar_r_subtalar_angle_r'
    mtp_angle_r = '/forceset/reserve_jointset_mtp_r_mtp_angle_r'
    
    hip_flexion_l = '/forceset/reserve_jointset_hip_l_hip_flexion_l'
    hip_adduction_l = '/forceset/reserve_jointset_hip_l_hip_adduction_l'
    hip_rotation_l = '/forceset/reserve_jointset_hip_l_hip_rotation_l'
    knee_angle_l = '/forceset/reserve_jointset_walker_knee_l_knee_angle_l'
    ankle_angle_l = '/forceset/reserve_jointset_ankle_l_ankle_angle_l'
    subtalar_angle_l = '/forceset/reserve_jointset_subtalar_l_subtalar_angle_l'
    mtp_angle_l = '/forceset/reserve_jointset_mtp_l_mtp_angle_l'

    plt.plot(df.time, df[hip_flexion_r]*config_file["actuator_magnitude"], label=f'hip_flexion_r')
    plt.plot(df.time, df[hip_adduction_r]*config_file["actuator_magnitude"], label=f'hip_adduction_r')
    plt.plot(df.time, df[hip_rotation_r]*config_file["actuator_magnitude"], label=f'hip_rotation_r')
    plt.plot(df.time, df[knee_angle_r]*config_file["actuator_magnitude"], label=f'knee_angle_r')
    plt.plot(df.time, df[ankle_angle_r]*config_file["actuator_magnitude"], label=f'ankle_angle_r')
    plt.plot(df.time, df[subtalar_angle_r]*config_file["actuator_magnitude"], label=f'subtalar_angle_r')
    plt.plot(df.time, df[mtp_angle_r]*config_file["actuator_magnitude"], label=f'mtp_angle_r')

    plt.plot(df.time, df[hip_flexion_l]*config_file["actuator_magnitude"], label=f'hip_flexion_l')
    plt.plot(df.time, df[hip_adduction_l]*config_file["actuator_magnitude"], label=f'hip_adduction_l')
    plt.plot(df.time, df[hip_rotation_l]*config_file["actuator_magnitude"], label=f'hip_rotation_l')
    plt.plot(df.time, df[knee_angle_l]*config_file["actuator_magnitude"], label=f'knee_angle_l')
    plt.plot(df.time, df[ankle_angle_l]*config_file["actuator_magnitude"], label=f'ankle_angle_l')
    plt.plot(df.time, df[subtalar_angle_l]*config_file["actuator_magnitude"], label=f'subtalar_angle_l')
    plt.plot(df.time, df[mtp_angle_l]*config_file["actuator_magnitude"], label=f'mtp_angle_l')

    plt.xlabel("Time [s]")
    plt.ylabel("Torque [Nm]")
    plt.title(f"Residual torques ({'assisted' if config_file['assistive_force'] else 'unassisted'})")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show()

def plot_residual_torques_leg_right(df: pd.DataFrame, config_file: dict):

    hip_flexion_r = '/forceset/reserve_jointset_hip_r_hip_flexion_r'
    hip_adduction_r = '/forceset/reserve_jointset_hip_r_hip_adduction_r'
    hip_rotation_r  = '/forceset/reserve_jointset_hip_r_hip_rotation_r'
    knee_angle_r = '/forceset/reserve_jointset_walker_knee_r_knee_angle_r'
    ankle_angle_r = '/forceset/reserve_jointset_ankle_r_ankle_angle_r'
    subtalar_angle_r = '/forceset/reserve_jointset_subtalar_r_subtalar_angle_r'
    mtp_angle_r = '/forceset/reserve_jointset_mtp_r_mtp_angle_r'

    plt.plot(df.time, df[hip_flexion_r]*config_file["actuator_magnitude"], label=f'hip_flexion_r')
    plt.plot(df.time, df[hip_adduction_r]*config_file["actuator_magnitude"], label=f'hip_adduction_r')
    plt.plot(df.time, df[hip_rotation_r]*config_file["actuator_magnitude"], label=f'hip_rotation_r')
    plt.plot(df.time, df[knee_angle_r]*config_file["actuator_magnitude"], label=f'knee_angle_r')
    plt.plot(df.time, df[ankle_angle_r]*config_file["actuator_magnitude"], label=f'ankle_angle_r')
    plt.plot(df.time, df[subtalar_angle_r]*config_file["actuator_magnitude"], label=f'subtalar_angle_r')
    plt.plot(df.time, df[mtp_angle_r]*config_file["actuator_magnitude"], label=f'mtp_angle_r')

    plt.xlabel("Time [s]")
    plt.ylabel("Torque [Nm]")
    plt.title(f"Residual torques ({'assisted' if config_file['assistive_force'] else 'unassisted'})")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show() 

def plot_residual_torques_leg_left(df: pd.DataFrame, config_file: dict):
    
    hip_flexion_l = '/forceset/reserve_jointset_hip_l_hip_flexion_l'
    hip_adduction_l = '/forceset/reserve_jointset_hip_l_hip_adduction_l'
    hip_rotation_l = '/forceset/reserve_jointset_hip_l_hip_rotation_l'
    knee_angle_l = '/forceset/reserve_jointset_walker_knee_l_knee_angle_l'
    ankle_angle_l = '/forceset/reserve_jointset_ankle_l_ankle_angle_l'
    subtalar_angle_l = '/forceset/reserve_jointset_subtalar_l_subtalar_angle_l'
    mtp_angle_l = '/forceset/reserve_jointset_mtp_l_mtp_angle_l'

    plt.plot(df.time, df[hip_flexion_l]*config_file["actuator_magnitude"], label=f'hip_flexion_l')
    plt.plot(df.time, df[hip_adduction_l]*config_file["actuator_magnitude"], label=f'hip_adduction_l')
    plt.plot(df.time, df[hip_rotation_l]*config_file["actuator_magnitude"], label=f'hip_rotation_l')
    plt.plot(df.time, df[knee_angle_l]*config_file["actuator_magnitude"], label=f'knee_angle_l')
    plt.plot(df.time, df[ankle_angle_l]*config_file["actuator_magnitude"], label=f'ankle_angle_l')
    plt.plot(df.time, df[subtalar_angle_l]*config_file["actuator_magnitude"], label=f'subtalar_angle_l')
    plt.plot(df.time, df[mtp_angle_l]*config_file["actuator_magnitude"], label=f'mtp_angle_l')

    plt.xlabel("Time [s]")
    plt.ylabel("Torque [Nm]")
    plt.title(f"Residual torques ({'assisted' if config_file['assistive_force'] else 'unassisted'})")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show() 

def plot_residual_forces_torques_all(df: pd.DataFrame, config_file: dict, filter_key_infos: bool, alpha=None):
    
    # Create a figure with a 2x2 grid, and split the lower right cell into two
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])

    tx = '/forceset/reserve_jointset_ground_pelvis_pelvis_tx'
    ty = '/forceset/reserve_jointset_ground_pelvis_pelvis_ty'
    tz = '/forceset/reserve_jointset_ground_pelvis_pelvis_tz'
    tilt = '/forceset/reserve_jointset_ground_pelvis_pelvis_tilt'
    list = '/forceset/reserve_jointset_ground_pelvis_pelvis_list'
    rotation = '/forceset/reserve_jointset_ground_pelvis_pelvis_rotation'

    hip_flexion_r = '/forceset/reserve_jointset_hip_r_hip_flexion_r'
    hip_adduction_r = '/forceset/reserve_jointset_hip_r_hip_adduction_r'
    hip_rotation_r  = '/forceset/reserve_jointset_hip_r_hip_rotation_r'
    knee_angle_r = '/forceset/reserve_jointset_walker_knee_r_knee_angle_r'
    ankle_angle_r = '/forceset/reserve_jointset_ankle_r_ankle_angle_r'
    subtalar_angle_r = '/forceset/reserve_jointset_subtalar_r_subtalar_angle_r'
    mtp_angle_r = '/forceset/reserve_jointset_mtp_r_mtp_angle_r'
    
    hip_flexion_l = '/forceset/reserve_jointset_hip_l_hip_flexion_l'
    hip_adduction_l = '/forceset/reserve_jointset_hip_l_hip_adduction_l'
    hip_rotation_l = '/forceset/reserve_jointset_hip_l_hip_rotation_l'
    knee_angle_l = '/forceset/reserve_jointset_walker_knee_l_knee_angle_l'
    ankle_angle_l = '/forceset/reserve_jointset_ankle_l_ankle_angle_l'
    subtalar_angle_l = '/forceset/reserve_jointset_subtalar_l_subtalar_angle_l'
    mtp_angle_l = '/forceset/reserve_jointset_mtp_l_mtp_angle_l'

    # Plot 1: Pelvis trans
    ax1 = fig.add_subplot(gs[0, 0])
    
    if config_file["assist_with_reserve_pelvis_txy"]:
        print("actuator_magnitude pelvis z", config_file["actuator_magnitude"])
        print("actuator_magnitude pelvis x/y", config_file["reserve_pelvis_opt_value"])
        if filter_key_infos == False:
            ax1.plot(df.time, df[tx]*config_file["reserve_pelvis_opt_value"], label='Pelvis x')
            ax1.plot(df.time, df[ty]*config_file["reserve_pelvis_opt_value"], label='Pelvis y')
            ax1.plot(df.time, df[tz]*config_file["actuator_magnitude"], label='Pelvis z')
        else:
            ax1.plot(df.time, df[tx]*config_file["reserve_pelvis_opt_value"], label='Pelvis x')
            ax1.plot(df.time, df[ty]*config_file["reserve_pelvis_opt_value"], label='Pelvis y')
            ax1.plot(df.time, df[tz]*config_file["actuator_magnitude"], label='Pelvis z', alpha=alpha)
    else:
        print("actuator_magnitude pelvis x/y/z", config_file["actuator_magnitude"])
        if filter_key_infos == False:
            ax1.plot(df.time, df[tx]*config_file["actuator_magnitude"], label='Pelvis x')
            ax1.plot(df.time, df[ty]*config_file["actuator_magnitude"], label='Pelvis y')
            ax1.plot(df.time, df[tz]*config_file["actuator_magnitude"], label='Pelvis z')
        else:
            ax1.plot(df.time, df[tx]*config_file["actuator_magnitude"], label='Pelvis x')
            ax1.plot(df.time, df[ty]*config_file["actuator_magnitude"], label='Pelvis y')
            ax1.plot(df.time, df[tz]*config_file["actuator_magnitude"], label='Pelvis z', alpha=alpha)
    
    ax1.set_xlabel("Time [s]", fontsize=16)
    ax1.set_ylabel("Force [N]", fontsize=16)
    ax1.set_title(f"Pelvis Residual Forces", fontsize=18)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='large')

    # Plot 2: Pelvis rot
    ax2 = fig.add_subplot(gs[0, 1])
    print("actuator_magnitude pelvis list/rot/tilt", config_file["actuator_magnitude"])
    if filter_key_infos == False:
        ax2.plot(df.time, df[list]*config_file["actuator_magnitude"], label='Pelvis list')
        ax2.plot(df.time, df[rotation]*config_file["actuator_magnitude"], label='Pelvis rotation')
        ax2.plot(df.time, df[tilt]*config_file["actuator_magnitude"], label='Pelvis tilt')
    else:
        ax2.plot(df.time, df[list]*config_file["actuator_magnitude"], label='Pelvis list', alpha=alpha)
        ax2.plot(df.time, df[rotation]*config_file["actuator_magnitude"], label='Pelvis rotation', alpha=alpha)
        ax2.plot(df.time, df[tilt]*config_file["actuator_magnitude"], label='Pelvis tilt')
    ax2.set_xlabel("Time [s]", fontsize=16)
    ax2.set_ylabel("Torque [Nm]", fontsize=16)
    ax2.set_title(f"Pelvis Residual Torques", fontsize=18)
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='large')

    # Plot 3: Left Leg
    ax3 = fig.add_subplot(gs[1, 0])
    print("actuator_magnitude legs", config_file["actuator_magnitude"])
    if filter_key_infos == False:
        ax3.plot(df.time, df[hip_flexion_l]*config_file["actuator_magnitude"], label=f'hip_flexion')
        ax3.plot(df.time, df[hip_adduction_l]*config_file["actuator_magnitude"], label=f'hip_adduction')
        ax3.plot(df.time, df[hip_rotation_l]*config_file["actuator_magnitude"], label=f'hip_rotation')
        ax3.plot(df.time, df[knee_angle_l]*config_file["actuator_magnitude"], label=f'knee_angle')
        ax3.plot(df.time, df[ankle_angle_l]*config_file["actuator_magnitude"], label=f'ankle_angle')
        ax3.plot(df.time, df[subtalar_angle_l]*config_file["actuator_magnitude"], label=f'subtalar_angle')
        ax3.plot(df.time, df[mtp_angle_l]*config_file["actuator_magnitude"], label=f'mtp_angle')
    else:
        ax3.plot(df.time, df[hip_flexion_l]*config_file["actuator_magnitude"], label=f'hip_flexion')
        ax3.plot(df.time, df[hip_adduction_l]*config_file["actuator_magnitude"], label=f'hip_adduction', alpha=alpha)
        ax3.plot(df.time, df[hip_rotation_l]*config_file["actuator_magnitude"], label=f'hip_rotation', alpha=alpha)
        ax3.plot(df.time, df[knee_angle_l]*config_file["actuator_magnitude"], label=f'knee_angle')
        ax3.plot(df.time, df[ankle_angle_l]*config_file["actuator_magnitude"], label=f'ankle_angle')
        ax3.plot(df.time, df[subtalar_angle_l]*config_file["actuator_magnitude"], label=f'subtalar_angle', alpha=alpha)
        ax3.plot(df.time, df[mtp_angle_l]*config_file["actuator_magnitude"], label=f'mtp_angle', alpha=alpha)
    ax3.set_xlabel("Time [s]", fontsize=16)
    ax3.set_ylabel("Torque [Nm]", fontsize=16)
    ax3.set_title(f"Left Leg Residual Torques", fontsize=18)
    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='large')

    # Plot 4: Right Leg
    ax4 = fig.add_subplot(gs[1, 1])
    if filter_key_infos == False:
        ax4.plot(df.time, df[hip_flexion_r]*config_file["actuator_magnitude"], label='hip_flexion')
        ax4.plot(df.time, df[hip_adduction_r]*config_file["actuator_magnitude"], label='hip_adduction')
        ax4.plot(df.time, df[hip_rotation_r]*config_file["actuator_magnitude"], label=f'hip_rotation')
        ax4.plot(df.time, df[knee_angle_r]*config_file["actuator_magnitude"], label=f'knee_angle')
        ax4.plot(df.time, df[ankle_angle_r]*config_file["actuator_magnitude"], label=f'ankle_angle')
        ax4.plot(df.time, df[subtalar_angle_r]*config_file["actuator_magnitude"], label=f'subtalar_angle')
        ax4.plot(df.time, df[mtp_angle_r]*config_file["actuator_magnitude"], label=f'mtp_angle')
    else:
        ax4.plot(df.time, df[hip_flexion_r]*config_file["actuator_magnitude"], label='hip_flexion')
        ax4.plot(df.time, df[hip_adduction_r]*config_file["actuator_magnitude"], label='hip_adduction', alpha=alpha)
        ax4.plot(df.time, df[hip_rotation_r]*config_file["actuator_magnitude"], label=f'hip_rotation', alpha=alpha)
        ax4.plot(df.time, df[knee_angle_r]*config_file["actuator_magnitude"], label=f'knee_angle')
        ax4.plot(df.time, df[ankle_angle_r]*config_file["actuator_magnitude"], label=f'ankle_angle')
        ax4.plot(df.time, df[subtalar_angle_r]*config_file["actuator_magnitude"], label=f'subtalar_angle', alpha=alpha)
        ax4.plot(df.time, df[mtp_angle_r]*config_file["actuator_magnitude"], label=f'mtp_angle', alpha=alpha)
    ax4.set_xlabel("Time [s]", fontsize=16)
    ax4.set_ylabel("Torque [Nm]", fontsize=16)
    ax4.set_title(f"Right Leg Residual Torques", fontsize=18)
    ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='large')

    # Adjust layout
    plt.tight_layout()
    #fig.subplots_adjust(hspace=0.3)  # Adjust this value as needed to prevent overlap

    # Add a main title
    fig.suptitle(f"Residual Forces and Torques ({'assisted' if config_file['assistive_force'] else 'unassisted'})", fontsize=22, y=1.02)

    # Show the plot
    plt.show()

def plot_coordactuator_torques_arm(df: pd.DataFrame, config_file: dict):

    shoulder_flex_r = '/forceset/shoulder_flex_r'
    shoulder_add_r = '/forceset/shoulder_add_r'
    shoulder_rot_r = '/forceset/shoulder_rot_r'
    elbow_flex_r = '/forceset/elbow_flex_r'
    pro_sup_r = '/forceset/pro_sup_r'
    
    shoulder_flex_l = '/forceset/shoulder_flex_l'
    shoulder_add_l = '/forceset/shoulder_add_l'
    shoulder_rot_l = '/forceset/shoulder_rot_l'
    elbow_flex_l = '/forceset/elbow_flex_l'
    pro_sup_l = '/forceset/pro_sup_l'

    #scale = config_file["coord_actuator_opt_value"]
    scale = 250

    plt.plot(df.time, df[shoulder_flex_r]*scale, label=f'arm_flex_r')
    plt.plot(df.time, df[shoulder_add_r]*scale, label=f'arm_add_r')
    plt.plot(df.time, df[shoulder_rot_r]*scale, label=f'arm_rot_r')
    plt.plot(df.time, df[elbow_flex_r]*scale, label=f'elbow_flex_r')
    plt.plot(df.time, df[pro_sup_r]*scale, label=f'pro_sup_r')

    plt.plot(df.time, df[shoulder_flex_l]*scale, label=f'arm_flex_l')
    plt.plot(df.time, df[shoulder_add_l]*scale, label=f'arm_add_l')
    plt.plot(df.time, df[shoulder_rot_l]*scale, label=f'arm_rot_l')
    plt.plot(df.time, df[elbow_flex_l]*scale, label=f'elbow_flex_l')
    plt.plot(df.time, df[pro_sup_l]*scale, label=f'pro_sup_l')

    plt.xlabel("Time [s]")
    plt.ylabel("Torque [Nm]")
    plt.title(f"Coordinate torques ({'assisted' if config_file['assistive_force'] else 'unassisted'})")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show()

def plot_coordactuator_torques_back(df: pd.DataFrame, config_file: dict):
          
    lumbar_extension = '/forceset/lumbar_ext'
    lumbar_bending = '/forceset/lumbar_bend'
    lumbar_rotation = '/forceset/lumbar_rot'

    scale = config_file["coord_actuator_opt_value"]
    #scale = 250

    if config_file["assistive_force"]:
        #scale = 10
        print("scale ",scale)
        plt.plot(df.time, df[lumbar_extension]*scale, label=f'lumbar_extension')
        plt.plot(df.time, df[lumbar_bending]*scale, label=f'lumbar_bending')
        plt.plot(df.time, df[lumbar_rotation]*scale, label=f'lumbar_rotation')
    else:
        print("scale ",scale)
        plt.plot(df.time, df[lumbar_extension]*scale, label=f'lumbar_extension')
        plt.plot(df.time, df[lumbar_bending]*scale, label=f'lumbar_bending')
        plt.plot(df.time, df[lumbar_rotation]*scale, label=f'lumbar_rotation')

    plt.xlabel("Time [s]")
    plt.ylabel("Torque [Nm]")
    plt.title(f"Coordinate torques ({'assisted' if config_file['assistive_force'] else 'unassisted'})")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show()

def plot_GRF_osim(df: pd.DataFrame, config_file: dict, title: str):
    # Create a figure with a 2x2 grid, and split the lower right cell into two
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
    gs_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 1])

    # Plot 1: Ground reaction forces (upper left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df.time, df['ground_force_r_vx'], label='Right GRF x')
    ax1.plot(df.time, df['ground_force_r_vy'], label='Right GRF y')
    ax1.plot(df.time, df['ground_force_r_vz'], label='Right GRF z')
    ax1.plot(df.time, df['ground_force_l_vx'], label='Left GRF x')
    ax1.plot(df.time, df['ground_force_l_vy'], label='Left GRF y')
    ax1.plot(df.time, df['ground_force_l_vz'], label='Left GRF z')
    ax1.set_xlabel("Time [s]", fontsize=16)
    ax1.set_ylabel("Force [N]", fontsize=16)
    ax1.set_title(f"Ground Forces", fontsize=18)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='large')

    # Plot 2: Ground torques (upper right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df.time, df['ground_torque_r_x'], label='Right GRT x')
    ax2.plot(df.time, df['ground_torque_r_y'], label='Right GRT y')
    ax2.plot(df.time, df['ground_torque_r_z'], label='Right GRT z')
    ax2.plot(df.time, df['ground_torque_l_x'], label='Left GRT x')
    ax2.plot(df.time, df['ground_torque_l_y'], label='Left GRT y')
    ax2.plot(df.time, df['ground_torque_l_z'], label='Left GRT z')
    ax2.set_xlabel("Time [s]", fontsize=16)
    ax2.set_ylabel("Torque [Nm]", fontsize=16)
    ax2.set_title(f"Ground Torques", fontsize=18)
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='large')

    # Plot 3: GRF trajectory (lower left)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(df['ground_force_r_px'], df['ground_force_r_pz'], label='Right GRF')
    ax3.plot(df['ground_force_l_px'], df['ground_force_l_pz'], label='Left GRF')
    # Add markers
    ax3.scatter(df['ground_force_r_px'], df['ground_force_r_pz'], color='blue', s=30, marker='o')
    ax3.scatter(df['ground_force_l_px'], df['ground_force_l_pz'], color='orange', s=30, marker='o')
    ax3.set_xlabel("x [m]", fontsize=16)
    ax3.set_ylabel("z [m]", fontsize=16)
    ax3.set_title(f"GRF Trajectory", fontsize=18)
    ax3.axis('equal')
    #ax3.axis('square')
    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='large')

    # Plot 4: Right GRF positions over time (lower right, top)
    ax4 = fig.add_subplot(gs_right[0])
    ax4.plot(df.time, df['ground_force_r_px'], label='Right x')
    ax4.plot(df.time, df['ground_force_l_px'], label='Left x')
    #ax4.set_xlabel("Time [s]", fontsize=16)
    ax4.set_ylabel("Position [m]", fontsize=16)
    ax4.set_title(f"GRF Positions", fontsize=18)
    ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='large')

    # Plot 5: Left GRF positions over time (lower right, bottom)
    ax5 = fig.add_subplot(gs_right[1])
    ax5.plot(df.time, df['ground_force_r_pz'], label='Right z')
    ax5.plot(df.time, df['ground_force_l_pz'], label='Left z')
    ax5.set_xlabel("Time [s]", fontsize=16)
    ax5.set_ylabel("Position [m]", fontsize=16)
    #ax5.set_title(f"Left GRF Positions", fontsize=18)
    ax5.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='large')

    # Adjust layout
    plt.tight_layout()
    #fig.subplots_adjust(hspace=0.3)  # Adjust this value as needed to prevent overlap

    # Add a main title
    fig.suptitle(title, fontsize=22, y=1.02)

    # Show the plot
    plt.show()

def plot_GRF_osim2(df_feet: pd.DataFrame, df_chair: pd.DataFrame, config_file: dict, title: str):
    # Create a figure with a 2x2 grid, and split the lower right cell into two
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])

    # Plot 1: Ground reaction forces feet (upper left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df_feet.time, df_feet['ground_force_r_vx'], label='Right GRF x')
    ax1.plot(df_feet.time, df_feet['ground_force_r_vy'], label='Right GRF y')
    ax1.plot(df_feet.time, df_feet['ground_force_r_vz'], label='Right GRF z')
    ax1.plot(df_feet.time, df_feet['ground_force_l_vx'], label='Left GRF x')
    ax1.plot(df_feet.time, df_feet['ground_force_l_vy'], label='Left GRF y')
    ax1.plot(df_feet.time, df_feet['ground_force_l_vz'], label='Left GRF z')
    ax1.set_xlabel("Time [s]", fontsize=16)
    ax1.set_ylabel("Force [N]", fontsize=16)
    ax1.set_title(f"Feet Ground Forces", fontsize=18)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='large')

    # Plot 2: Ground torques feet (upper right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df_feet.time, df_feet['ground_torque_r_x'], label='Right GRT x')
    ax2.plot(df_feet.time, df_feet['ground_torque_r_y'], label='Right GRT y')
    ax2.plot(df_feet.time, df_feet['ground_torque_r_z'], label='Right GRT z')
    ax2.plot(df_feet.time, df_feet['ground_torque_l_x'], label='Left GRT x')
    ax2.plot(df_feet.time, df_feet['ground_torque_l_y'], label='Left GRT y')
    ax2.plot(df_feet.time, df_feet['ground_torque_l_z'], label='Left GRT z')
    ax2.set_xlabel("Time [s]", fontsize=16)
    ax2.set_ylabel("Torque [Nm]", fontsize=16)
    ax2.set_title(f"Feet Ground Torques", fontsize=18)
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='large')

    # Plot 3: Ground reaction forces chair (lower left)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(df_chair.time, df_chair['ground_force_r_vx'], label='Right GRF x')
    ax3.plot(df_chair.time, df_chair['ground_force_r_vy'], label='Right GRF y')
    ax3.plot(df_chair.time, df_chair['ground_force_r_vz'], label='Right GRF z')
    ax3.plot(df_chair.time, df_chair['ground_force_l_vx'], label='Left GRF x')
    ax3.plot(df_chair.time, df_chair['ground_force_l_vy'], label='Left GRF y')
    ax3.plot(df_chair.time, df_chair['ground_force_l_vz'], label='Left GRF z')
    ax3.set_xlabel("Time [s]", fontsize=16)
    ax3.set_ylabel("Force [N]", fontsize=16)
    ax3.set_title(f"Chair Ground Forces", fontsize=18)
    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='large')
    

    # Plot 4: Right GRF positions over time (lower right, top)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(df_chair.time, df_chair['ground_torque_r_x'], label='Right GRT x')
    ax4.plot(df_chair.time, df_chair['ground_torque_r_y'], label='Right GRT y')
    ax4.plot(df_chair.time, df_chair['ground_torque_r_z'], label='Right GRT z')
    ax4.plot(df_chair.time, df_chair['ground_torque_l_x'], label='Left GRT x')
    ax4.plot(df_chair.time, df_chair['ground_torque_l_y'], label='Left GRT y')
    ax4.plot(df_chair.time, df_chair['ground_torque_l_z'], label='Left GRT z')
    ax4.set_xlabel("Time [s]", fontsize=16)
    ax4.set_ylabel("Torque [Nm]", fontsize=16)
    ax4.set_title(f"Chair Ground Torques", fontsize=18)
    ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='large')

    # Adjust layout
    plt.tight_layout()
    #fig.subplots_adjust(hspace=0.3)  # Adjust this value as needed to prevent overlap

    # Add a main title
    fig.suptitle(title, fontsize=22, y=1.02)

    # Show the plot
    plt.show()

def plot_GRF_osim2_averaged(df_feet: pd.DataFrame, df_chair: pd.DataFrame, config_file: dict, save:bool, title:str):
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])

    def plot_averaged_forces(ax, df, prefix, title, y_label, is_feet):
        colors = {
            'x': '#c3c3c3' if is_feet else '#e5bd93',  # Light brown / Dark gray
            'y': '#999999' if is_feet else '#b45f06',  # Specified brown / Specified gray
            'z': '#3c3c3c' if is_feet else '#513519'   # Very light brown / Light gray
        }
        for axis in ['x', 'y', 'z']:
            right_force = df[f'{prefix}_r_v{axis}']
            left_force = df[f'{prefix}_l_v{axis}']
            avg_force = (right_force + left_force) / 2
            ax.plot(df.percentage, right_force, color=colors[axis], alpha=0.3)
            ax.plot(df.percentage, left_force, color=colors[axis], alpha=0.3)
            ax.plot(df.percentage, avg_force, label=f'GRF {axis}', linewidth=2.5, color=colors[axis])
        ax.set_ylim(-100, 500) if is_feet else ax.set_ylim(-50, 350)
        ax.set_xlabel("Sit-to-Stand Phase (%)")
        ax.set_ylabel(y_label)
        ax.set_title(title, fontsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)

    def plot_averaged_torques(ax, df, prefix, title, y_label, is_feet):
        colors = {
            'x': '#c3c3c3' if is_feet else '#e5bd93',  # Dark brown / Dark gray
            'y': '#999999' if is_feet else '#b45f06',  # Specified brown / Specified gray
            'z': '#3c3c3c' if is_feet else '#513519'   # Light brown / Light gray
        }
        for axis in ['x', 'y', 'z']:
            right_torque = df[f'{prefix}_r_{axis}']
            left_torque = df[f'{prefix}_l_{axis}']
            avg_torque = (right_torque + left_torque) / 2
            ax.plot(df.percentage, right_torque, color=colors[axis], alpha=0.3)
            ax.plot(df.percentage, left_torque, color=colors[axis], alpha=0.3)
            ax.plot(df.percentage, avg_torque, label=f'GRT {axis}', linewidth=2.5, color=colors[axis])
        ax.set_ylim(-5, 28) if is_feet else ax.set_ylim(-30, 10)
        ax.set_xlabel("Sit-to-Stand Phase (%)")
        ax.set_ylabel(y_label)
        ax.set_title(title, fontsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)

    # Plot 1: Average ground reaction forces feet (upper left)
    ax1 = fig.add_subplot(gs[0, 0])
    plot_averaged_forces(ax1, df_feet, 'ground_force', "Average Feet Ground Reaction Forces", "Force (N)", True)

    # Plot 2: Average ground torques feet (upper right)
    ax2 = fig.add_subplot(gs[0, 1])
    plot_averaged_torques(ax2, df_feet, 'ground_torque', "Average Feet Ground Reaction Torques", "Torque (Nm)", True)

    # Plot 3: Average ground reaction forces chair (lower left)
    ax3 = fig.add_subplot(gs[1, 0])
    plot_averaged_forces(ax3, df_chair, 'ground_force', "Average Chair Ground Reaction Forces", "Force (N)", False)

    # Plot 4: Average ground torques chair (lower right)
    ax4 = fig.add_subplot(gs[1, 1])
    plot_averaged_torques(ax4, df_chair, 'ground_torque', "Average Chair Ground Reaction Torques", "Torque (Nm)", False)

    plt.tight_layout()
    fig.suptitle(f"Simulated Ground Reaction Forces and Torques ({'Assisted Case' if config_file['assistive_force'] else 'Unassisted Case'})", fontsize=16, y=1.02)
    if save:
        plt.savefig(f"/Users/camilleguillaume/Documents/MasterThesis/figures_rapport/{title}", bbox_inches="tight")
    plt.show()

def plot_GRF_osim3(df_feet_before: pd.DataFrame, df_chair_before: pd.DataFrame, df_feet: pd.DataFrame, df_chair: pd.DataFrame, config_file: dict, title: str):
    # Create a figure with a 2x2 grid, and split the lower right cell into two
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])

    force_data = [
        ('Right GRF x', 'ground_force_r_vx'),
        ('Right GRF y', 'ground_force_r_vy'),
        ('Right GRF z', 'ground_force_r_vz'),
        ('Left GRF x', 'ground_force_l_vx'),
        ('Left GRF y', 'ground_force_l_vy'),
        ('Left GRF z', 'ground_force_l_vz')
    ]
    torque_data = [
        ('Right GRT x', 'ground_torque_r_x'),
        ('Right GRT y', 'ground_torque_r_y'),
        ('Right GRT z', 'ground_torque_r_z'),
        ('Left GRT x', 'ground_torque_l_x'),
        ('Left GRT y', 'ground_torque_l_y'),
        ('Left GRT z', 'ground_torque_l_z')
    ]

    # Plot 1: Ground reaction forces feet (upper left)
    ax1 = fig.add_subplot(gs[0, 0])
    for i, (label, column) in enumerate(force_data):
        ax1.plot(df_feet.time, df_feet[column], label=label, color=f'C{i}')
        ax1.plot(df_feet_before.time, df_feet_before[column], color=f'C{i}', linestyle='--')
    ax1.set_xlabel("Time [s]", fontsize=16)
    ax1.set_ylabel("Force [N]", fontsize=16)
    ax1.set_title(f"Feet Ground Forces", fontsize=18)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='large')

    # Plot 2: Ground torques feet (upper right)
    ax2 = fig.add_subplot(gs[0, 1])
    for i, (label, column) in enumerate(torque_data):
        ax2.plot(df_feet.time, df_feet[column], label=label, color=f'C{i}')
        ax2.plot(df_feet_before.time, df_feet_before[column], color=f'C{i}', linestyle='--')
    ax2.set_xlabel("Time [s]", fontsize=16)
    ax2.set_ylabel("Torque [Nm]", fontsize=16)
    ax2.set_title(f"Feet Ground Torques", fontsize=18)
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='large')

    # Plot 3: Ground reaction forces chair (lower left)
    ax3 = fig.add_subplot(gs[1, 0])
    for i, (label, column) in enumerate(force_data):
        ax3.plot(df_chair.time, df_chair[column], label=label, color=f'C{i}')
        ax3.plot(df_chair_before.time, df_chair_before[column], color=f'C{i}', linestyle='--')
    ax3.set_xlabel("Time [s]", fontsize=16)
    ax3.set_ylabel("Force [N]", fontsize=16)
    ax3.set_title(f"Chair Ground Forces", fontsize=18)
    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='large')
    
    # Plot 4: Right GRF positions over time (lower right, top)
    ax4 = fig.add_subplot(gs[1, 1])
    for i, (label, column) in enumerate(torque_data):
        ax4.plot(df_chair.time, df_chair[column], label=label, color=f'C{i}')
        ax4.plot(df_chair_before.time, df_chair_before[column], color=f'C{i}', linestyle='--')
    ax4.set_xlabel("Time [s]", fontsize=16)
    ax4.set_ylabel("Torque [Nm]", fontsize=16)
    ax4.set_title(f"Chair Ground Torques", fontsize=18)
    ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='large')

    # Create custom legend elements for assistance
    custom_lines = [Line2D([0], [0], color='gray', lw=2),
                    Line2D([0], [0], color='gray', lw=2, linestyle='--')]
    # Add the assistance legend to the figure
    fig.legend(custom_lines, ['With assistance', 'Without assistance'], loc='lower center', ncol=2, fontsize='large', title='Assistance')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Adjust the bottom margin to make room for the assistance legend
    fig.suptitle(title, fontsize=22, y=1.02)
    plt.show()

def plot_GRF_osim4(df_feet_before: pd.DataFrame, df_chair_before: pd.DataFrame, df_feet: pd.DataFrame, df_chair: pd.DataFrame, config_file: dict, title: str):
    # Create a figure with a 2x2 grid, and split the lower right cell into two
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])

    force_data = [
        ('GRF x', ('ground_force_r_vx', 'ground_force_l_vx')),
        ('GRF y', ('ground_force_r_vy', 'ground_force_l_vy')),
        ('GRF z', ('ground_force_r_vz', 'ground_force_l_vz'))
    ]
    torque_data = [
        ('GRT x', ('ground_torque_r_x', 'ground_torque_l_x')),
        ('GRT y', ('ground_torque_r_y', 'ground_torque_l_y')),
        ('GRT z', ('ground_torque_r_z', 'ground_torque_l_z'))
    ]

    # Plot 1: Ground reaction forces feet (upper left)
    ax1 = fig.add_subplot(gs[0, 0])
    for i, (label, (right_col, left_col)) in enumerate(force_data):
        # Calculate average for "with assistance" data
        avg_with = (df_feet[right_col] + df_feet[left_col]) / 2
        ax1.plot(df_feet.time, avg_with, label=label, color=f'C{i}')
        # Calculate average for "without assistance" data
        avg_without = (df_feet_before[right_col] + df_feet_before[left_col]) / 2
        ax1.plot(df_feet_before.time, avg_without, color=f'C{i}', linestyle='--')
    ax1.set_xlabel("Time [s]", fontsize=16)
    ax1.set_ylabel("Force [N]", fontsize=16)
    ax1.set_title(f"Average Feet Ground Forces", fontsize=18)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='large', title='Coordinates')

    # Plot 2: Ground torques feet (upper right)
    ax2 = fig.add_subplot(gs[0, 1])
    lines = []
    for i, (label, (right_col, left_col)) in enumerate(torque_data):
        # Calculate average for "with assistance" data
        avg_with = (df_feet[right_col] + df_feet[left_col]) / 2
        line, = ax2.plot(df_feet.time, avg_with, label=label, color=f'C{i}')
        lines.append(line)
        # Calculate average for "without assistance" data
        avg_without = (df_feet_before[right_col] + df_feet_before[left_col]) / 2
        ax2.plot(df_feet_before.time, avg_without, color=f'C{i}', linestyle='--')

    ax2.set_xlabel("Time [s]", fontsize=16)
    ax2.set_ylabel("Force [N]", fontsize=16)
    ax2.set_title(f"Average Feet Ground Torques", fontsize=18)
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='large', title='Coordinates')

    # Plot 3: Ground reaction forces chair (lower left)
    ax3 = fig.add_subplot(gs[1, 0])
    for i, (label, (right_col, left_col)) in enumerate(force_data):
        # Calculate average for "after" data
        avg_after = (df_chair[right_col] + df_chair[left_col]) / 2
        ax3.plot(df_chair.time, avg_after, label=label, color=f'C{i}')
        # Calculate average for "before" data
        avg_before = (df_chair_before[right_col] + df_chair_before[left_col]) / 2
        ax3.plot(df_chair_before.time, avg_before, color=f'C{i}', linestyle='--')
    ax3.set_xlabel("Time [s]", fontsize=16)
    ax3.set_ylabel("Force [N]", fontsize=16)
    ax3.set_title(f"Average Chair Ground Forces", fontsize=18)
    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='large', title='Coordinates')
    
    # Plot 4: Right GRF positions over time (lower right, top)
    ax4 = fig.add_subplot(gs[1, 1])
    for i, (label, (right_col, left_col)) in enumerate(torque_data):
        # Calculate average for "after" data
        avg_after = (df_chair[right_col] + df_chair[left_col]) / 2
        ax4.plot(df_chair.time, avg_after, label=label, color=f'C{i}')
        # Calculate average for "before" data
        avg_before = (df_chair_before[right_col] + df_chair_before[left_col]) / 2
        ax4.plot(df_chair_before.time, avg_before, color=f'C{i}', linestyle='--')
    ax4.set_xlabel("Time [s]", fontsize=16)
    ax4.set_ylabel("Torque [Nm]", fontsize=16)
    ax4.set_title(f"Average Chair Ground Torques", fontsize=18)
    ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='large', title='Coordinates')

    # Create custom legend elements for assistance
    custom_lines = [Line2D([0], [0], color='gray', lw=2),
                    Line2D([0], [0], color='gray', lw=2, linestyle='--')]
    # Add the assistance legend to the figure
    fig.legend(custom_lines, ['With assistance', 'Without assistance'], loc='lower center', ncol=2, fontsize='large', title='Assistance')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Adjust the bottom margin to make room for the assistance legend
    fig.suptitle(title, fontsize=22, y=1.02)
    plt.show()


def extract_muscle_activations(df: pd.DataFrame, apply_filter: bool=True) -> pd.DataFrame:
    activations = []

    for column in df.columns:
        if "activation" in column:
            activations.append(column)
    
    df_activations = df[activations]
    df_activations.columns = [column.split("/")[2] for column in df_activations.columns]

    if apply_filter:
        df_activations = df_activations.loc[:, abs(df_activations.std(axis=0)) > 0.01]

    return df_activations

def plot_muscles_activations(df: pd.DataFrame, config_file: dict):
    fig, axs = plt.subplots(figsize=(8, 4), sharex=True)

    fig.suptitle(f"STS OpenSim muscles activations ({'assisted' if config_file['assistive_force'] else 'unassisted'})")
    axs.plot(df, label=df.columns)
    axs.set_xlabel("Time (s)")
    axs.set_ylabel("Activation (%)") # percentage
    axs.set_ylim(0, 1)
    axs.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, symbol=None))
    handles, labels = axs.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.95, 0.5), ncol=3, fontsize=10)
    plt.tight_layout()
    #plt.savefig("/Users/camilleguillaume/Documents/MasterThesis/figures/opensim/osim_activations.png", dpi=500, bbox_inches='tight')
    plt.show()

def plot_muscles_group_activations(df: pd.DataFrame, muscles_set1, muscles_set2, muscles_set3, muscles_set4, config_file: dict):

    fig, axs = plt.subplots(2, 2, figsize=(16, 8), sharex=True)
    fig.suptitle(f"STS OpenSim muscles groups activations ({'assisted' if config_file['assistive_force'] else 'unassisted'})", fontsize=16)

    # Subplot 1: Anterior (quadriceps)
    axs[0, 0].plot(df[muscles_set1], label=df[muscles_set1].columns)
    axs[0, 0].set_title("Anterior (quadriceps)", fontsize=14)
    axs[0, 0].set_xlabel("Time (s)")
    axs[0, 0].set_ylabel("Activation (%)")
    axs[0, 0].set_ylim(0, 1)
    axs[0, 0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, symbol=None))
    handles, labels = axs[0, 0].get_legend_handles_labels()
    axs[0, 0].legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

    # Subplot 2: Anterior (psoas)
    axs[0, 1].plot(df[muscles_set2], label=df[muscles_set2].columns)
    axs[0, 1].set_title("Anterior (psoas)", fontsize=14)
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].set_ylabel("Activation (%)")
    axs[0, 1].set_ylim(0, 1)
    axs[0, 1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, symbol=None))
    handles, labels = axs[0, 1].get_legend_handles_labels()
    axs[0, 1].legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

    # Subplot 3: Posterior (hamstring, gluteal)
    axs[1, 0].plot(df[muscles_set3], label=df[muscles_set3].columns)
    axs[1, 0].set_title("Posterior (hamstring, gluteal)", fontsize=14)
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("Activation (%)")
    axs[1, 0].set_ylim(0, 1)
    axs[1, 0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, symbol=None))
    handles, labels = axs[1, 0].get_legend_handles_labels()
    axs[1, 0].legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

    # Subplot 4: Posterior (calf)
    axs[1, 1].plot(df[muscles_set4], label=df[muscles_set4].columns)
    axs[1, 1].set_title("Posterior (calf) and anterior (tibia)", fontsize=14)
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("Activation (%)")
    axs[1, 1].set_ylim(0, 1)
    axs[1, 1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, symbol=None))
    handles, labels = axs[1, 1].get_legend_handles_labels()
    axs[1, 1].legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

    plt.tight_layout()
    plt.show()

def plot_muscles_group_activations2(df: pd.DataFrame, muscles_set1, muscles_set2, muscles_set3, muscles_set4, config_file: dict, motion_percentage):
    # Create custom palettes for each muscle set
    palette_set1 = sns.color_palette("Reds", n_colors=len(muscles_set1))
    palette_set2 = sns.color_palette("YlOrBr", n_colors=len(muscles_set2))
    palette_set3 = sns.color_palette("crest", n_colors=len(muscles_set3))
    palette_set4 = sns.color_palette("ch:s=-.2,r=.6", n_colors=len(muscles_set4))

    fig, axs = plt.subplots(2, 2, figsize=(16, 8), sharex=False)
    fig.suptitle(f"Simulation Muscles Activations ({'Assisted' if config_file['assistive_force'] else 'Unassisted'})", fontsize=16)

    # Subplot 1: Anterior (quadriceps)
    for muscle, color in zip(muscles_set1, palette_set1):
        axs[0, 0].plot(motion_percentage, df[muscle], label=muscle, color=color)
    axs[0, 0].set_title("Anterior (quadriceps)", fontsize=14)
    axs[0, 0].set_xlabel("Sit-to-Stand Phase (%)")
    axs[0, 0].set_ylabel("Activation (%)")
    axs[0, 0].set_ylim(0, 1)
    axs[0, 0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, symbol=None))
    handles, labels = axs[0, 0].get_legend_handles_labels()
    axs[0, 0].legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    axs[0, 0].spines['top'].set_visible(False)
    axs[0, 0].spines['right'].set_visible(False)

    # Subplot 2: Anterior (psoas)
    for muscle, color in zip(muscles_set2, palette_set2):
        axs[0, 1].plot(motion_percentage, df[muscle], label=muscle, color=color)
    axs[0, 1].set_title("Anterior (psoas)", fontsize=14)
    axs[0, 1].set_xlabel("Sit-to-Stand Phase (%)")
    axs[0, 1].set_ylabel("Activation (%)")
    axs[0, 1].set_ylim(0, 1)
    axs[0, 1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, symbol=None))
    handles, labels = axs[0, 1].get_legend_handles_labels()
    axs[0, 1].legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    axs[0, 1].spines['top'].set_visible(False)
    axs[0, 1].spines['right'].set_visible(False)


    # Subplot 3: Posterior (hamstring, gluteal)
    for muscle, color in zip(muscles_set3, palette_set3):
        axs[1, 0].plot(motion_percentage, df[muscle], label=muscle, color=color)
    axs[1, 0].set_title("Posterior (hamstring, gluteal)", fontsize=14)
    axs[1, 0].set_xlabel("Sit-to-Stand Phase (%)")
    axs[1, 0].set_ylabel("Activation (%)")
    axs[1, 0].set_ylim(0, 1)
    axs[1, 0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, symbol=None))
    handles, labels = axs[1, 0].get_legend_handles_labels()
    axs[1, 0].legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    axs[1, 0].spines['top'].set_visible(False)
    axs[1, 0].spines['right'].set_visible(False)

    # Subplot 4: Posterior (calf)
    for muscle, color in zip(muscles_set4, palette_set4):
        axs[1, 1].plot(motion_percentage, df[muscle], label=muscle, color=color)
    axs[1, 1].set_title("Posterior (calf) and anterior (tibia)", fontsize=14)
    axs[1, 1].set_xlabel("Sit-to-Stand Phase (%)")
    axs[1, 1].set_ylabel("Activation (%)")
    axs[1, 1].set_ylim(0, 1)
    axs[1, 1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, symbol=None))
    handles, labels = axs[1, 1].get_legend_handles_labels()
    axs[1, 1].legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    axs[1, 1].spines['top'].set_visible(False)
    axs[1, 1].spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

def plot_muscles_group_activations_mean2(df: pd.DataFrame, muscles_set1, muscles_set2, muscles_set3, muscles_set4, config_file: dict, motion_percentage, save:bool, title:str):
    # Create custom palettes for each muscle set
    palette_set1 = sns.color_palette("Reds", n_colors=len(muscles_set1)//2)
    palette_set2 = sns.color_palette("YlOrBr", n_colors=len(muscles_set2)//2)
    palette_set3 = sns.color_palette("crest", n_colors=len(muscles_set3)//2)
    palette_set4 = sns.color_palette("ch:s=-.2,r=.6", n_colors=len(muscles_set4)//2)

    fig, axs = plt.subplots(2, 2, figsize=(16, 8), sharex=False)
    fig.suptitle(f"Simulated Muscle Activations ({'Assisted Case' if config_file['assistive_force'] else 'Unassisted Case'})", fontsize=16)

    def plot_mean_activation(ax, muscles, palette, title):
        for i in range(0, len(muscles), 2):
            muscle_name = muscles[i].rsplit('_', 1)[0]  # Remove '_r' or '_l' suffix
            mean_activation = (df[muscles[i]] + df[muscles[i+1]]) / 2
            ax.plot(motion_percentage, df[muscles[i]], color=palette[i//2], alpha=0.3)
            ax.plot(motion_percentage, df[muscles[i+1]], color=palette[i//2], alpha=0.3)
            ax.plot(motion_percentage, mean_activation, label=muscle_name, color=palette[i//2], linewidth=2.5, alpha=0.8)
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Sit-to-Stand Phase (%)")
        ax.set_ylabel("Activation (%)")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, symbol=None))
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Subplot 1: Anterior (quadriceps)
    plot_mean_activation(axs[0, 0], muscles_set1, palette_set1, "Anterior (quadriceps)")

    # Subplot 2: Anterior (psoas)
    plot_mean_activation(axs[0, 1], muscles_set2, palette_set2, "Anterior (psoas)")

    # Subplot 3: Posterior (hamstring, gluteal)
    plot_mean_activation(axs[1, 0], muscles_set3, palette_set3, "Posterior (hamstring, gluteal)")

    # Subplot 4: Posterior (calf)
    plot_mean_activation(axs[1, 1], muscles_set4, palette_set4, "Posterior (calf) and anterior (tibia)")

    plt.tight_layout()
    if save:
        plt.savefig(f'/Users/camilleguillaume/Documents/MasterThesis/figures_rapport/{title}', bbox_inches="tight")
    plt.show()
  

def plot_muscles_group_activation_differences(df_assisted: pd.DataFrame, df_unassisted: pd.DataFrame, 
                                              muscles_set1, muscles_set2, muscles_set3, muscles_set4):
    fig, axs = plt.subplots(2, 2, figsize=(16, 8), sharex=True)
    fig.suptitle("Differences in STS OpenSim muscles groups activations (Assisted - Unassisted)", fontsize=16)

    muscle_sets = [muscles_set1, muscles_set2, muscles_set3, muscles_set4]
    titles = ["Anterior (quadriceps)", "Anterior (psoas)", "Posterior (hamstring, gluteal)", "Posterior (calf) and anterior (tibia)"]
    
    for i, (ax, muscles, title) in enumerate(zip(axs.flatten(), muscle_sets, titles)):
        for muscle in muscles:
            difference = df_assisted[muscle] - df_unassisted[muscle]
            ax.plot(df_assisted.index, difference, label=muscle)
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Sit-to-Stand Phase (%)")
        ax.set_ylabel("Activation difference")
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)  # Add a horizontal line at y=0
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
        
        # Set y-axis limits symmetrically
        y_max = max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]))
        ax.set_ylim(-y_max, y_max)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, symbol='%'))

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.ticker as mticker

def plot_muscles_group_activation_differences2(df_assisted: pd.DataFrame, df_unassisted: pd.DataFrame,
                                              muscles_set1, muscles_set2, muscles_set3, muscles_set4):
    # Create custom palettes for each muscle set
    palette_set1 = sns.color_palette("Reds", n_colors=len(muscles_set1))
    palette_set2 = sns.color_palette("YlOrBr", n_colors=len(muscles_set2))
    palette_set3 = sns.color_palette("crest", n_colors=len(muscles_set3))
    palette_set4 = sns.color_palette("ch:s=-.2,r=.6", n_colors=len(muscles_set4))

    fig, axs = plt.subplots(2, 2, figsize=(16, 8), sharex=True)
    fig.suptitle("Differences in STS OpenSim muscles groups activations (Assisted - Unassisted)", fontsize=16)

    muscle_sets = [muscles_set1, muscles_set2, muscles_set3, muscles_set4]
    titles = ["Anterior (quadriceps)", "Anterior (psoas)", "Posterior (hamstring, gluteal)", "Posterior (calf) and anterior (tibia)"]
    palettes = [palette_set1, palette_set2, palette_set3, palette_set4]

    for i, (ax, muscles, title, palette) in enumerate(zip(axs.flatten(), muscle_sets, titles, palettes)):
        for muscle, color in zip(muscles, palette):
            difference = df_assisted[muscle] - df_unassisted[muscle]
            ax.plot(df_assisted.index, difference, label=muscle, color=color)
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Sit-to-Stand Phase (%)")
        ax.set_ylabel("Activation difference")
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)  # Add a horizontal line at y=0
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
        
        # Set y-axis limits symmetrically
        y_max = max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]))
        ax.set_ylim(-y_max, y_max)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, symbol='%'))

    plt.tight_layout()
    plt.show()

# def create_torque_plot(torques, feasible_profiles, l1, l2):
#     # Normalizing color map based on peak torque values
#     cNorm = colors.Normalize(vmin=np.min([torques.tau_1.min(), torques.tau_2.min()]), 
#                              vmax=np.max([torques.tau_1.max(), torques.tau_2.max()]))
#     scalarMap = cmx.ScalarMappable(norm=cNorm, cmap="YlOrRd")

#     # Highlight point and label
#     highlight = (l1, l2)
#     highlight_label = f"({l1:.2f}, {l2:.2f})"

#     # Creating subplots
#     fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
#     fig.suptitle(f"L1: {l1:.2f}, L2: {l2:.2f}", x=0.45)

#     # First subplot for Tau 1
#     sc1 = axs[0].scatter(feasible_profiles.l1, feasible_profiles.l2, c=scalarMap.to_rgba(torques.tau_1))
#     axs[0].scatter(*highlight)
#     axs[0].axvline(x=highlight[0], linestyle="--", color='grey')
#     axs[0].axhline(y=highlight[1], linestyle="--", color='grey')
#     axs[0].set_xlabel("L1")
#     axs[0].set_ylabel("L2")
#     axs[0].set_title(r"$\tau_1$")

#     # Second subplot for Tau 2
#     sc2 = axs[1].scatter(feasible_profiles.l1, feasible_profiles.l2, c=scalarMap.to_rgba(torques.tau_2))
#     axs[1].scatter(*highlight)
#     axs[1].axvline(x=highlight[0], linestyle="--", color='grey')
#     axs[1].axhline(y=highlight[1], linestyle="--", color='grey')
#     axs[1].set_xlabel("L1")
#     axs[1].set_title(r"$\tau_2$")

#     # Adding color bar
#     fig.colorbar(scalarMap, ax=axs.ravel().tolist())
#     plt.show()
#     plt.savefig("torque_profiles.svg", dpi=500, format="svg")

def create_torque_plot(torques, feasible_profiles, l1, l2, fig_path: Path=None):
    # Grid points for interpolation
    grid_x, grid_y = np.mgrid[min(feasible_profiles.l1):max(feasible_profiles.l1):200j, 
                              min(feasible_profiles.l2):max(feasible_profiles.l2):200j]

    # Interpolate torque values
    grid_z1 = griddata((feasible_profiles.l1, feasible_profiles.l2), torques.tau_1, (grid_x, grid_y), method='cubic')
    grid_z2 = griddata((feasible_profiles.l1, feasible_profiles.l2), torques.tau_2, (grid_x, grid_y), method='cubic')

    # Normalizing color map based on peak torque values
    cNorm = colors.Normalize(vmin=min(torques.tau_1.min(), torques.tau_2.min()), 
                             vmax=max(torques.tau_1.max(), torques.tau_2.max()))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap="YlOrRd")

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(f"L1: {l1:.2f}, L2: {l2:.2f}")

    # 3D surface plot for Tau 1
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(grid_x, grid_y, grid_z1, cmap=scalarMap.cmap, linewidth=0, antialiased=True)
    ax1.view_init(elev=20, azim=-15)  # Adjusting the view angle
    ax1.set_xlabel("L1 (m)")
    ax1.set_ylabel("L2 (m)")
    ax1.set_zlabel(r"$\tau_1$")
    ax1.set_title("Torque 1 Surface")

    # 3D surface plot for Tau 2
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(grid_x, grid_y, grid_z2, cmap=scalarMap.cmap, linewidth=10, antialiased=True)
    ax2.view_init(elev=10, azim=30)  # Adjusting the view angle
    ax2.set_xlabel("L1 (m)")
    ax2.set_ylabel("L2 (m)")
    ax2.set_zlabel(r"$\tau_2$")
    ax2.set_title("Torque 2 Surface")

    # Adding color bar
    cbar = fig.colorbar(scalarMap, ax=[ax1, ax2], shrink=0.5, aspect=5)
    cbar.set_label('Torque')

    if fig_path:
        fig.savefig(fig_path, dpi=500, format='png')

    plt.show()
