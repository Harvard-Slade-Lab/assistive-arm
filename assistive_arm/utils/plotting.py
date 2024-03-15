import pandas as pd
import numpy as np

from typing import List
from pathlib import Path

import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt

from scipy.interpolate import griddata
import matplotlib.colors as colors

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

def plot_res_assist_forces(time: pd.Series, dataframes: List[pd.DataFrame], config: dict, output_path: Path=None, figsize: tuple=(12, 5)):
    coords = ['x', 'y']
    
    assist_true = dataframes["assist_true"]
    # assist_false = dataframes["assist_false"]
    grf = dataframes["ground_forces"]

    fig, axs = plt.subplots(len(coords), figsize=figsize, sharex=True)

    fig.suptitle('Residual, ground and assistive forces')

    for i, coord in enumerate(coords):
        axs[i].plot()

        # Pelvis T_coord
        # axs[i].plot(time, assist_false[f'/forceset/reserve_jointset_ground_pelvis_pelvis_t{coord}']*config["reserve_actuator_force"], label=f'Residual {coord.upper()} (unassisted)')
        # axs[i].plot(time, assist_true[f'/forceset/reserve_jointset_ground_pelvis_pelvis_t{coord}']*config["reserve_actuator_force"], label=f'Residual {coord.upper()} (assisted)')
        axs[i].plot(time, assist_true[f"/forceset/assistive_force_{coord}"]*config["actuator_magnitude"], label=f"Assistive Force {coord.upper()}")
        axs[i].plot(grf.time, grf[f'ground_force_l_v{coord}'], label=f'Ground Force {coord.upper()}')
        axs[i].set_title(coord.upper())
        axs[i].grid()
        axs[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    for ax in axs.flat:
        ax.set(ylabel='Force (N)')
    if output_path:
        fig.savefig(output_path, dpi=300)
    plt.show()

def plot_residual_forces(df: pd.DataFrame, config_file: dict):
    tx = 'reserve_pelvis_tx' if config_file["minimal_actuators"] else '/forceset/reserve_jointset_ground_pelvis_pelvis_tx'
    ty = 'reserve_pelvis_ty' if config_file["minimal_actuators"] else '/forceset/reserve_jointset_ground_pelvis_pelvis_ty'
    tz = 'reserve_pelvis_tz' if config_file["minimal_actuators"] else '/forceset/reserve_jointset_ground_pelvis_pelvis_tz'

    plt.plot(df.time, df[tx]*config_file["actuator_magnitude"], label=f'Residual X')
    plt.plot(df.time, df[ty]*config_file["actuator_magnitude"], label=f'Residual Y')
    plt.plot(df.time, df[tz]*config_file["actuator_magnitude"], label=f'Residual Z')
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.title(f"Residual forces ({'assisted' if config_file['assistive_force'] else 'unassisted'})")
    plt.legend()
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

def create_torque_plot(torques, feasible_profiles, l1, l2):
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
    ax1.view_init(elev=25, azim=-45)  # Adjusting the view angle
    ax1.set_xlabel("L1")
    ax1.set_ylabel("L2")
    ax1.set_zlabel(r"$\tau_1$")
    ax1.set_title("Torque 1 Surface")

    # 3D surface plot for Tau 2
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(grid_x, grid_y, grid_z2, cmap=scalarMap.cmap, linewidth=0, antialiased=True)
    ax2.view_init(elev=25, azim=-45)  # Adjusting the view angle
    ax2.set_xlabel("L1")
    ax2.set_ylabel("L2")
    ax2.set_zlabel(r"$\tau_2$")
    ax2.set_title("Torque 2 Surface")

    # Adding color bar
    cbar = fig.colorbar(scalarMap, ax=[ax1, ax2], shrink=0.5, aspect=5)
    cbar.set_label('Torque')

    plt.show()