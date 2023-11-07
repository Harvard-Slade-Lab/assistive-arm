import pandas as pd

from typing import List
from pathlib import Path

import matplotlib.pyplot as plt

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

def plot_mocap_forces(opencap_markers: pd.DataFrame, mocap_markers: pd.DataFrame, mocap_forces: pd.DataFrame, force_plates: dict, output_path: Path=None):
    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(10, 10))
    # Add 339 NaN values to the left knee values
    # Top subplot with the mocap left knee and opencap left knee
    axs[0].plot(mocap_markers.Time.t, mocap_markers.Knee.Y, label="mocap left knee")
    axs[0].plot(opencap_markers.Time.t, opencap_markers.LKnee.Y, label="opencap left knee")
    axs[0].plot(opencap_markers.Time.t, opencap_markers["L.PSIS_study"].Y, label="opencap L.PSIS")
    axs[0].set_title('Marker position over time')
    axs[0].set_ylabel('Position (X)')
    axs[0].legend()
    axs[0].grid(True)

    for i, coord in enumerate(["x", "y", "z"]):
        axs[i+1].plot(mocap_forces.time, mocap_forces[f"ground_force_l_v{coord}"], label="mocap left force")
        axs[i+1].plot(mocap_forces.time, mocap_forces[f"ground_force_r_v{coord}"], label="mocap right force")
        if "chair" in force_plates.keys():
            axs[i+1].plot(mocap_forces.time, mocap_forces[f"ground_force_chair_v{coord}"], label="mocap chair force")
        axs[i+1].set_title(coord)
        axs[i+1].set_ylabel('Force (N)')
        axs[i+1].grid(True)
        axs[i+1].legend()
    axs[-1].set_xlabel('Time (s)')

    if output_path:
        fig.savefig(output_path, dpi=300)

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
    assist_false = dataframes["assist_false"]
    grf = dataframes["ground_forces"]

    fig, axs = plt.subplots(len(coords), figsize=figsize, sharex=True)

    fig.suptitle('Residual, ground and assistive forces')

    for i, coord in enumerate(coords):
        axs[i].plot()

        # Pelvis T_coord
        axs[i].plot(time, assist_false[f'/forceset/reserve_jointset_ground_pelvis_pelvis_t{coord}']*config["reserve_actuator_force"], label=f'Residual {coord.upper()} (unassisted)')
        # axs[i].plot(time, assist_true[f'/forceset/reserve_jointset_ground_pelvis_pelvis_t{coord}']*config["reserve_actuator_force"], label=f'Residual {coord.upper()} (assisted)')
        axs[i].plot(time, assist_true[f"/forceset/assistive_force_{coord}"]*config["assistive_force_magnitude"], label=f"Assistive Force {coord.upper()}")
        axs[i].plot(grf.time, grf[f'ground_force_l_v{coord}'], label=f'Ground Force {coord.upper()}')
        axs[i].set_title(coord.upper())
        axs[i].grid()
        axs[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    for ax in axs.flat:
        ax.set(ylabel='Force (N)')
    if output_path:
        fig.savefig(output_path, dpi=300)
    plt.show()


def plot_residual_forces(df: pd.DataFrame, config_file: dict, output_path: Path=None) -> None:
    # Extract columns corresponding to reserve actuators
    reserve_columns = [col for col in df.columns if "reserve" in col]
    df_assist_true_reserve = df[reserve_columns]
    df_assist_true_reserve.columns = [col.split("/")[2] for col in reserve_columns]

    fig, axs = plt.subplots(5, 1, figsize=(15, 10), sharex=False)

    axs[0].bar(df_assist_true_reserve.columns, df_assist_true_reserve.std(axis=0)*config_file["reserve_actuator_force"])
    axs[0].set_ylabel("Standard deviation [N]")
    axs[0].set_xticks(range(len(df_assist_true_reserve.columns)))
    axs[0].set_xticklabels(range(len(df_assist_true_reserve.columns)))

    axs[1].bar(df_assist_true_reserve.columns, df_assist_true_reserve.mean(axis=0)*config_file["reserve_actuator_force"])
    axs[1].set_ylabel("Mean [N]")
    axs[1].set_xticks(range(len(df_assist_true_reserve.columns)))
    axs[1].set_xticklabels(range(len(df_assist_true_reserve.columns)))

    axs[2].bar(df_assist_true_reserve.columns, df_assist_true_reserve.min(axis=0)*config_file["reserve_actuator_force"])
    axs[2].set_ylabel("Min [N]")
    axs[2].set_xticks(range(len(df_assist_true_reserve.columns)))
    axs[2].set_xticklabels(range(len(df_assist_true_reserve.columns)))

    axs[3].bar(df_assist_true_reserve.columns, df_assist_true_reserve.max(axis=0)*config_file["reserve_actuator_force"])
    axs[3].set_ylabel("Max [N]")
    axs[3].set_xticks(range(len(df_assist_true_reserve.columns)))
    axs[3].set_xticklabels(range(len(df_assist_true_reserve.columns)))


    
    means = df_assist_true_reserve.mean(axis=0)*config_file["reserve_actuator_force"]
    q1 = df_assist_true_reserve.quantile(0.25, axis=0)*config_file["reserve_actuator_force"]
    q3 = df_assist_true_reserve.quantile(0.75, axis=0)*config_file["reserve_actuator_force"]
    whislo = q1 - 1.5 * (q3 - q1)
    whishi = q3 + 1.5 * (q3 - q1)

    keys = ['med', 'q1', 'q3', 'whislo', 'whishi']
    stats = [dict(zip(keys, vals)) for vals in zip(means, q1, q3, whislo, whishi)]
    axs[4].bxp(stats, showfliers=False)
    axs[4].set_xticks(range(1, len(stats) + 1))
    axs[4].set_xticklabels(df_assist_true_reserve.columns, rotation=45, ha="right")
    
    if output_path:
        fig.savefig(output_path, dpi=300)

    # axs[3].bar(df_assist_true_reserve.columns, df_assist_true_reserve.min(axis=0)*350)
    # axs[3].set_title("Min of reserve actuator forces")
    # axs[3].set_ylabel("Min [N]")
    # axs[3].set_xticklabels(df_assist_true_reserve.columns, rotation=45, ha="right")