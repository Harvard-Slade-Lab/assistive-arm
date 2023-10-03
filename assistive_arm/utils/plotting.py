import pandas as pd
import matplotlib.pyplot as plt

from typing import List

def extract_reserve_columns(df: pd.DataFrame):
    reserve_actuators = [column for column in df.columns if "reserve" in column]
    df_reserve = df[reserve_actuators]
    df_reserve.columns = [column.split("/")[2] for column in df_reserve.columns]

    return df_reserve


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

def plot_res_assist_forces(time: pd.Series, dataframes: List[pd.DataFrame], figsize=(12, 5)):
    coords = ['x', 'y']
    
    assist_true = dataframes["assist_true"]
    assist_false = dataframes["assist_false"]
    grf = dataframes["ground_forces"]

    fig, axs = plt.subplots(len(coords), figsize=figsize, sharex=True)

    fig.suptitle('Residual, ground and assistive forces')

    for i, coord in enumerate(coords):
        axs[i].plot()

        # Pelvis T_coord
        axs[i].plot(time, assist_true[f'/forceset/reserve_jointset_ground_pelvis_pelvis_t{coord}']*350, label=f'Residual {coord.upper()} (assisted)')
        axs[i].plot(time, assist_true[f"/forceset/assistive_force_{coord}"]*400, label=f"Assistive Force {coord.upper()}")
        axs[i].plot(time, assist_false[f'/forceset/reserve_jointset_ground_pelvis_pelvis_t{coord}']*350, label=f'Residual {coord.upper()} (unassisted)')
        axs[i].plot(grf.time, grf[f'ground_force_r_v{coord}'], label=f'Ground Force {coord.upper()}')
        axs[i].set_title(coord.upper())
        axs[i].grid()
        axs[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    for ax in axs.flat:
        ax.set(ylabel='Force (N)')

    plt.show()