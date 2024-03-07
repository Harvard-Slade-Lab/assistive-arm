import pandas as pd
import numpy as np

from typing import Literal
from pathlib import Path

import matplotlib.pyplot as plt

def plot_every_muscle(title: str, muscles: list, dfs: list[pd.DataFrame], freq: float=6):
    """ Plot EMG relevant for a given muscle (unfiltered vs. filtered)

    Args:
        title (str): plot title
        dfs (list[pd.DataFrame]): list of dataframes
        freq (float, optional): Filter frequency. Defaults to 6.

    Raises:
        ValueError: _description_
    """
    num_rows = len(dfs)
    single_row = num_rows == 1

    cols = dfs[0].columns

    fig, axs = plt.subplots(num_rows, len(muscles), figsize=(10, num_rows * 3 if not single_row else 5))
    fig.suptitle(title, fontsize=16)

    for i, emg_df in enumerate(dfs):
        time_unfiltered = dfs[i].index.values

        for j, muscle in enumerate(muscles):
            filtered_cols = [rel_col for rel_col in cols if muscle in rel_col]

            axs[i, j].set_title(f'Iteration {i+1}, {muscle}')
            axs[-1, j].set_xlabel("Time (s)")
            axs[i, 0].set_ylabel("EMG\n(% MVIC)")
            axs[i, j].set_ylim(0, 1)
            
            for col in filtered_cols:
                axs[i, j].plot(time_unfiltered, emg_df[col], label=col)
                axs[i, j].legend()

    plt.tight_layout()
    plt.show()


def plot_muscle_emg(
        title: str,
        target_muscle: Literal["RF", "VM", "BF"],
        unfiltered_dfs: list[pd.DataFrame],
        filtered_dfs: list[pd.DataFrame],
        freq: float=6,
        fig_path: Path=None,
        show: bool=True):
    """ Plot EMG relevant for a given muscle (untiltered vs. filtered)

    Args:
        title (str): plot title
        target_muscle (Literal["RF", "VM", "BF"]): muscle we want to plot
        unfiltered_dfs (list[pd.DataFrame]): list of unfiltered dataframes
        filtered_dfs (list[pd.DataFrame]): list of filtered dataframes
        freq (float, optional): Filter frequency. Defaults to 6.

    Raises:
        ValueError: _description_
    """
   
    if len(unfiltered_dfs) != len(filtered_dfs):
        raise ValueError("The number of unfiltered and filtered DataFrames must be the same.")

    num_rows = len(unfiltered_dfs)
    single_row = num_rows == 1

    fig, axs = plt.subplots(num_rows, 2, figsize=(10, num_rows * 4 if not single_row else 5))
    fig.suptitle(title, fontsize=16)
    
    axs = axs.reshape(num_rows, -1)  # Ensure axs is always 2D

    for i in range(num_rows):
        time_unfiltered = unfiltered_dfs[i].index.values
        time_filtered = filtered_dfs[i].index.values

        for j in range(2):
            axs[i, j].set_title(f'Iteration {i+1}, {"Filtered" if j == 1 else "Unfiltered"}, {target_muscle}')
            axs[i, j].grid(True)

            if j == 0 or j == 2:
                axs[i, j].set_ylabel("EMG signal (mV)")
            axs[i, j].set_xlabel("Time (s)")

        axs[i, 0].plot(time_unfiltered, unfiltered_dfs[i][f'{target_muscle}_LEFT'], label=f"Unfiltered {target_muscle}_LEFT")
        axs[i, 0].plot(time_unfiltered, unfiltered_dfs[i][f'{target_muscle}_RIGHT'], label=f"Unfiltered {target_muscle}_RIGHT")
        axs[i, 0].legend()

        axs[i, 1].plot(time_filtered, filtered_dfs[i][f'{target_muscle}_LEFT'], label=f"Filtered {target_muscle}_LEFT")
        axs[i, 1].plot(time_filtered, filtered_dfs[i][f'{target_muscle}_RIGHT'], label=f"Filtered {target_muscle}_RIGHT")
        axs[i, 1].legend()

    if fig_path:
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')

    if show:
        plt.tight_layout(pad=2.0, w_pad=2.0, h_pad=2.0)
        plt.show()
    
