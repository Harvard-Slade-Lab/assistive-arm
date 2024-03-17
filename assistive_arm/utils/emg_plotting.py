import pandas as pd
import numpy as np

from typing import Literal
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def plot_every_muscle(title: str, muscles: list, dfs: list[pd.DataFrame], freq: float=6, fig_path: Path=None):
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

    fig, axs = plt.subplots(len(muscles), num_rows, figsize=(num_rows * 2, 5), sharex='col', sharey='row' )
    fig.suptitle(title)

    for i, muscle in enumerate(muscles):
        for j, emg_df in enumerate(dfs):
            time_unfiltered = dfs[j].index.values

            filtered_cols = [rel_col for rel_col in cols if muscle in rel_col]

            axs[0, j].set_title(f'Iteration {j+1}')
            axs[-1, j].set_xlabel("Time (s)")
            axs[i, 0].set_ylabel(f"{muscle} EMG\n(% MVIC)")
            axs[i, j].set_ylim(0, 1)
            
            for col in filtered_cols:
                axs[i, j].plot(time_unfiltered, emg_df[col], label=col.split('_')[1])
            axs[i, j].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncols=len(labels), bbox_to_anchor=(0.5, 0.9))    

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if fig_path:
        plt.savefig(fig_path, dpi=500, format='svg')
        plt.savefig(fig_path.with_suffix(".png"), dpi=500, format='png')
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

    fig, axs = plt.subplots(2, num_rows, figsize=(num_rows * 3, 6), sharex='col', sharey='row')
    fig.suptitle(title)

    for i in range(num_rows):
        time_unfiltered = unfiltered_dfs[i].index.values
        time_filtered = filtered_dfs[i].index.values

        axs[0, i].set_title(f"Iteration {i+1}")
 
        axs[0, 0].set_ylabel(f"{target_muscle} UNFILTERED\n(mV)")
        axs[1, 0].set_ylabel(f"{target_muscle} FILTERED\n(mV)")
        axs[1, i].set_xlabel("Time (s)")

        plot_every = 30

        axs[0, i].plot(time_unfiltered[::plot_every], unfiltered_dfs[i][::plot_every][f'{target_muscle}_LEFT'], label=f"LEFT")
        axs[0, i].plot(time_unfiltered[::plot_every], unfiltered_dfs[i][::plot_every][f'{target_muscle}_RIGHT'], label=f"RIGHT")

        axs[1, i].plot(time_filtered[::plot_every], filtered_dfs[i][::plot_every][f'{target_muscle}_LEFT'], label=f"LEFT")
        axs[1, i].plot(time_filtered[::plot_every], filtered_dfs[i][::plot_every][f'{target_muscle}_RIGHT'], label=f"RIGHT")

        axs[1, i].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, symbol=None))
        axs[1, i].set_ylim(0, 0.7)

        handles, labels = axs[0, i].get_legend_handles_labels()

    fig.legend(handles, labels, loc='upper center', ncols=len(labels), bbox_to_anchor=(0.5, 0.9))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if fig_path:
        plt.savefig(fig_path, bbox_inches='tight', format='svg')
        plt.savefig(fig_path.with_suffix('.png'), bbox_inches='tight', format='png')

    fig.legend(handles, labels, loc='upper center', ncols=len(labels), bbox_to_anchor=(0.5, 0.9))

    if fig_path and not fig_path.exists():
        plt.savefig(fig_path, bbox_inches='tight', format='svg' if fig_path.suffix == ".svg" else "png")
    elif fig_path and fig_path.exists():
        print(f"Figure {fig_path} already exists, skipping...")
    else:
        print("No path provided, skipping saving...")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if fig_path:
        plt.savefig(fig_path, bbox_inches='tight', format='svg')
        plt.savefig(fig_path.with_suffix('.png'), bbox_inches='tight', format='png')

    if show:
        plt.show()
    
