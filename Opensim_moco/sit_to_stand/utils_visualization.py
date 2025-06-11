import os
import json
import csv
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.interpolate import PchipInterpolator

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
  
def plot_GRF_osim_v2(df_sim_floor, df_sim_chair, body_weight, fontsize, figsize, fig_path):
    
    # Define custom colors
    colors_floor = {  # GREY TONES for foot
        'x': '#d3d3d3',  # light grey
        'y': '#a9a9a9',  # dark grey
        'z': '#2f2f2f'   # almost black
    }
    colors_chair = {  # ORANGE-BROWN for chair
        'x': '#f4a582',  # light orange
        'y': '#d6604d',  # medium orange-red
        'z': '#8c2d04'   # dark brown
    }

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize, sharex=True)
    axes = axes.flatten()

    # Plot 1: Ground reaction forces (left)
    axes[0].plot(df_sim_floor.time_perc, (df_sim_floor['ground_force_r_vx']+df_sim_floor['ground_force_l_vx'])/body_weight, linewidth=2, color = colors_floor['x'], label='GRF x')
    axes[0].plot(df_sim_floor.time_perc, (df_sim_floor['ground_force_r_vy']+df_sim_floor['ground_force_l_vy'])/body_weight, linewidth=2,  color = colors_floor['y'],  label='GRF y')
    axes[0].plot(df_sim_floor.time_perc, (df_sim_floor['ground_force_r_vz']+df_sim_floor['ground_force_l_vz'])/body_weight, linewidth=2,  color = colors_floor['z'],  label='GRF z')
    axes[0].set_xlabel("Task completion [%]", fontsize=fontsize)
    axes[0].set_ylabel("Force [BW]", fontsize=fontsize)
    axes[0].set_ylim([-0.5, 1.5])
    axes[0].set_title(f"Ground forces (Floor)", fontsize=fontsize)
    axes[0].legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))

    # print max value
    print(f"maximum floor GRF: {np.max(df_sim_floor['ground_force_r_vy']+df_sim_floor['ground_force_l_vy'])/body_weight}")

    # Plot 2: Ground reaction forces (right)
    axes[1].plot(df_sim_chair.time_perc, (df_sim_chair['ground_force_r_vx']+df_sim_chair['ground_force_l_vx'])/body_weight, linewidth=2, color = colors_chair['x'], label='GRF x')
    axes[1].plot(df_sim_chair.time_perc, (df_sim_chair['ground_force_r_vy']+df_sim_chair['ground_force_l_vy'])/body_weight, linewidth=2, color = colors_chair['y'], label='GRF y')
    axes[1].plot(df_sim_chair.time_perc, (df_sim_chair['ground_force_r_vz']+df_sim_chair['ground_force_l_vz'])/body_weight, linewidth=2, color = colors_chair['z'], label='GRF z')
    axes[1].set_ylim([-0.5, 1.5])
    axes[1].set_xlabel("Task completion [%]", fontsize=fontsize)
    axes[1].set_ylabel("Force [BW]", fontsize=fontsize)
    axes[1].set_title(f"Ground forces (Chair)", fontsize=fontsize)
    axes[1].legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=fontsize)

    fig_title = 'sim_ground_reaction_force_torque.png'  # include file extension
    full_path = os.path.join(fig_path, fig_title)      # combine path and filename

    plt.tight_layout()
    plt.savefig(full_path, dpi=300)

def plot_GRF_osim(df_sim_floor, df_sim_chair, fontsize, figsize, fig_path):
    
    # Define custom colors
    colors_floor = {  # GREY TONES for foot
        'x': '#d3d3d3',  # light grey
        'y': '#a9a9a9',  # dark grey
        'z': '#2f2f2f'   # almost black
    }
    colors_chair = {  # ORANGE-BROWN for chair
        'x': '#f4a582',  # light orange
        'y': '#d6604d',  # medium orange-red
        'z': '#8c2d04'   # dark brown
    }

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize, sharex=True)
    axes = axes.flatten()

    # Plot 1: Ground reaction forces (upper left)
    axes[0].plot(df_sim_floor.time_perc, (df_sim_floor['ground_force_r_vx']+df_sim_floor['ground_force_l_vx'])/2, linewidth=3, color = colors_floor['x'], label='GRF x')
    axes[0].plot(df_sim_floor.time_perc, (df_sim_floor['ground_force_r_vy']+df_sim_floor['ground_force_l_vy'])/2, linewidth=3,  color = colors_floor['y'],  label='GRF y')
    axes[0].plot(df_sim_floor.time_perc, (df_sim_floor['ground_force_r_vz']+df_sim_floor['ground_force_l_vz'])/2, linewidth=3,  color = colors_floor['z'],  label='GRF z')

    axes[0].plot(df_sim_floor.time_perc, df_sim_floor['ground_force_r_vx'], color = colors_floor['x'], alpha=0.5)
    axes[0].plot(df_sim_floor.time_perc, df_sim_floor['ground_force_l_vx'], color = colors_floor['x'], alpha=0.5)
    axes[0].plot(df_sim_floor.time_perc, df_sim_floor['ground_force_r_vy'], color = colors_floor['y'], alpha=0.5)
    axes[0].plot(df_sim_floor.time_perc, df_sim_floor['ground_force_l_vy'], color = colors_floor['y'], alpha=0.5)
    axes[0].plot(df_sim_floor.time_perc, df_sim_floor['ground_force_r_vz'], color = colors_floor['z'], alpha=0.5)
    axes[0].plot(df_sim_floor.time_perc, df_sim_floor['ground_force_l_vz'], color = colors_floor['z'], alpha=0.5)

    axes[0].set_xlabel("Task completion [%]", fontsize=fontsize)
    axes[0].set_ylabel("Force [N]", fontsize=fontsize)
    axes[0].set_title(f"Ground forces (Floor)", fontsize=fontsize)
    axes[0].legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))

    # # Plot 2: Ground torques (upper right)
    axes[1].plot(df_sim_floor.time_perc, (df_sim_floor['ground_torque_r_x']+df_sim_floor['ground_torque_l_x'])/2, linewidth=3, color = colors_floor['x'],  label='GRT x')
    axes[1].plot(df_sim_floor.time_perc, (df_sim_floor['ground_torque_r_y']+df_sim_floor['ground_torque_l_y'])/2, linewidth=3, color = colors_floor['y'],  label='GRT y')
    axes[1].plot(df_sim_floor.time_perc, (df_sim_floor['ground_torque_r_z']+df_sim_floor['ground_torque_l_z'])/2, linewidth=3, color = colors_floor['z'],  label='GRT z')

    axes[1].plot(df_sim_floor.time_perc, df_sim_floor['ground_torque_r_x'], color = colors_floor['x'], alpha=0.5)
    axes[1].plot(df_sim_floor.time_perc, df_sim_floor['ground_torque_r_y'], color = colors_floor['y'], alpha=0.5)
    axes[1].plot(df_sim_floor.time_perc, df_sim_floor['ground_torque_r_z'], color = colors_floor['z'], alpha=0.5)
    axes[1].plot(df_sim_floor.time_perc, df_sim_floor['ground_torque_l_x'], color = colors_floor['x'], alpha=0.5)
    axes[1].plot(df_sim_floor.time_perc, df_sim_floor['ground_torque_l_y'], color = colors_floor['y'], alpha=0.5)
    axes[1].plot(df_sim_floor.time_perc, df_sim_floor['ground_torque_l_z'], color = colors_floor['z'], alpha=0.5)

    axes[1].set_xlabel("Task completion [%]", fontsize=fontsize)
    axes[1].set_ylabel("Torque [Nm]", fontsize=fontsize)
    axes[1].set_title(f"Ground torque (Floor)", fontsize=fontsize)
    axes[1].legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))

    # Plot 1: Ground reaction forces (upper left)
    axes[2].plot(df_sim_chair.time_perc, (df_sim_chair['ground_force_r_vx']+df_sim_chair['ground_force_l_vx'])/2, linewidth=3, color = colors_chair['x'], label='GRF x')
    axes[2].plot(df_sim_chair.time_perc, (df_sim_chair['ground_force_r_vy']+df_sim_chair['ground_force_l_vy'])/2, linewidth=3, color = colors_chair['y'], label='GRF y')
    axes[2].plot(df_sim_chair.time_perc, (df_sim_chair['ground_force_r_vz']+df_sim_chair['ground_force_l_vz'])/2, linewidth=3, color = colors_chair['z'], label='GRF z')
    axes[2].set_xlabel("Task completion [%]", fontsize=fontsize)
    axes[2].set_ylabel("Force [N]", fontsize=fontsize)
    axes[2].set_title(f"Ground forces (Chair)", fontsize=fontsize)
    axes[2].legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))

    axes[2].plot(df_sim_chair.time_perc, df_sim_chair['ground_force_r_vx'], color = colors_chair['x'], alpha=0.5)
    axes[2].plot(df_sim_chair.time_perc, df_sim_chair['ground_force_l_vx'], color = colors_chair['x'], alpha=0.5)
    axes[2].plot(df_sim_chair.time_perc, df_sim_chair['ground_force_r_vy'], color = colors_chair['y'], alpha=0.5)
    axes[2].plot(df_sim_chair.time_perc, df_sim_chair['ground_force_l_vy'], color = colors_chair['y'], alpha=0.5)
    axes[2].plot(df_sim_chair.time_perc, df_sim_chair['ground_force_r_vz'], color = colors_chair['z'], alpha=0.5)
    axes[2].plot(df_sim_chair.time_perc, df_sim_chair['ground_force_l_vz'], color = colors_chair['z'], alpha=0.5)

    # # Plot 2: Ground torques (upper right)
    axes[3].plot(df_sim_chair.time_perc, (df_sim_chair['ground_torque_r_x']+df_sim_chair['ground_torque_l_x'])/2, linewidth=3, color = colors_chair['x'], label='GRT x')
    axes[3].plot(df_sim_chair.time_perc, (df_sim_chair['ground_torque_r_y']+df_sim_chair['ground_torque_l_y'])/2, linewidth=3, color = colors_chair['y'], label='GRT y')
    axes[3].plot(df_sim_chair.time_perc, (df_sim_chair['ground_torque_r_z']+df_sim_chair['ground_torque_l_z'])/2, linewidth=3, color = colors_chair['z'], label='GRT z')
    axes[3].set_xlabel("Task completion [%]", fontsize=fontsize)
    axes[3].set_ylabel("Torque [Nm]", fontsize=fontsize)
    axes[3].set_title(f"Ground torque (Chair)", fontsize=fontsize)
    axes[3].legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))

    axes[3].plot(df_sim_chair.time_perc, df_sim_chair['ground_torque_r_x'], color = colors_chair['x'], alpha=0.5)
    axes[3].plot(df_sim_chair.time_perc, df_sim_chair['ground_torque_r_y'], color = colors_chair['y'], alpha=0.5)
    axes[3].plot(df_sim_chair.time_perc, df_sim_chair['ground_torque_r_z'], color = colors_chair['z'], alpha=0.5)
    axes[3].plot(df_sim_chair.time_perc, df_sim_chair['ground_torque_l_x'], color = colors_chair['x'], alpha=0.5)
    axes[3].plot(df_sim_chair.time_perc, df_sim_chair['ground_torque_l_y'], color = colors_chair['y'], alpha=0.5)
    axes[3].plot(df_sim_chair.time_perc, df_sim_chair['ground_torque_l_z'], color = colors_chair['z'], alpha=0.5)

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=fontsize)

    fig_title = 'sim_ground_reaction_force_torque.png'  # include file extension
    full_path = os.path.join(fig_path, fig_title)      # combine path and filename

    plt.tight_layout()
    plt.savefig(full_path, dpi=300)

def plot_muscle_activation_grid(df_sim, ref_m_activation, fontsize, figsize, fig_path):
    time = df_sim['time_perc']

    # List of (muscle_abbreviation, full_name) tuples
    muscles = [
        ('recfem',  'Rectus Femoris'),
        ('vasmed',  'Vastus Medialis'),
        ('bflh',    'Biceps Femoris (long head)'),
        ('soleus',  'Soleus'),
        ('gasmed',  'Gastrocnemius Medialis'),
        ('tibant',  'Tibialis Anterior'),
    ]

    muscle_colors = {
        'recfem':  'blue',
        'vasmed':  'green',
        'bflh':    'orange',
        'soleus':  'brown',
        'gasmed':  'red',
        'tibant':  'purple',
    }
    n_muscles = len(muscles)
    n_cols = math.ceil(math.sqrt(n_muscles))
    n_rows = math.ceil(n_muscles / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten()

    for i, (muscle_abbr, muscle_fullname) in enumerate(muscles):
        left = f'/forceset/{muscle_abbr}_l'
        right = f'/forceset/{muscle_abbr}_r'
        color = muscle_colors.get(muscle_abbr, 'gray')

        ax = axes[i]

        if left in df_sim and right in df_sim:
            l = df_sim[left] * 100  # convert to percentage
            r = df_sim[right] * 100
            avg = (l + r) / 2

            # Plot average activation
            ax.plot(time, avg, label='average', color=color, linewidth=2)

            # ▶️ Add mean and peak activation as text below plot
            mean_activation = np.mean(avg)
            peak_activation = np.max(avg)
            stats_text = f"Mean: {mean_activation:.1f}%, Peak: {peak_activation:.1f}%"
            ax.text(0.5, 1, stats_text, transform=ax.transAxes,
                    ha='center', va='top', fontsize=fontsize)

        # Overlay reference curve if available
        for cur_file in os.listdir(ref_m_activation):
            if muscle_abbr[:5] in cur_file:
                ref_path = os.path.join(ref_m_activation, cur_file)
                df_ref = pd.read_csv(ref_path).values
                ref_time = df_ref[:, 0]
                ref_activation = df_ref[:, 1]

                ax.plot(ref_time, ref_activation, alpha=1, linewidth=1,
                        color='k', linestyle='--', label='ref')

        ax.set_title(muscle_fullname, fontsize=fontsize)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=fontsize)

    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Shared axis labels
    fig.text(0.5, 0.04, 'Task Completion (%)', ha='center', fontsize=fontsize)
    fig.text(0.04, 0.5, 'Muscle Activation (%)', va='center', rotation='vertical', fontsize=fontsize)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', fontsize=fontsize, frameon=False)

    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])

    # Save figure
    fig_title = 'muscle_activation_grid.png'
    full_path = os.path.join(fig_path, fig_title)
    plt.savefig(full_path, dpi=300)
    plt.close()

def plot_muscle_activation_grid_original(df_sim, ref_m_activation, fontsize, figsize, fig_path):
    time = df_sim['time_perc']

    # List of (muscle_abbreviation, full_name) tuples
    muscles = [
        ('recfem',  'Rectus Femoris'),
        ('vasmed',  'Vastus Medialis'),
        ('vaslat',  'Vastus Lateralis'),
        ('psoas',   'Psoas'),
        ('bflh',    'Biceps Femoris (long head)'),
        ('bfsh',    'Biceps Femoris (short head)'),
        ('glmax1',  'Gluteus Maximus (upper)'),
        ('glmax2',  'Gluteus Maximus (middle)'),
        ('glmax3',  'Gluteus Maximus (lower)'),
        ('glmed1',  'Gluteus Medius (front)'),
        ('glmed2',  'Gluteus Medius (center)'),
        ('glmed3',  'Gluteus Medius (back)'),
        ('soleus',  'Soleus'),
        ('gasmed',  'Gastrocnemius Medialis'),
        ('tibant',  'Tibialis Anterior'),
    ]

    # Muscle colors
    muscle_colors = {
        'recfem':  '#f78181',
        'vasmed':  '#f44141',
        'vaslat':  '#cc0000',
        'psoas':   '#ffa500',
        'bflh':    '#86c7b8',
        'bfsh':    '#40b4a6',
        'glmax1':  '#58acb2',
        'glmax2':  '#2f7fa5',
        'glmax3':  '#0b5c94',
        'glmed1':  '#92d0b5',
        'glmed2':  '#5ca891',
        'glmed3':  '#006b8f',
        'soleus':  '#e0c8a8',
        'gasmed':  '#cc8899',
        'tibant':  '#2d2d53',
    }

    n_muscles = len(muscles)
    n_cols = math.ceil(math.sqrt(n_muscles))
    n_rows = math.ceil(n_muscles / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten()

    for i, (muscle_abbr, muscle_fullname) in enumerate(muscles):
        left = f'/forceset/{muscle_abbr}_l'
        right = f'/forceset/{muscle_abbr}_r'
        color = muscle_colors.get(muscle_abbr, 'gray')

        ax = axes[i]
        if left in df_sim and right in df_sim:
            l = df_sim[left] * 100
            r = df_sim[right] * 100
            avg = (l + r) / 2
            ax.plot(time, avg, label='average', color=color, linewidth=2)
            # ax.plot(time, l, linestyle='-', alpha=0.5, linewidth=1, color=color, label='left')
            # ax.plot(time, r, linestyle='--', alpha=0.5, linewidth=1, color=color, label='right')
            # ax.set_ylim([0, 100])

        for cur_file in os.listdir(ref_m_activation):
            if muscle_abbr[:5] in cur_file:
                ref_path = os.path.join(ref_m_activation, cur_file)
                df_ref = pd.read_csv(ref_path).values
                ref_time = df_ref[:,0]
                ref_activation = df_ref[:,1]

                ax.plot(ref_time, ref_activation, alpha=1, linewidth=1, color='k', linestyle='--', label='ref')
        ax.set_title(muscle_fullname, fontsize=fontsize)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=fontsize * 0.8)

    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Shared axis labels
    fig.text(0.5, 0.04, 'Task Completion (%)', ha='center', fontsize=fontsize)
    fig.text(0.04, 0.5, 'Muscle Activation (%)', va='center', rotation='vertical', fontsize=fontsize)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', fontsize=fontsize * 0.8, frameon=False)

    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])

    fig_title = 'muscle_activation_grid.png'
    full_path = os.path.join(fig_path, fig_title)
    plt.savefig(full_path, dpi=300)
    plt.close()

def plot_muscle_activation(df_sim, fontsize, figsize, fig_path):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize, sharex=True)
    axes = axes.flatten()

    time = df_sim['time_perc']

    # Color map derived from provided image
    muscle_colors = {
        'recfem':  '#f78181',
        'vasmed':  '#f44141',
        'vaslat':  '#cc0000',
        'psoas':   '#ffa500',
        'bflh':    '#86c7b8',
        'bfsh':    '#40b4a6',
        'glmax1':  '#58acb2',
        'glmax2':  '#2f7fa5',
        'glmax3':  '#0b5c94',
        'glmed1':  '#92d0b5',
        'glmed2':  '#5ca891',
        'glmed3':  '#006b8f',
        'soleus':  '#e0c8a8',
        'gasmed':  '#cc8899',
        'tibant':  '#2d2d53',
    }

    # Panel 0: Anterior Quadriceps
    quad_muscles = ['vaslat', 'vasmed', 'recfem']
    for muscle in quad_muscles:
        left = f'/forceset/{muscle}_l'
        right = f'/forceset/{muscle}_r'
        color = muscle_colors.get(muscle, 'gray')
        if left in df_sim and right in df_sim:
            l = df_sim[left] * 100
            r = df_sim[right] * 100
            avg = (l + r) / 2
            axes[0].plot(time, avg, label=muscle, color=color, linewidth=2)
            axes[0].plot(time, l, linestyle='-', alpha=0.5, linewidth=1, color=color)
            axes[0].plot(time, r, linestyle='-', alpha=0.5, linewidth=1, color=color)
    axes[0].set_title('Anterior (quadriceps)')
    axes[0].legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fontsize * 0.9)

    # Panel 1: Psoas
    for muscle in ['psoas']:
        left = f'/forceset/{muscle}_l'
        right = f'/forceset/{muscle}_r'
        color = muscle_colors.get(muscle, 'gray')
        if left in df_sim and right in df_sim:
            l = df_sim[left] * 100
            r = df_sim[right] * 100
            avg = (l + r) / 2
            axes[1].plot(time, avg, label=muscle, color=color, linewidth=2)
            axes[1].plot(time, l, linestyle='-', alpha=0.5, linewidth=1, color=color)
            axes[1].plot(time, r, linestyle='-', alpha=0.5, linewidth=1, color=color)
    axes[1].set_title('Anterior (psoas)')
    axes[1].legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fontsize * 0.9)

    # Panel 2: Posterior (hamstring, gluteal)
    posterior = [
        'bflh', 'bfsh',
        'glmax1', 'glmax2', 'glmax3',
        'glmed1', 'glmed2', 'glmed3',
    ]
    for muscle in posterior:
        left = f'/forceset/{muscle}_l'
        right = f'/forceset/{muscle}_r'
        color = muscle_colors.get(muscle, 'gray')
        if left in df_sim and right in df_sim:
            l = df_sim[left] * 100
            r = df_sim[right] * 100
            avg = (l + r) / 2
            axes[2].plot(time, avg, label=muscle, color=color, linewidth=2)
            axes[2].plot(time, l, linestyle='-', alpha=0.5, linewidth=1, color=color)
            axes[2].plot(time, r, linestyle='-', alpha=0.5, linewidth=1, color=color)
    axes[2].set_title('Posterior (hamstring, gluteal)')
    axes[2].legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fontsize * 0.9)

    # Panel 3: Posterior (calf) and anterior (tibia)
    for muscle in ['soleus', 'gasmed', 'tibant']:
        left = f'/forceset/{muscle}_l'
        right = f'/forceset/{muscle}_r'
        color = muscle_colors.get(muscle, 'gray')
        if left in df_sim and right in df_sim:
            l = df_sim[left] * 100
            r = df_sim[right] * 100
            avg = (l + r) / 2
            axes[3].plot(time, avg, label=muscle, color=color, linewidth=2)
            axes[3].plot(time, l, linestyle='-', alpha=0.5, linewidth=1, color=color)
            axes[3].plot(time, r, linestyle='-', alpha=0.5, linewidth=1, color=color)
        elif right in df_sim:
            axes[3].plot(time, df_sim[right], label=f'{muscle}_r', color=color, linewidth=2)
    axes[3].set_title('Posterior (calf) and anterior (tibia)')
    axes[3].legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fontsize * 0.9)

    # Formatting
    for ax in axes:
        ax.set_xlabel('Task completion (%)')
        ax.set_ylabel('Activation (%)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=fontsize)

    plt.tight_layout()
    fig_title = 'muscle_activation.png'  # include file extension
    full_path = os.path.join(fig_path, fig_title)      # combine path and filename
    plt.savefig(full_path, dpi=300)
    # plt.show()

def plot_residual_force_torque_others(df_sim, sim_params, grf_info, fontsize, figsize, fig_path):

    # 1% of COM height times maximum external forces
    residual_torque_threshold = np.max(grf_info['ground_force_r_vy']+grf_info['ground_force_l_vy']) * 0.01 * 1.77 * 0.55

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize, sharex=True)

    # Make sure axes is a flat 1D array (just in case):
    axes = axes.flatten()

    axes[0].plot(df_sim['time_perc'], df_sim['/forceset/reserve_jointset_hip_l_hip_adduction_l']*sim_params['opt_force_reserve_actuator'], label='hip_adduction_l')
    axes[0].plot(df_sim['time_perc'], df_sim['/forceset/reserve_jointset_hip_l_hip_rotation_l']*sim_params['opt_force_reserve_actuator'], label='hip_rotation_l')
    axes[0].plot(df_sim['time_perc'], df_sim['/forceset/reserve_jointset_walker_knee_l_knee_angle_l']*sim_params['opt_force_reserve_actuator'], label='knee_angle_l')
    axes[0].plot(df_sim['time_perc'], df_sim['/forceset/reserve_jointset_ankle_l_ankle_angle_l']*sim_params['opt_force_reserve_actuator'], label='ankle_angle_l')
    axes[0].plot(df_sim['time_perc'], df_sim['/forceset/reserve_jointset_subtalar_l_subtalar_angle_l']*sim_params['opt_force_reserve_actuator'], label='subtalar_angle_l')
    axes[0].plot(df_sim['time_perc'], df_sim['/forceset/reserve_jointset_mtp_l_mtp_angle_l']*sim_params['opt_force_reserve_actuator'], label='mtp_angle_l')
    axes[0].axhline(y=residual_torque_threshold, linestyle='--', color='k')
    axes[0].axhline(y=-residual_torque_threshold, label='Residual torque\nthreshold', linestyle='--', color='k')

    axes[0].set_xlabel('Task completion (%)')
    axes[0].set_ylabel('Torque (Nm)')
    # axes[0].set_ylim([-20, 20])
    axes[0].legend(frameon=False, bbox_to_anchor=(1, 0.5), loc='best', fontsize=8)
    axes[0].set_title('Left leg Residual Torques')

    axes[1].plot(df_sim['time_perc'], df_sim['/forceset/reserve_jointset_hip_r_hip_adduction_r']*sim_params['opt_force_reserve_actuator'], label='hip_adduction_r')
    axes[1].plot(df_sim['time_perc'], df_sim['/forceset/reserve_jointset_hip_r_hip_rotation_r']*sim_params['opt_force_reserve_actuator'], label='hip_rotation_r')
    axes[1].plot(df_sim['time_perc'], df_sim['/forceset/reserve_jointset_walker_knee_r_knee_angle_r']*sim_params['opt_force_reserve_actuator'], label='knee_angle_r')
    axes[1].plot(df_sim['time_perc'], df_sim['/forceset/reserve_jointset_ankle_r_ankle_angle_r']*sim_params['opt_force_reserve_actuator'], label='ankle_angle_r')
    axes[1].plot(df_sim['time_perc'], df_sim['/forceset/reserve_jointset_subtalar_r_subtalar_angle_r']*sim_params['opt_force_reserve_actuator'], label='subtalar_angle_r')
    axes[1].plot(df_sim['time_perc'], df_sim['/forceset/reserve_jointset_mtp_r_mtp_angle_r']*sim_params['opt_force_reserve_actuator'], label='mtp_angle_r')
    axes[1].axhline(y=residual_torque_threshold, linestyle='--', color='k')
    axes[1].axhline(y=-residual_torque_threshold, label='Residual torque\nthreshold', linestyle='--', color='k')

    axes[1].set_xlabel('Task completion (%)')
    axes[1].set_ylabel('Torque (Nm)')
    # axes[1].set_ylim([-20, 20])
    axes[1].legend(frameon=False, bbox_to_anchor=(1, 0.5), loc='best', fontsize=8)
    axes[1].set_title('Right leg Residual Torques')

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=fontsize)

    fig_title = 'pelvis_residual_force_torque_others.png'  # include file extension
    full_path = os.path.join(fig_path, fig_title)      # combine path and filename

    plt.tight_layout()
    plt.savefig(full_path, dpi=300)    

def plot_assistive_force(df_sim, sim_params, fontsize, figsize, fig_path):

    fig, ax = plt.subplots(figsize=figsize, sharex=True)

    # Plot both x and y assistive forces if available:
    if '/forceset/assistive_force_x' in df_sim.columns:
        ax.plot(df_sim['time_perc'],
                df_sim['/forceset/assistive_force_x'] * sim_params['opt_force_robot_assistance'],
                linewidth=2, label='Assistive force x')
    if '/forceset/assistive_force_y' in df_sim.columns:
        ax.plot(df_sim['time_perc'],
                df_sim['/forceset/assistive_force_y'] * sim_params['opt_force_robot_assistance'],
                linewidth=2, label='Assistive force y')

    ax.set_xlabel('Task completion (%)', fontsize=fontsize)
    ax.set_ylabel('Force (N)', fontsize=fontsize)
    ax.legend(frameon=False, loc='best', fontsize=fontsize)
    ax.set_title('Assistive Force', fontsize=fontsize)

    # Hide top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=fontsize)

    fig_title = 'Assistive_force.png'  # include file extension
    full_path = os.path.join(fig_path, fig_title)      # combine path and filename

    plt.tight_layout()
    plt.savefig(full_path, dpi=300)
    plt.close(fig)
    
def parameterize_assistive_force(df_sim, sim_params, fontsize, figsize, fig_path=None):

    # Extract/compute data
    motion_percentage = df_sim['time_perc'].values
    assist_x_force = df_sim['/forceset/assistive_force_x'].values * sim_params['force_ext_opt_value']
    assist_y_force = df_sim['/forceset/assistive_force_y'].values * sim_params['force_ext_opt_value']

    # Clip negative values to zero
    assist_x_force_clipped = np.clip(assist_x_force, 0, None)
    assist_y_force_clipped = np.clip(assist_y_force, 0, None)

    # Fit cubic Hermite (PCHIP) splines
    hermite_x = PchipInterpolator(motion_percentage, assist_x_force_clipped)
    hermite_y = PchipInterpolator(motion_percentage, assist_y_force_clipped)

    # Generate smooth time/frame values
    t_smooth = np.linspace(motion_percentage.min(), motion_percentage.max(), 100)
    fx_spline = hermite_x(t_smooth)
    fy_spline = hermite_y(t_smooth)

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=figsize, sharex=True)

    # X direction
    axs[0].plot(motion_percentage, assist_x_force_clipped, 'o', label='Clipped assist x')
    axs[0].plot(t_smooth, fx_spline, '-', label='Hermite spline x', linewidth=2)
    axs[0].set_title('Assistive Force X', fontsize=fontsize)
    axs[0].set_xlabel('Task completion (%)', fontsize=fontsize)
    axs[0].set_ylabel('Force (N)', fontsize=fontsize)
    axs[0].legend(frameon=False)
    axs[0].tick_params(labelsize=fontsize)

    # Y direction
    axs[1].plot(motion_percentage, assist_y_force_clipped, 'o', label='Clipped assist y')
    axs[1].plot(t_smooth, fy_spline, '-', label='Hermite spline y', linewidth=2)
    axs[1].set_title('Assistive Force Y', fontsize=fontsize)
    axs[1].set_xlabel('Task completion (%)', fontsize=fontsize)
    axs[1].legend(frameon=False)
    axs[1].tick_params(labelsize=fontsize)

    plt.suptitle('Assistive Force (Clipped & Hermite spline)', fontsize=fontsize+2)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save figure if a path is given, else just show
    if fig_path is not None:
        plt.savefig(os.path.join(fig_path, "Assistive_force_spline.png"), dpi=300)
    # plt.show()
    plt.close(fig)

    # Optionally, return the spline interpolators for later use
    return hermite_x, hermite_y

def plot_residual_force_torque_pelvis(df_sim, sim_params, grf_info, fontsize, figsize, fig_path):

    # 5% of maximum external force magnitude
    residual_force_threshold = np.max(grf_info['ground_force_r_vy']+grf_info['ground_force_l_vy']) * 0.05
    
    # 1% of COM height times maximum external forces
    residual_torque_threshold = np.max(grf_info['ground_force_r_vy']+grf_info['ground_force_l_vy']) * 0.01 * 1.77 * 0.55

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize, sharex=True)

    # Make sure axes is a flat 1D array (just in case):
    axes = axes.flatten()

    axes[0].plot(df_sim['time_perc'], df_sim['/forceset/reserve_jointset_ground_pelvis_pelvis_tx']*sim_params['opt_force_reserve_actuator'], linewidth = 3, label='Pelvis x')
    axes[0].plot(df_sim['time_perc'], df_sim['/forceset/reserve_jointset_ground_pelvis_pelvis_ty']*sim_params['opt_force_reserve_actuator'], linewidth = 3, label='Pelvis y')
    axes[0].plot(df_sim['time_perc'], df_sim['/forceset/reserve_jointset_ground_pelvis_pelvis_tz']*sim_params['opt_force_reserve_actuator'], linewidth = 3, label='Pelvis z')
    
    axes[0].axhline(y=residual_force_threshold, linestyle='--', color='k')
    axes[0].axhline(y=-residual_force_threshold, label='Residual force\nthreshold', linestyle='--', color='k')

    axes[0].set_xlabel('Task completion (%)')
    axes[0].set_ylabel('Force (N)')
    # axes[0].set_ylim([-300, 700])
    axes[0].legend(frameon=False, loc='best')
    axes[0].set_title('Pelvis Residual Forces')

    axes[1].plot(df_sim['time_perc'], df_sim['/forceset/reserve_jointset_ground_pelvis_pelvis_tilt']*sim_params['opt_force_reserve_actuator'], linewidth = 3, label='Pelvis tilt')
    axes[1].plot(df_sim['time_perc'], df_sim['/forceset/reserve_jointset_ground_pelvis_pelvis_list']*sim_params['opt_force_reserve_actuator'], linewidth = 3, label='Pelvis list')
    axes[1].plot(df_sim['time_perc'], df_sim['/forceset/reserve_jointset_ground_pelvis_pelvis_rotation']*sim_params['opt_force_reserve_actuator'], linewidth = 3, label='Pelvis rotation')
    axes[1].axhline(y=residual_torque_threshold, linestyle='--', color='k')
    axes[1].axhline(y=-residual_torque_threshold, label='Residual torque\nthreshold', linestyle='--', color='k')

    axes[1].set_xlabel('Task completion (%)')
    axes[1].set_ylabel('Torque (Nm)')
    # axes[1].set_ylim([-100, 30])

    axes[1].legend(frameon=False, loc='best')
    axes[1].set_title('Pelvis Residual Torques')

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=fontsize)

    fig_title = 'pelvis_residual_force_torque_pelvis.png'  # include file extension
    full_path = os.path.join(fig_path, fig_title)      # combine path and filename

    plt.tight_layout()
    plt.savefig(full_path, dpi=300)

def plot_pelvis_kinematics(df_sim, df_ref, fontsize, figsize, fig_path):
    plt.rcParams.update({'font.size': fontsize})

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize, sharex=True)
    time_sim = df_sim['time_perc']

    # Interpolate reference to match simulation time
    pelvis_tx_ref_interp = np.interp(time_sim, df_ref['time_perc'], df_ref['pelvis_tx'])
    pelvis_ty_ref_interp = np.interp(time_sim, df_ref['time_perc'], df_ref['pelvis_ty'])
    tilt_ref_interp = np.interp(time_sim, df_ref['time_perc'], df_ref['pelvis_tilt'])  # already in degrees

    # Pelvis Tx
    tx_sim = df_sim['/jointset/ground_pelvis/pelvis_tx/value']
    tx_rmse = np.sqrt(np.mean((tx_sim - pelvis_tx_ref_interp) ** 2))
    axes[0].plot(time_sim, tx_sim, label='Sim')
    axes[0].plot(time_sim, pelvis_tx_ref_interp, label='Ref')
    axes[0].set_ylabel('Position (m)')
    axes[0].set_xlabel('Task completion (%)')
    axes[0].set_title('Pelvis Tx')
    axes[0].text(0.05, 0.95, f'RMSE = {tx_rmse:.4f} m', transform=axes[0].transAxes,
                 verticalalignment='top', fontsize=fontsize*0.9)
    axes[0].legend(frameon=False)

    # Pelvis Ty
    ty_sim = df_sim['/jointset/ground_pelvis/pelvis_ty/value']
    ty_rmse = np.sqrt(np.mean((ty_sim - pelvis_ty_ref_interp) ** 2))
    axes[1].plot(time_sim, ty_sim, label='Sim')
    axes[1].plot(time_sim, pelvis_ty_ref_interp, label='Ref')
    axes[1].set_ylabel('Position (m)')
    axes[1].set_xlabel('Task completion (%)')
    axes[1].set_title('Pelvis Ty')
    axes[1].text(0.05, 0.95, f'RMSE = {ty_rmse:.4f} m', transform=axes[1].transAxes,
                 verticalalignment='top', fontsize=fontsize*0.9)
    axes[1].legend(frameon=False)

    # Pelvis Tilt
    tilt_sim_rad = df_sim['/jointset/ground_pelvis/pelvis_tilt/value']
    tilt_sim_deg = np.rad2deg(tilt_sim_rad)   # convert to degrees
    tilt_rmse = np.sqrt(np.mean((tilt_sim_deg - tilt_ref_interp) ** 2))
    axes[2].plot(time_sim, tilt_sim_deg, label='Sim')
    axes[2].plot(time_sim, tilt_ref_interp, label='Ref')
    axes[2].set_ylabel('Angle (deg)')
    axes[2].set_xlabel('Task completion (%)')
    axes[2].set_title('Pelvis Tilt')
    axes[2].text(0.05, 0.95, f'RMSE = {tilt_rmse:.2f}°', transform=axes[2].transAxes,
                 verticalalignment='top', fontsize=fontsize*0.9)
    axes[2].legend(frameon=False)

    # Clean up all axes
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig_title = 'pelvis_kinematics_tx_ty_tilt.png'  # include file extension
    full_path = os.path.join(fig_path, fig_title)      # combine path and filename

    plt.tight_layout()
    plt.savefig(full_path, dpi=300)
    # plt.show()

def plot_joint_kinematics(df_sim, df_ref, fontsize, figsize, side, fig_path):
    plt.rcParams.update({'font.size': fontsize})

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize, sharex=True)
    time_sim = df_sim['time_perc']

    # Joint names and column keys
    joints = {
        'Hip flexion': {
            'sim_col': f'/jointset/hip_{side}/hip_flexion_{side}/value',
            'ref_col': f'hip_flexion_{side}',
            'axis': axes[0]
        },
        'Knee flexion': {
            'sim_col': f'/jointset/walker_knee_{side}/knee_angle_{side}/value',
            'ref_col': f'knee_angle_{side}',
            'axis': axes[1]
        },
        'Ankle flexion': {
            'sim_col': f'/jointset/ankle_{side}/ankle_angle_{side}/value',
            'ref_col': f'ankle_angle_{side}',
            'axis': axes[2]
        }
    }

    for joint_label, info in joints.items():
        sim_angle_rad = df_sim[info['sim_col']]
        ref_angle_deg_interp = np.interp(time_sim, df_ref['time_perc'], df_ref[info['ref_col']])
        sim_angle_deg = np.rad2deg(sim_angle_rad)

        # Calculate RMSE
        rmse = np.sqrt(np.mean((sim_angle_deg - ref_angle_deg_interp) ** 2))

        # Plot
        ax = info['axis']
        ax.plot(time_sim, sim_angle_deg, label='Sim')
        ax.plot(time_sim, ref_angle_deg_interp, label='Ref')
        ax.set_ylabel('Angle (deg)')
        ax.set_xlabel('Task completion (%)')
        ax.set_title(f'{joint_label} ({side.upper()})')
        ax.text(0.05, 0.95, f'RMSE = {rmse:.2f}°',
                transform=ax.transAxes, verticalalignment='top', fontsize=fontsize*0.9)
        ax.legend(frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


    fig_title = f'joint_kinematics_hip_knee_ankle_{side}.png'  # include file extension
    full_path = os.path.join(fig_path, fig_title)      # combine path and filename

    plt.tight_layout()
    plt.savefig(full_path, dpi=300)
    # plt.show()

def read_save_sto_metadata(sto_path, save_path, save_basename='moco_metadata'):
    """
    Reads the header metadata (key=value pairs) from a Moco .sto file,
    and saves the result in both .json and .csv formats.

    Parameters:
        sto_path (str): Path to the .sto file.
        save_path (str): Directory where metadata files should be saved.
        save_basename (str): Base name (without extension) for saved files.

    Returns:
        dict: Metadata keys and their values as strings, numbers, or booleans.
    """
    metadata = {}

    # --- Read Metadata ---
    with open(sto_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('endheader') or line.startswith('time'):
                break
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                # Convert value to bool, int, or float if possible
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                else:
                    try:
                        value = float(value) if '.' in value or 'e' in value.lower() else int(value)
                    except ValueError:
                        pass  # Keep value as string
                metadata[key] = value

    # --- Prepare save directory ---
    os.makedirs(save_path, exist_ok=True)  # Create folder if not exist

    # --- Build file paths ---
    json_path = os.path.join(save_path, f'{save_basename}.json')
    csv_path = os.path.join(save_path, f'{save_basename}.csv')

    # --- Save as JSON ---
    # with open(json_path, 'w') as f_json:
        # json.dump(metadata, f_json, indent=4)

    # --- Save as CSV ---
    with open(csv_path, 'w', newline='') as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=metadata.keys())
        writer.writeheader()
        writer.writerow(metadata)

    print(f"Metadata saved to:\n  - {json_path}\n  - {csv_path}")
    return metadata


def read_sto_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find the index of the 'endheader' line
    header_end = next(i for i, line in enumerate(lines) if line.strip() == 'endheader')
    
    # The next line after 'endheader' contains the column names
    column_names = lines[header_end + 1].strip().split('\t')
    
    # Read the data, starting from the line after the column names
    df = pd.read_csv(file_path, 
                     delimiter='\t', 
                     skiprows=header_end + 2,  # Skip header + column names row
                     names=column_names)  # Use the extracted column names
    return df

# def readMotionFile(filename):
#     """ Reads OpenSim .sto files.
#     Parameters
#     ----------
#     filename: absolute path to the .sto file
#     Returns
#     -------
#     header: the header of the .sto
#     labels: the labels of the columns
#     data: an array of the data
#     """

#     if not os.path.exists(filename):
#         print('file do not exists')

#     file_id = open(filename, 'r')

#     # read header
#     next_line = file_id.readline()
#     print(file_id)
#     exit()
#     header = [next_line]
#     nc = 0
#     nr = 0
#     while not 'endheader' in next_line:
#         if 'datacolumns' in next_line:
#             nc = int(next_line[next_line.index(' ') + 1:len(next_line)])
#         elif 'datarows' in next_line:
#             nr = int(next_line[next_line.index(' ') + 1:len(next_line)])
#         elif 'nColumns' in next_line:
#             nc = int(next_line[next_line.index('=') + 1:len(next_line)])
#         elif 'nRows' in next_line:
#             nr = int(next_line[next_line.index('=') + 1:len(next_line)])

#         next_line = file_id.readline()
#         header.append(next_line)

#     # process column labels
#     next_line = file_id.readline()
#     if next_line.isspace() == True:
#         next_line = file_id.readline()

#     labels = next_line.split()

#     # get data
#     data = []
#     for i in range(1, nr + 1):
#         d = [float(x) for x in file_id.readline().split()]
#         data.append(d)

#     file_id.close()

#     return header, labels, data

def apply_low_pass (df, cutoff, order):

    fs = 1 / np.mean(np.diff(df['time']))  # sampling frequency in Hz
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist       # normalized cutoff

    # Design Butterworth low-pass filter
    b, a = butter(N=order, Wn=normal_cutoff, btype='low', analog=False)

    # Apply filter to each signal column
    df_filtered = df.copy()
    signal_columns = [col for col in df.columns if col != 'time' or 'time_perc']  # exclude 'time' column
    
    for col in signal_columns:
        df_filtered[col] = filtfilt(b, a, df[col])  # zero-phase filtering

    return df_filtered

