import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
  
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
            l = df_sim[left]
            r = df_sim[right]
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
            l = df_sim[left]
            r = df_sim[right]
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
            l = df_sim[left]
            r = df_sim[right]
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
            l = df_sim[left]
            r = df_sim[right]
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
        ax.set_ylabel('Activation')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=fontsize)

    plt.tight_layout()
    fig_title = 'muscle_activation.png'  # include file extension
    full_path = os.path.join(fig_path, fig_title)      # combine path and filename
    plt.savefig(full_path, dpi=300)
    # plt.show()

def plot_residual_force_torque(df_sim, fontsize, figsize, fig_path):
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize, sharex=True)

    # Make sure axes is a flat 1D array (just in case):
    axes = axes.flatten()

    axes[0].plot(df_sim['time_perc'], df_sim['/forceset/reserve_jointset_ground_pelvis_pelvis_tx'], label='Pelvis x')
    axes[0].plot(df_sim['time_perc'], df_sim['/forceset/reserve_jointset_ground_pelvis_pelvis_ty'], label='Pelvis y')
    axes[0].plot(df_sim['time_perc'], df_sim['/forceset/reserve_jointset_ground_pelvis_pelvis_tz'], label='Pelvis z')
    axes[0].set_xlabel('Task completion (%)')
    axes[0].set_ylabel('Force (N)')
    axes[0].legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
    axes[0].set_title('Pelvis Residual Forces')

    axes[1].plot(df_sim['time_perc'], df_sim['/forceset/reserve_jointset_ground_pelvis_pelvis_tilt'], label='Pelvis tilt')
    axes[1].plot(df_sim['time_perc'], df_sim['/forceset/reserve_jointset_ground_pelvis_pelvis_list'], label='Pelvis list')
    axes[1].plot(df_sim['time_perc'], df_sim['/forceset/reserve_jointset_ground_pelvis_pelvis_rotation'], label='Pelvis rotation')
    axes[1].set_xlabel('Task completion (%)')
    axes[1].set_ylabel('Torque (Nm)')
    axes[1].legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
    axes[1].set_title('Pelvis Residual Torques')

    axes[2].plot(df_sim['time_perc'], df_sim['/forceset/reserve_jointset_hip_l_hip_adduction_l'], label='hip_adduction_l')
    axes[2].plot(df_sim['time_perc'], df_sim['/forceset/reserve_jointset_hip_l_hip_rotation_l'], label='hip_rotation_l')
    axes[2].plot(df_sim['time_perc'], df_sim['/forceset/reserve_jointset_walker_knee_l_knee_angle_l'], label='knee_angle_l')
    axes[2].plot(df_sim['time_perc'], df_sim['/forceset/reserve_jointset_ankle_l_ankle_angle_l'], label='ankle_angle_l')
    axes[2].plot(df_sim['time_perc'], df_sim['/forceset/reserve_jointset_subtalar_l_subtalar_angle_l'], label='subtalar_angle_l')
    axes[2].plot(df_sim['time_perc'], df_sim['/forceset/reserve_jointset_mtp_l_mtp_angle_l'], label='mtp_angle_l')

    axes[2].set_xlabel('Task completion (%)')
    axes[2].set_ylabel('Torque (Nm)')
    axes[2].legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
    axes[2].set_title('Left leg Residual Torques')

    axes[3].plot(df_sim['time_perc'], df_sim['/forceset/reserve_jointset_hip_r_hip_adduction_r'], label='hip_adduction_r')
    axes[3].plot(df_sim['time_perc'], df_sim['/forceset/reserve_jointset_hip_r_hip_rotation_r'], label='hip_rotation_r')
    axes[3].plot(df_sim['time_perc'], df_sim['/forceset/reserve_jointset_walker_knee_r_knee_angle_r'], label='knee_angle_r')
    axes[3].plot(df_sim['time_perc'], df_sim['/forceset/reserve_jointset_ankle_r_ankle_angle_r'], label='ankle_angle_r')
    axes[3].plot(df_sim['time_perc'], df_sim['/forceset/reserve_jointset_subtalar_r_subtalar_angle_r'], label='subtalar_angle_r')
    axes[3].plot(df_sim['time_perc'], df_sim['/forceset/reserve_jointset_mtp_r_mtp_angle_r'], label='mtp_angle_r')

    axes[3].set_xlabel('Task completion (%)')
    axes[3].set_ylabel('Torque (Nm)')
    axes[3].legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
    axes[3].set_title('Right leg Residual Torques')

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=fontsize)

    fig_title = 'pelvis_residual_force_torque.png'  # include file extension
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

def readMotionFile(filename):
    """ Reads OpenSim .sto files.
    Parameters
    ----------
    filename: absolute path to the .sto file
    Returns
    -------
    header: the header of the .sto
    labels: the labels of the columns
    data: an array of the data
    """

    if not os.path.exists(filename):
        print('file do not exists')

    file_id = open(filename, 'r')

    # read header
    next_line = file_id.readline()
    print(file_id)
    exit()
    header = [next_line]
    nc = 0
    nr = 0
    while not 'endheader' in next_line:
        if 'datacolumns' in next_line:
            nc = int(next_line[next_line.index(' ') + 1:len(next_line)])
        elif 'datarows' in next_line:
            nr = int(next_line[next_line.index(' ') + 1:len(next_line)])
        elif 'nColumns' in next_line:
            nc = int(next_line[next_line.index('=') + 1:len(next_line)])
        elif 'nRows' in next_line:
            nr = int(next_line[next_line.index('=') + 1:len(next_line)])

        next_line = file_id.readline()
        header.append(next_line)

    # process column labels
    next_line = file_id.readline()
    if next_line.isspace() == True:
        next_line = file_id.readline()

    labels = next_line.split()

    # get data
    data = []
    for i in range(1, nr + 1):
        d = [float(x) for x in file_id.readline().split()]
        data.append(d)

    file_id.close()

    return header, labels, data

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

