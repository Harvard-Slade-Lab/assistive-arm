import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mticker
from matplotlib import rcParams
import matplotlib.gridspec as gridspec


from assistive_arm.utils.analyze_emg_data_processing import *



def plot_scores(subject_name, session_data, name_tag_mapping, save_path):
    """
    Plot all scores for a given subject and session, grouped by tags.

    Args:
        subject_name (str): Name of the subject.
        session_data (dict): Session data containing assisted trials and log info.
        name_tag_mapping (dict): Mapping of tags to their display names.
        save_path (Path): Directory path to save the plot.
    """
    # Create a new figure for the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(f"Subject {subject_name}, Scores over time")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Score")

    # Generate colors for each tag
    colors = plt.cm.viridis(np.linspace(0, 1, len(session_data["ASSISTED"]["FIRST_TAGS"])))

    # Plot scores for each tag
    i = 0
    for tag_index, tag in enumerate(session_data["ASSISTED"]["FIRST_TAGS"]):
        tag_info = session_data["ASSISTED"][tag]["LOG_INFO"]
        group_scores = [info[3] for info in tag_info]
        
        # Determine x-axis values
        x_values = np.arange(i, i + len(group_scores))
        i += len(group_scores)
        
        # Plot the group scores
        ax.plot(x_values, group_scores, color=colors[tag_index], label=name_tag_mapping[tag])

    # Customize plot
    plt.ylim((0, 3))
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path / f"{subject_name}_scores.svg", dpi=500, bbox_inches="tight", format="svg")
    # plt.show()



def plot_mvic_data(subject, session_data, emg_config, mvic_dir):
    """
    Plot the MVIC data for a subject's session.

    Args:
        subject (str): Subject name.
        session_data (dict): Session data containing MVIC information.
        emg_config (dict): EMG configuration containing sampling frequency.
        mvic_dir (Path): Directory to save the plot.
    """
    # Extract sampling information
    sampling_freq = emg_config["EMG_FREQUENCY"]
    sampling_interval = 1 / sampling_freq

    # Group columns by their base name
    grouped_columns = {}
    for col in session_data["MVIC"]["Filtered"].columns:
        base_name = col.rsplit("_", 1)[0]  # Extract base name
        grouped_columns.setdefault(base_name, []).append(col)

    n_cols = len(grouped_columns)
    fig, axs = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3), sharey=True)
    fig.suptitle(f"Subject {subject} MVIC")

    for j, (base_name, cols) in enumerate(grouped_columns.items()):
        if n_cols == 1:
            for col in cols:
                axs.plot(session_data["MVIC"]["Filtered"][col], label=col)
            axs.set_title(f"{base_name}")
            axs.set_xlabel("Time (s)")
            axs.set_ylabel("EMG signal (mV)")
            handles, labels = axs.get_legend_handles_labels()
        else:
            for col in cols:
                axs[j].plot(session_data["MVIC"]["Filtered"][col], label=col)
            axs[j].set_title(f"{base_name}")
            axs[j].set_xlabel("Time (s)")
            axs[0].set_ylabel("EMG signal (mV)")

    handles, labels = axs[0].get_legend_handles_labels() if n_cols > 1 else axs.get_legend_handles_labels()

    # Add a legend for the entire figure
    fig.legend(handles, labels, loc='upper right', fontsize=10, bbox_to_anchor=(0.85, 1))

    plt.tight_layout()
    plt.savefig(mvic_dir / f"{subject}_MVIC.svg", dpi=500, bbox_inches='tight', format='svg')



def plot_all_force_profiles(subjects, subject_data, subject_dirs):
    """
    Plot all force profiles for given subjects and their sessions.

    Args:
        subjects (list): List of subject objects.
        subject_data (dict): Dictionary containing subject data.
        subject_dirs (dict): Dictionary containing subject directory information.
    """
    for subject in subjects:
        for session in iter(subject_data[subject.name].keys()):
            # Prepare directories
            profile_plot_dir = subject_dirs[subject.name][session]["plot_dir"] / "profiles"
            profile_plot_dir.mkdir(parents=True, exist_ok=True)

            # Load session data and calibration data
            session_data = subject_data[subject.name][session]["session_data"]
            optimizer = session_data["OPTIMIZER"]

            motor_dir = subject_dirs[subject.name][session]["motor_dir"]
            yaml_path = motor_dir / "device_height_calibration.yaml"

            with open(yaml_path, "r") as f:
                calibration_data = yaml.safe_load(f)
                roll_angles = pd.DataFrame(calibration_data["roll_angles"])
                max_time = len(roll_angles)

            for direction in ["X", "Y"]:
                # Initialize a single plot for all profiles
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.set_title(f"All Profiles for Subject {subject.name} in {direction}")
                ax.set_xlabel("Time")
                ax.set_ylabel("Force (N)")

                # Iterate over optimizer rows to extract and plot profiles
                for index, row in optimizer.iterrows():
                    # Extract values from the optimizer row
                    t11_p = row["params.force1_end_time_p"]
                    f11_p = row["params.force1_peak_force_p"]
                    t21_p = row["params.force2_start_time_p"]
                    t22_p = row["params.force2_peak_time_p"]
                    t23_p = row["params.force2_end_time_p"]
                    f21_p = row["params.force2_peak_force_p"]

                    # Convert percentage values to actual values
                    t11, f11, t21, t22, f21, t23 = percentage_to_actual(t11_p, f11_p, t21_p, t22_p, f21_p, t23_p, max_time)

                    # Get the profile
                    profile = get_profile(t11, f11, t21, t22, f21, t23, roll_angles)

                    # Plot the force profile on the same axes
                    ax.plot(profile[f"force_{direction}"], label=f"Profile {index} - {direction} Force")

                # Save and display the plot
                plt.savefig(profile_plot_dir / f"{subject.name}_all_profiles_{direction}.svg", dpi=500, bbox_inches='tight', format='svg')


def plot_emg_and_force_profiles_with_means(session_data, name_tag_mapping, assisted_mean, unpowered_mean, save_dir, sampling_frequency):
    """
    Plot EMG means for each muscle in stacked subplots with overall means and a combined force profile plot at the bottom.
    Time is computed from the sampling frequency and used for the x-axis.
    
    Args:
        session_data (dict): Session data containing EMG and force profile information.
        name_tag_mapping (dict): Mapping of tags to human-readable names.
        assisted_mean (DataFrame): Overall mean for the assisted condition.
        unpowered_mean (DataFrame): Overall mean for the unpowered condition.
        save_dir (Path): Directory to save the generated plots.
        sampling_frequency (float): Sampling frequency in Hz.
    """
    # Calculate time from sampling frequency
    emg_length = len(assisted_mean)
    time = np.linspace(0, emg_length / sampling_frequency, emg_length)

    for tag in session_data["ASSISTED"]["FIRST_TAGS"]:
        tags = session_data["ASSISTED"][tag]["PROFILE_TAGS"]
        tag_mean = session_data["ASSISTED"][tag]["EMG"]["Mean"]
        individual_dfs = session_data["ASSISTED"][tag]["EMG"]["Filtered"]

        # Force profile (assumed already aligned)
        force_profile = session_data["ASSISTED"][tag]["FORCE_PROFILE"]

        # Create stacked subplots: one per muscle + 1 for force profiles
        muscle_groups = tag_mean.columns
        num_muscles = len(muscle_groups)
        fig, axes = plt.subplots(num_muscles + 1, 1, figsize=(12, 4 * (num_muscles + 1)), sharex=True)

        # Define colors
        colors = {"assisted_mean": "blue", "unpowered_mean": "red", "tag_mean": "black"}
        gray_shades = cm.gray(np.linspace(0.4, 0.8, len(individual_dfs)))

        # Plot EMG for each muscle
        for i, muscle in enumerate(muscle_groups):
            ax = axes[i]

            # Plot individual data
            for idx, df in enumerate(individual_dfs):
                cropped_mean = detect_peak_and_crop(df)
                cropped_time = np.linspace(0, len(cropped_mean) / sampling_frequency, len(cropped_mean))
                ax.plot(cropped_time, cropped_mean[muscle], linestyle="--", alpha=0.6, color=gray_shades[idx])

            # Plot overall means and tag mean
            ax.plot(
                time, unpowered_mean[muscle], label="Unpowered Mean", color=colors["unpowered_mean"], linewidth=2
            )
            ax.plot(
                time, assisted_mean[muscle], label="Assisted Mean", color=colors["assisted_mean"], linewidth=2
            )
            ax.plot(
                time, tag_mean[muscle], label=f"{tag} Mean", color=colors["tag_mean"], linewidth=2
            )

            # Title and labels
            ax.set_title(f"{muscle} - EMG Signals")
            ax.set_ylabel("Mean EMG Signal")
            ax.grid(linestyle="--", alpha=0.5)
            if i == 0:
                ax.legend(fontsize="small")

        # Plot force profiles in the last subplot
        ax_force = axes[-1]
        force_time = np.linspace(0, len(force_profile) / sampling_frequency, len(force_profile))
        ax_force.plot(force_time, force_profile["force_X"], label="Force X", color="blue", linewidth=2)
        ax_force.plot(force_time, force_profile["force_Y"], label="Force Y", color="orange", linewidth=2)

        ax_force.set_title("Force Profiles")
        ax_force.set_xlabel("Time (s)")
        ax_force.set_ylabel("Force (N)")
        ax_force.grid(linestyle="--", alpha=0.5)
        ax_force.legend(fontsize="small")

        # Adjust layout and save the plot
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to avoid overlaps
        plt.suptitle(f"EMG and Force Profiles for {name_tag_mapping[tag]}")
        save_path = save_dir / f"{name_tag_mapping[tag]}_emg_force_profiles_with_means.svg"
        plt.savefig(save_path, dpi=500, bbox_inches="tight", format="svg")
        plt.show()


def plot_tag_means_with_individuals(session_data, assisted_mean, unpowered_mean, name_tag_mapping, save_dir):
    # Iterate through the assisted tags
    for tag in session_data["ASSISTED"]["FIRST_TAGS"]:
        tags = session_data["ASSISTED"][tag]["PROFILE_TAGS"]
        tag_mean = session_data["ASSISTED"][tag]["EMG"]["Mean"]
        individual_dfs = session_data["ASSISTED"][tag]["EMG"]["Filtered"]

        # Create a new figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()  # Flatten the 2x2 array of axes

        # Define muscle groups
        muscle_groups = tag_mean.columns 
        colors = ["blue", "red", "black"]

        num_individuals = len(individual_dfs)
        gray_shades = cm.gray(np.linspace(0.4, 0.8, num_individuals))

        # Plot data for each muscle group in its subplot
        for i, muscle in enumerate(muscle_groups):
            ax = axes[i]

            # Plot the overall assisted mean
            ax.plot(assisted_mean[muscle], label="Overall Assisted Mean", color=colors[0], linewidth=2)

            # Plot the overall unpowered mean
            ax.plot(unpowered_mean[muscle], label="Overall Unpowered Mean", color=colors[1], linewidth=2)

            # Plot the tag mean
            ax.plot(tag_mean[muscle], label=f"{tag} Mean", color=colors[2], linewidth=2)

            # Plot individual DataFrames for the tag
            for idx, df in enumerate(individual_dfs):
                cropped_mean = detect_peak_and_crop(df)
                ax.plot(cropped_mean[muscle], linestyle="--", alpha=0.6, color=gray_shades[idx], label=f"Individual {tags[idx]}")

            # Add title and legend for the subplot
            ax.set_title(muscle)
            ax.set_xlabel("Samples")  # Adjust as needed for your x-axis
            ax.set_ylabel("Mean EMG Signal")
            ax.grid(linestyle="--", alpha=0.5)
            if i == 0:
                ax.legend(fontsize="small")

        # Add a main title for the figure
        plt.suptitle(f"Comparison of {name_tag_mapping[tag]} Means with Overall Assisted and Unpowered Means")
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust spacing

        # Save the plot
        plt.savefig(save_dir / f"{name_tag_mapping[tag]}.svg", dpi=500, bbox_inches="tight", format="svg")
        plt.show()


def plot_means(assisted_mean, unpowered_mean, save_dir):
    # Create a new figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()  # Flatten the 2x2 array of axes

    # Define muscle groups
    muscle_groups = assisted_mean.columns  # Assumes DataFrame columns represent muscle groups
    colors = ["blue", "red"]

    # Plot data for each muscle group in its subplot
    for i, muscle in enumerate(muscle_groups):
        ax = axes[i]

        # Plot assisted mean
        ax.plot(assisted_mean[muscle], label="Assisted", color=colors[0], linewidth=2)

        # Plot unpowered mean
        ax.plot(unpowered_mean[muscle], label="Unpowered", color=colors[1], linewidth=2)

        # Add title and legend for the subplot
        ax.set_title(muscle)
        ax.set_xlabel("Samples")  # Adjust as needed for your x-axis
        ax.set_ylabel("Mean EMG Signal")
        ax.grid(linestyle="--", alpha=0.5)
        if i == 0:
            ax.legend(fontsize="small")

    # Add a main title for the figure
    plt.suptitle("Comparison of Assisted vs Unpowered EMG Means")
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust spacing

    # Save the plot
    plt.savefig(save_dir / "assisted_vs_unassisted_means.svg", dpi=500, bbox_inches="tight", format="svg")
    plt.show()
