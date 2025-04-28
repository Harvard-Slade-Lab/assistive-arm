import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mticker
from matplotlib import rcParams
import matplotlib.gridspec as gridspec
from matplotlib import colors as mcolors


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
    fig, ax = plt.subplots(figsize=(8, 4))
    # ax.set_title(f"Subject {subject_name}, Scores over time")
    ax.set_xlabel("Trial", fontsize=14)
    ax.set_ylabel("Score", fontsize=14)

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
        ax.plot(x_values, group_scores, color=colors[tag_index], label=name_tag_mapping[tag], linewidth=2)

    # Customize plot
    # plt.ylim((-0.1, 0.4))
    # ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path / f"{subject_name}_scores.pdf", format="pdf", bbox_inches="tight")
    # plt.show()


def plot_mean_std_score(subject_name, session_data, name_tag_mapping, save_path):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(f"Subject {subject_name}, Scores over time")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Score")

    # Generate colors for each tag
    colors = plt.cm.viridis(np.linspace(0, 1, len(session_data["ASSISTED"]["FIRST_TAGS"])))

    # Variables for pooled standard deviation calculation
    all_std_devs = []
    all_sample_sizes = []

    cv_values = []  # Store Coefficient of Variation (CV) for each tag
    peak_cv_values = []  # Store peak CV for each tag

    # Plot scores for each tag
    i = 0
    for tag_index, tag in enumerate(session_data["ASSISTED"]["FIRST_TAGS"]):
        tag_info = session_data["ASSISTED"][tag]["LOG_INFO"]
        group_scores = np.array([info["Score"] for info in tag_info])
        
        # Compute mean and standard deviation
        mean_score = np.mean(group_scores)
        # max_score = np.max(group_scores)
        std_score = np.std(group_scores, ddof=1)  # ddof=1 for sample std dev
        
        # Compute Coefficient of Variation (CV)
        if mean_score != 0:
            cv_values.append(std_score / mean_score)
            # peak_cv_values.append(std_score / max_score)

        
        # Store for pooled standard deviation
        all_std_devs.append(std_score)
        all_sample_sizes.append(len(group_scores))
        
        # Determine x-axis values
        x_values = np.arange(i, i + len(group_scores))
        i += len(group_scores)
        
        # Plot the group scores
        ax.plot(x_values, group_scores, color=colors[tag_index], label=name_tag_mapping[tag])
        
        # Plot mean as a horizontal line
        # ax.axhline(mean_score, color=colors[tag_index], linestyle="dashed", alpha=0.7)
        
        # Fill region for standard deviation
        ax.fill_between(
            x_values, mean_score - std_score, mean_score + std_score,
            color=colors[tag_index], alpha=0.2
        )

    # Compute average CV
    average_cv = np.mean(cv_values) if cv_values else 0
    peak_cv = np.mean(peak_cv_values) if peak_cv_values else 0

    # Print noise metrics
    print(f"Average Coefficient of Variation (CV): {average_cv:.4f}")
    print(f"Average Peak Coefficient of Variation (CV): {peak_cv:.4f}")

    # Customize plot
    # plt.ylim((-0.1, 0.4))
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path / f"{subject_name}_mean_std_scores.pdf", format="pdf", bbox_inches="tight")






def plot_mvic_data(subject, session_data, emg_config, mvic_dir):
    """
    Plot the MVIC data for a subject's session.

    Args:
        subject (str): Subject name.
        session_data (dict): Session data containing MVIC information.
        emg_config (dict): EMG configuration containing sampling frequency.
        mvic_dir (Path): Directory to save the plot.
    """
    fontsize = 20

    muscle_colors_hex = {
    "RF": "#006FFF",
    "VM": "#60BA46",
    "BF": "#FA9D00",
    "SO": "#946635",
    "TA": "#A400C7",
    "G": "#ED3624",
    }
    # Convert hex to RGB (0-1 scale) while keeping it in a dictionary
    muscle_colors = {muscle: mcolors.hex2color(hex_code) for muscle, hex_code in muscle_colors_hex.items()}


    # Extract sampling information
    sampling_freq = emg_config["EMG_FREQUENCY"]
    sampling_interval = 1 / sampling_freq

    # Group columns by their base name
    grouped_columns = {}
    for col in session_data["MVIC"]["Filtered"].columns:
        base_name = col.rsplit("_", 1)[0]  # Extract base name
        grouped_columns.setdefault(base_name, []).append(col)

    n_cols = len(grouped_columns)
    fig, axs = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3.5), sharey=True)
    # fig.suptitle(f"Subject {subject} MVIC")

    mvic = session_data["MVIC"]["Filtered"]
    # crop data from 0 to 5 seconds
    session_data["MVIC"]["Filtered"] = mvic.loc[(mvic.index >= 0) & (mvic.index <= 5)]

    # change order to match order of the colors
    grouped_columns = dict(sorted(grouped_columns.items(), key=lambda item: list(muscle_colors.keys()).index(item[0])))

    for j, (base_name, cols) in enumerate(grouped_columns.items()):
        if n_cols == 1:
            for col in cols:
                axs.plot(session_data["MVIC"]["Filtered"][col], label=col, linewidth=3, color=muscle_colors[base_name])
            axs.set_title(f"{base_name}", fontsize=fontsize)
            axs.set_xlabel("Time $[s]$", fontsize=fontsize)
            axs.set_ylabel("EMG signal $[mV]$", fontsize=fontsize)
            # handles, labels = axs.get_legend_handles_labels()
            axs.legend(loc='upper left', fontsize=12)
        else:
            for col in cols:
                # if "RIGHT" in col, set alpha to 0.5
                alpha = 0.5 if "RIGHT" in col else 1
                axs[j].plot(session_data["MVIC"]["Filtered"][col], label=col, linewidth=3, color=muscle_colors[base_name], alpha=alpha)
            axs[j].set_title(f"{base_name}", fontsize=fontsize)
            axs[j].set_xlabel("Time $[s]$", fontsize=fontsize)
            axs[0].set_ylabel("EMG signal $[mV]$", fontsize=fontsize)
            axs[j].legend(loc='upper left', fontsize=12)

    # handles, labels = axs[0].get_legend_handles_labels() if n_cols > 1 else axs.get_legend_handles_labels()

    # # remove the muscle name from the label
    # labels = [label.split("_")[0] for label in labels]
    # # change names to _LEFT and _RIGHT
    # labels = [label.replace("_", " ") for label in labels]
    # labels = [label.replace("LEFT", "Left") for label in labels]
    # labels = [label.replace("RIGHT", "Right") for label in labels]

    # Add a legend for the entire figure
    # fig.legend(handles, labels, loc='upper right', fontsize=10, bbox_to_anchor=(0.85, 1))

    plt.tight_layout()
    plt.savefig(mvic_dir / f"{subject}_MVIC.pdf", format="pdf", bbox_inches="tight")



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
                plt.savefig(profile_plot_dir / f"{subject.name}_all_profiles_{direction}.svg", dpi=500, bbox_inches='tight')
                plt.close()


def plot_emg_and_force_profiles_with_means(session_data, name_tag_mapping, assisted_mean, unpowered_means, save_dir, sampling_frequency):
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
                # cropped_mean = detect_peak_and_crop(df)
                cropped_mean = df
                cropped_time = np.linspace(0, len(cropped_mean) / sampling_frequency, len(cropped_mean))
                ax.plot(cropped_time, cropped_mean[muscle], linestyle="--", alpha=0.6, color=gray_shades[idx])

            # Plot overall means and tag mean
            for iter, unpowered_mean in enumerate(unpowered_means):
                ax.plot(unpowered_mean[muscle], label=f"Unpowered_{iter}", linewidth=2)
            ax.plot(time, assisted_mean[muscle], label="Assisted Mean", color=colors["assisted_mean"], linewidth=2)
            ax.plot(time, tag_mean[muscle], label=f"{tag} Mean", color=colors["tag_mean"], linewidth=2)

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
        plt.savefig(save_path, dpi=500, bbox_inches="tight")
        plt.show()
        plt.close()


def plot_tag_means_with_individuals(session_data, assisted_mean, unpowered_means, name_tag_mapping, save_dir):
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

            # Plot unpowered mean
            for iter, unpowered_mean in enumerate(unpowered_means):
                ax.plot(unpowered_mean[muscle], label=f"Unpowered_{iter}", linewidth=2)

            # Plot the tag mean
            ax.plot(tag_mean[muscle], label=f"{tag} Mean", color=colors[2], linewidth=2)

            # Plot individual DataFrames for the tag
            for idx, df in enumerate(individual_dfs):
                # cropped_mean = detect_peak_and_crop(df)
                cropped_mean = df
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
        plt.savefig(save_dir / f"{name_tag_mapping[tag]}.svg", dpi=500, bbox_inches="tight")
        plt.close()


def plot_means(assisted_mean, unpowered_means, save_dir):
    # Create a new figure with 2x2 subplots
    n = len(assisted_mean.columns)
    fig, axes = plt.subplots(int(np.round(n/2)), 2, figsize=(12, int(np.round(n/2))*4))
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
        for iter, unpowered_mean in enumerate(unpowered_means):
            ax.plot(unpowered_mean[muscle], label=f"Unpowered_{iter}", linewidth=2)

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
    plt.savefig(save_dir / "assisted_vs_unassisted_means.svg", dpi=500, bbox_inches="tight")
    plt.close()


def plot_means_MVIC(assisted_mean, unpowered_means, save_dir, max_values):
    # Create a new figure with 2x2 subplots
    n = len(assisted_mean.columns)
    fig, axes = plt.subplots(int(np.round(n/2)), 2, figsize=(12, int(np.round(n/2))*4))
    axes = axes.flatten()  # Flatten the 2x2 array of axes

    # Define muscle groups
    muscle_groups = assisted_mean.columns  # Assumes DataFrame columns represent muscle groups
    colors = ["blue", "red"]

    # Plot data for each muscle group in its subplot
    for i, muscle in enumerate(muscle_groups):
        ax = axes[i]
        max_val = max_values[muscle]  # Get max value for scaling
        
        # Plot assisted mean (scaled)
        ax.plot(assisted_mean[muscle] / max_val, label="Assisted", color=colors[0], linewidth=2)

        # Plot unpowered mean (scaled)
        for iter, unpowered_mean in enumerate(unpowered_means):
            ax.plot(unpowered_mean[muscle] / max_val, label=f"Unpowered_{iter}", linewidth=2)

        # Add title and legend for the subplot
        ax.set_title(muscle)
        ax.set_xlabel("Samples")  # Adjust as needed for your x-axis
        ax.set_ylabel("Normalized Mean EMG Signal")
        ax.grid(linestyle="--", alpha=0.5)
        if i == 0:
            ax.legend(fontsize="small")

    # Add a main title for the figure
    plt.suptitle("Comparison of Assisted vs Unpowered EMG Means (Normalized)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust spacing

    # Save the plot
    plt.savefig(save_dir / "assisted_vs_unassisted_means.svg", dpi=500, bbox_inches="tight")
    plt.close()


def plot_overall_deltas(delta_dir, session_data, name_tag_mapping, muscles, subject, session):
    deltas = {muscle: [] for muscle in muscles}
    
    # Compute mean of unassisted trials
    unpowered_means = {muscle: 0 for muscle in muscles}
    num_unpowered = len(session_data["UNPOWERED"]["FIRST_TAGS"])

    for tag in session_data["UNPOWERED"]["FIRST_TAGS"]:
        for muscle in muscles:
            unpowered_means[muscle] += session_data["UNPOWERED"][tag]["EMG"]["Mean"][muscle]

    # Get the average unassisted values
    for muscle in muscles:
        unpowered_means[muscle] /= num_unpowered

    # Compute deltas for assisted trials
    for tag in session_data["ASSISTED"]["FIRST_TAGS"]:
        profile_name = name_tag_mapping[tag]
        profile_mean = session_data["ASSISTED"][tag]["EMG"]["Mean"]

        for muscle in muscles:
            delta = (unpowered_means[muscle] - profile_mean[muscle]) / unpowered_means[muscle]
            deltas[muscle].append(delta)

    # Compute mean delta per muscle
    mean_deltas = {muscle: np.mean(deltas[muscle]) for muscle in muscles}

    # Plot results
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["RF", "VM"]
    left_means = [mean_deltas["RF_L"], mean_deltas["VM_L"]]
    right_means = [mean_deltas["RF_R"], mean_deltas["VM_R"]]

    x = np.arange(len(labels))  # Label locations
    width = 0.35  # Bar width

    rects1 = ax.bar(x - width/2, left_means, width, label="Left")
    rects2 = ax.bar(x + width/2, right_means, width, label="Right")

    ax.set_ylabel("Mean EMG Reduction (%)")
    ax.set_title(f"EMG Reduction for {subject.name} - Session {session}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.savefig(delta_dir / f"emg_deltas_{subject.name}_{session}.png")
    plt.close()




def plot_all_motor_data(session_data, name_tag_mapping, plot_dir):

    for profile in session_data["ASSISTED"]["FIRST_TAGS"]:
        i = 0
        for motor_log in session_data["ASSISTED"][profile]["MOTOR_DATA"]:
            i += 1

            colors = iter(rcParams['axes.prop_cycle'].by_key()['color'])

            filename = plot_dir / f"motor_{name_tag_mapping[profile]}_{i}.svg"

            # Use GridSpec to control subplot layouts
            fig = plt.figure(figsize=(8, 9))
            gs = gridspec.GridSpec(5, 1, height_ratios=[1, 1, 1, 1, 1.2])  # Last subplot is slightly taller
            fig.suptitle(name_tag_mapping[profile])

            # Subplots with shared x-axis
            ax0 = fig.add_subplot(gs[0])
            ax1 = fig.add_subplot(gs[1], sharex=ax0)
            ax2 = fig.add_subplot(gs[2], sharex=ax0)
            ax3 = fig.add_subplot(gs[3], sharex=ax0)

            # Subplot with independent x-axis
            ax4 = fig.add_subplot(gs[4])

            # Plot theta_2
            ax0.plot(motor_log.index, motor_log["theta_2"], label=r"$\theta_2$", color=colors.__next__())
            ax0.set_ylabel(r"$\theta_2$ (rad)")
            handles0, labels0 = ax0.get_legend_handles_labels()

            # Plot torques
            ax1.plot(motor_log.index, motor_log["target_tau_1"], label=r"Target $\tau_1$", color=colors.__next__())
            ax1.plot(motor_log.index, motor_log["measured_tau_1"], label=r"Measured $\tau_1$", color=colors.__next__())
            ax1.plot(motor_log.index, motor_log["target_tau_2"], label=r"Target $\tau_2$", color=colors.__next__())
            ax1.plot(motor_log.index, motor_log["measured_tau_2"], label=r"Measured $\tau_2$", color=colors.__next__())
            ax1.set_ylabel("Torques (Nm)", fontsize=12)
            handles1, labels1 = ax1.get_legend_handles_labels()

            # Plot forces F_X and F_Y
            ax2.plot(motor_log.index, motor_log["F_X"], label=r"$F_X$", color=colors.__next__())
            ax2.plot(motor_log.index, motor_log["F_Y"], label=r"$F_Y$", color=colors.__next__())
            ax2.set_ylabel("Forces (N)")
            handles2, labels2 = ax2.get_legend_handles_labels()

            # Plot STS percentage
            ax3.plot(motor_log.index, motor_log["Percentage"], label="STS %", color=colors.__next__())
            ax3.axhline(y=100, linestyle="--", color="black")
            ax3.axhline(y=0, linestyle="--", color="black")
            ax3.set_ylabel("STS %")
            ax3.set_xlabel('Time (s)')
            handles3, labels3 = ax3.get_legend_handles_labels()

            # Plot actual forces with a different x-axis
            profile_data = session_data["ASSISTED"][profile]["FORCE_PROFILE"]
            ax4.plot(profile_data.index, profile_data["force_X"], label=r"Actual $F_X$", color=colors.__next__())
            ax4.plot(profile_data.index, profile_data["force_Y"], label=r"Actual $F_Y$", color=colors.__next__())
            ax4.set_ylabel("Forces (N)")
            ax4.set_xlabel('Index')
            handles4, labels4 = ax4.get_legend_handles_labels()

            # Combine all legend handles and labels
            handles = handles0 + handles1 + handles2 + handles3 + handles4
            labels = labels0 + labels1 + labels2 + labels3 + labels4
            fig.legend(handles, labels, loc='upper center', ncols=6, bbox_to_anchor=(0.5, 0.97), fontsize=10)

            # Adjust layout
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            # Save the plot
            plt.savefig(filename.with_suffix('.png'), dpi=500, format='png')
            plt.close()
