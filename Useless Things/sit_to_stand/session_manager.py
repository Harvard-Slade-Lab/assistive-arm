import os
import re
import csv
import yaml
import zipfile
import pandas as pd
from socket_server import SocketServer

from pathlib import Path
from datetime import datetime

# Directory to directly save the logs to remote host (laptop used for data analysis), also need to add Host in ~/.ssh/config
PROJECT_DIR_REMOTE = Path("/Users/nathanirniger/Desktop/MA/Project/Code/assistive-arm")

class SessionManager:
    """Handles session logging and data storage."""
    
    def __init__(self, subject_id):
        self.subject_folder = Path(f"./subject_logs/subject_{subject_id}")
        self.session_dir, self.session_remote_dir = self.set_up_logging_dir()
        self.roll_angles = None
        self.max_roll_angle = None
        self.yaml_path = None

        self.get_yaml_path("device_height_calibration")
        self.load_device_height_calibration()

    def set_up_logging_dir(self):
        """Set up directories for logging."""
        current_date = datetime.now()
        session_dir = self.subject_folder / f"{current_date.strftime('%B_%d')}" / "Motor"
        session_dir.mkdir(parents=True, exist_ok=True)
        
        session_remote_dir = Path(f"{PROJECT_DIR_REMOTE}/subject_logs") / session_dir.relative_to("subject_logs")
        try:
            os.system(f"ssh macbook 'mkdir -p {session_remote_dir}'")
        except Exception as e:
            print(f"Error creating remote directory: {e}")
        
        return session_dir, session_remote_dir

    def save_log_or_delete(self, log_path: Path, successful: bool):
        """Save or delete log file based on success of the session."""
        # if successful:
        print("\nSending logfile to Mac...")
        try:
            os.system(f"scp {log_path} macbook:{self.session_remote_dir}")
        except Exception as e:
            print(f"Error transferring file: {e}")
        # else:
        #     print(f"Removing {log_path}")
        #     os.remove(log_path)

    def load_device_height_calibration(self) -> pd.Series:
        """Load the device height calibration data."""
        # If the file exists, load the calibration data
        if self.yaml_path is not None:
            if self.yaml_path.exists():
                with open(self.yaml_path, "r") as f:
                    calibration_data = yaml.safe_load(f)
                self.roll_angles = pd.DataFrame(calibration_data["roll_angles"])
                self.roll_angles.index = calibration_data["Percentage"]
                self.roll_angles.columns = ["roll_angles"]
                self.max_roll_angle = self.roll_angles["roll_angles"].max()
                return 1
            else:
                print("No calibration data found.")
                return None

    def get_yaml_path(self, yaml_name: str) -> Path:
        yaml_file = f"{yaml_name}.yaml"
        self.yaml_path = self.session_dir / yaml_file
        return self.session_dir / yaml_file
    

def get_logger(log_name: str, session_manager: SessionManager, socket_server: SocketServer = None) -> tuple[Path, csv.writer]:
    """
    Set up a logger for various tasks in the script. Return the log file path and logger.

    Args:
        log_name (str): Name for the log file.
        session_manager (SessionManager): Instance managing the session directory.
        socket_server (SocketServer): Instance of the command socket server.
    Returns:
        tuple[Path, csv.writer]: The path to the log file and the CSV writer instance.
    """
    
    # Variables to be logged
    logged_vars = ["Percentage", "roll_angle", "target_tau_1", "measured_tau_1", "theta_1", "velocity_1", 
                   "target_tau_2", "measured_tau_2", "theta_2", "velocity_2", "EE_X", "EE_Y"]

    # Use session_manager to determine the next sample number
    if not socket_server:
        sample_num = get_next_sample_number(session_manager.session_dir, log_name)
    # Currently doesn't matter if server is passed or not, but maybe we want to add more info in the future
    else: 
        sample_num = get_next_sample_number(session_manager.session_dir, f"{log_name}")
    
    # Format log file path
    if not socket_server:
        log_file = f"{log_name}_{sample_num:02}.csv"
        # log_file = f"{log_name}.csv"
    else:
        log_file = f"{log_name}_{sample_num:02}.csv"  # session_manager.profile_name
        # log_file = f"{log_name}_Profile_{server.profile_name}.csv"  # session_manager.profile_name
    log_path = session_manager.session_dir / log_file
    log_path.touch(exist_ok=True)  # Ensure the file is created

    # Extract profile details from log_name CURERNTLY NOT USED
    # parts = log_name.split('_')
    # numbers = [int(parts[i + 1]) for i in range(0, len(parts) - 1, 2) if parts[i + 1].isdigit()]

    # Set up the CSV writer
    with open(log_path, "w") as fd:
        writer = csv.writer(fd)
        
        # if numbers:
        #     # Add profile details force1_end_time, force1_peak_force, force2_start_time, force2_peak_time, force2_peak_force, force2_end_time
        #     writer.writerow(["force1_end_time", numbers[0]])
        #     writer.writerow(["force1_peak_force", numbers[1]])
        #     writer.writerow(["force2_start_time", numbers[2]])
        #     writer.writerow(["force2_peak_time", numbers[3]])
        #     writer.writerow(["force2_peak_force", numbers[4]])
        #     writer.writerow(["force2_end_time", numbers[5]])
        
        # Write headers
        writer.writerow(["time"] + logged_vars)

    # Open the CSV file for appending and set up the logger
    csv_file = open(log_path, "a").__enter__()
    task_logger = csv.writer(csv_file)

    return log_path, task_logger


def get_next_sample_number(session_dir: Path, log_name: str) -> int:
    pattern = re.compile(rf"{log_name}_(\d+).csv")
    max_sample = 0
    for file in session_dir.iterdir():
        match = pattern.match(file.name)
        if match:
            sample_number = int(match.group(1))
            max_sample = max(max_sample, sample_number)
    return max_sample + 1  # First sample will be 1 if no files match


def save_calibrated_profiles(theta_2_scaled, session_manager, yaml_path, calibration_data):
    """
    Save the calibrated profiles to both local and remote directories.

    Args:
        theta_2_scaled (pd.Series): Scaled theta_2 values.
        profile_dir (Path): Directory where the profiles are stored.
        session_manager (SessionManager): Instance for handling session directory and remote sync.
        yaml_path (Path): Path to the YAML file for calibration data.
        calibration_data (dict): Calibration data to save in YAML.
    """
    spline_path = Path("./torque_profiles")
    calibrated_dir = spline_path / "calibrated"
    calibrated_dir.mkdir(parents=True, exist_ok=True)

    # Zip and prepare for transfer
    zip_file_path = calibrated_dir / "calibrated_profiles.zip"
    with zipfile.ZipFile(zip_file_path, "w") as zip_file:
        for profile in spline_path.iterdir():
            if profile.suffix == ".csv" and "calibrated" not in profile.stem:
                # Read and calibrate profile data
                spline_profile = pd.read_csv(profile, index_col="Percentage")
                spline_profile['theta_2'] = theta_2_scaled
                
                # Save calibrated profile in the target directory
                calibrated_profile_path = calibrated_dir / f"{profile.stem}_calibrated.csv"
                spline_profile.to_csv(calibrated_profile_path, index=True)

                # Add calibrated profile to zip
                zip_file.write(calibrated_profile_path, calibrated_profile_path.name)

    # Transfer to remote host
    remote_path = session_manager.session_remote_dir / "calibrated"
    try:
        os.system(f"ssh macbook 'mkdir -p {remote_path}'")
    except Exception as e:
        print(f"Error creating remote directory: {e}")
        
    try:
        os.system(f"scp {zip_file_path} macbook:{remote_path}")
        os.system(f"ssh macbook 'unzip -oq {remote_path / zip_file_path.name} -d {remote_path}'")
        os.remove(zip_file_path)
    except Exception as e:
        print(f"Error transferring file: {e}")

    # Save calibration data in YAML and sync with the remote directory
    with open(yaml_path, "w") as f:
        yaml.dump(calibration_data, f)
    try:
        os.system(f"scp {yaml_path} macbook:{remote_path}")
    except Exception as e:
        print(f"Error transferring YAML file: {e}")
