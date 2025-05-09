import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import subprocess

PROJECT_DIR_REMOTE = Path("/Users/filippo.mariani/Desktop/Universita/Harvard/Third_Arm_Data")

class DataExporter:
    def __init__(self, parent):
        self.parent = parent

    def export_data_to_csv(self, filename_emg="EMG_data.csv", filename_acc="ACC_data.csv", filename_gyro="GYRO_data.csv", filename_or="ORIENTATION_data.csv", filename_or_debug="ORIENTATION_data_debug.csv", filename_euler="Euler_Angles.csv"):
        print("Exporting collected data...")

        # Export EMG data
        with open(filename_emg, 'w') as f_emg:
            headers = []
            for sensor_label in self.parent.complete_emg_data:
                headers.append(f'EMG {self.parent.sensor_names.get(sensor_label, sensor_label)}')
            f_emg.write(','.join(headers) + '\n')

            max_length = max(len(data) for data in self.parent.complete_emg_data.values()) if self.parent.complete_emg_data else 0
            for row_idx in range(max_length):
                row = []
                for data in self.parent.complete_emg_data.values():
                    if row_idx < len(data):
                        row.append(str(data[row_idx]))
                    else:
                        row.append("")
                f_emg.write(",".join(row) + '\n')

        # Export ACC data
        with open(filename_acc, 'w') as f_acc:
            # Build headers
            headers = []
            for sensor_label in self.parent.complete_acc_data:
                for axis in ['X', 'Y', 'Z']:
                    headers.append(f'ACC {self.parent.sensor_names.get(sensor_label, sensor_label)} {axis}')
            f_acc.write(','.join(headers) + '\n')

            # Find max length
            max_length = 0
            for sensor_data in self.parent.complete_acc_data.values():
                max_length = max(max_length, max(len(axis_data) for axis_data in sensor_data.values()))
            # Write data
            for row_idx in range(max_length):
                row = []
                for sensor_data in self.parent.complete_acc_data.values():
                    for axis in ['X', 'Y', 'Z']:
                        data = sensor_data[axis]
                        if row_idx < len(data):
                            row.append(str(data[row_idx]))
                        else:
                            row.append("")
                f_acc.write(",".join(row) + '\n')

        # Export GYRO data
        with open(filename_gyro, 'w') as f_gyro:
            # Build headers
            headers = []
            for sensor_label in self.parent.complete_gyro_data:
                for axis in ['X', 'Y', 'Z']:
                    headers.append(f'GYRO {self.parent.sensor_names.get(sensor_label, sensor_label)} {axis}')
            f_gyro.write(','.join(headers) + '\n')

            # Find max length
            max_length = 0
            for sensor_data in self.parent.complete_gyro_data.values():
                max_length = max(max_length, max(len(axis_data) for axis_data in sensor_data.values()))
            # Write data
            for row_idx in range(max_length):
                row = []
                for sensor_data in self.parent.complete_gyro_data.values():
                    for axis in ['X', 'Y', 'Z']:
                        data = sensor_data[axis]
                        if row_idx < len(data):
                            row.append(str(data[row_idx]))
                        else:
                            row.append("")
                f_gyro.write(",".join(row) + '\n')

        # Export ORIENTATION data
        with open(filename_or, 'w') as f_or:
            # Build headers
            headers = []
            for sensor_label in self.parent.complete_or_data:
                for axis in ['W', 'X', 'Y', 'Z']:
                    headers.append(f'ORIENTATION {self.parent.sensor_names.get(sensor_label, sensor_label)} {axis}')
            f_or.write(','.join(headers) + '\n')

            # Find max length
            max_length = 0
            for sensor_data in self.parent.complete_or_data.values():
                max_length = max(max_length, max(len(axis_data) for axis_data in sensor_data.values()))
            # Write data
            for row_idx in range(max_length):
                row = []
                for sensor_data in self.parent.complete_or_data.values():
                    for axis in ['W', 'X', 'Y', 'Z']:
                        data = sensor_data[axis]
                        if row_idx < len(data):
                            row.append(str(data[row_idx]))
                        else:
                            row.append("")
                f_or.write(",".join(row) + '\n')

        # Export ORIENTATION data (debug)
        with open(filename_or_debug, 'w') as f_or_debug:
            # Build headers
            headers = []
            for sensor_label in self.parent.complete_or_data_debug:
                for axis in ['W', 'X', 'Y', 'Z']:
                    headers.append(f'ORIENTATION {self.parent.sensor_names.get(sensor_label, sensor_label)} {axis}')
            f_or_debug.write(','.join(headers) + '\n')

            # Find max length
            max_length = 0
            for sensor_data in self.parent.complete_or_data_debug.values():
                max_length = max(max_length, max(len(axis_data) for axis_data in sensor_data.values()))
            # Write data
            for row_idx in range(max_length):
                row = []
                for sensor_data in self.parent.complete_or_data_debug.values():
                    for axis in ['W', 'X', 'Y', 'Z']:
                        data = sensor_data[axis]
                        if row_idx < len(data):
                            row.append(str(data[row_idx]))
                        else:
                            row.append("")
                f_or_debug.write(",".join(row) + '\n')

        # Export Euler Angles data
        with open(filename_euler, 'w') as f_euler:
            # Build headers
            headers = ['Roll', 'Pitch', 'Yaw']
            f_euler.write(','.join(headers) + '\n')

            # Write each row of Euler angle data
            for entry in self.parent.complete_euler_angles_data:
                # Ensure each entry is a numpy array of shape (1, 3)
                if isinstance(entry, np.ndarray) and entry.shape == (1, 3):
                    row = entry.flatten()  # Convert to 1D array: [roll, pitch, yaw]
                    f_euler.write(','.join(str(value) for value in row) + '\n')
                else:
                    # Handle unexpected shape or data type
                    f_euler.write(',,\n')  # Blank row in case of data error


        ################Removed to safe time####################
        # Make remote copy of the exported files
        # current_date = datetime.now()
        # month = current_date.strftime("%B")
        # day = current_date.strftime("%d")

        # remote_dir_emg = Path(PROJECT_DIR_REMOTE / "subject_logs" / f"subject_{self.parent.subject_number}" / f"{month}_{day}" / "EMG" / "Raw").as_posix()
        # remote_dir_log = Path(PROJECT_DIR_REMOTE / "subject_logs" / f"subject_{self.parent.subject_number}" / f"{month}_{day}" / "Log").as_posix()
        # try:
        #     os.system(f"ssh macbook mkdir -p {remote_dir_emg}")
        #     os.system(f"ssh macbook mkdir -p {remote_dir_log}")
        # except Exception as e:
        #     print(f"Error creating remote directories: {e}")

        # Export log data to CSV
        print("Exporting log data...")
        log_df = pd.DataFrame(self.parent.log_entries)
        # Save log data to CSV in the local subject folder
        subject_folder_log = os.path.join(self.parent.data_directory, f"subject_{self.parent.subject_number}", "Log")
        if not os.path.exists(subject_folder_log):
            os.makedirs(subject_folder_log)
        log_filename = os.path.join(subject_folder_log, f"Log_Data_Profile_{self.parent.assistive_profile_name}_Trial_{self.parent.trial_number}.csv")
        log_df.to_csv(log_filename, index=False)

        # try:
        #     os.system(f"scp {log_filename} macbook:{remote_dir_log}")
        #     os.system(f"scp {filename_emg} macbook:{remote_dir_emg}")
        #     os.system(f"scp {filename_acc} macbook:{remote_dir_emg}")
        #     os.system(f"scp {filename_gyro} macbook:{remote_dir_emg}")
        # except Exception as e:
        #     print(f"Error exporting data to remote host: {e}")
        # print("Data exported to Host.")


    def export_to_host(self):
        current_date = datetime.now()
        month = current_date.strftime("%B")
        day = current_date.strftime("%d")

        # Create remote directories


        remote_dir_emg = Path(PROJECT_DIR_REMOTE / "subject_logs" / f"subject_{self.parent.subject_number}" / f"{month}_{day}" / "EMG").as_posix()
        remote_dir_log = Path(PROJECT_DIR_REMOTE / "subject_logs" / f"subject_{self.parent.subject_number}" / f"{month}_{day}").as_posix()
        try:
            private_key_path = os.path.expanduser("~/.ssh/id_rsa_filipo_mbp")
            subprocess.run(["ssh", "-i", private_key_path, "macbook", "mkdir", "-p", remote_dir_emg])
            subprocess.run(["ssh", "-i", private_key_path, "macbook", "mkdir", "-p", remote_dir_log])

        except Exception as e:
            print(f"Error creating remote directories: {e}")

        # Export complete log folder to remote host
        local_log_folder = os.path.join(self.parent.data_directory, f"subject_{self.parent.subject_number}", "Log")
        try:
            os.system(f"scp -r {local_log_folder} macbook:{remote_dir_log}")
        except Exception as e:
            print(f"Error exporting log folder to remote host: {e}")
        print("Log folder exported to Host.")

        # Export complete Raw EMG folder to remote host
        local_emg_folder = os.path.join(self.parent.data_directory, f"subject_{self.parent.subject_number}", "Raw")
        try:
            os.system(f"scp -r {local_emg_folder} macbook:{remote_dir_emg}")
        except Exception as e:
            print(f"Error exporting EMG folder to remote host: {e}")
        print("EMG folder exported to Host.")

        # Export complete STS folder to remote host
        local_sts_folder = os.path.join(self.parent.data_directory, f"subject_{self.parent.subject_number}", "STS")
        try:
            os.system(f"scp -r {local_sts_folder} macbook:{remote_dir_emg}")
        except Exception as e:
            print(f"Error exporting STS folder to remote host: {e}")

        print("Subject folder exported to Host.")


    def export_sts_data_to_csv(self, emg_data_df, sensor_label):
        print("Exporting STS IMU data...")
        filename_emg = f"IMU_STS_Profile_{self.parent.assistive_profile_name}_Trial_{self.parent.trial_number}_Sensor_{sensor_label}_stsnumber{len(self.parent.log_entries)+1}.csv"

        # Save EMG data to CSV in the local subject folder
        subject_folder_sts = os.path.join(self.parent.data_directory, f"subject_{self.parent.subject_number}", "STS")
        if not os.path.exists(subject_folder_sts):
            os.makedirs(subject_folder_sts)
        filepath_emg = os.path.join(subject_folder_sts, filename_emg)  
        emg_data_df.to_csv(filepath_emg, index=False)

        # Make remote copy of the exported files
        # current_date = datetime.now()
        # month = current_date.strftime("%B")
        # day = current_date.strftime("%d")
        # remote_dir_emg_sts = Path(PROJECT_DIR_REMOTE / "subject_logs" / f"subject_{self.parent.subject_number}" / f"{month}_{day}" / "EMG" / "sts").as_posix()
        # try:
        #     os.system(f"ssh macbook mkdir -p {remote_dir_emg_sts}")
        #     os.system(f"scp {filepath_emg} macbook:{remote_dir_emg_sts}")
        # except Exception as e:
        #     print(f"Error exporting STS data to remote host: {e}")

    
    def export_all_sts_data_to_csv(self, emg_data, what):
        print("Exporting STS emg data...")
        print(len(emg_data))
        filename_emg = f"EMG_STS_Profile_{self.parent.assistive_profile_name}_Trial_{self.parent.trial_number}_all_Sensors_stsnumber_{len(self.parent.log_entries)+1}_{what}.csv"

        # Save EMG data to CSV in the local subject folder
        subject_folder_sts = os.path.join(self.parent.data_directory, f"subject_{self.parent.subject_number}", "STS")
        if not os.path.exists(subject_folder_sts):
            os.makedirs(subject_folder_sts)

        filepath_emg = os.path.join(subject_folder_sts, filename_emg)  
        emg_data.to_csv(filepath_emg, index=False)

        # Make remote copy of the exported files
        # current_date = datetime.now()
        # month = current_date.strftime("%B")
        # day = current_date.strftime("%d")
        # remote_dir_emg_sts = Path(PROJECT_DIR_REMOTE / "subject_logs" / f"subject_{self.parent.subject_number}" / f"{month}_{day}" / "EMG" / "sts").as_posix()
        # try:
        #     os.system(f"ssh macbook mkdir -p {remote_dir_emg_sts}")
        #     os.system(f"scp {filepath_emg} macbook:{remote_dir_emg_sts}")
        # except Exception as e:
        #     print(f"Error exporting STS data to remote host: {e}")


    def export_unassisted_mean_to_npy(self, unassisted_mean):
        # Save unassisted mean as a single-value npy file locally
        filename_unassisted_mean = "most_recent_unassisted_mean.npy"
        
        # Ensure subject folder exists
        subject_folder = os.path.join(self.parent.data_directory, f"subject_{self.parent.subject_number}")
        if not os.path.exists(subject_folder):
            os.makedirs(subject_folder)
        
        # Write the unassisted mean to a npy file
        filepath_unassisted = os.path.join(subject_folder, filename_unassisted_mean)
        np.save(filepath_unassisted, unassisted_mean)
            

    def load_unassisted_mean_from_npy(self):
        filename_unassisted_mean = "most_recent_unassisted_mean.npy"
        filepath_unassisted = os.path.join(self.parent.data_directory, f"subject_{self.parent.subject_number}", filename_unassisted_mean)
        
        # Load unassisted mean from npy if it exists
        try:
            return np.load(filepath_unassisted)
        except Exception as e:
            print(f"Error loading unassisted mean from file: {e}, record unassisted data first")
            return None


    def export_roll_angle_limits_to_npy(self, min_roll_angle, max_roll_angle):
        # Save roll angle limits as a two-value npy file locally
        filename_roll_angle = "roll_angle_limits.npy"
        
        # Ensure subject folder exists
        subject_folder = os.path.join(self.parent.data_directory, f"subject_{self.parent.subject_number}")
        if not os.path.exists(subject_folder):
            os.makedirs(subject_folder)
        
        # Combine min and max roll angles into an array and write to a npy file
        roll_angles = np.array([min_roll_angle, max_roll_angle])
        filepath_roll = os.path.join(subject_folder, filename_roll_angle)
        np.save(filepath_roll, roll_angles)


    def load_roll_angle_limits_from_npy(self):
        filename_roll_angle = "roll_angle_limits.npy"
        filepath_roll = os.path.join(self.parent.data_directory, f"subject_{self.parent.subject_number}", filename_roll_angle)
        
        # Load roll angles from npy if it exists
        try:
            roll_angles = np.load(filepath_roll)
            return roll_angles[0], roll_angles[1]  # Return min and max roll angles
        except Exception as e:
            print(f"Error loading roll angles from file: {e}, record roll angle limits first")
            return None, None
