import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
PROJECT_DIR_REMOTE = Path("/Users/nathanirniger/Desktop/MA/Project/Code/assistive-arm")

class DataExporter:
    def __init__(self, parent):
        self.parent = parent

    def export_data_to_csv(self, filename_emg="EMG_data.csv", filename_acc="ACC_data.csv", filename_gyro="GYRO_data.csv", filename_or="ORIENTATION_data.csv"):
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
                for axis in ['X', 'Y', 'Z', 'W']:
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
                    for axis in ['X', 'Y', 'Z', 'W']:
                        data = sensor_data[axis]
                        if row_idx < len(data):
                            row.append(str(data[row_idx]))
                        else:
                            row.append("")
                f_or.write(",".join(row) + '\n')

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
        log_filename = os.path.join(subject_folder_log, f"{self.parent.current_date}_Log_Data_Profile_{self.parent.assistive_profile_name}_Trial_{self.parent.trial_number}.csv")
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
            os.system(f"ssh macbook mkdir -p {remote_dir_emg}")
            os.system(f"ssh macbook mkdir -p {remote_dir_log}")
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
        filename_emg = f"{self.parent.current_date}_IMU_STS_Profile_{self.parent.assistive_profile_name}_Trial_{self.parent.trial_number}_Sensor_{sensor_label}_stsnumber{len(self.parent.log_entries)+1}.csv"

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

    
    def export_all_sts_data_to_csv(self, emg_data):
        print("Exporting STS emg data...")
        filename_emg = f"{self.parent.current_date}_EMG_STS_Profile_{self.parent.assistive_profile_name}_Trial_{self.parent.trial_number}_all_Sensors_stsnumber{len(self.parent.log_entries)+1}.csv"

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


    def export_unassisted_mean_to_csv(self, unassisted_mean):
        # Save unassisted mean as a single-value CSV file locally
        filename_unassisted_mean = "most_recent_unassisted_mean.npy"
        
        # Ensure subject folder exists
        subject_folder_unassisted = os.path.join(self.parent.data_directory, f"subject_{self.parent.subject_number}")
        if not os.path.exists(subject_folder_unassisted):
            os.makedirs(subject_folder_unassisted)
        
        # Write the unassisted mean to a npy file
        filepath_unassisted = os.path.join(subject_folder_unassisted, filename_unassisted_mean)
        np.save(filepath_unassisted, unassisted_mean)
            

    def load_unassisted_mean_from_csv(self):
        filename_unassisted_mean = "most_recent_unassisted_mean.npy"
        filepath_unassisted = os.path.join(self.parent.data_directory, f"subject_{self.parent.subject_number}", filename_unassisted_mean)
        
        # Load unassisted mean from npy if it exists
        try:
            return np.load(filepath_unassisted)
        except Exception as e:
            print(f"Error loading unassisted mean from file: {e}, record unassisted data first")
            return None
