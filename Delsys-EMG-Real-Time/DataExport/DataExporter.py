import os
import pandas as pd
from pathlib import Path
from datetime import datetime
PROJECT_DIR_REMOTE = Path("/Users/nathanirniger/Desktop/MA/Project/Code/assistive-arm")

class DataExporter:
    def __init__(self, parent):
        self.parent = parent

    def export_data_to_csv(self, filename_emg="EMG_data.csv", filename_acc="ACC_data.csv", filename_gyro="GYRO_data.csv"):
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

        # Make remote copy of the exported files
        current_date = datetime.now()
        month = current_date.strftime("%B")
        day = current_date.strftime("%d")

        remote_dir_emg = Path(PROJECT_DIR_REMOTE / "subject_logs" / f"subject_{self.parent.subject_number}" / f"{month}_{day}" / "EMG" / "Raw").as_posix()
        remote_dir_log = Path(PROJECT_DIR_REMOTE / "subject_logs" / f"subject_{self.parent.subject_number}" / f"{month}_{day}" / "Log").as_posix()
        os.system(f"ssh macbook mkdir -p {remote_dir_emg}")
        os.system(f"ssh macbook mkdir -p {remote_dir_log}")

        # Export log data to CSV
        print("Exporting log data...")
        log_df = pd.DataFrame(self.parent.log_entries)
        # Save log data to CSV in the local subject folder
        subject_folder_log = os.path.join(self.parent.data_directory, f"subject_{self.parent.subject_number}", "Logs")
        if not os.path.exists(subject_folder_log):
            os.makedirs(subject_folder_log)
        log_filename = os.path.join(subject_folder_log, f"{self.parent.current_date}_Log_Data_{self.parent.trial_number}.csv")
        log_df.to_csv(log_filename, index=False)

        os.system(f"scp {log_filename} macbook:{remote_dir_log}")
        os.system(f"scp {filename_emg} macbook:{remote_dir_emg}")
        os.system(f"scp {filename_acc} macbook:{remote_dir_emg}")
        os.system(f"scp {filename_gyro} macbook:{remote_dir_emg}")
        print("Data exported to Host.")



    def export_sts_data_to_csv(self, emg_data, sensor_label):
        print("Exporting STS emg data...")
        filename_emg = f"{self.parent.current_date}_EMG_STS_Trial_{self.parent.trial_number}_Sensor_{sensor_label}.csv"

        emg_data_df = pd.DataFrame(emg_data)

        # Save EMG data to CSV in the local subject folder
        subject_folder_sts = os.path.join(self.parent.data_directory, f"subject_{self.parent.subject_number}", "STS")
        if not os.path.exists(subject_folder_sts):
            os.makedirs(subject_folder_sts)
        filepath_emg = os.path.join(subject_folder_sts, filename_emg)  
        emg_data_df.to_csv(filepath_emg, index=False)


        # Make remote copy of the exported files
        current_date = datetime.now()
        month = current_date.strftime("%B")
        day = current_date.strftime("%d")
        remote_dir_emg_sts = Path(PROJECT_DIR_REMOTE / "subject_logs" / f"subject_{self.parent.subject_number}" / f"{month}_{day}" / "EMG" / "sts").as_posix()
        os.system(f"ssh macbook mkdir -p {remote_dir_emg_sts}")

        os.system(f"scp {filepath_emg} macbook:{remote_dir_emg_sts}")

