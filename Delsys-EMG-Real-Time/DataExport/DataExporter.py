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
