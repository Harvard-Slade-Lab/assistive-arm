import threading
import queue
import numpy as np

class NoPlotDataProcessor:
    def __init__(self, parent):
        self.parent = parent

    def process_remaining_data(self):
        """Process any remaining data in the queue after stopping the collection."""
        while not self.parent.data_queue.empty():
            try:
                data_batch = self.parent.data_queue.get(timeout=1)
                self.process_data_batch(data_batch)
                self.parent.data_queue.task_done()
            except queue.Empty:
                break

    def process_data(self):
        """Process data from the queue."""
        while not self.parent.pauseFlag or not self.parent.data_queue.empty():
            try:
                data_batch = self.parent.data_queue.get(timeout=1)
                self.process_data_batch(data_batch)
                self.parent.data_queue.task_done()
            except queue.Empty:
                continue

    def process_data_batch(self, data_batch):
        """Process a single data batch."""
        if not isinstance(data_batch, list) or not data_batch:
            print("Received empty or invalid data_batch")
            return

        # Update plot data buffer and collect data for exporting
        with self.parent.plot_data_lock:
            # ORIENTATION Data
            for sensor_label, sensor_info in self.parent.or_channels_per_sensor.items():
                indices = sensor_info['indices']
                labels = sensor_info['labels']
                for idx, ch_label in zip(indices, labels):
                    axis = ch_label[-1]
                    if idx < len(data_batch):
                        data = data_batch[idx]
                        self.parent.complete_or_data[sensor_label][axis].extend(data)
                        self.parent.complete_or_data_debug[sensor_label][axis].extend([data[-1]])
                    else:
                        print(f"ORIENTATION Channel index {idx} out of range in data_batch.")
            # EMG Data
            for idx, channel_idx in enumerate(self.parent.base.emgChannelsIdx):
                sensor_label = self.parent.base.emgChannelSensors[idx]
                if channel_idx < len(data_batch):
                    data = data_batch[channel_idx]
                    self.parent.complete_emg_data[sensor_label].extend(data)
                else:
                    print(f"EMG Channel index {channel_idx} out of range in data_batch.")

            # ACC Data
            for sensor_label, sensor_info in self.parent.acc_channels_per_sensor.items():
                indices = sensor_info['indices']
                labels = sensor_info['labels']
                for idx, ch_label in zip(indices, labels):
                    axis = ch_label[-1]  # Assuming labels end with 'X', 'Y', 'Z'
                    if idx < len(data_batch):
                        data = data_batch[idx]
                        self.parent.complete_acc_data[sensor_label][axis].extend(data)
                    else:
                        print(f"ACC Channel index {idx} out of range in data_batch.")

            # GYRO Data
            for sensor_label, sensor_info in self.parent.gyro_channels_per_sensor.items():
                indices = sensor_info['indices']
                labels = sensor_info['labels']
                for idx, ch_label in zip(indices, labels):
                    axis = ch_label[-1]  # Assuming labels end with 'X', 'Y', 'Z'
                    if idx < len(data_batch):
                        data = data_batch[idx]
                        self.parent.complete_gyro_data[sensor_label][axis].extend(data)
                    else:
                        print(f"GYRO Channel index {idx} out of range in data_batch.")



        
