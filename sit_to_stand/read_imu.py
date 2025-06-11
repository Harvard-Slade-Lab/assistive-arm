import struct
import can
import os
import threading

# CAN message IDs for IMU data
IMU_DATA1 = 0x011  # Roll, Pitch
IMU_DATA2 = 0x012  # Yaw
IMU_DATA3 = 0x013  # AccX, AccY
IMU_DATA4 = 0x014  # AccZ
IMU_DATA5 = 0x015  # GyroX, GyroY
IMU_DATA6 = 0x016  # GyroZ

class IMUData:
    """Class to hold IMU data."""
    def __init__(self):
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.accX = 0.0
        self.accY = 0.0
        self.accZ = 0.0
        self.gyroX = 0.0
        self.gyroY = 0.0
        self.gyroZ = 0.0

class IMUDataHistory:
    """Class history of IMU Data."""
    def __init__(self):
        self.roll = []
        self.pitch = []
        self.yaw = []
        self.accX = []
        self.accY = []
        self.accZ = []
        self.gyroX = []
        self.gyroY = []
        self.gyroZ = []

class IMUReader:
    def __init__(self, channel='can1', bustype='socketcan', bitrate=1000000):
        self.imu_data_history = IMUDataHistory()
        self.imu_data = IMUData()
        self.channel = channel
        self.bustype = bustype
        self.bitrate = bitrate
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

    def convert_bytes_to_float(self, byte_array):
        """Convert 4 bytes to a float using struct unpacking."""
        return struct.unpack('>f', byte_array)[0]

    def process_imu_data(self, std_id, data):
        """Process the IMU data based on the standard ID."""
        imu_type = std_id & 0x0FF  # Extract lower three digits

        with self._lock:
            if imu_type == IMU_DATA1:  # Roll, Pitch
                # History
                self.imu_data_history.roll.append(self.convert_bytes_to_float(data[0:4]))
                self.imu_data_history.pitch.append(self.convert_bytes_to_float(data[4:8]))
                # Current
                self.imu_data.roll = self.convert_bytes_to_float(data[0:4])
                self.imu_data.pitch = self.convert_bytes_to_float(data[4:8])
            elif imu_type == IMU_DATA2:  # Yaw
                # History
                self.imu_data_history.yaw.append(self.convert_bytes_to_float(data[0:4]))
                # Current
                self.imu_data.yaw = self.convert_bytes_to_float(data[0:4])
            elif imu_type == IMU_DATA3:  # AccX, AccY
                # History
                self.imu_data_history.accX.append(self.convert_bytes_to_float(data[0:4]))
                self.imu_data_history.accY.append(self.convert_bytes_to_float(data[4:8]))
                # Current
                self.imu_data.accX = self.convert_bytes_to_float(data[0:4])
                self.imu_data.accY = self.convert_bytes_to_float(data[4:8])
            elif imu_type == IMU_DATA4:  # AccZ
                # History
                self.imu_data_history.accZ.append(self.convert_bytes_to_float(data[0:4]))
                # Current
                self.imu_data.accZ = self.convert_bytes_to_float(data[0:4])
            elif imu_type == IMU_DATA5:  # GyroX, GyroY
                # History
                self.imu_data_history.gyroX.append(self.convert_bytes_to_float(data[0:4]))
                self.imu_data_history.gyroY.append(self.convert_bytes_to_float(data[4:8]))
                # Current
                self.imu_data.gyroX = self.convert_bytes_to_float(data[0:4])
                self.imu_data.gyroY = self.convert_bytes_to_float(data[4:8])
            elif imu_type == IMU_DATA6:  # GyroZ
                # History
                self.imu_data_history.gyroZ.append(self.convert_bytes_to_float(data[0:4]))
                # Current
                self.imu_data.gyroZ = self.convert_bytes_to_float(data[0:4])

    def read_imu_data(self):
        """Continuously read and process IMU data from the CAN bus."""
        try:
            bus = can.interface.Bus(channel=self.channel, bustype=self.bustype, bitrate=self.bitrate)
            print(f"Listening for IMU messages on {self.channel}...")

            while self._running:
                msg = bus.recv(timeout=1.0)  # Timeout in seconds

                if msg:
                    # print(f"Received message: ID={msg.arbitration_id}, Data={msg.data.hex()}")
                    self.process_imu_data(msg.arbitration_id, msg.data)
                    # print(f"Roll: {self.imu_data.roll:.2f}, Pitch: {self.imu_data.pitch:.2f}")
                else:
                    print("No wired IMU message received. Waiting...")

        except KeyboardInterrupt:
            print("\nStopping the IMU data reader.")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            if 'bus' in locals():
                bus.shutdown()

    def setup_can_bus(self):
        """Set up the CAN bus interface."""
        print(f"Setting up CAN bus on {self.channel} with bitrate {self.bitrate}...")
        os.system(f"sudo ip link set {self.channel} down")
        os.system(f"sudo ip link set {self.channel} up type can bitrate {self.bitrate}")
        print(f"CAN bus {self.channel} is up.")

    def shutdown_can_bus(self):
        """Shut down the CAN bus interface."""
        print(f"Shutting down CAN bus on {self.channel}...")
        os.system(f"sudo ip link set {self.channel} down")
        print(f"CAN bus {self.channel} is down.")

    def start_reading_imu_data(self):
        """Start a thread to read IMU data."""
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self.read_imu_data)
            self._thread.start()
            print("IMU data reading started.")
        else:
            print("IMU data reading is already running.")

    def stop_reading_imu_data(self):
        """Stop reading IMU data."""
        if self._running:
            self._running = False
            self._thread.join()
            print("IMU data reading stopped.")
        else:
            print("IMU data reading is not running.")

