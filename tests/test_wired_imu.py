import struct
import can
import os

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

def convert_bytes_to_float(byte_array):
    """Convert 4 bytes to a float using struct unpacking."""
    return struct.unpack('>f', byte_array)[0]

def process_imu_data(imu, std_id, data):
    """Process the IMU data based on the standard ID."""
    imu_type = std_id & 0x0FF  # Extract lower three digits

    if imu_type == IMU_DATA1:  # Roll, Pitch
        imu.roll = convert_bytes_to_float(data[0:4])
        imu.pitch = convert_bytes_to_float(data[4:8])
    elif imu_type == IMU_DATA2:  # Yaw
        imu.yaw = convert_bytes_to_float(data[0:4])
    elif imu_type == IMU_DATA3:  # AccX, AccY
        imu.accX = convert_bytes_to_float(data[0:4])
        imu.accY = convert_bytes_to_float(data[4:8])
    elif imu_type == IMU_DATA4:  # AccZ
        imu.accZ = convert_bytes_to_float(data[0:4])
    elif imu_type == IMU_DATA5:  # GyroX, GyroY
        imu.gyroX = convert_bytes_to_float(data[0:4])
        imu.gyroY = convert_bytes_to_float(data[4:8])
    elif imu_type == IMU_DATA6:  # GyroZ
        imu.gyroZ = convert_bytes_to_float(data[0:4])

def read_imu_data(channel='can1', bustype='socketcan', bitrate=1000000):
    """
    Reads and decodes IMU data from a CAN bus.

    Parameters:
        channel (str): The CAN channel to use.
        bustype (str): The type of CAN interface.
        bitrate (int): The bitrate of the CAN bus.

    Returns:
        None
    """
    imu_data = IMUData()

    try:
        bus = can.interface.Bus(channel=channel, bustype=bustype, bitrate=bitrate)
        print(f"Listening for IMU messages on {channel}...")

        while True:
            msg = bus.recv(timeout=1.0)  # Timeout in seconds

            if msg:
                print(f"Received message: ID={msg.arbitration_id}, Data={msg.data.hex()}")
                process_imu_data(imu_data, msg.arbitration_id, msg.data)
                # print(f"IMU Data: {vars(imu_data)}")
                print(f"Roll: {imu_data.roll:.2f}, Pitch: {imu_data.pitch:.2f}, Yaw: {imu_data.yaw:.2f}, ")
                print(f"AccX: {imu_data.accX:.2f}, AccY: {imu_data.accY:.2f}, AccZ: {imu_data.accZ:.2f}, ")
                print(f"GyroX: {imu_data.gyroX:.2f}, GyroY: {imu_data.gyroY:.2f}, GyroZ: {imu_data.gyroZ:.2f}")
            else:
                print("No message received. Waiting...")

    except KeyboardInterrupt:
        print("\nStopping the IMU data reader.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'bus' in locals():
            bus.shutdown()


if __name__ == "__main__":
    os.system(f"sudo ip link set can1 down")
    os.system(f"sudo ip link set can1 up type can bitrate 1000000")
    read_imu_data()
    os.system(f"sudo ip link set can1 down")
