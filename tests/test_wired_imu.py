import struct
import can
import os
import matplotlib.pyplot as plt


# CAN message IDs for IMU data
IMU_DATA1 = 0x011  # Roll, Pitch
IMU_DATA2 = 0x012  # Yaw
IMU_DATA3 = 0x013  # AccX, AccY
IMU_DATA4 = 0x014  # AccZ
IMU_DATA5 = 0x015  # GyroX, GyroY
IMU_DATA6 = 0x016  # GyroZ

class IMUData:
    """Class to hold IMU data and their history."""
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

def convert_bytes_to_float(byte_array):
    """Convert 4 bytes to a float using struct unpacking."""
    return struct.unpack('>f', byte_array)[0]

def process_imu_data(imu, std_id, data):
    """Process the IMU data and append to history."""
    imu_type = std_id & 0x0FF

    if imu_type == IMU_DATA1:  # Roll, Pitch
        imu.roll.append(convert_bytes_to_float(data[0:4]))
        imu.pitch.append(convert_bytes_to_float(data[4:8]))
    elif imu_type == IMU_DATA2:  # Yaw
        imu.yaw.append(convert_bytes_to_float(data[0:4]))
    elif imu_type == IMU_DATA3:  # AccX, AccY
        imu.accX.append(convert_bytes_to_float(data[0:4]))
        imu.accY.append(convert_bytes_to_float(data[4:8]))
    elif imu_type == IMU_DATA4:  # AccZ
        imu.accZ.append(convert_bytes_to_float(data[0:4]))
    elif imu_type == IMU_DATA5:  # GyroX, GyroY
        imu.gyroX.append(convert_bytes_to_float(data[0:4]))
        imu.gyroY.append(convert_bytes_to_float(data[4:8]))
    elif imu_type == IMU_DATA6:  # GyroZ
        imu.gyroZ.append(convert_bytes_to_float(data[0:4]))


def read_imu_data(channel='can1', bustype='socketcan', bitrate=1000000):
    """
    Reads and decodes IMU data from a CAN bus.
    Returns the full IMUData object containing history.
    """
    imu_data = IMUData()

    try:
        bus = can.interface.Bus(channel=channel, bustype=bustype, bitrate=bitrate)
        print(f"Listening for IMU messages on {channel}...")

        while True:
            msg = bus.recv(timeout=1.0)

            if msg:
                print(f"Received message: ID={msg.arbitration_id}, Data={msg.data.hex()}")
                process_imu_data(imu_data, msg.arbitration_id, msg.data)
                if imu_data.roll:
                    print(f"Roll: {imu_data.roll[-1]:.2f}")
            else:
                print("No message received. Waiting...")

    except KeyboardInterrupt:
        print("\nStopping the IMU data reader.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'bus' in locals():
            bus.shutdown()

    return imu_data



if __name__ == "__main__":
    os.system(f"sudo ip link set can1 down")
    os.system(f"sudo ip link set can1 up type can bitrate 1000000")
    
    imu_data = read_imu_data()
    
    print(f"Collected {len(imu_data.roll)} samples.")
    
    os.system(f"sudo ip link set can1 down")

    # === PLOT SECTION ===
    print(f"Plotting {len(imu_data.roll)} samples...")

    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Plot orientation
    axs[0].plot(imu_data.roll, label='Roll')
    axs[0].plot(imu_data.pitch, label='Pitch')
    axs[0].plot(imu_data.yaw, label='Yaw')
    axs[0].set_ylabel("Angle (deg)")
    axs[0].set_title("Orientation")
    axs[0].legend()
    axs[0].grid(True)

    # Plot acceleration
    axs[1].plot(imu_data.accX, label='AccX')
    axs[1].plot(imu_data.accY, label='AccY')
    axs[1].plot(imu_data.accZ, label='AccZ')
    axs[1].set_ylabel("Acceleration (m/s²)")
    axs[1].set_title("Acceleration")
    axs[1].legend()
    axs[1].grid(True)

    # Plot gyroscope
    axs[2].plot(imu_data.gyroX, label='GyroX')
    axs[2].plot(imu_data.gyroY, label='GyroY')
    axs[2].plot(imu_data.gyroZ, label='GyroZ')
    axs[2].set_ylabel("Angular velocity (deg/s)")
    axs[2].set_title("Gyroscope")
    axs[2].legend()
    axs[2].grid(True)

    axs[2].set_xlabel("Sample Index")

    plt.tight_layout()
    plt.show()

    print("Plotting complete.")



