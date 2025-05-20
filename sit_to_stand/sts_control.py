import csv
import os
import sys
import time
import zipfile
import numpy as np
import pandas as pd
import yaml
import threading
from typing import Literal

from pathlib import Path

import RPi.GPIO as GPIO

from NeuroLocoMiddleware.SoftRealtimeLoop import SoftRealtimeLoop
from assistive_arm.motor_control import CubemarsMotor
from assistive_arm.robotic_arm import calculate_ee_pos, get_target_torques
from session_manager import SessionManager, get_logger
from socket_server import SocketServer
from read_imu import IMUReader
from sts_utils import await_trigger_signal, countdown
from datetime import datetime

import joblib
from Phase_Estimation import DataLoaderIMU
from Phase_Estimation import MatrixCreator
from Phase_Estimation.Regression_Methods import SVR_Reg


# DA CAMBIARE_--------------------------
training_segmentation_flag = False

# Global Variables:
current_model = None
phase_baseline = None


def calibrate_height(
        freq: int,
        session_manager: SessionManager,
        mode: Literal["TRIGGER", "ENTER", "SOCKET"],
        socket_server: SocketServer,
        imu_reader: IMUReader):
    """
    Perform height calibration by running the sit-to-stand motion.

    Args:
        freq (int): Control loop frequency (Hz).
        session_manager (SessionManager): Manages session and logging paths.
        mode (Literal["TRIGGER", "ENTER", "SOCKET"]): Start trigger mode.
        socket_server (SocketServer): Socket server instance for communication.
        imu_reader (IMUReader): IMU reader instance for roll angle data.
    """
    global current_model, phase_baseline
    # Use session_manager to retrieve the YAML path for calibration data
    yaml_path = session_manager.yaml_path

    # Set up the real-time control loop
    loop = SoftRealtimeLoop(dt=1 / freq, report=False, fade=0)

    # Data for calibration
    calibration_data = dict()
    roll_angles = []
    start_time = 0

    subject_name = input("Enter the subject name: ")
    try:
        for training in range(5):
            print(f"\nTraining iteration {training + 1} of 5\n")

            # Wait for trigger signal and start recording based on mode
            await_trigger_signal(mode=mode, socket_server=socket_server) # --> using SOCKET mode, you have to press the Button on the EMG Laptop

            # Start reading from the IMU
            if imu_reader is not None:
                imu_reader.start_reading_imu_data()

            countdown(duration=3)

            print("\nTraining started. Please perform the sit-to-stand motion.\n")

            sts_start = time.time()
            
            for t in loop:
                if socket_server is not None:
                    if socket_server.mode_flag or socket_server.kill_flag:
                        print("Stopped recording, exiting...")
                        break
                if mode == "TRIGGER" and GPIO.input(17):
                    print("Stopped recording, exiting...")
                    break
                elif mode == "SOCKET" and not socket_server.collect_flag:
                    print("Stopped recording, exiting...")
                    break
                


                # with threading.Lock():
                #     if imu_reader is not None:
                #         roll_angle = imu_reader.imu_data.pitch
                #     else:
                #         roll_angle = socket_server.roll_angle

                # if roll_angle is not None:
                #     roll_angles.append(roll_angle)

                # if t - start_time >= 0.05:
                #     print(f"Roll angle: {roll_angle}", end="\r")
                #     start_time = t




            # Stop the IMU reader
            if imu_reader is not None:
                imu_reader.stop_reading_imu_data()

            # Extract the acquired DATA from the Wired IMU
            roll_angle = imu_reader.imu_data_history.roll
            pitch_angle = imu_reader.imu_data_history.pitch
            yaw_angle = imu_reader.imu_data_history.yaw
            accX = imu_reader.imu_data_history.accX
            accY = imu_reader.imu_data_history.accY
            accZ = imu_reader.imu_data_history.accZ
            gyroX = imu_reader.imu_data_history.gyroX
            gyroY = imu_reader.imu_data_history.gyroY
            gyroZ = imu_reader.imu_data_history.gyroZ

            # Print the length of all the single vectors for DEBUG:
            # print(f"Length of roll angle: {len(roll_angle)}")
            # print(f"Length of pitch angle: {len(pitch_angle)}")
            # print(f"Length of yaw angle: {len(yaw_angle)}")
            # print(f"Length of accX: {len(accX)}")
            # print(f"Length of accY: {len(accY)}")
            # print(f"Length of accZ: {len(accZ)}")
            # print(f"Length of gyroX: {len(gyroX)}")
            # print(f"Length of gyroY: {len(gyroY)}")
            # print(f"Length of gyroZ: {len(gyroZ)}")

            sts_duration = time.time() - sts_start

            # Get the current date and time
            current_date = datetime.now().strftime("%Y-%m-%d")

            # Generate the file name with the current date and trial number
            file_name = f"IMU_Profile_{current_date}_Trial_{training + 1}.csv"

            length_data = min(len(roll_angle), len(pitch_angle), len(yaw_angle), len(accX), len(accY), len(accZ), len(gyroX), len(gyroY), len(gyroZ))
            # Export data to CSV
            with open(file_name, "w") as f:
                f.write("Roll,Pitch,Yaw,AccX,AccY,AccZ,GyroX,GyroY,GyroZ\n")
                for i in range(length_data):
                    f.write(f"{roll_angle[i]},{pitch_angle[i]},{yaw_angle[i]},"
                            f"{accX[i]},{accY[i]},{accZ[i]},"
                            f"{gyroX[i]},{gyroY[i]},{gyroZ[i]}\n")
            print(f"Data exported to '{file_name}'.")

            # Save the file to the remote directory
            PROJECT_DIR_REMOTE = Path("/Users/filippo.mariani/Desktop/Universita/Harvard/Third_Arm_Data/subject_logs")
            # Create a new folder with subject_name
            session_remote_dir = PROJECT_DIR_REMOTE / f"Subject_{subject_name}"
            # Check remotely if the folder exists; if not, create it
            check_and_create_cmd = f'ssh macbook "[ -d \\"{session_remote_dir}\\" ] || mkdir -p \\"{session_remote_dir}\\""'
            os.system(check_and_create_cmd)
            # Save the file to the remote directory
            os.system(f"scp {file_name} macbook:{session_remote_dir}")
            print(f"Data file '{file_name}' sent to remote directory.")

        # Copy the folder from the macbook to the Raspi
        folder_remote = session_remote_dir
        folder_local = "/home/xabier/Documents/Data_AssistiveArm/Training"
        folder_local = Path(folder_local)  # Convert string to Path
        folder_training_raspi = folder_local / f"Subject_{subject_name}"
        # Check if the folder exists on the Raspi
        if not os.path.exists(folder_training_raspi):
            os.system(f"scp -r macbook:{folder_remote} {folder_local}")
        else:
            print(f"Error! Folder {folder_training_raspi} already exists.")
            print("Using old data for training.")


        # Load the Data for the Training:
        frequencies = [200, 200, 200]
        folder_path = session_remote_dir
        if not folder_path:
            print("No folder selected. Training cancelled.")
            
        segment_choice = input("Select segmentation method:\n1. for One Shot\n2. for Cyclic\n ")
   

        # Load and process files
        acc_data, gyro_data, or_data, acc_files, gyro_files, or_files = DataLoaderIMU.load_and_process_files(folder_training_raspi)
        print(f"Loaded {len(acc_files)} ACC files, {len(gyro_files)} GYRO files, and {len(or_files)} OR files")

        grouped_indices = DataLoaderIMU.group_files_by_timestamp(acc_files, gyro_files, or_files)
        print(f"Found {len(grouped_indices)} complete data sets")   

        # If you want to add fictitious trials, there is a section in create_matrices that can be modified
        X, Y, timestamps, segment_lengths, feature_names = MatrixCreator.create_matrices(
            acc_data, gyro_data, or_data, grouped_indices,
            segment_choice, frequencies,
            biasPlot_flag=training_segmentation_flag
        )

        print(f"Created X matrix with shape {X.shape} and Y matrix with length {len(Y)}")

        print("\nColumn information:")
        for i, name in enumerate(feature_names):
            print(f"Column {i}: {name}")

        MatrixCreator.visualize_matrices(X, Y, timestamps, segment_lengths, feature_names)

        print("Do you want to use FAST computation for the SVR grid search?")
        fast_comput = input("yes or no? ")

        if fast_comput == 'yes':
            param_grid = {
                'svr__C': [100],
                'svr__epsilon': [0.01],
                'svr__gamma': [0.01]
            }
        else:
            param_grid = {
                'svr__C': np.logspace(-3, 3, 7),
                'svr__epsilon': np.logspace(-3, 0, 4),
                'svr__gamma': np.logspace(-4, 1, 6)
            }

        svr_model, y_svr = SVR_Reg.enhanced_svr_regression(
            X, Y, kernel='rbf',
            param_grid=param_grid,
            plot=True,
            frequencies=frequencies
        )

        joblib.dump(svr_model, 'svr_phase_model.joblib')
        current_model = svr_model['model']
        print("Training Finished!")

        num_entries = length_data
        phase_baseline = np.linspace(0, 100, num_entries)


        # # Calibration calculations
        # roll_angles = np.array(roll_angles)
        # new_max = roll_angles.max() - 1
        # new_min = roll_angles.min() + 1

        # # Clip roll angles
        # roll_angles = np.clip(roll_angles, new_min, new_max)

        # # Calculate the number of entries
        # num_entries = int(sts_duration * freq)

        # # Store calibration data
        # calibration_data["new_range"] = {"min": float(new_min), "max": float(new_max)}
        # # Generate the roll angles array
        # calibration_data["roll_angles"] = np.linspace(new_min, new_max, num=num_entries).tolist()
        # # Generate the percentage array
        # calibration_data["Percentage"] = np.linspace(0, 100, num=num_entries).tolist()

        # remote_path = session_manager.session_remote_dir

        # # Save calibration data in YAML and sync with the remote directory
        # with open(yaml_path, "w") as f:
        #     yaml.dump(calibration_data, f)
        # try:
        #     os.system(f"scp {yaml_path} macbook:{remote_path}")
        # except Exception as e:
        #     print(f"Error transferring YAML file: {e}")
        
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Shutting down...")

def control_loop_and_log(
        motor_1: CubemarsMotor,
        motor_2: CubemarsMotor,
        logger: csv.writer,
        profile: pd.DataFrame,
        freq: int,
        mode: Literal["TRIGGER", "ENTER", "SOCKET"],
        apply_force: bool,
        log_path: Path,
        socket_server: SocketServer,
        imu_reader: IMUReader,
        session_manager: SessionManager):
    
    global current_model, phase_baseline
    # Reset the motor buffers
    motor_1.new_run = True
    motor_2.new_run = True

    print("Recording started. Please perform the sit-to-stand motion.")
    print("Press Ctrl + C or trigger to stop recording.\n")
    print_time = 0
    start_time = time.time()

    loop = SoftRealtimeLoop(dt=1 / freq, report=False, fade=0)
    success = True
    printed = False

    for t in loop:
        if socket_server is not None:
            if socket_server.mode_flag or socket_server.kill_flag:
                print("Stopped recording, exiting...")
                break
        if mode == "TRIGGER" and GPIO.input(17):
            print("Stopped recording, exiting...")
            break
        elif mode == "SOCKET" and not socket_server.collect_flag:
            print("Stopped recording, exiting...")
            break

        cur_time = time.time()

        if motor_1._emergency_stop or motor_2._emergency_stop:
            success = False
            break

        # Maybe want to move this into the function
        with threading.Lock():
            if imu_reader is not None:
                # here we can extract all imu data and compute the phase
                roll_angle = imu_reader.imu_data.roll
                pitch_angle = imu_reader.imu_data.pitch
                yaw_angle = imu_reader.imu_data.yaw
                accX = imu_reader.imu_data.accX
                accY = imu_reader.imu_data.accY
                accZ = imu_reader.imu_data.accZ
                gyroX = imu_reader.imu_data.gyroX
                gyroY = imu_reader.imu_data.gyroY
                gyroZ = imu_reader.imu_data.gyroZ

                # Compute the phase using the SVR model
                current_phase = current_model.predict(np.array([[accX, accY, accZ, gyroX, gyroY, gyroZ, roll_angle, pitch_angle]]))*100
                # Saturate the phase between 0 and 100
                current_phase = max(0, min(current_phase[0], 100))
            else:
                roll_angle = socket_server.roll_angle
        
        if motor_1.swapped_motors:
            tau_1, tau_2, P_EE, index = get_target_torques(theta_1=motor_2.position, theta_2=motor_1.position, current_phase=current_phase, profiles=profile)
        else:
            tau_1, tau_2, P_EE, index = get_target_torques(theta_1=motor_1.position, theta_2=motor_2.position, current_phase=current_phase, profiles=profile)

        # Stop if the roll angle is larger than the maximum roll angle
        if current_phase > 99 and t > 0.1:
            # Wait for emg collection to stop, so there is no mess with the file naming
            if socket_server is not None:
                if apply_force:
                    print("Maximum phase exceeded. Setting torques to zero.")
                    apply_force = False
            else:
                print("Maximum phase exceeded. Stopping...")
                break

        if apply_force and t >= 0.1:
            if not printed:
                print("\nGO!")
                printed = True
            motor_1.send_torque(desired_torque=tau_1, safety=False)
            motor_2.send_torque(desired_torque=tau_2, safety=False)
        else:
            if not printed:
                print("\nGO!")
                printed = True
            motor_1.send_torque(desired_torque=0, safety=False)
            motor_2.send_torque(desired_torque=0, safety=False)

        if t - print_time >= 0.05:
            print(f"{motor_1.type}: Angle: {np.rad2deg(motor_1.position):.3f} Torque: {motor_1.measured_torque:.3f} Temperature: {motor_1.temperature}")
            print(f"{motor_2.type}: Angle: {np.rad2deg(motor_2.position):.3f} Torque: {motor_2.measured_torque:.3f} Temperature: {motor_2.temperature}")
            print(f"Body height: {-P_EE[0]}")
            print(f"Movement: {index:.0f}%. tau_1: {tau_1}, tau_2: {tau_2}")
            sys.stdout.write(f"\x1b[4A\x1b[2K")
                    
            print_time = t
        if not motor_1.swapped_motors:
            logger.writerow([cur_time - start_time, index, roll_angle, tau_1, motor_1.measured_torque, motor_1.position, motor_1.velocity, tau_2, motor_2.measured_torque, motor_2.position, motor_2.velocity, P_EE[0], P_EE[1]])
        else:
            logger.writerow([cur_time - start_time, index, roll_angle, tau_2, motor_2.measured_torque, motor_2.position, motor_2.velocity, tau_1, motor_1.measured_torque, motor_1.position, motor_1.velocity, P_EE[0], P_EE[1]])
    del loop

    motor_1.send_torque(desired_torque=0, safety=False)
    motor_2.send_torque(desired_torque=0, safety=False)

    if not success:
        motor_1._emergency_stop = False
        motor_2._emergency_stop = False
        print("\nSomething went wrong. Repeating the iteration...")

    session_manager.save_log_or_delete(log_path=log_path, successful=success)

    return success


def apply_simulation_profile(
        motor_1: CubemarsMotor,
        motor_2: CubemarsMotor,
        freq: int,
        session_manager: SessionManager,
        profile: pd.DataFrame,
        profile_name: str,
        mode: Literal["TRIGGER", "ENTER", "SOCKET"],
        socket_server: SocketServer,
        imu_reader: IMUReader):
    """
    Apply a calibrated profile to simulate an assistive arm motion.

    Args:
        motor_1 (CubemarsMotor): First motor in the assistive arm.
        motor_2 (CubemarsMotor): Second motor in the assistive arm.
        freq (int): Control loop frequency (Hz).
        session_manager (SessionManager): Manages session and logging paths.
        profile: Most recent profile.
        profile_name (str): Name of the profile to apply.
        mode (Literal["TRIGGER", "ENTER", "SOCKET"]): Start trigger mode.
        server (SocketServer): Socket server instance for communication.
        imu_reader (IMUReader): IMU reader instance for roll angle data.
    """

    # Wait for trigger signal and start recording based on mode
    await_trigger_signal(mode=mode, socket_server=socket_server)

    if socket_server is not None:
        if socket_server.mode_flag or socket_server.kill_flag:
            return
        
    # Start reading from the IMU
    if imu_reader is not None:
        imu_reader.start_reading_imu_data()

    # Set up logging for this iteration
    log_path, logger = get_logger(log_name=f"{profile_name}", session_manager=session_manager, socket_server=socket_server)
    
    # Countdown before starting
    countdown(duration=1)

    try:
        # Run the control loop, passing session_manager and server
        success = control_loop_and_log(
            motor_1=motor_1,
            motor_2=motor_2,
            logger=logger,
            profile=profile,
            freq=freq,
            mode=mode,
            apply_force=True,
            log_path=log_path,
            socket_server=socket_server,
            imu_reader=imu_reader,
            session_manager=session_manager
        )

        # Save or delete the log based on success
        session_manager.save_log_or_delete(log_path=log_path, successful=success)

        # Stop reading from the IMU
        if imu_reader is not None:
            imu_reader.stop_reading_imu_data()

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Stopping...")


def collect_unpowered_data(
        motor_1: CubemarsMotor,
        motor_2: CubemarsMotor,
        freq: int,
        iterations: int,
        session_manager: SessionManager,
        mode: Literal["TRIGGER", "ENTER"],
        socket_server: SocketServer,
        imu_reader: IMUReader):
    """
    Collect unpowered data for EMG synchronization.

    Args:
        motor_1 (CubemarsMotor): The first motor for the assistive arm.
        motor_2 (CubemarsMotor): The second motor for the assistive arm.
        freq (int): Control loop frequency (Hz).
        session_manager (SessionManager): Manages session and logging paths.
        mode (Literal["TRIGGER", "ENTER"]): Start trigger mode.
        socket_server (SocketServer): Socket server instance for connection control.
        imu_reader (IMUReader): IMU reader instance for roll angle data.
    """ 
    global current_model, phase_baseline

    profile = pd.DataFrame(phase_baseline)
    profile.columns = ["phase_baseline"]
    # Add columns with zero force
    profile["force_X"] = 0
    profile["force_Y"] = 0

    i = 1

    # Loop over the number of iterations for unpowered data collection
    while i <= iterations:  # Collect data across 5 iterations
        print(f"\nIteration number: {i}")
        success = False
        profile_name = socket_server.profile_name
        
        # Wait until successful data collection
        while not success:
            try:
                # Wait for trigger signal to start based on the selected mode, or kill the process/exit
                await_trigger_signal(mode=mode, socket_server=socket_server)
                # Check if the server is in a mode that requires stopping the process
                if socket_server is not None:
                    if socket_server.mode_flag or socket_server.kill_flag:
                        return
                    
                # Start reading from the IMU
                if imu_reader is not None:
                    imu_reader.start_reading_imu_data()
                
                # Prepare the logger for the current iteration
                log_path, logger = get_logger(
                    log_name=f"unpowered_device_tag_{profile_name}_iteration_{i}",
                    session_manager=session_manager,
                    socket_server=socket_server
                )

                # Start countdown before data collection
                countdown(duration=1)
                print(f"Recording to {log_path}")

                # Start the control loop for data collection without force application
                success = control_loop_and_log(
                    motor_1=motor_1,
                    motor_2=motor_2,
                    logger=logger,
                    profile=profile,
                    freq=freq,
                    mode=mode,
                    apply_force=False,
                    log_path=log_path,
                    socket_server=socket_server,
                    imu_reader=imu_reader,
                    session_manager=session_manager
                )

                # Save or delete the log based on the success of data collection
                session_manager.save_log_or_delete(log_path=log_path, successful=success)

                # Stop reading from the IMU
                if imu_reader is not None:
                    imu_reader.stop_reading_imu_data()

            except Exception as e:
                print(f"An error occurred: {e}. Repeating the iteration...")

            except KeyboardInterrupt:
                print("Keyboard interrupt detected. Exiting...")
                if imu_reader is not None:
                    imu_reader.stop_reading_imu_data()
                return
        # Will miss repetition if last iteration fails (maybe add getting the profile tag)
        if socket_server.repeat_flag:
            print("Iteration has to be repeated.")
            socket_server.repeat_flag = False
        else:
            i += 1
