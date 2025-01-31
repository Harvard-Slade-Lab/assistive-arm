import can
import csv
import numpy as np
import os
import sys
import time
import traceback
import yaml

from datetime import datetime
from typing import Literal
from pathlib import Path
from functools import wraps

import logging


with open("./motor_config.yaml", "r") as f:
    MOTOR_PARAMS = yaml.load(f, Loader=yaml.FullLoader)

def init_can_port(channel, bitrate=1000000):
    try:
        os.system(f"sudo ip link set {channel} up type can bitrate {bitrate}")
        print(f"CAN port {channel} initialized successfully with bitrate {bitrate}")
    except Exception as e:
        print(f"Failed to initialize CAN port {channel}: {e}")

def shutdown_can_port(channel):
    try:
        os.system(f"sudo ip link set {channel} down")
        print(f"CAN port {channel} shutdown successfully")
    except Exception as e:
        print(f"Failed to shutdown CAN port {channel}: {e}")

def setup_can_and_motors():
    max_retries = 5
    for attempt in range(max_retries):
        try:
            print(f"Attempting to initialize CAN bus and connect motors (Attempt {attempt + 1}/{max_retries})...")
            init_can_port("can0")
            can_bus = can.interface.Bus(channel="can0", bustype="socketcan")

            motor_1 = CubemarsMotor("AK70-10", frequency=200, can_bus=can_bus)
            motor_2 = CubemarsMotor("AK60-6", frequency=200, can_bus=can_bus)

            if not motor_1.connected or not motor_2.connected:
                raise Exception("Failed to connect to both motors")
            
            print("Both motors connected successfully.")
            return can_bus, motor_1, motor_2

        except Exception as e:
            print(f"Error during motor setup: {e}")
            print("Reinitializing CAN bus...")
            shutdown_can_port("can0")
            time.sleep(1)  # Delay before retrying
    
    print("Failed to connect both motors after multiple attempts.")
    return None  # Only return None if all retries fail

def shutdown_can_and_motors(can_bus, motor_1, motor_2):
    """Shutdown the motors and CAN bus."""
    try:
        motor_1.shutdown()
    except Exception as e:
        print(f"Error shutting down motor 1: {e}")
    try:
        motor_2.shutdown()
    except Exception as e:
        print(f"Error shutting down motor 2: {e}")
    try:
        can_bus.shutdown()
    except Exception as e:
        print(f"Error shutting down CAN bus: {e}")
    shutdown_can_port("can0")

def uint_to_float(x, xmin, xmax, bits):
    span = xmax - xmin
    int_val = float(x) * span / (float((1 << bits) - 1)) + xmin
    return int_val

def float_to_uint(x, xmin, xmax, bits):
    span = xmax - xmin
    if x < xmin:
        x = xmin
    elif x > xmax:
        x = xmax
    convert = int((x - xmin) * (((1 << bits) - 1) / span))
    return convert

class CubemarsMotor:
    def __init__(
        self,
        motor_type: Literal["AK60-6", "AK70-10"],
        frequency: int,
        can_bus: can.Bus,  # Shared CAN bus instance
    ) -> None:
        self.type = motor_type
        self.params = MOTOR_PARAMS[motor_type]
        if motor_type == "AK60-6":
            self.other_params = MOTOR_PARAMS["AK70-10"]
        else:
            self.other_params = MOTOR_PARAMS["AK60-6"]
        self.log_vars = ["position", "velocity", "torque"]
        self.frequency = frequency
        self.can_bus = can_bus
        self.swapped_motors = False
        self.switch_now = False
        self.previous_id = None

        self.position = 0
        self.prev_velocity = 0
        self.velocity = 0
        self.temperature = 0
        self.csv_file_name = None

        self.measured_torque = 0

        self.buffer_index = 0
        self.buffer_size = int(0.1 * self.frequency)  # Buffer size for X secs

        self.position_buffer = [0] * self.buffer_size
        self.velocity_buffer = [0] * self.buffer_size
        self.torque_buffer = [0] * self.buffer_size
        self.temperature_buffer = [0] * self.buffer_size

        self._emergency_stop = False
        self.new_run = True

        self.logging = False
        if self.logging:
            self.logger = self._setup_logger()

        self.connected = False
        self._connect_motor()

    def _setup_logger(self):
        logger = logging.getLogger(f"Motor_{self.params['ID']}")
        if not logger.hasHandlers():  # Avoid duplicate handlers
            logger.setLevel(logging.INFO)
            file_handler = logging.FileHandler(f"motor_{self.params['ID']}.log")
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        return logger
    
    def _connect_motor(self, delay: float = 0.005, zero: bool = False) -> None:
        """Establish connection with motor
        Args:
            can_bus (can.Bus): can bus object corresponding to motor
            motor_id (hex): motor id in hexadecimal
            delay (float, optional): waiting time. Defaults to 0.001.
        """
        self._start_time = time.time()
        self._last_update_time = self._start_time

        # Command to enter motor control mode, can also be used to read the current state in stateless manner
        start_motor_mode = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFC]

        tries = 0
        while not self.connected:
            print("\nSending starting command...")
            try:
                self.can_bus.send(self._send_message(data=start_motor_mode))
                print("Waiting for response...")
                time.sleep(0.1)
                response = self.can_bus.recv(delay)

                if response:
                    P, V, T, Te = self._read_motor_msg(response.data)
                    print(f"Motor {self.type} connected successfully")
                    self.connected = True
                    if zero:
                        self.send_zero_position()
                else:
                    raise AttributeError("No response from motor")

            except (can.CanError, AttributeError) as e:
                tries += 1
                if tries > 5:
                    print(f"Failed to connect to motor {self.type}")
                    break

    def _stop_motor(self) -> None:
        """Stop motor"""
        # Command to exit motor control mode
        stop_motor_mode = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFD]
        # can_name = self.can_bus.channel_info.split(" ")[-1]
        print(f"\nShutting down motor {self.type}...")

        try:
            self.can_bus.send(self._send_message(stop_motor_mode))  # disable motor mode
            # self.can_bus.shutdown()
            print(f"Motor {self.type} shutdown successful")
        except:
            traceback.print_exc()
            print(f"Failed to shutdown motor {self.type}")

    def shutdown(self,):
        self._stop_motor()

    def check_safety_speed_limit(self):
        if abs(self.velocity) > self.params["Vel_limit"]:
            self._emergency_stop = True
            self.send_velocity(0)
            self.velocity = 0
            time.sleep(0.5)
            self.send_torque(0)
            print("Motor speed exceeded speed limit")
            # self._stop_motor()

    def check_temperature_limit(self):
        # Actually returns the mosfet temperature
        # TODO not completely correct, get weird signal from 70-10
        if self.temperature > 80:
            self._emergency_stop = True
            self.send_velocity(0)
            self.velocity = 0
            time.sleep(0.5)
            self.send_torque(0)
            print("Motor temperature exceeded 80 degrees")
            # self._stop_motor()

    def send_zero_position(self) -> None:
        # Set current Position as zero position
        zero_position = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE]

        self.can_bus.send(self._send_message(zero_position))
        # Sleep for 2.5s to allow motor to zero
        response = self.can_bus.recv(1.5)
        print("Zeroing position...")
        print("Pos, Vel, Torque, Temperature: ", self._read_motor_msg(response.data))

        zero_cmd = [0, 0, 0, 0, 0]

        self._update_motor(cmd=zero_cmd, wait_time=0.001)

    def send_angle(self, angle: float) -> None:
        """Send angle to motor (degrees)
        Args:
            angle (float): angle (degrees)

        Returns:
            None
        """
        cmd = [np.deg2rad(angle), 0, 5, 0.2, 0]

        self._update_motor(cmd=cmd)

    def send_velocity(self, desired_vel: float) -> None:
        """Send velocity to motor in rad/s
        Args:
            desired_vel (float): target speed in rad/s

        Returns:
            tuple: pos [rad], vel [rad/s], torque [Nm]
        """
        vel_gain = 5 if self.type == "AK70-10" else 2.5
        cmd = [0, desired_vel, 0, vel_gain, 0]
        self._update_motor(cmd=cmd)

    def send_torque(self, desired_torque: float, safety: bool=True) -> tuple:
        """ Send torque to motor

        Args:
            desired_torque (float): target torque
            safety (bool, optional): Safety clipping. Defaults to True.

        Returns:
            tuple: _description_
        """
        # Clip the filtered torque to the torque limits
        filtered_torque = np.clip(desired_torque, self.params['T_min'], self.params['T_max'])

        # Hard code safety
        if safety:
            filtered_torque = np.clip(filtered_torque, -3, 3)

        cmd = [0, 0, 0, 0, filtered_torque]

        self._update_motor(cmd=cmd)

    def clear_buffers(self):
        """
        Clear the circular buffers, so that the next call to update_motor will not use old data
        """
        self.position_buffer = [0] * self.buffer_size
        self.velocity_buffer = [0] * self.buffer_size
        self.torque_buffer = [0] * self.buffer_size

    def _update_motor(self, cmd: list[hex], wait_time: float = 0.001) -> bool:
        if len(cmd) != 5:
            print("Too many or too few arguments")
            return

        # Invert sign of position, velocity or torque for AK60-6
        if self.type == "AK60-6":
            cmd[0] *= -1
            cmd[1] *= -1
            cmd[-1] *= -1

        packed_cmd = self._pack_cmd(*cmd)

        self.can_bus.send(self._send_message(packed_cmd))
        new_msg = self.can_bus.recv(wait_time)

        if self.logging:
            self.logger.info(f"Data: {new_msg}")

        if new_msg is None:
            return

        motor_id = new_msg.data[0]

        if motor_id != self.params["ID"]:
            self.swapped_motors = True
            if motor_id != self.previous_id:
                self.switch_now = True
                print(f"SWAPPED MOTORS, {self.params["ID"]}")
            else:
                self.switch_now = False
        else:
            self.swapped_motors = False
            if motor_id != self.previous_id:
                self.switch_now = True
                print(f"SWAPPED MOTORS, {self.params["ID"]}")
            else:
                self.switch_now = False
        
        self.previous_id = motor_id

        # Must be after sending message to ensure motor stops
        if self._emergency_stop:
            return

        self.check_safety_speed_limit()
        self.check_temperature_limit()
        try:
            # Read position, velocity, and torque from the received message
            p, v, t, te = self._read_motor_msg(new_msg.data)
            if not self.swapped_motors:
                p *= -1 if self.type == "AK60-6" else 1
                v *= -1 if self.type == "AK60-6" else 1
                t *= -1 if self.type == "AK60-6" else 1
            else:
                p *= -1 if self.type == "AK70-10" else 1
                v *= -1 if self.type == "AK70-10" else 1
                t *= -1 if self.type == "AK70-10" else 1

            # Update the circular buffers
            self.position_buffer[self.buffer_index] = p
            self.velocity_buffer[self.buffer_index] = v
            self.torque_buffer[self.buffer_index] = t
            self.temperature_buffer[self.buffer_index] = te

            self.buffer_index = (self.buffer_index + 1) % self.buffer_size

            # Calculate the moving average position and velocity
            if self._emergency_stop:
                self.velocity_buffer = [0] * self.buffer_size
            
            if self.new_run or self.switch_now: 
                self.position_buffer = [p] * self.buffer_size
                self.velocity_buffer = [v] * self.buffer_size
                self.torque_buffer = [t] * self.buffer_size
                self.temperature_buffer = [te] * self.buffer_size
                self.new_run = False
            
            avg_position = sum(self.position_buffer) / self.buffer_size
            avg_velocity = sum(self.velocity_buffer) / self.buffer_size
            avg_temperature = sum(self.temperature_buffer) / self.buffer_size

            self.position = avg_position
            self.velocity = avg_velocity
            self.temperature = avg_temperature
            self.measured_torque = t

        except AttributeError as e:
            traceback.print_exc()
            return True

        self.prev_velocity = self.velocity
        self._last_update_time = time.time()


    def _read_motor_msg(self, data: can.Message) -> tuple:
        """Read motor message
        Args:
            data (can.Message): can message response
        Returns:
            tuple: pos, vel, torque
        """
        p_int = (data[1] << 8) | data[2]
        v_int = (data[3] << 4) | (data[4] >> 4)
        t_int = ((data[4] & 0xF) << 8) | data[5]
        # Temperature (Range from -40 to 215)
        te_int = data[6]

        # convert to floats
        p = uint_to_float(p_int, self.params["P_min"], self.params["P_max"], 16)
        v = uint_to_float(v_int, self.params["V_min"], self.params["V_max"], 12)
        if not self.swapped_motors:
            t = uint_to_float(t_int, self.params["T_min"], self.params["T_max"], 12)
        else:
            t = uint_to_float(t_int, self.other_params["T_min"], self.other_params["T_max"], 12)
        te = uint_to_float(te_int, -40, 215, 8)

        return p, v, t, te  # position, velocity, torque, temperature

    def _pack_cmd(self, p_des: int, v_des: int, kp: int, kd: int, t_ff: int):
        # convert floats to ints
        p_int = float_to_uint(
            p_des, self.params["P_min"], self.params["P_max"], 16
        )
        v_int = float_to_uint(
            v_des, self.params["V_min"], self.params["V_max"], 12
        )
        kp_int = float_to_uint(
            kp, self.params["Kp_min"], self.params["Kp_max"], 12
        )
        kd_int = float_to_uint(
            kd, self.params["Kd_min"], self.params["Kd_max"], 12
        )
        t_int = float_to_uint(
            t_ff, self.params["T_min"], self.params["T_max"], 12
        )
        # pack ints into buffer message
        msg = []
        msg.append(p_int >> 8)
        msg.append(p_int & 0xFF)
        msg.append(v_int >> 4)
        msg.append((((v_int & 0xF) << 4)) | (kp_int >> 8))
        msg.append(kp_int & 0xFF)
        msg.append(kd_int >> 4)
        msg.append((((kd_int & 0xF) << 4)) | (t_int >> 8))
        msg.append(t_int & 0xFF)
        return msg

    def _send_message(self, data):
        if self.logging:
            self.logger.info(f"Sent - ID: {self.params['ID']}, Data: {data}")
        return can.Message(arbitration_id=self.params["ID"], data=data, is_extended_id=False)

