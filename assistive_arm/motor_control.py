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


with open("./motor_config.yaml", "r") as f:
    MOTOR_PARAMS = yaml.load(f, Loader=yaml.FullLoader)


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
    ) -> None:
        self.type = motor_type
        self.params = MOTOR_PARAMS[motor_type]
        self.log_vars = ["position", "velocity", "torque"]
        self.frequency = frequency

        self.position = 0
        self.prev_velocity = 0
        self.velocity = 0
        self.csv_file_name = None

        self.measured_torque = 0

        self.buffer_index = 0
        self.buffer_size = int(0.1 * self.frequency)  # Buffer size for X secs

        self.position_buffer = [0] * self.buffer_size
        self.velocity_buffer = [0] * self.buffer_size
        self.torque_buffer = [0] * self.buffer_size

        self._emergency_stop = False
        self._first_run = True

    def __enter__(self):
        self._init_can_ports()
        self.can_bus = can.interface.Bus(
            channel=self.params["CAN"], bustype="socketcan"
        )
        self._connect_motor()
        self._start_time = time.time()
        self._last_update_time = self._start_time
        return self

    def __exit__(self, exc_type: None, exc_value: None, trb: None):
        if self._emergency_stop:
            print("\n\nEmergency stop triggered. Shutting down...\n\n")
        else:
            self._stop_motor()
            self._stop_can_port()
    
        if exc_type is not None:
            traceback.print_exc()

    def _init_can_ports(self) -> None:
        os.system(f"sudo ip link set {self.params['CAN']} type can bitrate 1000000")
        os.system(f"sudo ifconfig {self.params['CAN']} up")

    def _stop_can_port(self) -> None:
        os.system(f"sudo ifconfig {self.params['CAN']} down")

    def _connect_motor(self, delay: float = 0.005, zero: bool = False) -> None:
        """Establish connection with motor
        Args:
            can_bus (can.Bus): can bus object corresponding to motor
            motor_id (hex): motor id in hexadecimal
            delay (float, optional): waiting time. Defaults to 0.001.
        """
        start_motor_mode = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFC]

        starting = False

        while not starting:
            print("\nSending starting command...")
            try:
                self.can_bus.send(self._send_message(data=start_motor_mode))
                print("Waiting for response...")
                time.sleep(0.1)
                response = self.can_bus.recv(delay)

                if response:
                    P, V, T = self._read_motor_msg(response.data)
                    print(f"Motor {self.type} connected successfully")
                    starting = True
                    if zero:
                        self.send_zero_position()
                else:
                    raise AttributeError("No response from motor")

            except (can.CanError, AttributeError) as e:
                print(f"Error: {e}, reinitializing CAN interface...")
                # Try moving this to line 115
                self._init_can_ports()  # Reinitialize the CAN interface
                time.sleep(1)  # Wait before retrying

    def _stop_motor(self) -> None:
        """Stop motor"""

        stop_motor_mode = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFD]
        can_name = self.can_bus.channel_info.split(" ")[-1]

        print(f"\nShutting down motor {self.type}...")

        try:
            self.can_bus.send(self._send_message(stop_motor_mode))  # disable motor mode
            self.can_bus.shutdown()
            print(f"Motor {self.type} shutdown successful")
        except:
            traceback.print_exc()
            print(f"Failed to shutdown motor {self.type} or {can_name}")

    def check_safety_speed_limit(self):
        if abs(self.velocity) > self.params["Vel_limit"]:
            self._emergency_stop = True
            self.send_velocity(0)
            self.velocity = 0
            time.sleep(0.5)
            self.send_torque(0)
            # self._stop_motor()

    def send_zero_position(self) -> None:
        zero_position = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE]

        self.can_bus.send(self._send_message(zero_position))
        # Sleep for 2.5s to allow motor to zero
        response = self.can_bus.recv(1.5)
        print("Zeroing position...")
        print("Pos, Vel, Torque: ", self._read_motor_msg(response.data))

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

        # Must be after sending message to ensure motor stops
        if self._emergency_stop:
            return

        self.check_safety_speed_limit()
        try:
            # Read position, velocity, and torque from the received message
            p, v, t = self._read_motor_msg(new_msg.data)

            p *= -1 if self.type == "AK60-6" else 1
            v *= -1 if self.type == "AK60-6" else 1
            t *= -1 if self.type == "AK60-6" else 1

            # Update the circular buffers
            self.position_buffer[self.buffer_index] = p
            self.velocity_buffer[self.buffer_index] = v
            self.torque_buffer[self.buffer_index] = t

            self.buffer_index = (self.buffer_index + 1) % self.buffer_size

            # Calculate the moving average position and velocity
            if self._emergency_stop:
                self.velocity_buffer = [0] * self.buffer_size
            
            if self._first_run: 
                self.position_buffer = [p] * self.buffer_size
                self.velocity_buffer = [v] * self.buffer_size
                self.torque_buffer = [t] * self.buffer_size
                self._first_run = False
            
            avg_position = sum(self.position_buffer) / self.buffer_size
            avg_velocity = sum(self.velocity_buffer) / self.buffer_size

            self.position = avg_position
            self.velocity = avg_velocity
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
        if data == None:
            return None
        # else:
        #    print(data[0],data[1], data)
        id_val = data[0]
        p_int = (data[1] << 8) | data[2]
        v_int = (data[3] << 4) | (data[4] >> 4)
        t_int = ((data[4] & 0xF) << 8) | data[5]
        # convert to floats
        p = uint_to_float(p_int, self.params["P_min"], self.params["P_max"], 16)
        v = uint_to_float(v_int, self.params["V_min"], self.params["V_max"], 12)
        t = uint_to_float(t_int, self.params["T_min"], self.params["T_max"], 12)

        return p, v, t  # position, velocity, torque

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
        return can.Message(
            arbitration_id=self.params["ID"], data=data, is_extended_id=False
        )