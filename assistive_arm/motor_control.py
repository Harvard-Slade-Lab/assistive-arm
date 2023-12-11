import can
import csv
import numpy as np
import os
import time
import traceback
import yaml

from typing import Literal
from pathlib import Path

from assistive_arm.motor_helper import read_motor_msg, pack_cmd


with open("./motor_config.yaml", "r") as f:
        MOTOR_PARAMS = yaml.load(f, Loader=yaml.FullLoader)

class CubemarsMotor:
    def __init__(self, motor_type: Literal["AK60-6", "AK70-10"], csv_file: Path=None, frequency: int=200) -> None:
        self.type = motor_type
        self.params = MOTOR_PARAMS[motor_type]
        self.log_vars = ["position", "velocity", "torque"]


        self.position = 0
        self.velocity = 0
        self.torque = 0

        self.frequency = frequency
        self.csv_file_name = csv_file.with_name(f"{self.type}_" + csv_file.name)
        self._start_time = time.time()

        os.system(f"touch {self.csv_file_name}")

    
    def __enter__(self):
        if self.csv_file_name is not None:
            with open(self.csv_file_name,'w') as fd:
                writer = csv.writer(fd)
                writer.writerow(["pi_time"]+self.log_vars)
            self.csv_file = open(self.csv_file_name,'a').__enter__()
            self.csv_writer = csv.writer(self.csv_file)
        self._init_can_ports()
        self.can_bus = can.interface.Bus(channel=self.params['CAN'], bustype='socketcan')
        self._connect_motor()

        return self

    def __exit__(self, exc_type, exc_value, trb):
        self._stop_motor()
        self._stop_can_port()

        if not (exc_type is None):
            traceback.print_exc()

    def _init_can_ports(self) -> None:
        os.system(f"sudo ip link set {self.params['CAN']} type can bitrate 1000000")
        os.system(f"sudo ifconfig {self.params['CAN']} up")

    def _stop_can_port(self) -> None:
        os.system(f"sudo ifconfig {self.params['CAN']} down")


    def _connect_motor(self, delay: float = 0.001, zero: bool=False) -> None:
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

            self.can_bus.send(self._send_message(data=start_motor_mode))
            print("Waiting for response...", )
            time.sleep(0.1)
            response = self.can_bus.recv(delay)  # time

            try:
                P, V, T = read_motor_msg(response.data)
                print(f"Motor {self.type} connected successfully")
                starting = True
                if zero:
                    self.send_zero_position()
            except AttributeError:
                print("Received no response")
    
    def _stop_motor(self) -> None:
        """ Stop motor """
        
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

# Move the cursor up and clear these lines

    def send_zero_position(self) -> None:
        zero_position = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE]

        self.can_bus.send(self._send_message(zero_position))
        # Sleep for 2.5s to allow motor to zero
        response = self.can_bus.recv(1.5)
        print("Zeroing position...")
        print("Pos, Vel, Torque: ", read_motor_msg(response.data))

        zero_cmd = [0, 0, 0, 0, 0]

        self._update_motor(cmd=zero_cmd, wait_time=0.001)

    def send_angle(self, angle: float) -> None:
        """ Send angle to motor (degrees)
        Args:
            angle (float): angle (degrees)

        Returns:
            None
        """
        cmd = [np.deg2rad(angle), 0, 5, 0.2, 0]
        
        self._update_motor(cmd=cmd)
    
    def send_velocity(self, desired_vel: float) -> None:
        """ Send velocity to motor in rad/s
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
        filtered_torque = np.clip(desired_torque, self.params['T_min'], self.params['T_max'])
        
        # Hard code safety
        if safety:
            filtered_torque = np.clip(filtered_torque, -3, 3)
            
        cmd = [0, 0, 0, 0, filtered_torque]
        
        self._update_motor(cmd=cmd)

    def _update_motor(self, cmd: list[hex], wait_time: float = 0.001) -> tuple:
        if len(cmd) != 5:
            print("Too many or too few arguments")
            return None, None, None

        # Invert sign of position, velocity or torque for AK60-6
        if self.type == "AK60-6":
            cmd[0] *= -1
            cmd[1] *= -1
            cmd[-1] *= -1

        packed_cmd = pack_cmd(*cmd)

        self.can_bus.send(self._send_message(packed_cmd))
        new_msg = self.can_bus.recv(wait_time)
        self._last_update_time = time.time()

        try:
            self.position, self.velocity, self.torque = read_motor_msg(new_msg.data)
            if self.csv_writer:
                self.csv_writer.writerow([self._last_update_time - self._start_time] + [self.position, self.velocity, self.torque])

            self.position *= -1 if self.type == "AK60-6" else 1
            self.velocity *= -1 if self.type == "AK60-6" else 1
            self.torque *= -1 if self.type == "AK60-6" else 1
        
        except AttributeError as e:
            return 0, 0, 0

    def _send_message(self, data):
        return can.Message(arbitration_id=self.params['ID'], data=data, is_extended_id=False)





