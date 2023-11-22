import can
import csv
import numpy as np
import os
import sys
import traceback

import time
from typing import Literal

from assistive_arm.motor_helper import read_motor_msg, pack_cmd


MOTOR_PARAMS = {
    'AK70-10' : {
            'P_min' : -12.5,
            'P_max' : 12.5,
            'V_min' : -50.0,
            'V_max' : 50.0,
            'T_min' : -25.0,
            'T_max' : 25.0,
            'Kp_min': 0.0,
            'Kp_max': 500.0,
            'Kd_min': 0.0,
            'Kd_max': 5.0,
            'Kt_TMotor' : 0.095, # from TMotor website (actually 1/Kvll)
            'Current_Factor' : 0.59, # # UNTESTED CONSTANT!
            'Kt_actual': 0.122, # UNTESTED CONSTANT!
            'GEAR_RATIO': 10.0,
            'Use_derived_torque_constants': False, # true if you have bettermodel
            'CAN': "can0",
            'ID': 0x01
        },
    'AK60-6':{
            'P_min' : -12.5,
            'P_max' : 12.5,
            'V_min' : -50.0,
            'V_max' : 50.0,
            'T_min' : -15.0,
            'T_max' : 15.0,
            'Kp_min': 0.0,
            'Kp_max': 500.0,
            'Kd_min': 0.0,
            'Kd_max': 5.0,
            'Kt_TMotor' : 0.068, # from TMotor website (actually 1/Kvll)
            'Current_Factor' : 0.59, # # UNTESTED CONSTANT!
            'Kt_actual': 0.087, # UNTESTED CONSTANT!
            'GEAR_RATIO': 6.0, 
            'Use_derived_torque_constants': False, # true if you have a better model
            'CAN': "can1",
            'ID': 0x02
        }
}

class CubemarsMotor:
    def __init__(self, motor_type: Literal["AK60-6", "AK70-10"], csv_file: str=None, frequency: int=200) -> None:
        self.type = motor_type
        self.params = MOTOR_PARAMS[motor_type]
        self.log_vars = ["position", "velocity", "torque"]

        self.position = 0
        self.velocity = 0
        self.torque = 0

        self.frequency = frequency
        self.csv_file_name = csv_file
        self._start_time = time.time()
    
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

    def print_state(self, other_motor=None, P_EE=None) -> None:
        """ Print state of this motor, optionally the state of another motor, and optionally P_EE """
        lines_to_move_up = 3 if P_EE is not None else 2
        sys.stdout.write(f'\x1b[{lines_to_move_up}A\x1b[2K')  # Move the cursor up and clear these lines

        print(f"{self.type}: Angle: {self.position:.3f} Velocity: {self.velocity:.3f} Torque: {self.torque:.3f}")
        if other_motor:
            print(f"{other_motor.type}: Angle: {other_motor.position:.3f} Velocity: {other_motor.velocity:.3f} Torque: {other_motor.torque:.3f}")

        if P_EE is not None:
            print(f"P_EE: x:{P_EE[0]:.3f} y:{P_EE[1]:.3f} theta_1+theta_2:{np.rad2deg(P_EE[2]):.3f}")

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
    
    def send_torque(self, desired_torque: float) -> tuple:
        filtered_torque = np.clip(desired_torque, self.params['T_min'], self.params['T_max'])
        cmd = [0, 0, 0, 0, filtered_torque]
        
        self._update_motor(cmd=cmd)

    def _update_motor(self, cmd: list[hex], wait_time: float = 0.001) -> tuple:
        if len(cmd) != 5:
            print("Too many or too few arguments")
            return None

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
            self.csv_writer.writerow([self._last_update_time - self._start_time] + [self.position, self.velocity, self.torque])

            self.position *= -1 if self.type == "AK60-6" else 1
            self.velocity *= -1 if self.type == "AK60-6" else 1
            self.torque *= -1 if self.type == "AK60-6" else 1
        
        except AttributeError as e:
            return 0, 0, 0

    def _send_message(self, data):
        return can.Message(arbitration_id=self.params['ID'], data=data, is_extended_id=False)





