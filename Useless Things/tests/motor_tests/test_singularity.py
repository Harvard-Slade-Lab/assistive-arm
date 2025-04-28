import os
import sys
import time
import numpy as np

from pathlib import Path

from NeuroLocoMiddleware.SoftRealtimeLoop import SoftRealtimeLoop
from assistive_arm.motor_control import CubemarsMotor

np.set_printoptions(precision=3, suppress=True)


def avoidance_torque(theta_1: float, theta_2: float, d_sing: float=0.175) -> float:
    
    l1 = 0.44
    l2 = 0.41
    jacobian = np.array([
                [-l1*np.sin(theta_1)-l2*np.sin(theta_1+theta_2), - l2*np.sin(theta_1+theta_2)],
                [l1*np.cos(theta_1)+l2*np.cos(theta_1+theta_2), l2*np.cos(theta_1+theta_2)]])

    Ks = 150 # Singularity avoidance gain

    manipulability = l1*l2*abs(np.sin(theta_2))

    force = np.array([0, 0])

    if abs(theta_2) > d_sing:
        force = np.array([0, 0]).T
    else:
        force = np.array(
            [0,
             Ks*manipulability* l1 * l2 * np.sin(2*theta_2) / (2*abs(np.sin(theta_2 + 0.001)))]
        ).T
    
    torque = jacobian.T @ force

    return force


def main(motor_1: CubemarsMotor, motor_2: CubemarsMotor):
    # Start control loop
    freq = 200  # Hz

    loop = SoftRealtimeLoop(dt=1 / freq, report=True, fade=0)
    start_time = 0

    # General control loop
    l1 = 0.44
    l2 = 0.41


    try:
        for t in loop:

            # motor_1.send_torque(desired_torque=0, safety=False)
            # motor_2.send_torque(desired_torque=0, safety=False)
            avoid_torque = avoidance_torque(theta_1=motor_1.position, theta_2=motor_2.position)
            motor_1.send_torque(desired_torque=avoid_torque[0], safety=False)
            motor_2.send_torque(desired_torque=avoid_torque[1], safety=False)

            sing_prox = l1*l2*abs(np.sin(motor_2.position))

            if t - start_time > 0.1:
                print(
                    f"{motor_1.type}: Angle: {np.rad2deg(motor_1.position):.3f} Velocity: {motor_1.velocity:.3f} Torque: {motor_1.measured_torque:.3f}"
                )
                print(f"Dist singularity: {sing_prox:.3f}, Avoidance torque: {avoid_torque}")

                sys.stdout.write(f"\x1b[2A\x1b[2K")
                start_time = t

            if motor_1._emergency_stop or motor_2._emergency_stop:
                break
        del loop

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Shutting down...")


if __name__ == "__main__":
    with CubemarsMotor("AK70-10", logging=True) as motor_1:
        with CubemarsMotor("AK60-6", logging=True) as motor_2:
            main(motor_1, motor_2)
