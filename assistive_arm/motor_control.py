import can
import os
import traceback

from time import sleep

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
            'Use_derived_torque_constants': False, # true if you have a better model
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
        }
}

class CubemarsMotor:
    def __init__(self) -> None:
        pass




def send_message(arb_id, data):
    return can.Message(arbitration_id=arb_id, data=data, is_extended_id=False)


def connect_motor(can_bus: can.Bus, motor_id: hex, delay: float = 0.001) -> None:
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
        print("CMD: ", start_motor_mode)

        can_bus.send(send_message(motor_id, start_motor_mode))
        print("Waiting for response...", )
        sleep(0.01)
        response = can_bus.recv(delay)  # time
        print("Response: \n", response)

        try:
            P, V, T = read_motor_msg(response.data)
            print(f"Motor {motor_id} connected successfully")
            starting = True
            send_zero_position(can_bus=can_bus, motor_id=motor_id)
        except AttributeError:
            print("Received no response")



def stop_motor(can_bus: can.Bus, motor_id: hex) -> None:
    """Stop motor
    Args:
        can_bus (can.Bus): can bus object corresponding to motor
        motor_id (hex): motor id in hexadecimal
    """
    stop_motor_mode = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFD]

    can_name = can_bus.channel_info.split(" ")[-1]

    print(f"\nShutting down motor {motor_id}...")
    try:
        can_bus.send(send_message(motor_id, stop_motor_mode))  # disable motor mode
        can_bus.shutdown()
        print(f"Motor {motor_id} shutdown successful")
    except:
        traceback.print_exc()
        print(f"Failed to shutdown motor {motor_id} or {can_name}")

def update_motor(
    can_bus: can.Bus, motor_id: int, cmd: list[hex], wait_time: float = 0.001
) -> tuple:
    if len(cmd) != 5:
        print("Too many or too few arguments")
        return None

    packed_cmd = pack_cmd(*cmd)

    can_bus.send(send_message(motor_id, packed_cmd))
    new_msg = can_bus.recv(wait_time)

    try:
        pos, vel, torque = read_motor_msg(new_msg.data)
        return pos, vel, torque
    
    except AttributeError:
        print("Trying to access a 'None' object")


def send_zero_position(can_bus: can.Bus, motor_id: hex, delay: float = 2.5) -> None:
    zero_position = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE]

    can_bus.send(send_message(motor_id, zero_position))
    # Sleep for 2.5s to allow motor to zero
    response = can_bus.recv(delay)
    print("Zeroing position...")
    print("Pos, Vel, Torque: ", read_motor_msg(response.data))

    zero_cmd = [0, 0, 0, 0, 0]

    update_motor(can_bus=can_bus, motor_id=motor_id, cmd=zero_cmd, wait_time=0.001)
