import numpy as np
import can
from chspy import CubicHermiteSpline
import os
from natsort import natsorted
import time

# constants for motor stuff
PMIN = -12.5
PMAX = -PMIN
VMIN = -50.0  # 46.57 # -23.24 # becomes 46.57 for 48V!!!
VMAX = -VMIN
TMIN = -54.0
TMAX = -TMIN
KPMIN = 0.0
KPMAX = 500.0
KDMIN = 0.0
KDMAX = 5.0

def zeroMotor(can0, m_id, zero_position, mpos, mvel, mtorq, wait_time):
    can0.send(message(m_id, zero_position))
    time.sleep(2.5)
    new_msg = can0.recv(wait_time)  # time is timeout waiting for message
    mpos, mvel, mtorq = readMotorMsg(new_msg, mpos, mvel, mtorq)
    print("Zeroing position.")
    cmd_arr = [0, 0, 0, 0, 0]
    mpos, mvel, mtorq, _, _, _, _ = motorUpdate(
        can0, m_id, cmd_arr, mpos, mvel, mtorq, wait_time
    )
    return mpos, mvel, mtorq


def killMotor(can0, m_id, stop_motor_mode):
    try:
        can0.send(message(m_id, stop_motor_mode))
        time.sleep(0.03)
        os.system("sudo ifconfig can0 down")
    except:
        print("Error with killing motor...")
    return False, False  # set enable_motor to False


def desiredTorque(
    tau_ramp, heelstrike_timing_adjust, torq_prof, tstride_avg, tstride, init_tau, dt
):
    phase = (tstride + heelstrike_timing_adjust) / tstride_avg * 100  # current phase
    des_torq = torq_prof.get_state(phase)[0] * tau_ramp  # tau for current
    if dt != 0.0:
        phase2 = (
            (tstride + dt + heelstrike_timing_adjust) / tstride_avg * 100
        )  # phase for desired motor torque in the future
        des_future_torq = torq_prof.get_state(phase2)[0] * tau_ramp
    else:
        des_future_torq = des_torq
    return phase, des_torq, des_future_torq


def slacktrack(force, force_des, mpos, k_zt, phase=0):
    posmdes = mpos + (force_des - force) / k_zt
    return posmdes


def motorUpdate(
    can0,
    m_id,
    cmd_arr,
    mpos,
    mvel,
    mtorq,
    wait_time,
    mpos_off=0.0,
    mpos_prev=-100,
    mvel_off=0.0,
    mvel_prev=-100,
    mvel_max=50,
    posmax=12.5,
):
    # cmd array: p_des, v_des, kp, kd, t_ff
    cmd = pack_cmd(cmd_arr[0], cmd_arr[1], cmd_arr[2], cmd_arr[3], cmd_arr[4])
    can0.send(message(m_id, cmd))
    new_msg = can0.recv(wait_time)
    mpos, mvel, mtorq = readMotorMsg(new_msg, mpos, mvel, mtorq)
    rel_mpos = mpos
    rel_mvel = mvel
    mpos, mpos_off = absPos(mpos_off, mpos, mpos_prev, pos_max=posmax)
    mvel, mvel_off = absPos(mvel_off, mvel, mvel_prev, pos_max=mvel_max)
    return mpos, mvel, mtorq, mpos_off, rel_mpos, mvel_off, rel_mvel


# wrap around motor position to avoid out of bounds errors
def absPos(mpos_off, mpos, mpos_prev, pos_max=12.5):
    if mpos_prev == -100:
        return mpos, mpos_off
    else:
        abs_pos = mpos
        if np.abs(mpos - mpos_prev) > pos_max:
            if mpos > 0:
                mpos_off -= 2 * pos_max
                abs_pos -= 2 * pos_max
            else:
                mpos_off += 2 * pos_max
                abs_pos += 2 * pos_max  # mpos_off
        return abs_pos, mpos_off


def float_to_uint(x, xmin, xmax, bits):
    span = xmax - xmin
    if x < xmin:
        x = xmin
    elif x > xmax:
        x = xmax
    convert = int((x - xmin) * (((1 << bits) - 1) / span))
    return convert


def uint_to_float(x, xmin, xmax, bits):
    span = xmax - xmin
    int_val = float(x) * span / (float((1 << bits) - 1)) + xmin
    return int_val


def read_motor_msg(data: can.Message) -> tuple:
    """ Read motor message
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
    p = uint_to_float(p_int, PMIN, PMAX, 16)
    v = uint_to_float(v_int, VMIN, VMAX, 12)
    t = uint_to_float(t_int, TMIN, TMAX, 12)

    return p, v, t  # position, velocity, torque


def pack_cmd(p_des, v_des, kp, kd, t_ff):
    # convert floats to ints
    p_int = float_to_uint(p_des, PMIN, PMAX, 16)
    v_int = float_to_uint(v_des, VMIN, VMAX, 12)
    kp_int = float_to_uint(kp, KPMIN, KPMAX, 12)
    kd_int = float_to_uint(kd, KDMIN, KDMAX, 12)
    t_int = float_to_uint(t_ff, TMIN, TMAX, 12)
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


def readMotorMsg(new_msg: can.Message):
    try:
        P, V, T = read_motor_msg(new_msg.data)
        return P, V, T
    except:
        print("Motor message error...")
        print(new_msg.data)


def message(arb_id, data):
    return can.Message(arbitration_id=arb_id, data=data, is_extended_id=False)
