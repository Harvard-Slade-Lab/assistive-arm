import os
import json
import logging
import numpy as np
import timeit
import time
import zmq


from datetime import datetime
from collections import namedtuple

from assistive_arm.utils.logging import print_elapsed_time

forces = namedtuple(
    "forces",
    ["x", "y", "z", "moment_x", "moment_y", "moment_z", "acc_x", "acc_y", "acc_z"],
)


def connect_to_publisher(logger: logging.Logger = None) -> zmq.Socket:
    """Connect to publisher socket and return subscriber socket
    
    Args:
        logger (logging.Logger, optional): Logger, defaults to None.
    Returns:
        zmq.Socket: subscriber socket
    """
    if logger:
        logger.info("Connecting to publisher...")
    context = zmq.Context()

    # Set up subscriber
    subscriber = context.socket(zmq.SUB)
    subscriber.connect("tcp://10.245.250.27:5555")
    subscriber.setsockopt_string(zmq.SUBSCRIBE, "")

    return subscriber


def setup_client_logger() -> logging.Logger:
    """Setup logger for client"""
    # Generate logfile name
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logfile = f"./logs/{current_time}_client_marker_processing.log"

    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    logging.basicConfig(
        filename=logfile, format="%(asctime)s %(message)s", filemode="w"
    )

    logger = logging.getLogger("Client logger")
    logger.setLevel(logging.DEBUG)

    return logger


# @print_elapsed_time()
def get_qrt_data(logger: logging.Logger, socket: zmq.Socket) -> dict:
    """Get marker and force data from Motion Capture

    Args:
        logger (logging.Logger): logger
        socket (zmq.Socket): zmq publisher socket

    Returns:
        dict: contains organized marker and force data
    """
    # Retrieve data from server
    marker_data = {}
    force_data = {}
    analog_data = {}

    rt_data = read_mocap_data(logger=logger, socket=socket)

    # Measure time to build dicts
    for rt_id, data in rt_data.items():
        if "plate" in rt_id:
            force_data[rt_id] = [forces(*sensor) for sensor in data]
        elif "marker" in rt_id:
            marker_data[rt_id] = np.array(data)
        elif "analog" in rt_id:
            analog_data[rt_id] = np.array(data)
    # logger.info(f"timestamp: \n{time.time()}")
    # logger.info(f"Force data: \n{force_data}")
    logger.info(f"Marker data: \n{marker_data}")
    logger.info(f"Analog data: \n{analog_data}")

    return marker_data, force_data, analog_data


# @print_elapsed_time()
def read_mocap_data(logger: logging.Logger, socket: zmq.Socket) -> dict:
    """ Read mocap data from publisher node

    Args:
        logger (logging.Logger): logger
        socket (zmq.Socket): zmq socket
    Returns:
        dict: contains raw data from server
    """
    try:
        # Read data from publisher
        message = socket.recv_string()
        try:
            rt_data = json.loads(message)
        except json.JSONDecodeError as error:
            logger.error(f"An error occurred while decoding JSON: {error}")
            return rt_data

    except Exception as general_error:
        logger.error(f"An unexpected error occurred: {general_error}")

    return rt_data
