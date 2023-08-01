import os
import json
import logging
import numpy as np
import timeit
import zmq


from datetime import datetime
from collections import namedtuple

from assistive_arm.utils import print_elapsed_time

forces = namedtuple(
    "forces",
    ["x", "y", "z", "moment_x", "moment_y", "moment_z", "acc_x", "acc_y", "acc_z"],
)


def connect_to_server(logger: logging.Logger = None) -> zmq.Socket:
    """Connect to server
    Args:
        logger (logging.Logger, optional): Logger, defaults to None.
    Returns:
        zmq.Socket: socket
    """
    context = zmq.Context()
    logger.info("Connecting to server...")
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://10.245.250.27:5555")

    return socket


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

@print_elapsed_time()
def get_qrt_data(logger: logging.Logger, socket: zmq.Socket) -> dict:
    """Get marker and force data from Motion Capture

    Args:
        logger (logging.Logger): logger
        socket (zmq.Socket): zmq socket

    Returns:
        dict: contains organized marker and force data
    """
    # Retrieve data from server
    marker_data = {}
    force_data = {}

    rt_data = request_data(logger, socket)

    # Measure time to build dicts
    start_time = timeit.default_timer()
    for rt_id, data in rt_data.items():
        if "plate" in rt_id:
            force_data[rt_id] = [forces(*sensor) for sensor in data]
        elif "marker" in rt_id:
            marker_data[rt_id] = np.array(data)
    end_time = timeit.default_timer()
    print(f"Elapsed time for building dicts: {end_time - start_time} seconds")

    logger.info(f"Force data: \n{force_data}")
    logger.info(f"Marker data: \n{marker_data}")

    return marker_data, force_data

@print_elapsed_time()
def request_data(logger: logging.Logger, socket: zmq.Socket) -> dict:
    """Request data from server

    Args:
        logger (logging.Logger): logger
        socket (zmq.Socket): zmq socket
    Returns:
        dict: contains raw data from server
    """
    try:
        logger.info("Sending request...")
        socket.send_string("Retrieve marker data")
        message = socket.recv_string()

        try:
            rt_data = json.loads(message)
        except json.JSONDecodeError as error:
            logger.error(f"An error occurred while decoding JSON: {error}")
            return rt_data

    except Exception as general_error:
        logger.error(f"An unexpected error occurred: {general_error}")

    return rt_data
