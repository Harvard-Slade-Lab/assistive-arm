import os
import json
import logging
import numpy as np
import zmq


from datetime import datetime
from collections import namedtuple

coordinates = namedtuple("coordinates", ["x", "y", "z"])
forces = namedtuple(
    "forces",
    ["x", "y", "z", "moment_x", "moment_y", "moment_z", "acc_x", "acc_y", "acc_z"],
)


def connect_to_server(logger: logging.Logger=None) -> zmq.Socket:
    context = zmq.Context()
    logger.info("Connecting to server...")
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://10.245.250.27:5555")

    return socket

def setup_client_logger() -> logging.Logger:
    """Setup logger for client"""
    # Generate logfile name
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logfile = f"../logs/{current_time}_client_marker_processing.log"

    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    logging.basicConfig(
        filename=logfile, format="%(asctime)s %(message)s", filemode="w"
    )

    logger = logging.getLogger("Client logger")
    logger.setLevel(logging.DEBUG)

    return logger


def get_qrt_data(logger: logging.Logger, socket: zmq.Socket) -> dict:
    markers = {}
    force_data = {}

    # Retrieve data from server
    rt_data = request_data(logger, socket)

    markers = {
        key: np.array(value) for key, value in rt_data.items() if "marker" in key
    }
    force_data = {
        key: forces(**value) for key, value in rt_data.items() if "plate" in key
    }
    logger.info("Marker 3D position: \n", markers)
    logger.info("Force plate data: \n", force_data)

    return markers, force_data


def request_data(logger, socket):
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
