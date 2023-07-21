import json
import logging

from datetime import datetime
from collections import namedtuple


def setup_logger() -> logging.Logger:
    """Setup logger for client"""
    # Generate logfile name
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logfile = f"../../logs/{current_time}_client_marker_processing.log"

    logging.basicConfig(
        filename=logfile, format="%(asctime)s %(message)s", filemode="w"
    )

    logger = logging.getLogger("Client logger")
    logger.setLevel(logging.DEBUG)

    return logger

def request_data(logger, socket):
    coordinates = namedtuple("coordinates", ["x", "y", "z"])
    forces = namedtuple("forces", ["x", "y", "z", "moment_x", "moment_y", "moment_z", "acc_x", "acc_y", "acc_z"])
    markers = {}
    force_data = {}

    try:
        logger.info("Sending request...")
        socket.send_string("Retrieve marker data")
        message = socket.recv_string()

        try:
            rt_data = json.loads(message)
        except json.JSONDecodeError as error:
            logger.error(f"An error occurred while decoding JSON: {error}")
            return markers, force_data

        markers = {key: coordinates(**value) for key, value in rt_data.items() if "marker" in key}
        force_data = {key: forces(**value) for key, value in rt_data.items() if "plate" in key}
        logger.info("Marker 3D position: \n", markers)
        logger.info("Force plate data: \n", force_data)

    except Exception as general_error:
        logger.error(f"An unexpected error occurred: {general_error}")

    return markers, force_data