import zmq

from time import sleep

from assistive_arm.network.client import setup_logger, get_qrt_data


def main():
    logger = setup_logger()
    context = zmq.Context()
    logger.info("Connecting to server...")
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://10.245.250.27:5555")


    try:
        while True:
            markers, forces = get_qrt_data(logger=logger, socket=socket)
            sleep(0.1)
    except KeyboardInterrupt:
        exit(0)

if __name__ == "__main__":
    main()
