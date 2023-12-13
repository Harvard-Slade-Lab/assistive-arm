import time
import timeit

from assistive_arm.network.client import (
    setup_client_logger,
    get_qrt_data,
    connect_to_publisher,
)


def main():
    client_logger = setup_client_logger()
    subscriber = connect_to_publisher(logger=client_logger)

    try:
        while True:
            cur_time = timeit.default_timer()
            markers, forces, analog_data = get_qrt_data(logger=client_logger, socket=subscriber)
            print("markers: ", markers)
            print("analog data: ", analog_data)
            # print(f"Elapsed time: {timeit.default_timer() - cur_time}s")

    except KeyboardInterrupt:
        exit(0)


if __name__ == "__main__":
    main()
