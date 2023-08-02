import time
import timeit

from assistive_arm.network.client import (
    setup_client_logger,
    get_qrt_data,
    connect_to_server,
)


def main():
    client_logger = setup_client_logger()
    socket = connect_to_server(logger=client_logger)

    try:
        while True:
            cur_time = timeit.default_timer()
            markers, forces = get_qrt_data(logger=client_logger, socket=socket)
            print(f"Elapsed time: {timeit.default_timer() - cur_time}s")
            time.sleep(0.1)
    except KeyboardInterrupt:
        exit(0)


if __name__ == "__main__":
    main()
