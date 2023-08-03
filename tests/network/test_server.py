import time
import zmq

def main():
    """Server function."""
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.bind("tcp://*:5555")

    while True:
        # Publish message to all subscribers
        freq = 10 #Hz
        publisher.send_string(f"{time.time()}")
        time.sleep(1/freq)

if __name__ == "__main__":
    main()
