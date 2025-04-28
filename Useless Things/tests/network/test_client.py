import time
import zmq

def main():
    """Client function."""
    context = zmq.Context()

    # Socket to talk to server
    print("Connecting to hello world server...")
    subscriber = context.socket(zmq.SUB)
    subscriber.connect("tcp://10.245.250.27:5555")

    subscriber.setsockopt_string(zmq.SUBSCRIBE, "")

    while True:
        # Receive message from publisher
        message = subscriber.recv_string()
        print("Time elapsed: ", time.time() - float(message))

if __name__ == "__main__":
    main()
