import time
import zmq

def main():
    """Client function."""
    context = zmq.Context()

    # Socket to receive messages from the server
    subscriber = context.socket(zmq.SUB)
    subscriber.connect("tcp://10.245.250.27:5555")
    subscriber.setsockopt_string(zmq.SUBSCRIBE, "")

    # Socket to send trigger commands to the server
    publisher = context.socket(zmq.PUB)
    publisher.connect("tcp://10.245.250.27:5556")  # Assuming the server listens on a different port for triggers

    while True:
        # Receive message from the server
        message = subscriber.recv_string()
        print("Time elapsed: ", time.time() - float(message))

        # Send a trigger command to the server
        # This can be triggered based on some condition or input
        trigger_command = "TRIGGER"
        publisher.send_string(trigger_command)
        time.sleep(1)  # Adjust the frequency of sending triggers as needed

if __name__ == "__main__":
    main()
