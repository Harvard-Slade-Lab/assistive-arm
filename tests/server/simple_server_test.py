import zmq

def main():
    """Server function."""
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")

    while True:
        message = socket.recv_string()
        print("Received request: %s" % message)

        # Process the request and Send reply back to client.
        socket.send_string("Hello from server! Your message was: %s" % message)

if __name__ == "__main__":
    main()
