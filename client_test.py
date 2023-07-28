import zmq


def main():
    """Client function."""
    context = zmq.Context()

    # Socket to talk to server
    print("Connecting to hello world server...")
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://10.250.137.101:5555")

    for request in range(10):
        print("Sending request %s ..." % request)
        socket.send_string("Hello from client! Request number: %s" % request)

        # Wait for the reply.
        message = socket.recv_string()
        print("Received reply %s [ %s ]" % (request, message))

if __name__ == "__main__":
    main()
