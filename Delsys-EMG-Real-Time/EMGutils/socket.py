import socket
import struct

class SocketServer:
    """
    Handles socket communication with the Raspberry Pi server.
    """
    def __init__(self, host='10.250.16.32', port=3003):
        self.host = host
        self.port = port
        self.socket = None

    def connect_to_server(self):
        """Connects to the Raspberry Pi server."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))

    def send_data(self, data):
        """Sends data to the Raspberry Pi."""
        try:
            # if self.socket:
            byte_data = data.encode('utf-8')
            self.socket.sendall(byte_data)
        except BrokenPipeError:
            print("Connection closed by the server.")

    def send_roll_angle(self, data):
        """Sends roll angle data to the Raspberry Pi, faster than using utf."""
        try:
            message = struct.pack('!cf', b'r', data)
            self.socket.sendall(message)
        except BrokenPipeError:
            print("Connection closed by the server.")

    def close_connection(self):
        """Closes the connection to the server."""
        if self.socket:
            self.socket.close()
            self.socket = None

