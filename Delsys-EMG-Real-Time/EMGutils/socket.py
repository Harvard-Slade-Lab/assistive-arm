import socket
import struct

class SocketServer:
    """
    Handles socket communication with the Raspberry Pi server.
    Manages separate connections for roll angles and other data.
    """
    def __init__(self, host='10.250.116.196', roll_port=3001, data_port=3002):
        self.host = host
        self.roll_port = roll_port
        self.data_port = data_port
        self.roll_socket = None
        self.data_socket = None

    def connect_to_server(self):
        """Connects to the Raspberry Pi server on both ports."""
        # Connect for roll angle data
        self.roll_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.roll_socket.connect((self.host, self.roll_port))
        
        # Connect for other data
        self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.data_socket.connect((self.host, self.data_port))

    def send_roll_angle_to_pi(self, data):
        """Sends roll angle data to the Raspberry Pi."""
        print(f"Sending roll angle: {data}")
        try:
            message = struct.pack('!cf', b'r', data)
            self.roll_socket.sendall(message)
        except BrokenPipeError:
            print("Connection for roll angle closed by the server.")

    def send_data(self, data):
        """Sends other data to the Raspberry Pi."""
        try:
            byte_data = data.encode('utf-8')
            self.data_socket.sendall(byte_data)
        except BrokenPipeError:
            print("Connection for other data closed by the server.")

    def close_connection(self):
        """Closes all connections to the server."""
        if self.roll_socket:
            self.roll_socket.close()
            self.roll_socket = None
        if self.data_socket:
            self.data_socket.close()
            self.data_socket = None

# Example usage:
# server = SocketServer()
# server.connect_to_server()
# server.send_roll_angle_to_pi(45.0)
# server.send_data("Some other data")
# server.close_connection()
