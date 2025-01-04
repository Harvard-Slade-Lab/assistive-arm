# Network socket server
import socket
import threading
import struct
# import time

# Configuration
HOST = '0.0.0.0'  # Listen on all available interfaces
PORT = 3003      # Arbitrary non-privileged port

class SocketServer:
    """Handles the socket connection and communication."""
    
    def __init__(self, host=HOST, port=PORT):
        self.host = host
        self.port = port
        self.collect_flag = False
        self.mode_flag = False
        self.repeat_flag = False
        self.kill_flag = False
        self.profile_name = None
        self.score = None
        self.roll_angle = None
        # self.score_receival_time = None
        self.score_tag = None
        self.stop_server = False
        self.conn = None
        self.server_thread = threading.Thread(target=self.start_server)
        self.server_thread.start()

    def start_server(self):
        """Start the socket server and accept connections."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            # Allow the socket to be reused immediately
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            server_socket.bind((self.host, self.port))
            server_socket.listen()
            print(f"Server listening on {self.host}:{self.port}")

            while not self.stop_server:
                try:
                    server_socket.settimeout(1.0)  # Set a timeout to check stop_server periodically
                    self.conn, addr = server_socket.accept()
                    with self.conn:
                        print(f"Connected by {addr}")
                        while not self.stop_server:
                            data = self.conn.recv(1024)
                            if not data:
                                break
                            # Quick check if it is roll angle data
                            if data[0:1] == b'r':  # If the first byte indicates roll angle
                                roll_angle = struct.unpack('!f', data[1:5])[0]
                                self.roll_angle = roll_angle
                            else:
                                data_decoded = data.decode('utf-8', errors='replace')
                                self.process_data(data_decoded)
                except socket.timeout:
                    # Timeout reached, loop back to check stop_server
                    continue


    def process_data(self, data):
        """Process received data to control the session state."""
        if data == "Start":
            self.collect_flag = True
        elif data == "Stop":
            self.collect_flag = False
        elif data == "Mode":
            self.mode_flag = True
        elif data == "Kill":
            print("\nClosing connection and exiting...")
            self.kill_flag = True
            self.stop()
        elif "Repeat" in data:
            parts = data.split("_")
            self.repeat_flag = True
            # This is needed to verify, that the repeat command is received for the current iteration
            self.score_tag = parts[1]
        elif "Profile" in data:
            self.profile_name = data.split(":")[1]
        elif "Score" in data:
            # self.score = float(data.split(":")[1])
            # self.score_receival_time = time.time()
            parts = data.split("_")
            self.score = float(parts[1])
            self.score_tag = parts[3]
            print(f"\nScore received: {self.score}", f"Tag: {self.score_tag}")

    def stop(self):
        """Stop the server and terminate the active session."""
        self.stop_server = True
        self.collect_flag = False
        
        # Notify the client before closing the connection
        if self.conn:
            try:
                self.conn.sendall(b"Server shutting down\n")
                self.conn.shutdown(socket.SHUT_RDWR)
            except (socket.error, OSError):
                print("Error notifying client or shutting down the connection.")
            finally:
                self.conn.close()
                self.conn = None
        
        # Stop the server thread
        if self.server_thread.is_alive():
            self.server_thread.join(timeout=5)
