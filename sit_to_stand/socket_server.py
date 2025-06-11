import socket
import threading
import struct
import numpy as np
import time

# Configuration
HOST = '0.0.0.0'  # Listen on all available interfaces
PORT_ROLL_ANGLE = 3001  # Port for roll angles
PORT_COMMANDS = 3002  # Port for commands

# ---------------------------------------------------------------------
# This is another socket server that sends data from Raspberry Pi to MacBook
# Socket server for sending Start and Stop data to the MacBook
import socket
import pickle
# Socket setup (do this once)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
mac_ip = '10.250.76.72'  # e.g., '192.168.1.100'
mac_port = 9999

def send_data(data):
    """Send any serializable object to Mac"""
    sock.sendto(pickle.dumps(data), (mac_ip, mac_port))
    time.sleep(0.01)  # small pause to prevent flooding
# ---------------------------------------------------------------------

class SocketServer:
    """Handles both roll angle and command servers."""

    def __init__(self, host=HOST, port_commands=PORT_COMMANDS, port_roll_angle=PORT_ROLL_ANGLE):
        self.host = host
        self.port_commands = port_commands
        self.port_roll_angle = port_roll_angle

        # Shared state variables
        self.roll_angle = None
        self.roll_lock = threading.Lock()
        self.collect_flag = False
        self.mode_flag = False
        self.repeat_flag = False
        self.kill_flag = False
        self.profile_name = None
        self.score = None
        self.score_tag = None

        # Control flags
        self.stop_server = False

        # Threads for each server
        self.command_thread = threading.Thread(target=self.start_command_server, daemon=True)
        self.roll_angle_thread = threading.Thread(target=self.start_roll_angle_server, daemon=True)

        # Start both servers
        self.command_thread.start()
        self.roll_angle_thread.start()

    def start_command_server(self):
        """Start the command socket server."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.host, self.port_commands))
            server_socket.listen()
            print(f"Command Server listening on {self.host}:{self.port_commands}")

            while not self.stop_server:
                try:
                    server_socket.settimeout(1.0)
                    conn, addr = server_socket.accept()
                    with conn:
                        print(f"Command Server connected by {addr}")
                        while not self.stop_server:
                            data = conn.recv(1024)
                            print(f"Received data: {data}")

                            # Forward the raw command to the Mac via UDP
                            send_data(data)

                            if not data:
                                break
                            self.process_command_data(data.decode('utf-8', errors='replace').strip())
                except socket.timeout:
                    continue

    def start_roll_angle_server(self):
        """Start the roll angle socket server."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.host, self.port_roll_angle))
            server_socket.listen()
            print(f"Roll Angle Server listening on {self.host}:{self.port_roll_angle}")

            while not self.stop_server:
                try:
                    server_socket.settimeout(1.0)
                    conn, addr = server_socket.accept()
                    with conn:
                        print(f"Roll Angle Server connected by {addr}")
                        while not self.stop_server:
                            data = conn.recv(8)
                            if not data:
                                break
                            if data[0:1] == b'r':  # If the first byte indicates roll angle
                                roll_angle = struct.unpack('!f', data[1:5])[0]
                                with self.roll_lock:
                                    self.roll_angle = np.degrees(roll_angle)
                except socket.timeout:
                    continue

    def process_command_data(self, data):
        """Process received command data."""
        if "Start" in data:
            self.collect_flag = True
        elif "Stop" in data:
            self.collect_flag = False
        elif "Mode" in data:
            self.mode_flag = True
        elif "Kill" in data:
            print("\nReceived Kill Command. Shutting down...")
            self.kill_flag = True
            self.stop()
        elif "Repeat" in data:
            parts = data.split("_")
            self.repeat_flag = True
            self.score_tag = parts[1]
        elif "Profile" in data:
            self.profile_name = data.split(":")[1]
        elif "Score" in data:
            parts = data.split("_")
            self.score = float(parts[1])
            self.score_tag = parts[3]
            print(f"Score received: {self.score}, Tag: {self.score_tag}")

    def stop(self):
        """Stop both servers and their threads."""
        self.stop_server = True
        if self.command_thread.is_alive():
            self.command_thread.join(timeout=5)
        if self.roll_angle_thread.is_alive():
            self.roll_angle_thread.join(timeout=5)



###################Separate classes###############
# import socket
# import threading
# import struct

# # Configuration
# HOST = '0.0.0.0'  # Listen on all available interfaces
# PORT_ROLL_ANGLE = 3003  # Port for roll angles
# PORT_COMMANDS = 3004  # Port for commands

# class RollAngleSocketServer:
#     """Handles roll angle data."""
    
#     def __init__(self, host=HOST, port=PORT_ROLL_ANGLE):
#         self.host = host
#         self.port = port
#         self.roll_angle = None
#         self.stop_server = False
#         self.server_thread = threading.Thread(target=self.start_server)
#         self.server_thread.start()

#     def start_server(self):
#         """Start the roll angle socket server."""
#         with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
#             server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#             server_socket.bind((self.host, self.port))
#             server_socket.listen()
#             print(f"Roll Angle Server listening on {self.host}:{self.port}")

#             while not self.stop_server:
#                 try:
#                     server_socket.settimeout(1.0)  # Set a timeout to check stop_server periodically
#                     conn, addr = server_socket.accept()
#                     with conn:
#                         print(f"Roll Angle Server connected by {addr}")
#                         while not self.stop_server:
#                             data = conn.recv(1024)
#                             if not data:
#                                 break
#                             if data[0:1] == b'r':  # If the first byte indicates roll angle
#                                 roll_angle = struct.unpack('!f', data[1:5])[0]
#                                 with threading.Lock():
#                                     self.roll_angle = roll_angle
#                 except socket.timeout:
#                     continue

#     def stop(self):
#         """Stop the roll angle server."""
#         self.stop_server = True
#         if self.server_thread.is_alive():
#             self.server_thread.join(timeout=5)


# class CommandSocketServer:
#     """Handles all other commands."""
    
#     def __init__(self, host=HOST, port=PORT_COMMANDS):
#         self.host = host
#         self.port = port
#         self.collect_flag = False
#         self.mode_flag = False
#         self.repeat_flag = False
#         self.kill_flag = False
#         self.profile_name = None
#         self.score = None
#         self.score_tag = None
#         self.stop_server = False
#         self.conn = None
#         self.server_thread = threading.Thread(target=self.start_server)
#         self.server_thread.start()

#     def start_server(self):
#         """Start the command socket server."""
#         with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
#             server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#             server_socket.bind((self.host, self.port))
#             server_socket.listen()
#             print(f"Command Server listening on {self.host}:{self.port}")

#             while not self.stop_server:
#                 try:
#                     server_socket.settimeout(1.0)  # Set a timeout to check stop_server periodically
#                     self.conn, addr = server_socket.accept()
#                     with self.conn:
#                         print(f"Command Server connected by {addr}")
#                         while not self.stop_server:
#                             data = self.conn.recv(1024)
#                             if not data:
#                                 break
#                             data_decoded = data.decode('utf-8', errors='replace').strip()
#                             self.process_data(data_decoded)
#                 except socket.timeout:
#                     continue

#     def process_data(self, data):
#         """Process received data to control the session state."""
#         if "Start" in data:
#             self.collect_flag = True
#         elif "Stop" in data:
#             self.collect_flag = False
#         elif "Mode" in data:
#             self.mode_flag = True
#         elif "Kill" in data:
#             print("\nClosing connection and exiting...")
#             self.kill_flag = True
#             self.stop()
#         elif "Repeat" in data:
#             parts = data.split("_")
#             self.repeat_flag = True
#             self.score_tag = parts[1]
#         elif "Profile" in data:
#             self.profile_name = data.split(":")[1]
#         elif "Score" in data:
#             parts = data.split("_")
#             self.score = float(parts[1])
#             self.score_tag = parts[3]
#             print(f"\nScore received: {self.score}", f"Tag: {self.score_tag}")

#     def stop(self):
#         """Stop the command server."""
#         self.stop_server = True
#         if self.conn:
#             try:
#                 self.conn.sendall(b"Server shutting down\n")
#                 self.conn.shutdown(socket.SHUT_RDWR)
#             except (socket.error, OSError):
#                 print("Error notifying client or shutting down the connection.")
#             finally:
#                 self.conn.close()
#                 self.conn = None
        
#         if self.server_thread.is_alive():
#             self.server_thread.join(timeout=5)
