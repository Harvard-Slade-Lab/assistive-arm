# Network socket server
import socket
import threading

# Configuration
HOST = '0.0.0.0'  # Listen on all available interfaces
PORT = 3003      # Arbitrary non-privileged port

class SocketServer:
    """Handles the socket connection and communication."""
    
    def __init__(self, host=HOST, port=PORT):
        self.host = host
        self.port = port
        self.collect_flag = False
        self.profile_name = None
        self.score = None
        self.stop_server = False
        self.conn = None
        self.server_thread = threading.Thread(target=self.start_server)
        self.server_thread.start()

    def start_server(self):
        """Start the socket server and accept connections."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
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
        elif data == "Kill":
            print("\nClosing connection and exiting...")
            self.stop()
            # if self.conn:
            #     self.conn.close()
            # self.server_thread.join()
        elif "Profile" in data:
            self.profile_name = data.split(":")[1]
        elif "Score" in data:
            self.score = float(data.split(":")[1])

    def stop(self):
        """Stop the server."""
        self.stop_server = True
        self.collect_flag = False
        if self.conn:
            self.conn.close()
        if self.server_thread.is_alive():
            self.server_thread.join()
