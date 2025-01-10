import socket
import time
import struct

# Configuration
SERVER_HOST = '10.250.16.32'  # Server IP address (replace with actual server IP)
SERVER_PORT = 3003       # Port that matches the server's port

def connect_to_server(retries=1, delay=2):
    """ Function to connect to the server with retry logic. """
    for attempt in range(retries):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((SERVER_HOST, SERVER_PORT))
            print(f"Connected to server on attempt {attempt + 1}")
            return sock
        except ConnectionRefusedError:
            print(f"Connection failed (attempt {attempt + 1}). Retrying in {delay} seconds...")
            time.sleep(delay)
    print(f"Failed to connect after {retries} attempts.")
    return None


def main():
    sock = connect_to_server()
    if sock:
        message = "Start"
        sock.sendall(message.encode('utf-8'))

        time.sleep(1)
        message = struct.pack('!cf', b'r', 2.145)
        sock.sendall(message)

        # Wait for 5 seconds
        time.sleep(5)
        message = 5
        sock.sendall(str(message).encode('utf-8'))

        time.sleep(1)
        message = "Stop"
        sock.sendall(message.encode('utf-8'))

        sock.close()

if __name__ == "__main__":
    main()