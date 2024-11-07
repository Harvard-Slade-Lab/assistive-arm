# Network socket server
# server.py
import socket
import threading
import logging

# Configuration
HOST = '0.0.0.0'  # Listen on all available interfaces
PORT = 65432      # Arbitrary non-privileged port

# # Set up logging to file
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s [%(levelname)s] %(message)s',
#     handlers=[
#         logging.FileHandler("server.log"),
#         logging.StreamHandler()
#     ]
# )

# def handle_client(conn, addr):
#     logging.info(f'Connected by {addr}')
#     try:
#         while True:
#             data = conn.recv(1024)
#             if not data:
#                 break
#             # Log the received message
#             logging.info(f'Received from {addr}: {data.decode("utf-8", errors="replace")}')
            
#             # Optionally echo the data back to the client
#             conn.sendall(data)
#     except ConnectionResetError:
#         logging.warning(f'Connection reset by {addr}')
#     finally:
#         conn.close()
#         logging.info(f'Disconnected from {addr}')

# def start_server():
#     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#         s.bind((HOST, PORT))
#         s.listen()
#         logging.info(f'Server listening on {HOST}:{PORT}')
#         while True:
#             conn, addr = s.accept()
#             client_thread = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
#             client_thread.start()


def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, PORT))
        server_socket.listen()
        print(f"Server listening on {HOST}:{PORT}")

        while True:
            conn, addr = server_socket.accept()
            with conn:
                print(f"Connected by {addr}")
                while True:
                    data = conn.recv(1024)  # Buffer size of 1024 bytes
                    if not data:
                        break
                    # Process the received data (print, save, etc.)
                    print("Received data:", data.decode('utf-8'))



if __name__ == '__main__':
    start_server()
