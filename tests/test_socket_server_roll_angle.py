import socket
import threading
import time
import struct
from queue import Queue
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Configuration
HOST = '0.0.0.0'  # Listen on all available interfaces
PORT = 3003       # Arbitrary non-privileged port

control_flag = False
kill_flag = False
roll_angle = None
roll_angle_queue = Queue()  # Thread-safe queue for storing roll angles

def start_server():
    """ Start a TCP socket server and accept a connection. """
    global control_flag
    global kill_flag
    global roll_angle

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((HOST, PORT))
        server_socket.listen()
        print(f"Server listening on {HOST}:{PORT}")

        conn, addr = server_socket.accept()
        with conn:
            print(f"Connected by {addr}")
            while True:
                data = conn.recv(1024)  # Buffer size of 1024 bytes
                if not data:
                    break

                if data[0:1] == b'r':  # If the first byte indicates roll angle
                    roll_angle = struct.unpack('!f', data[1:5])[0]
                    print(f"\nRoll angle: {roll_angle}")
                else:
                    data_decoded = data.decode('utf-8', errors='replace').strip()
                    if data_decoded == "Start":
                        print("Start recording")
                        control_flag = True
                    elif data_decoded == "Stop":
                        print("Stop recording")
                        control_flag = False
                    elif data_decoded == "Kill":
                        kill_flag = True
                        break

def plot_roll_angle():
    """ Live plot of roll angles. """
    fig, ax = plt.subplots()
    x_data, y_data = [], []

    def update(frame):
        while not roll_angle_queue.empty():
            roll_angle = roll_angle_queue.get()
            x_data.append(len(x_data))  # Use the index as x-axis data
            y_data.append(roll_angle)
        ax.clear()
        ax.plot(x_data, y_data, label='Roll Angle')
        ax.set_title("Live Roll Angle Plot")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Roll Angle")
        ax.legend()

    ani = FuncAnimation(fig, update, interval=100)  # Update every 100ms
    plt.show()

def main():
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    while not control_flag:
        time.sleep(0.1) # Wait for the server to start

    while not kill_flag:
        time.sleep(1)
    # plot_roll_angle()  # Start live plotting

if __name__ == "__main__":
    main()

