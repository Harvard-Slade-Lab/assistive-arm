import socket
import threading
import time
import struct
import csv
from queue import Queue

# Both queues and locks are fast enough according to the following test
# Appearantly the lock is the bester option for this use case

# Configuration
HOST = '0.0.0.0'  # Listen on all available interfaces
ROLL_PORT = 3003   # Port for roll angle server
DATA_PORT = 3004   # Port for other data

# Shared variable and lock
roll_angle = None
roll_angle_lock = threading.Lock()

control_flag = False
kill_flag = False
roll_angle_queue = Queue()


def roll_angle_server():
    """ Socket server to receive roll angle data. """
    global roll_angle

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((HOST, ROLL_PORT))
        server_socket.listen()
        print(f"Roll angle server listening on {HOST}:{ROLL_PORT}")

        conn, addr = server_socket.accept()
        with conn:
            print(f"Connected by {addr} (Roll angle)")
            # start_time = time.time()  
            # count = 0 
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                if data[0:1] == b'r':  # If the first byte indicates roll angle
                    new_roll_angle = struct.unpack('!f', data[1:5])[0]
                    # Use queue to update the shared variable
                    roll_angle_queue.put(new_roll_angle)

                    # Use lock to update the shared variable
                    # with roll_angle_lock:
                    #     roll_angle = new_roll_angle

                    # Use queue

                    # count += 1
                    # if time.time() - start_time > 10:
                    #     print(f"Received at: {count/10}Hz")
                    #     count = 0
                    #     start_time = time.time()

                    # print(f"Roll Angle: {new_roll_angle}")


def data_server():
    """ Socket server to receive other data. """
    global control_flag, kill_flag

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((HOST, DATA_PORT))
        server_socket.listen()
        print(f"Data server listening on {HOST}:{DATA_PORT}")

        conn, addr = server_socket.accept()
        with conn:
            print(f"Connected by {addr} (Data)")
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                data_decoded = data.decode('utf-8', errors='replace').strip()
                if data_decoded == "Start":
                    print("Start recording")
                    control_flag = True
                elif data_decoded == "Stop":
                    print("Stop recording")
                    control_flag = False
                elif data_decoded == "Kill":
                    print("Kill signal received")
                    kill_flag = True
                    break


def main():
    # Start roll angle server in its own thread
    roll_thread = threading.Thread(target=roll_angle_server, daemon=True)
    roll_thread.start()

    # Start data server in its own thread
    data_thread = threading.Thread(target=data_server, daemon=True)
    data_thread.start()

    # with open("roll_angle_log.csv", mode="w", newline="") as csvfile:
    #     csv_writer = csv.writer(csvfile)
    #     csv_writer.writerow(["Timestamp", "Roll Angle"])  # Header row

    try:
        while not kill_flag:
            # if not roll_angle_queue.empty():
            #     latest_roll_angle = roll_angle_queue.get()
            # Access the latest roll angle using the lock

            # count = 0
            # start_time = time.time()
            while control_flag:
                if not roll_angle_queue.empty():
                    latest_roll_angle = roll_angle_queue.get()
                    print(f"Roll Angle: {latest_roll_angle}")

                # with roll_angle_lock:
                #     print(f"Roll Angle: {roll_angle}")
                    # if roll_angle is not None and control_flag:
                    #     csv_writer.writerow([time.time(), roll_angle])

                    # count += 1
                    # if time.time() - start_time > 10:
                    #     print(f"Received at: {count/10}Hz")
                    #     count = 0
                    #     start_time = time.time()

                time.sleep(0.01)  # Polling interval
    except KeyboardInterrupt:
            print("Exiting...")

    print("Shutting down servers...")


if __name__ == "__main__":
    main()
