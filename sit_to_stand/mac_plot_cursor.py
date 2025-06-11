import matplotlib.pyplot as plt
import numpy as np
import socket
import pickle
import threading
import pandas as pd
import queue
import time

# Message queue for thread-safe communication
data_queue = queue.Queue()

# Setup UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", 9999))

# Plot placeholders
profiles = None
cursor_fx = None
cursor_fy = None
ax = None

def socket_thread():
    while True:
        data, _ = sock.recvfrom(65536)
        try:
            payload = pickle.loads(data)
            data_queue.put(payload)
        except Exception as e:
            print("Failed to parse data:", e)

# Start the socket listener
threading.Thread(target=socket_thread, daemon=True).start()

# Main thread loop
fig, ax = plt.subplots()
plt.ion()
plt.show()

while True:
    try:
        while not data_queue.empty():
            payload = data_queue.get()

            if isinstance(payload, dict) and payload.get("type") == "profile":
                profiles = pd.DataFrame(payload["data"])
                ax.clear()
                ax.plot(profiles["force_X"].values, label="Force X")
                ax.plot(profiles["force_Y"].values, label="Force Y")
                cursor_fx, = ax.plot([], [], 'ro', label='Cursor X')
                cursor_fy, = ax.plot([], [], 'bo', label='Cursor Y')
                ax.set_title("Real-time Force Visualization")
                ax.set_xlabel("Profile Index")
                ax.set_ylabel("Force")
                ax.legend()

            elif isinstance(payload, dict) and payload.get("type") == "cursor" and profiles is not None:
                idx = payload["index"]
                fx, fy = payload["force"]
                cursor_fx.set_data([idx], [fx])
                cursor_fy.set_data([idx], [fy])

        # Always refresh (even if no new data)
        if profiles is not None:
            ax.relim()
            ax.autoscale_view()
            plt.pause(0.05)

        time.sleep(0.01)

    except KeyboardInterrupt:
        break
