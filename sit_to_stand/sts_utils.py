from datetime import datetime
import time
import RPi.GPIO as GPIO

from typing import Literal
from sit_to_stand.socket_server import SocketServer

            
def await_trigger_signal(mode: Literal["TRIGGER", "ENTER", "SOCKET"], server: SocketServer=None):
    """ Wait for trigger signal OR Enter to start recording """
    if mode == "ENTER": 
        input("\nPress Enter to start recording...")

    if mode == "TRIGGER":
        print("\nPress trigger to start recording P_EE...")
        while not GPIO.input(17):
            pass

    elif mode == "SOCKET" and server:
        print("\nWaiting for socket data to start recording")
        while not server.collect_flag:
            time.sleep(0.1)

def countdown(duration: int=3):
    for i in range(duration, 0, -1):
        print(f"Recording in {i} seconds...", end="\r")
        time.sleep(1)
    print("\nGO!")
