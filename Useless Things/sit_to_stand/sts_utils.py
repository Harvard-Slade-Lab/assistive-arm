from datetime import datetime
import time
import signal
import RPi.GPIO as GPIO

from typing import Literal
from sit_to_stand.socket_server import SocketServer

interrupt_flag = False

def handle_interrupt(signum, frame):
    print("\nKeyboard interrupt received. Changing mode_flag.")
    global interrupt_flag
    interrupt_flag = True
            
def await_trigger_signal(mode: Literal["TRIGGER", "ENTER", "SOCKET"], socket_server: SocketServer=None):
    """ Wait for trigger signal OR Enter to start Trial """
    signal.signal(signal.SIGINT, handle_interrupt)

    if mode == "ENTER": 
        input("\nPress Enter to start Trial...")

    if mode == "TRIGGER":
        print("\nPress trigger to start Trial")
        while not GPIO.input(17):
            pass

    elif mode == "SOCKET" and socket_server:
        print("\nWaiting for socket start signal to start Trial")
        # Make sure we can exit if connection is lost, kind of an ugly solution, also tried demon thread with shared flag
        # as well as an exception handling approach, but this seems to be the most reliable.
        while not socket_server.collect_flag:
            if socket_server.kill_flag:
                socket_server.stop()
                break
            if socket_server.mode_flag:
                break
            time.sleep(0.1)

            if interrupt_flag:
                # This is hacky but makes sense to exit to where we want to
                socket_server.mode_flag = True

def countdown(duration: int=3):
    for i in range(duration, 0, -1):
        print(f"Recording in {i} seconds...", end="\r")
        time.sleep(1)
    # print("\nGO!")
