import RPi.GPIO as GPIO
import time

# Use Broadcom SOC Pin numbers
GPIO.setmode(GPIO.BCM)

# Set up GPIO pin 17 as an input
GPIO.setup(17, GPIO.IN)

try:
    print("Waiting for the first pulse signal on GPIO pin 17...")

    # First loop: Wait for the first pulse signal to turn on
    while not GPIO.input(17):
        pass
    print("Signal turned on.")

    print("In second loop, waiting for the signal to turn off.")

    # Second loop: Wait for the signal to turn off
    while True:
        if not GPIO.input(17):  # Detects if signal turns off (low signal)
            print("Signal turned off, exiting.")
            break
        time.sleep(0.1)  # Short delay

except KeyboardInterrupt:
    print("Script interrupted by the user")

finally:
    # Clean up GPIO settings before exiting
    GPIO.cleanup()

print("Finished monitoring GPIO pin 17.")
