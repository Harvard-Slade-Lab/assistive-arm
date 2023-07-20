import numpy as np

from collections import namedtuple

import RPi.GPIO as GPIO

servo_range = namedtuple("servo_range", ["min", "max"])


class ServoControl:
    def __init__(self, pin: int):
        # TODO Make sure that there is no other servo connected at this pin
        self._gpio_pin = pin
        self._servo_min = 2
        self._servo_max = 10
        self._pwm_cycle = 50  # Hz
        self.angle_range = servo_range(0, 180)

        GPIO.setmode(GPIO.BOARD)  # Set GPIO numbering mode
        GPIO.setup(self._gpio_pin, GPIO.OUT)
        self.pwm = GPIO.PWM(self._gpio_pin, self._pwm_cycle)

        self.pwm.start(2)  # Start PWM running, set servo to 0 degree.

    def set_angle(self, angle: int) -> None:
        clamp_angle = np.clip(angle, self.angle_range.min, self.angle_range.max)
        self.pwm.ChangeDutyCycle(self._servo_min + self._servo_max * clamp_angle / 180)

    def stop(self) -> None:
        self.pwm.stop()

    # TODO Move this to somewhere else in the program
    def cleanup(self) -> None:
        response = input(
            "This will cleanup all the GPIO ports. Do you want to continue? (Y/N): "
        )
        if response.lower() != "n":
            GPIO.cleanup()
        else:
            print("Did not clean GPIO ports.")
