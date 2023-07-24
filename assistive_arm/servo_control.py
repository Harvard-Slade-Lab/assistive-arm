import numpy as np

from time import sleep
from abc import ABC, abstractmethod, abstractproperty

import RPi.GPIO as GPIO

class ServoBase(ABC):
    """Base class for future servos"""
    
    @abstractproperty
    def angle_range(self) -> np.array:
        """Range of angles that the servo can move to"""
        pass

    @abstractproperty
    def _gpio_pin(self) -> int:
        """GPIO pin that the servo is connected to"""
        pass
    
    @abstractproperty
    def _servo_min(self) -> int:
        """Minimum duty cycle for the servo"""
        pass
    
    @abstractproperty
    def _servo_max(self) -> int:
        """Maximum duty cycle for the servo"""
        pass
    
    @abstractproperty
    def _pwm_cycle(self) -> int:
        """PWM cycle for the servo"""
        pass

    @abstractmethod
    def set_angle(self, angle: float) -> None:
        """Set the angle of the servo

        Args:
            angle (float): Angle in radians
        """
        pass

    @abstractmethod
    def start(self) -> None:
        """Start the servo"""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop the servo"""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup the servo ports"""
        pass

class ServoControl:
    def __init__(self, pin: int):
        # TODO Make sure that there is no other servo connected at this pin
        self._gpio_pin = pin
        self._servo_min = 2
        self._servo_max = 10
        self._pwm_cycle = 50  # Hz
        self.angle_range = np.array([-90, 90])

        GPIO.setmode(GPIO.BOARD)  # Set GPIO numbering mode
        GPIO.setup(self._gpio_pin, GPIO.OUT)
        self.pwm = GPIO.PWM(self._gpio_pin, self._pwm_cycle)
        self.start()  # Start PWM running, set servo to 0 degree.

    def set_angle(self, angle: float) -> None:
        # Angle is between -90 and 90, convert to 0 and 180
        clamped_angle = np.clip(angle, self.angle_range.min, self.angle_range.max)
        degree = 90 - clamped_angle
        self.pwm.ChangeDutyCycle(self._servo_min + self._servo_max * degree / 180)
        sleep(0.1)

    def start(self) -> None:
        self.pwm.start(0)

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
