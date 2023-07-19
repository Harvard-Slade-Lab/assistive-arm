import RPi.GPIO as GPIO
import time

# Servo motor is connected to GPIO pin 11.
servo_pin = 11

# Degrees to PWM duty cycle mapping
degree_45 = 7
degree_90 = 12

# Set GPIO numbering mode
GPIO.setmode(GPIO.BOARD)

# Set pin 11 as an output, and define as servo
GPIO.setup(servo_pin, GPIO.OUT)
servo = GPIO.PWM(servo_pin, 50) # 50 Hz PWM signal

# Start PWM running (set servo to mid position)
servo.start(0)

try:
    while True:
        # Turn servo to 45 degrees
        servo.ChangeDutyCycle(degree_45)
        time.sleep(2)
        
        # Turn servo to 90 degrees
        servo.ChangeDutyCycle(degree_90)
        time.sleep(2)

except KeyboardInterrupt:
    print("Ctrl-C Pressed: Exiting Program")

finally:
    # Stop the PWM
    servo.stop()
    # Reset GPIO settings
    GPIO.cleanup()