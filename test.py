import RPi.GPIO as GPIO
import time

servo_pin = 11             #Servo motor is connected to GPIO pin 11.

GPIO.setmode(GPIO.BOARD)  #Set GPIO numbering mode

GPIO.setup(servo_pin, GPIO.OUT)

pwm_frequency= 50
servo = GPIO.PWM(servo_pin, pwm_frequency) #50 Hz PWM signal

servo.start(5)     #Start PWM running, set servo to 0 degree.

servo_min = 2
servo_max = 10

try:
    while True:
        #Sweep Servo from 0 to 180 and vice versa
        angle = 0
        servo.ChangeDutyCycle(servo_min + servo_max * angle / 180)  # changing duty cycle
        time.sleep(1)

        angle = 45
        servo.ChangeDutyCycle(servo_min + servo_max * angle / 180)  # changing duty cycle
        time.sleep(1)
        
        angle = 90
        servo.ChangeDutyCycle(servo_min + servo_max * angle / 180)  # changing duty cycle
        time.sleep(1)

        angle = 180
        servo.ChangeDutyCycle(servo_min + servo_max * angle / 180)  # changing duty cycle
        time.sleep(1)
        


except KeyboardInterrupt:
    servo.stop()
    GPIO.cleanup()