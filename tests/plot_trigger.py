import RPi.GPIO as GPIO
import time
import matplotlib.pyplot as plt

def collect_and_plot_signal(pin, duration=10, interval=0.01, filename="gpio_signal_plot.png"):
    """
    Collects the signal from the specified GPIO pin for a given duration and saves the plot to a file.

    :param pin: The GPIO pin number (BCM mode) to monitor.
    :param duration: The total time in seconds to monitor the pin.
    :param interval: Time interval in seconds between each signal check.
    :param filename: The name of the file to save the plot.
    """
    # Use Broadcom SOC Pin numbers
    GPIO.setmode(GPIO.BCM)
    
    # Set up the specified GPIO pin as an input
    GPIO.setup(pin, GPIO.IN)
    
    signal_data = []  # To store signal values (0 or 1)
    timestamps = []   # To store the corresponding timestamps
    
    start_time = time.time()
    
    try:
        print(f"Collecting signal data from GPIO pin {pin} for {duration} seconds...")
        while (time.time() - start_time) < duration:
            # Read the signal and timestamp
            signal = GPIO.input(pin)
            current_time = time.time() - start_time
            
            # Append the signal and timestamp to lists
            signal_data.append(signal)
            timestamps.append(current_time)
            
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print("Data collection interrupted by the user.")
    
    finally:
        # Clean up GPIO settings
        GPIO.cleanup()
        print(f"Finished collecting data. Now saving the plot as '{filename}'...")

        # Plotting the collected signal data
        plt.figure(figsize=(10, 4))
        plt.plot(timestamps, signal_data, label="GPIO Signal", drawstyle="steps-post")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Signal (HIGH=1, LOW=0)")
        plt.title(f"Signal from GPIO Pin {pin} Over Time")
        plt.grid(True)
        plt.legend()

        # Save the plot to a file instead of showing it
        plt.savefig(filename)
        print(f"Plot saved as {filename} in the current directory.")

# Example usage: Monitor GPIO pin 17 for 10 seconds with an interval of 0.1 seconds, and save the plot
collect_and_plot_signal(pin=17, duration=10, interval=0.1, filename="gpio_signal_plot.png")
