import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

def process_signals(gyro, acc, orientation):
    """Process and harmonize sensor signals with automatic interpolation and visualization."""
    
    def interpolate_signal(signal, target_length):
        old_t = np.arange(len(signal))
        new_t = np.linspace(0, len(signal) - 1, target_length)
        interp_signal = np.zeros((target_length, signal.shape[1]))
        for i in range(signal.shape[1]):
            f = interpolate.interp1d(old_t, signal[:, i], kind='cubic', fill_value='extrapolate')
            interp_signal[:, i] = f(new_t)
        return interp_signal

    lengths = [len(gyro), len(acc), len(orientation)]
    target_len = max(lengths)

    processed = [
        interpolate_signal(sensor, target_len) if len(sensor) != target_len else sensor
        for sensor in [gyro, acc, orientation]
    ]

    # Visualization
    fig, axs = plt.subplots(3, 1, figsize=(15, 15))
    titles = ['Gyroscope', 'Accelerometer', 'Orientation']
    components = [
        (['X', 'Y', 'Z'], ['Gyro X', 'Gyro Y', 'Gyro Z']),
        (['X', 'Y', 'Z'], ['Acc X', 'Acc Y', 'Acc Z']),
        (['Roll', 'Pitch', 'Yaw'], ['Roll', 'Pitch', 'Yaw'])
    ]

    for i, (sensor, title, (axes, labels)) in enumerate(zip(processed, titles, components)):
        for j, (axis, label) in enumerate(zip(axes, labels)):
            axs[i].plot(sensor[:, j], label=label)
        axs[i].set_title(f'{title} (Interpolated, {len(sensor)} points)')
        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout()
    plt.show()

    return processed[0], processed[1], processed[2]
