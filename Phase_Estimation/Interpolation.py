import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

def interpolate_sensor_signals(gyro, acc, orientation, frequencies=None):
    """
    Interpolate sensor signals to ensure they have the same length.
    
    Parameters:
    -----------
    gyro : pandas.DataFrame
        Gyroscope data after segmentation
    acc : pandas.DataFrame
        Accelerometer data after segmentation
    orientation : pandas.DataFrame
        Orientation data after segmentation
    frequencies : list, optional
        Sampling frequencies [gyro_freq, acc_freq, orientation_freq]
    
    Returns:
    --------
    gyro_interp, acc_interp, orientation_interp : pandas.DataFrame
        Interpolated signals with the same length
    time_interp : numpy.ndarray
        Time vector for the interpolated signals
    """
    # Get lengths of each signal
    len_gyro = len(gyro)
    len_acc = len(acc)
    len_orientation = len(orientation)
    
    print(f"Original signal lengths: Gyro={len_gyro}, Acc={len_acc}, Orientation={len_orientation}")
    
    # Create time vectors based on frequencies if provided
    if frequencies is not None:
        time_gyro = np.arange(len_gyro) / frequencies[0]
        time_acc = np.arange(len_acc) / frequencies[1]
        time_orientation = np.arange(len_orientation) / frequencies[2]
    else:
        # Normalized time if frequencies not provided
        time_gyro = np.linspace(0, 1, len_gyro)
        time_acc = np.linspace(0, 1, len_acc)
        time_orientation = np.linspace(0, 1, len_orientation)
    
    # Check if all signals have the same length
    if len_gyro == len_acc == len_orientation:
        print("All signals have the same length. No interpolation needed.")
        if frequencies is not None:
            time_interp = time_gyro
        else:
            time_interp = np.linspace(0, 1, len_gyro)
        return gyro, acc, orientation, time_interp
    
    # Find the target length (minimum of the three)
    target_length = min(len_gyro, len_acc, len_orientation)
    print(f"Target length for interpolation: {target_length}")
    
    # Determine time vector for interpolated signals
    if len_gyro == target_length:
        time_interp = time_gyro
    elif len_acc == target_length:
        time_interp = time_acc
    else:
        time_interp = time_orientation
    
    # Initialize interpolated dataframes
    gyro_interp = gyro.copy() if len_gyro == target_length else pd.DataFrame(index=range(target_length))
    acc_interp = acc.copy() if len_acc == target_length else pd.DataFrame(index=range(target_length))
    orientation_interp = orientation.copy() if len_orientation == target_length else pd.DataFrame(index=range(target_length))
    
    # Interpolate gyro if needed
    if len_gyro != target_length:
        for col in gyro.columns:
            f = interpolate.interp1d(time_gyro, gyro[col], kind='linear', fill_value='extrapolate')
            gyro_interp[col] = f(time_interp)
        print(f"Interpolated gyroscope data from {len_gyro} to {target_length} samples")
    
    # Interpolate acc if needed
    if len_acc != target_length:
        for col in acc.columns:
            f = interpolate.interp1d(time_acc, acc[col], kind='linear', fill_value='extrapolate')
            acc_interp[col] = f(time_interp)
        print(f"Interpolated accelerometer data from {len_acc} to {target_length} samples")
    
    # Interpolate orientation if needed
    if len_orientation != target_length:
        for col in orientation.columns:
            f = interpolate.interp1d(time_orientation, orientation[col], kind='linear', fill_value='extrapolate')
            orientation_interp[col] = f(time_interp)
        print(f"Interpolated orientation data from {len_orientation} to {target_length} samples")
    
    return gyro_interp, acc_interp, orientation_interp, time_interp

def plot_signal_lengths(gyro, acc, orientation, gyro_interp, acc_interp, orientation_interp, frequencies=None):
    """
    Visualize the lengths of original and interpolated signals.
    """
    signal_types = ['Gyroscope', 'Accelerometer', 'Orientation']
    original_lengths = [len(gyro), len(acc), len(orientation)]
    interp_lengths = [len(gyro_interp), len(acc_interp), len(orientation_interp)]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.arange(len(signal_types))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, original_lengths, width, label='Original')
    bars2 = ax.bar(x + width/2, interp_lengths, width, label='Interpolated')
    
    # Add frequency and length labels
    if frequencies is not None:
        for i, (freq, bar) in enumerate(zip(frequencies, bars1)):
            ax.text(i - width/2, bar.get_height() + 5, f"{bar.get_height()} ({freq} Hz)", 
                   ha='center', va='bottom')
    else:
        for i, bar in enumerate(bars1):
            ax.text(i - width/2, bar.get_height() + 5, str(bar.get_height()), 
                   ha='center', va='bottom')
    
    for i, bar in enumerate(bars2):
        ax.text(i + width/2, bar.get_height() + 5, str(bar.get_height()), 
               ha='center', va='bottom')
    
    ax.set_xticks(x)
    ax.set_xticklabels(signal_types)
    ax.set_ylabel('Number of Samples')
    ax.set_title('Signal Lengths Before and After Interpolation')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    return fig

def visualize_interpolation(original, interpolated, signal_type, time_original=None, time_interp=None):
    """
    Visualize original and interpolated signals.
    """
    if len(original) == len(interpolated) and np.array_equal(original.values, interpolated.values):
        return None  # No visualization if signals are identical
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{signal_type} - Original vs Interpolated")
    
    # Create x-axes for plotting
    x_original = time_original if time_original is not None else np.arange(len(original))
    x_interp = time_interp if time_interp is not None else np.arange(len(interpolated))
    
    # Plot original data
    for col in original.columns:
        ax1.plot(x_original, original[col], label=col)
    
    unit = "seconds" if time_original is not None and np.max(time_original) > 1 else "samples"
    ax1.set_title(f"Original ({len(original)} samples)")
    ax1.set_xlabel(f"Time [{unit}]")
    ax1.set_ylabel("Value")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot interpolated data
    for col in interpolated.columns:
        ax2.plot(x_interp, interpolated[col], label=col)
    
    unit = "seconds" if time_interp is not None and np.max(time_interp) > 1 else "samples"
    ax2.set_title(f"Interpolated ({len(interpolated)} samples)")
    ax2.set_xlabel(f"Time [{unit}]")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig

def compare_signal_segments(original, interpolated, signal_type, time_original=None, time_interp=None, segment_size=100):
    """
    Compare specific segments of original and interpolated signals.
    """
    if len(original) == len(interpolated) and np.array_equal(original.values, interpolated.values):
        return None  # No comparison needed if signals are identical
    
    # Create x-axes for plotting
    x_original = time_original if time_original is not None else np.arange(len(original))
    x_interp = time_interp if time_interp is not None else np.arange(len(interpolated))
    
    # Determine segment for visualization
    mid_point_orig = len(original) // 2
    start_orig = max(0, mid_point_orig - segment_size // 2)
    end_orig = min(len(original), mid_point_orig + segment_size // 2)
    
    # Find corresponding segment in interpolated signal
    if time_original is not None and time_interp is not None:
        start_time = x_original[start_orig]
        end_time = x_original[end_orig-1]
        start_interp = np.argmin(np.abs(x_interp - start_time))
        end_interp = np.argmin(np.abs(x_interp - end_time)) + 1
    else:
        # Use proportional scaling
        scale_factor = len(interpolated) / len(original)
        start_interp = int(start_orig * scale_factor)
        end_interp = int(end_orig * scale_factor)
    
    # Create figure with subplots
    num_cols = len(original.columns)
    fig, axs = plt.subplots(num_cols, 1, figsize=(10, 2*num_cols), sharex=True)
    if num_cols == 1:
        axs = [axs]
    
    fig.suptitle(f"{signal_type} - Detailed Segment Comparison")
    
    for i, col in enumerate(original.columns):
        axs[i].plot(x_original[start_orig:end_orig], 
                  original[col].iloc[start_orig:end_orig], 
                  'b-', label='Original')
        
        axs[i].plot(x_interp[start_interp:end_interp], 
                  interpolated[col].iloc[start_interp:end_interp], 
                  'r--', label='Interpolated')
        
        axs[i].set_title(f"{col}")
        unit = "seconds" if time_original is not None and np.max(time_original) > 1 else "samples"
        axs[i].set_ylabel("Value")
        axs[i].legend()
        axs[i].grid(True, alpha=0.3)
    
    # Add common x-label
    fig.text(0.5, 0.04, f"Time [{unit}]", ha='center')
    
    plt.tight_layout()
    
    return fig

def plot_all_signals_together(gyro_interp, acc_interp, orientation_interp, time_interp=None):
    """
    Plot all interpolated signals together to verify alignment.
    """
    x_axis = time_interp if time_interp is not None else np.arange(len(gyro_interp))
    unit = "seconds" if time_interp is not None and np.max(time_interp) > 1 else "samples"
    
    # Organize signals by type
    signal_groups = [
        ("Gyroscope", gyro_interp, 'b-'),
        ("Accelerometer", acc_interp, 'r-'),
        ("Orientation", orientation_interp, 'g-')
    ]
    
    # Create figure with time series plots
    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("All Interpolated Signals")
    
    # Plot each signal group in its own subplot
    for i, (name, data, style) in enumerate(signal_groups):
        for col in data.columns:
            axs[i].plot(x_axis, data[col], style, label=f"{col}")
        
        axs[i].set_title(f"{name}")
        axs[i].set_ylabel("Value")
        axs[i].legend()
        axs[i].grid(True, alpha=0.3)
    
    axs[-1].set_xlabel(f"Time [{unit}]")
    
    plt.tight_layout()
    
    return fig

def interpolate_and_visualize(gyro, acc, orientation, frequencies=None, plot_flag=True):
    """
    Main function to interpolate signals and visualize the results.

    Parameters:
    -----------
    gyro : pandas.DataFrame
        Gyroscope data after segmentation
    acc : pandas.DataFrame
        Accelerometer data after segmentation
    orientation : pandas.DataFrame
        Orientation data after segmentation
    frequencies : list or numpy.ndarray, optional
        Sampling frequencies [gyro_freq, acc_freq, orientation_freq] in Hz
    plot_flag : bool, optional
        Flag to control whether plots are generated (default is True)

    Returns:
    --------
    gyro_interp, acc_interp, orientation_interp : pandas.DataFrame
        Interpolated signals with the same length
    """
    # Process frequency information if provided
    if frequencies is not None:
        print(f"Using sampling frequencies: Gyro={frequencies[0]}Hz, Acc={frequencies[1]}Hz, Orientation={frequencies[2]}Hz")
    
    # Perform interpolation
    gyro_interp, acc_interp, orientation_interp, time_interp = interpolate_sensor_signals(
        gyro, acc, orientation, frequencies
    )
    
    if plot_flag:
        # Create visualization plots using subplots for compact presentation
        plot_signal_lengths(gyro, acc, orientation, gyro_interp, acc_interp, orientation_interp, frequencies)
        
        # Only create visualization plots if interpolation was performed
        if len(gyro) != len(gyro_interp):
            visualize_interpolation(gyro, gyro_interp, "Gyroscope", 
                                    time_original=None if frequencies is None else np.arange(len(gyro))/frequencies[0], 
                                    time_interp=time_interp)
        
        if len(acc) != len(acc_interp):
            visualize_interpolation(acc, acc_interp, "Accelerometer",
                                    time_original=None if frequencies is None else np.arange(len(acc))/frequencies[1], 
                                    time_interp=time_interp)
        
        if len(orientation) != len(orientation_interp):
            visualize_interpolation(orientation, orientation_interp, "Orientation",
                                    time_original=None if frequencies is None else np.arange(len(orientation))/frequencies[2], 
                                    time_interp=time_interp)
        
        # Show detailed segment comparison if signals were interpolated
        if len(gyro) != len(gyro_interp):
            compare_signal_segments(gyro, gyro_interp, "Gyroscope", 
                                    time_original=None if frequencies is None else np.arange(len(gyro))/frequencies[0],
                                    time_interp=time_interp)
        
        # Plot all signals together after interpolation
        plot_all_signals_together(gyro_interp, acc_interp, orientation_interp, time_interp)
    
    # Final verification
    final_lengths = [len(gyro_interp), len(acc_interp), len(orientation_interp)]
    if len(set(final_lengths)) == 1:
        print(f"Success: All signals now have the same length of {final_lengths[0]} samples")
    else:
        print("Warning: Signals still have different lengths after interpolation")
    
    return gyro_interp, acc_interp, orientation_interp