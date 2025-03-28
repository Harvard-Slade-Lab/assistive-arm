import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

def interpolate_sensor_signals(gyro, acc, orientation):
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
    
    Returns:
    --------
    gyro_interp, acc_interp, orientation_interp : pandas.DataFrame
        Interpolated signals with the same length
    """
    # Get lengths of each signal
    len_gyro = len(gyro)
    len_acc = len(acc)
    len_orientation = len(orientation)
    
    print(f"Original signal lengths: Gyro={len_gyro}, Acc={len_acc}, Orientation={len_orientation}")
    
    # Check if all signals have the same length
    if len_gyro == len_acc == len_orientation:
        print("All signals have the same length. No interpolation needed.")
        return gyro, acc, orientation
    
    # Find the target length (maximum of the three)
    target_length = max(len_gyro, len_acc, len_orientation)
    print(f"Target length for interpolation: {target_length}")
    
    # Create a new time index for the target length
    new_index = np.linspace(0, 1, target_length)
    
    # Initialize interpolated dataframes
    gyro_interp = gyro.copy()
    acc_interp = acc.copy()
    orientation_interp = orientation.copy()
    
    # Interpolate gyro if needed
    if len_gyro != target_length:
        old_index = np.linspace(0, 1, len_gyro)
        gyro_interp = pd.DataFrame(index=range(target_length))
        
        for col in gyro.columns:
            f = interpolate.interp1d(old_index, gyro[col], kind='linear', 
                                     fill_value='extrapolate')
            gyro_interp[col] = f(new_index)
        
        print(f"Interpolated gyroscope data from {len_gyro} to {target_length} samples")
    
    # Interpolate acc if needed
    if len_acc != target_length:
        old_index = np.linspace(0, 1, len_acc)
        acc_interp = pd.DataFrame(index=range(target_length))
        
        for col in acc.columns:
            f = interpolate.interp1d(old_index, acc[col], kind='linear', 
                                     fill_value='extrapolate')
            acc_interp[col] = f(new_index)
        
        print(f"Interpolated accelerometer data from {len_acc} to {target_length} samples")
    
    # Interpolate orientation if needed
    if len_orientation != target_length:
        old_index = np.linspace(0, 1, len_orientation)
        orientation_interp = pd.DataFrame(index=range(target_length))
        
        for col in orientation.columns:
            f = interpolate.interp1d(old_index, orientation[col], kind='linear', 
                                     fill_value='extrapolate')
            orientation_interp[col] = f(new_index)
        
        print(f"Interpolated orientation data from {len_orientation} to {target_length} samples")
    
    return gyro_interp, acc_interp, orientation_interp

def create_new_figure():
    """Create a new figure and return its number."""
    fig = plt.figure()
    return fig.number

def visualize_interpolation(original, interpolated, signal_type):
    """
    Visualize original and interpolated signals.
    """
    if len(original) == len(interpolated):
        return  # No visualization needed if lengths are the same
    
    fig_num = create_new_figure()
    fig = plt.figure(fig_num)
    fig.clf()  # Clear the figure
    
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    # Plot original data
    for col in original.columns:
        ax1.plot(original[col], label=col)
    
    ax1.set_title(f"Original {signal_type} Data ({len(original)} samples)")
    ax1.set_xlabel("Sample Index")
    ax1.set_ylabel("Value")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot interpolated data
    for col in interpolated.columns:
        ax2.plot(interpolated[col], label=col)
    
    ax2.set_title(f"Interpolated {signal_type} Data ({len(interpolated)} samples)")
    ax2.set_xlabel("Sample Index")
    ax2.set_ylabel("Value")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)  # Add a small pause to allow the plot to render

def compare_signal_segments(original, interpolated, signal_type, segment_size=100):
    """
    Compare specific segments of original and interpolated signals to show detail.
    """
    if len(original) == len(interpolated):
        return  # No comparison needed if lengths are the same
    
    scale_factor = len(interpolated) / len(original)
    mid_point_orig = len(original) // 2
    start_orig = max(0, mid_point_orig - segment_size // 2)
    end_orig = min(len(original), mid_point_orig + segment_size // 2)
    start_interp = int(start_orig * scale_factor)
    end_interp = int(end_orig * scale_factor)
    
    fig_num = create_new_figure()
    fig = plt.figure(fig_num)
    fig.clf()  # Clear the figure
    
    num_cols = len(original.columns)
    fig, axs = plt.subplots(num_cols, 1, num=fig_num, figsize=(12, 3*num_cols))
    
    if num_cols == 1:
        axs = [axs]
    
    for i, col in enumerate(original.columns):
        axs[i].plot(range(start_orig, end_orig), 
                    original[col].iloc[start_orig:end_orig], 
                    'b-', label='Original')
        axs[i].plot(np.linspace(start_orig, end_orig-1, end_interp-start_interp), 
                    interpolated[col].iloc[start_interp:end_interp], 
                    'r--', label='Interpolated')
        axs[i].set_title(f"{signal_type} - {col} (Detailed Segment)")
        axs[i].set_xlabel("Original Sample Index")
        axs[i].set_ylabel("Value")
        axs[i].legend()
        axs[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)  # Add a small pause to allow the plot to render

def plot_signal_lengths(gyro, acc, orientation, gyro_interp, acc_interp, orientation_interp):
    """
    Visualize the lengths of original and interpolated signals.
    """
    signal_types = ['Gyroscope', 'Accelerometer', 'Orientation']
    original_lengths = [len(gyro), len(acc), len(orientation)]
    interp_lengths = [len(gyro_interp), len(acc_interp), len(orientation_interp)]
    
    fig_num = create_new_figure()
    fig = plt.figure(fig_num)
    fig.clf()  # Clear the figure
    
    ax = fig.add_subplot(111)
    
    x = np.arange(len(signal_types))
    width = 0.35
    
    ax.bar(x - width/2, original_lengths, width, label='Original')
    ax.bar(x + width/2, interp_lengths, width, label='Interpolated')
    
    for i, v in enumerate(original_lengths):
        ax.text(i - width/2, v + 0.1, str(v), ha='center')
    
    for i, v in enumerate(interp_lengths):
        ax.text(i + width/2, v + 0.1, str(v), ha='center')
    
    ax.set_xticks(x)
    ax.set_xticklabels(signal_types)
    ax.set_ylabel('Number of Samples')
    ax.set_title('Signal Lengths Before and After Interpolation')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)  # Add a small pause to allow the plot to render
    
def interpolate_and_visualize(gyro, acc, orientation):
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
    
    Returns:
    --------
    gyro_interp, acc_interp, orientation_interp : pandas.DataFrame
        Interpolated signals with the same length
    """
    # Perform interpolation
    gyro_interp, acc_interp, orientation_interp = interpolate_sensor_signals(gyro, acc, orientation)
    
    # Visualize signal lengths
    plot_signal_lengths(gyro, acc, orientation, gyro_interp, acc_interp, orientation_interp)
    
    # Visualize full signals if interpolated
    if len(gyro) != len(gyro_interp):
        visualize_interpolation(gyro, gyro_interp, "Gyroscope")
        compare_signal_segments(gyro, gyro_interp, "Gyroscope")
    
    if len(acc) != len(acc_interp):
        visualize_interpolation(acc, acc_interp, "Accelerometer")
        compare_signal_segments(acc, acc_interp, "Accelerometer")
    
    if len(orientation) != len(orientation_interp):
        visualize_interpolation(orientation, orientation_interp, "Orientation")
        compare_signal_segments(orientation, orientation_interp, "Orientation")
    
    # Final verification
    final_lengths = [len(gyro_interp), len(acc_interp), len(orientation_interp)]
    if len(set(final_lengths)) == 1:
        print(f"Success: All signals now have the same length of {final_lengths[0]} samples")
    else:
        print("Warning: Signals still have different lengths after interpolation")
    
    return gyro_interp, acc_interp, orientation_interp
