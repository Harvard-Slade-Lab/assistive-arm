import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.signal import find_peaks, argrelextrema
import os


def load_data(file_path):
    # Skip the first rows containing metadata
    skip_rows = 8  # Adjust based on the exact number of metadata rows

    # Read the actual sensor data, assuming the delimiter is tab or spaces
    df = pd.read_csv(file_path, delimiter=',', skiprows=skip_rows)

    # print(df)
    # # Rename the columns based on the file format
    df.columns = ['EMG 1 (mV)', 'ACC X (G)', 'ACC Y (G)', 'ACC Z (G)', 'GYRO X (deg/s)', 'GYRO Y (deg/s)', 'GYRO Z (deg/s)']

    df = df.apply(pd.to_numeric, errors='coerce')

    # Extract the imu data once the first NaN appears in 'ACC X (G)'
    if df['GYRO Z (deg/s)'].isna().any():
        first_nan_index = df['GYRO Z (deg/s)'].index[df['GYRO Z (deg/s)'].isna()][0]
        imu_df = df[:first_nan_index]  # Slice the DataFrame until the first NaN in 'ACC X (G)'
        imu_df = imu_df.drop(columns=['EMG 1 (mV)'])

    # Create emg df
    emg_df = pd.DataFrame()
    emg_df['EMG 1 (mV)'] = df['EMG 1 (mV)']

    return imu_df, emg_df

# def apply_moving_average_filter(df: pd.DataFrame, window_size: int) -> pd.DataFrame:
#     """
#     Apply a moving average filter to the first 7 columns of the given DataFrame.

#     Parameters:
#     - df: pandas.DataFrame with the data to be filtered.
#     - window_size: The size of the moving average window.

#     Returns:
#     - df_filtered: DataFrame with the moving average filtered values.
#     """
#     # Apply a moving average filter with the given window size
#     df_filtered = df.copy()
#     df_filtered.iloc[:, :7] = df.iloc[:, :7].rolling(window=window_size, min_periods=1).mean()

#     return df_filtered


def apply_lowpass_filter(df: pd.DataFrame, cutoff_freq: float, sampling_freq: float, filter_order=4) -> pd.DataFrame:
    """
    Apply a low-pass filter to the first 7 columns of the given DataFrame.

    Parameters:
    - df: pandas.DataFrame with the data to be filtered.
    - cutoff_freq: The cutoff frequency for the low-pass filter (in Hz).
    - sampling_freq: The sampling frequency of the data (in Hz).
    - filter_order: The order of the Butterworth filter (default is 4).

    Returns:
    - df_filtered: DataFrame with the low-pass filtered values.
    """
    # Normalize the cutoff frequency with respect to Nyquist frequency
    nyquist_freq = sampling_freq / 2
    normalized_cutoff = cutoff_freq / nyquist_freq

    # Design a Butterworth low-pass filter
    b, a = sp.signal.butter(filter_order, normalized_cutoff, btype='low')

    # Create a copy of the DataFrame to hold the filtered data
    df_filtered = df.copy()

    # Apply the filter to data
    for col in df.columns:
        df_filtered[col] = sp.signal.filtfilt(b, a, df[col])

    return df_filtered


def filter_emg(unfiltered_df: pd.DataFrame, low_pass=4, sfreq=2000, high_band=20, low_band=450) -> pd.DataFrame:
    """ Filter EMG signals

    Args:
        unfiltered_df (pd.DataFrame): DataFrame containing the EMG data and time
        low_pass (int, optional): Low-pass cut off frequency. Defaults to 4.
        sfreq (int, optional): Sampling frequency. Defaults to 2000.
        high_band (int, optional): High-band frequency for bandpass filter. Defaults to 20.
        low_band (int, optional): Low-band frequency for bandpass filter. Defaults to 450.

    Returns:
        pd.DataFrame: filtered dataframe
    """
    emg_data = unfiltered_df.copy()

    # Normalize cut-off frequencies to sampling frequency
    high_band_normalized = high_band / (sfreq / 2)
    low_band_normalized = low_band / (sfreq / 2)
    low_pass_normalized = low_pass / (sfreq / 2)

    # Bandpass filter coefficients
    b1, a1 = sp.signal.butter(4, [high_band_normalized, low_band_normalized], btype='bandpass')

    # Lowpass filter coefficients
    b2, a2 = sp.signal.butter(4, low_pass_normalized, btype='lowpass')

    def process_emg(emg):
        # Handle NaNs: skip filtering for NaN segments
        if emg.isna().all():
            return emg  # Returns as is if all are NaNs

        # Correct mean for non-NaN values
        non_nan_emg = emg.dropna()
        emg_correctmean = non_nan_emg - non_nan_emg.mean()

        # Filter EMG: bandpass, rectify, lowpass for non-NaN values
        emg_filtered = sp.signal.filtfilt(b1, a1, emg_correctmean)
        emg_rectified = np.abs(emg_filtered)
        emg_envelope = sp.signal.filtfilt(b2, a2, emg_rectified)

        # Construct the resulting series, placing NaNs back in their original positions
        result = pd.Series(index=emg.index, data=np.nan)
        result[emg.notna()] = emg_envelope

        return result

    # Apply processing to each column
    envelopes = emg_data.apply(process_emg, axis=0)
    env_freq = int(low_pass_normalized * sfreq)

    return envelopes, env_freq



def extract_time(df):
    """
    Detect the start and end of a sit-to-stand motion based on angular acceleration in z-axis.
    
    Parameters:
    - df: pandas.DataFrame with the data
    Returns:
    start_time (float): Time when the motion starts.
    end_time (float): Time when the motion ends.
    """
    # First derivative (rate of change) to detect where the acceleration starts decreasing
    acc_z_diff = np.diff(df['ACC Z (G)'])

    # Debug plot
    # der = acc_z_diff
    # der = np.append(der, 0)
    # df["der"] = der
    # df["der"].plot()
    # plt.show()

    # Find all acceleration-change extremas
    maxima = argrelextrema(acc_z_diff, np.greater)[0]

    # Find the global minimum
    minimum = np.argmin(df['ACC Z (G)'])

    # Get the two maxima closest to the global minimum
    maxima.sort()
    start_idx = maxima[np.searchsorted(maxima, minimum) - 1]
    end_idx = maxima[np.searchsorted(maxima, minimum)]

    return start_idx, end_idx


    # Through change of signs in velocities
    # start_idx = None
    # end_idx = None

    # # find global velocity extremas
    # gyro_x_diff = np.diff(df['GYRO X (deg/s)'])
    # gyro_x_diff_diff = np.diff(gyro_x_diff)
    
    # minima = argrelextrema(gyro_x_diff_diff, np.less)[0]
    # for minimum in minima:
    #     if gyro_x_diff_diff[minimum] < -0.01 and start_idx is not None:
    #         end_idx = minimum
    #     if gyro_x_diff_diff[minimum] < -0.01 and start_idx is None:
    #         start_idx = minimum

    


def segmentation(imu_df, emg_df):
    """
    Segmentation of incoming data based on angular acceleration in z-axis.
    
    Parameters:
    - imu_df: pandas.DataFrame with the imu data
    
    Returns:
    - df_segmented: DataFrame with the segmented data.
    """

    # Find maxima in angular velocity in x-axis, that exceed a certain threshold
    maxima = find_peaks(imu_df['GYRO X (deg/s)'], height=50)[0]

    # Add first and last index to the maxima list
    maxima = np.insert(maxima, 0, 0)
    maxima = np.append(maxima, len(imu_df)-1)
    
    # Extract relevant times between two maxima
    for i in range(len(maxima)-1):
        start_idx = maxima[i]
        end_idx = maxima[i+1]

        # Create a new dataframe with the relevant data
        imu_df_segmented = imu_df.iloc[start_idx:end_idx].reset_index(drop=True)

        # Extract the start and end index of the motion
        start_idx2, end_idx2 = extract_time(imu_df_segmented)

        # Calculate some score of the emg data during the motion
        relevant_emg = imu_df_segmented.iloc[start_idx2:end_idx2].reset_index(drop=True)
        # Debug plots
        start_time = imu_df_segmented['Time (s)'][start_idx2]
        end_time = imu_df_segmented['Time (s)'][end_idx2]
        plot_data(imu_df, emg_df, file_path, start_time, end_time)


    # return df






def plot_data(imu_df, emg_df, file_path, start, end):
    filename = os.path.basename(file_path).replace('.csv', '')

    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Subplot for angular accelerations (ACC X, ACC Y, ACC Z)
    ax1.plot(imu_df['Time (s)'], imu_df['ACC X (G)'], label='ACC X (G)')
    ax1.plot(imu_df['Time (s)'], imu_df['ACC Y (G)'], label='ACC Y (G)')
    ax1.plot(imu_df['Time (s)'], imu_df['ACC Z (G)'], label='ACC Z (G)')
    ax1.axvline(x=start, color='r', linestyle='--', label='Start')
    ax1.axvline(x=end, color='g', linestyle='--', label='End')
    ax1.set_ylabel('Angular Acceleration (G)')
    ax1.set_title('Angular Accelerations')
    ax1.legend()
    ax1.grid(True)

    # Subplot for angular velocities (GYRO X, GYRO Y, GYRO Z)
    ax2.plot(imu_df['Time (s)'], imu_df['GYRO X (deg/s)'], label='GYRO X (deg/s)')
    ax2.plot(imu_df['Time (s)'], imu_df['GYRO Y (deg/s)'], label='GYRO Y (deg/s)')
    ax2.plot(imu_df['Time (s)'], imu_df['GYRO Z (deg/s)'], label='GYRO Z (deg/s)')
    ax2.axvline(x=start, color='r', linestyle='--', label='Start')
    ax2.axvline(x=end, color='g', linestyle='--', label='End')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Angular Velocity (deg/s)')
    ax2.set_title('Angular Velocities')
    ax2.legend()
    ax2.grid(True)

    # Subplot for EMG data
    ax3.plot(emg_df['EMG Time (s)'], emg_df['EMG 1 (mV)'], label='EMG 1 (mV)')
    ax3.axvline(x=start, color='r', linestyle='--', label='Start')
    ax3.axvline(x=end, color='g', linestyle='--', label='End')
    ax3.set_ylabel('EMG (mV)')
    ax3.set_title('EMG Data')
    ax3.legend()
    ax3.grid(True)

    # Show plot
    plt.tight_layout()
    plt.savefig(f'/Users/nathanirniger/Desktop/IMU_segmentation/{filename}.png')
    plt.show()


if __name__ == "__main__":
    file_path = '/Users/nathanirniger/Desktop/IMU_segmentation/RF_4.csv'

    imu_df, emg_df = load_data(file_path)
    imu_df = apply_lowpass_filter(imu_df, 1, 148.1481)

    imu_time_interval = 1 / 148.1481  # Time step based on ACC X sampling rate
    imu_df['Time (s)'] = imu_df.index * imu_time_interval  # Create a time column

    emg_time_interval = 1 / 1259.2593 # Time step based on EMG sampling rate
    emg_df['EMG Time (s)'] = emg_df.index * emg_time_interval  # Create a time column for EMG data
    # emg_df, env_freq = filter_emg(emg_df, sfreq=1259.2593)

    segmentation(imu_df, emg_df)
    # start, end = extract_time(df)
    # plot_data(df, file_path, start, end)