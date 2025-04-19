import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import DataLoader  # Assuming this is your custom module

# Configuration Constants
SAMPLE_RATE = 519  # Hz
PROCESSING_WINDOW = 0.3  # seconds for feature extraction
LOWPASS_CUTOFF = 10  # Hz
STATIONARY_ACCEL_VAR_THRESH = 0.15  # m/s² variance threshold
BIAS_CALC_WINDOW = int(SAMPLE_RATE * 0.5)  # 0.5s window for bias estimation

def butterworth_lowpass(data, cutoff, fs, order=4):
    """Apply Butterworth low-pass filter to 1D signal"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, data)

def quaternion_to_rotation_matrix(q):
    """Convert quaternion [w,x,y,z] to 3x3 rotation matrix"""
    w, x, y, z = q
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])

def process_sensor_data(gyro_df, accel_df, ori_df, timestamp):
    """
    Full processing pipeline for one timestamp dataset
    Returns: (features, labels, filtered_magnitude, timestamps)
    """
    print(f"\nProcessing {timestamp}")
    
    # 1. Synchronized data alignment
    non_zero_idx = max(
        (gyro_df != 0).any(axis=1).idxmax(),
        (accel_df != 0).any(axis=1).idxmax()
    )
    
    # 2. Stationary phase detection using accelerometer
    accel_trimmed = accel_df.iloc[non_zero_idx:].reset_index(drop=True)
    accel_magnitude = np.linalg.norm(accel_trimmed, axis=1)
    
    # Find low-variance windows (1s sliding window)
    stationary_mask = np.zeros(len(accel_magnitude), dtype=bool)
    window_size = int(SAMPLE_RATE * 1)
    for i in range(0, len(accel_magnitude)-window_size, window_size//2):
        window_var = np.var(accel_magnitude[i:i+window_size])
        if window_var < STATIONARY_ACCEL_VAR_THRESH:
            stationary_mask[i:i+window_size] = True

    # 3. Gyroscope bias calibration
    gyro_trimmed = gyro_df.iloc[non_zero_idx:].reset_index(drop=True)
    if np.any(stationary_mask):
        gyro_bias = gyro_trimmed[stationary_mask].mean()
    else:
        gyro_bias = gyro_trimmed.iloc[:BIAS_CALC_WINDOW].mean()
    
    # 4. Bias compensation
    gyro_clean = gyro_trimmed - gyro_bias.values

    # 5. Orientation compensation
    ori_trimmed = ori_df.iloc[non_zero_idx:].reset_index(drop=True)
    global_omega = []
    for idx, (_, gyro_row) in enumerate(gyro_clean.iterrows()):
        R = quaternion_to_rotation_matrix(ori_trimmed.iloc[idx])
        global_omega.append(R @ gyro_row.values)
    gyro_rotated = pd.DataFrame(global_omega, columns=gyro_clean.columns)

    # 6. Signal conditioning
    for axis in gyro_rotated.columns:
        gyro_rotated[f'{axis}_filt'] = butterworth_lowpass(
            gyro_rotated[axis], LOWPASS_CUTOFF, SAMPLE_RATE)
    
    # 7. Magnitude calculation
    gyro_rotated['mag'] = np.linalg.norm(
        gyro_rotated[['0_filt', '1_filt', '2_filt']], axis=1)

    # 8. Feature engineering
    window_size = int(SAMPLE_RATE * PROCESSING_WINDOW)
    half_window = window_size // 2
    features = []
    
    for i in range(half_window, len(gyro_rotated)-half_window):
        window = gyro_rotated['mag'].iloc[i-half_window:i+half_window]
        
        # Temporal features
        w_mean = np.mean(window)
        w_std = np.std(window)
        
        # Spectral features
        fft = np.abs(np.fft.rfft(window - w_mean))
        spectral_energy = np.sum(fft**2)
        
        # Shape features
        slope = (window.values[-1] - window.values[0]) / len(window)
        
        features.append([w_mean, w_std, spectral_energy, slope])

    # 9. Adaptive segmentation labeling
    noise_floor = gyro_rotated['mag'][stationary_mask].mean() if np.any(stationary_mask) else 0.1
    peaks, _ = find_peaks(gyro_rotated['mag'], 
                         height=5*noise_floor,
                         distance=int(SAMPLE_RATE*0.2))
    
    labels = np.zeros(len(gyro_rotated))
    for p in peaks:
        start = max(0, p - int(SAMPLE_RATE*0.1))
        end = min(len(labels), p + int(SAMPLE_RATE*0.1))
        labels[start:end] = 1
    labels = labels[half_window:-half_window]

    return np.array(features), labels, gyro_rotated['mag'].values, gyro_rotated.index/SAMPLE_RATE

def main():
    # Data loading using existing pipeline
    folder_path = DataLoader.select_folder()
    if not folder_path:
        print("No folder selected. Exiting...")
        return
    
    accel_data, gyro_data, ori_data, acc_files, gyro_files, or_files = DataLoader.load_and_process_files(folder_path)
    grouped_indices = DataLoader.group_files_by_timestamp(acc_files, gyro_files, or_files)
    
    # Initialize classifier with optimized parameters
    segment_classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight='balanced_subsample',
        n_jobs=-1,
        random_state=42
    )
    
    # Process each timestamp dataset
    for timestamp in sorted(grouped_indices.keys()):
        indices = grouped_indices[timestamp]
        
        # Extract sensor data
        gyro = gyro_data[indices["gyro"]]
        accel = accel_data[indices["acc"]]
        ori = ori_data[indices["or"]]
        
        # Process dataset
        features, labels, magnitudes, time = process_sensor_data(gyro, accel, ori, timestamp)
        
        if len(features) == 0 or len(labels) == 0:
            print(f"Skipping {timestamp} - insufficient data")
            continue
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.25, stratify=labels)
        
        # Model training
        segment_classifier.fit(X_train, y_train)
        
        # Generate predictions
        train_probs = segment_classifier.predict_proba(X_train)[:,1]
        test_probs = segment_classifier.predict_proba(X_test)[:,1]
        
        # Visualization
        plt.figure(figsize=(15, 8))
        
        # Raw signal and peaks
        plt.subplot(2,1,1)
        plt.plot(time, magnitudes, label='Filtered Magnitude')
        plt.ylabel('Angular Velocity (rad/s)')
        plt.title(f'Motion Profile: {timestamp}')
        
        # Segmentation results
        plt.subplot(2,1,2)
        plt.plot(time[len(magnitudes)//4:-len(magnitudes)//4], 
                segment_classifier.predict_proba(features)[::4,1], 
                label='Segment Probability')
        plt.plot(time[len(magnitudes)//4:-len(magnitudes)//4], 
                labels[::4], '--', label='Ground Truth')
        plt.xlabel('Time (s)')
        plt.ylabel('Segmentation Confidence')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Performance report
        print(f"\nSegmentation Performance - {timestamp}")
        print("Test Set Classification Report:")
        print(classification_report(y_test, segment_classifier.predict(X_test)))

if __name__ == "__main__":
    main()
