import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models, optimizers
from tkinter.filedialog import askdirectory
from scipy.signal import butter, filtfilt
from Interpolation import interpolate_and_visualize
import DataLoader
import MatrixCreator
import os

# ----------- NEURAL NETWORK SETTINGS -----------------
MODEL_PATH = "peak_detection_model.h5"
TRAINING_SAMPLES_NEEDED = 10  # Minimum samples before training
AUGMENTATION_FACTOR = 5  # Data augmentation multiplier

# ----------- HYPERPARAMETERS -----------------
bias_average_window = 1000
frequency = 519

# Initialize data collection
training_data = []
training_labels = []

def build_peak_detection_model(input_shape):
    """Build the neural network model"""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(32, 5, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 3, activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(2)
    ])
    model.compile(optimizer=optimizers.Adam(0.001),
                 loss='mse',
                 metrics=['mae'])
    return model

def on_click(event, magnitude, timestamp):
    """Handle mouse clicks for peak annotation"""
    global click_count, start_idx, end_idx
    if event.xdata is not None:
        if click_count == 0:
            start_idx = int(round(event.xdata))
            click_count = 1
            print(f"Start index set to: {start_idx}")
        else:
            end_idx = int(round(event.xdata))
            click_count = 0
            print(f"End index set to: {end_idx}")
            plt.close()
            
            # Store the training sample
            training_data.append(magnitude)
            training_labels.append([start_idx, end_idx])
            print(f"Saved sample for {timestamp} (Total samples: {len(training_data)})")
            
            # Check if we have enough samples to train
            if len(training_data) >= TRAINING_SAMPLES_NEEDED:
                train_model()

def normalize_signal(signal):
    """Normalize signal to 0-1 range"""
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

def augment_dataset(signals, labels):
    """Augment dataset with noise and shifts"""
    augmented_signals = []
    augmented_labels = []
    
    for signal, label in zip(signals, labels):
        # Original sample
        augmented_signals.append(signal)
        augmented_labels.append(label)
        
        # Augmented variations
        for _ in range(AUGMENTATION_FACTOR - 1):
            # Add noise
            noisy = signal + np.random.normal(0, 0.1, len(signal))
            
            # Random shift
            shift = np.random.randint(-5, 6)
            shifted = np.roll(noisy, shift)
            new_label = [max(0, label[0]+shift), max(0, label[1]+shift)]
            
            augmented_signals.append(shifted)
            augmented_labels.append(new_label)
    
    return np.array(augmented_signals), np.array(augmented_labels)

def train_model():
    """Train the neural network with collected data"""
    print("\nStarting model training...")
    
    # Normalize and augment data
    normalized = [normalize_signal(sig) for sig in training_data]
    X_aug, y_aug = augment_dataset(normalized, training_labels)
    
    # Convert to numpy arrays and add channel dimension
    X = np.array([sig.reshape(-1, 1) for sig in X_aug])
    y = np.array(y_aug)
    
    # Build or load model
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Loaded existing model for continued training")
    else:
        model = build_peak_detection_model((None, 1))
    
    # Train the model
    model.fit(X, y, epochs=50, batch_size=4, validation_split=0.2,
             callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)])
    
    # Save the model
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

# ----------- MAIN PROCESSING LOOP -----------------
folder_path = DataLoader.select_folder()
if not folder_path:
    print("No folder selected. Exiting...")
else:
    print(f"Selected folder: {folder_path}")
    acc_data, gyro_data, or_data, acc_files, gyro_files, or_files = DataLoader.load_and_process_files(folder_path)
    grouped_indices = DataLoader.group_files_by_timestamp(acc_files, gyro_files, or_files)
    sorted_timestamps = sorted(grouped_indices.keys())
    
    # Try to load existing model
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Loaded trained model for inference")
    else:
        model = None
        print("No trained model found - will collect training data")

    for timestamp in sorted_timestamps:
        indices = grouped_indices[timestamp]
        gyro = gyro_data[indices["gyro"]]
        print(f"\nProcessing data set from timestamp: {timestamp}")

        # ------------------------- BIAS REMOVAL ----------------------------
        non_zero_index = (gyro != 0).any(axis=1).idxmax()
        sample_size = bias_average_window
        
        if non_zero_index + sample_size <= len(gyro):
            means = gyro.iloc[non_zero_index:non_zero_index + sample_size].mean()
            gyro_data_centered = gyro - means
            gyro_data_trimmed = gyro_data_centered.iloc[non_zero_index:].reset_index(drop=True)
            
            # Compute magnitude
            raw_magnitude = np.sqrt(gyro_data_trimmed.iloc[:,0]**2 + 
                                   gyro_data_trimmed.iloc[:,1]**2 + 
                                   gyro_data_trimmed.iloc[:,2]**2)
            
            # Filter magnitude
            sampling_rate = frequency
            nyquist = 0.5 * sampling_rate
            cutoff = 5
            normal_cutoff = cutoff / nyquist
            b, a = butter(4, normal_cutoff, btype='low', analog=False)
            magnitude = filtfilt(b, a, raw_magnitude)

            # ------------------- PEAK DETECTION --------------------------
            plt.figure(figsize=(15, 5))
            plt.plot(magnitude, label='Filtered Magnitude', color='blue')
            
            if model:
                # Predict with neural network
                norm_signal = normalize_signal(magnitude).reshape(1, -1, 1)
                prediction = model.predict(norm_signal)[0]
                start, end = prediction.astype(int)
                plt.axvline(start, color='green', linestyle='--', label='Predicted Start')
                plt.axvline(end, color='red', linestyle='--', label='Predicted End')
            else:
                # Collect training data
                global click_count
                click_count = 0
                plt.title(f'Click to annotate peak START then END for {timestamp}')
                plt.connect('button_press_event', lambda event: on_click(event, magnitude, timestamp))
            
            plt.title(f'Filtered Magnitude for {timestamp}')
            plt.xlabel('Samples')
            plt.ylabel('Magnitude')
            plt.legend()
            plt.grid()
            plt.show(block=True)

# After processing all files
if len(training_data) > 0 and len(training_data) < TRAINING_SAMPLES_NEEDED:
    print(f"\nCollected {len(training_data)} samples - need {TRAINING_SAMPLES_NEEDED} to train")
