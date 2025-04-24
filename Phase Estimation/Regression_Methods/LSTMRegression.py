import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import time
import os


def prepare_lstm_data(X, y, time_steps):
    """
    Prepare data for LSTM model by reshaping into [samples, time_steps, features]
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix of shape [samples, features]
    y : numpy.ndarray
        Target values of shape [samples]
    time_steps : int
        Number of time steps to consider for each prediction
        
    Returns:
    --------
    X_lstm : numpy.ndarray
        Reshaped feature matrix of shape [samples-time_steps+1, time_steps, features]
    y_lstm : numpy.ndarray
        Target values corresponding to X_lstm
    """
    X_lstm, y_lstm = [], []
    for i in range(len(X) - time_steps + 1):
        X_lstm.append(X[i:(i + time_steps), :])
        y_lstm.append(y[i + time_steps - 1])
    return np.array(X_lstm), np.array(y_lstm)

def lstm_regression(X, y, time_steps=10, epochs=100, batch_size=32, 
                    validation_split=0.2, patience=20, verbose=1, plot=True, 
                    frequencies=None):
    """
    Applies LSTM Regression for gait phase estimation
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix of shape [samples, features]
    y : numpy.ndarray
        Target values of shape [samples]
    time_steps : int, optional
        Number of time steps to consider for each prediction. Default is 10.
    epochs : int, optional
        Number of training epochs. Default is 100.
    batch_size : int, optional
        Batch size for training. Default is 32.
    validation_split : float, optional
        Fraction of data to be used for validation. Default is 0.2.
    patience : int, optional
        Number of epochs with no improvement after which training will be stopped. Default is 20.
    verbose : int, optional
        Verbosity mode (0, 1, or 2). Default is 1.
    plot : bool, optional
        Whether to plot the results. Default is True.
    frequencies : list, optional
        List of sensor frequencies. Default is None.
        
    Returns:
    --------
    dict
        Dictionary containing the model, history, predictions, and evaluation metrics
    """
    # Extract minimum frequency for plotting
    min_frequency = np.min(frequencies) if frequencies is not None else None
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Prepare data for LSTM
    X_lstm, y_lstm = prepare_lstm_data(X_scaled, y, time_steps)
    
    # Define the model architecture
    n_features = X.shape[1]
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(time_steps, n_features), return_sequences=True),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    # Define callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        ModelCheckpoint('best_lstm_model.h5', save_best_only=True, monitor='val_loss', mode='min')
    ]
    
    # Train the model
    history = model.fit(
        X_lstm, y_lstm,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=verbose
    )
    
    # Make predictions and calculate metrics
    # [Code omitted for brevity]
    
    # Return results dictionary
    return {
        'model': model,
        'history': history.history,
        'predictions': final_pred,
        'residuals': residuals,
        'mse': mse,
        'r2': r2,
        'scaler': scaler,
        'time_steps': time_steps
    }

def test_lstm(model_results, X_new, frequencies=None, plot=True):
    """
    Test the LSTM model with new data.
    
    Parameters:
    -----------
    model_results : dict
        Dictionary returned by lstm_regression function containing model and other information.
    X_new : numpy.ndarray
        New feature matrix.
    frequencies : list, optional
        List of sensor frequencies. Default is None.
    plot : bool, optional
        Whether to plot the results. Default is True.
    
    Returns:
    --------
    predictions : numpy.ndarray
        Predicted values from the model.
    """
    min_frequency = np.min(frequencies) if frequencies is not None else None
    
    # Extract model and other information
    model = model_results['model']
    scaler = model_results['scaler']
    time_steps = model_results['time_steps']
    
    # Scale the features
    X_scaled = scaler.transform(X_new)
    
    # Prepare data for LSTM
    X_lstm = []
    for i in range(len(X_scaled) - time_steps + 1):
        X_lstm.append(X_scaled[i:(i + time_steps), :])
    X_lstm = np.array(X_lstm)
    
    # Predict using the model
    y_pred_lstm = model.predict(X_lstm)
    
    # Reshape predictions to match original shape
    y_new = np.zeros(len(X_new))
    y_new[time_steps-1:] = y_pred_lstm.flatten()
    
    # For the first (time_steps-1) values, use the first prediction
    if time_steps > 1:
        y_new[:time_steps-1] = y_pred_lstm[0]
    
    # Create expected linear progression (0-100% over time)
    n_samples = len(X_new)
    y_expected = np.linspace(0, 1, n_samples)
    
    # Calculate metrics
    mse = mean_squared_error(y_expected[time_steps-1:], y_pred_lstm.flatten())
    r2 = r2_score(y_expected[time_steps-1:], y_pred_lstm.flatten())
    
    print(f"Test Mean Squared Error: {mse:.4e}")
    print(f"Test R² Score: {r2:.4f}")
    
    # Create and display visualization
    # [Code omitted for brevity]
    
    return y_new, y_expected, mse



# Example usage
if __name__ == "__main__":
    # 1. Load your segmented gait cycle data
    # X_train should contain gyro x,y,z; acc x,y,z; orientation data
    # y_train should contain linear progression from 0 to 1 (0% to 100%)
    
    # 2. Train an LSTM model
    model_results = lstm_regression(X_train, y_train, time_steps=10, epochs=100)
    
    # 3. Test the model on new data
    test_lstm(model_results, X_test)
    
    # 4. Optional: Tune hyperparameters for better performance
    tuned_model = lstm_hyperparameter_tuning(X_train, y_train)
    
    # 5. Save the best model for later use
    save_lstm_model(tuned_model, 'gait_phase_lstm_model')
