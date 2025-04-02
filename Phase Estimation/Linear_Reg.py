import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline

def perform_regression(X, y, frequencies, plot=True):
    # Use minimum frequency from the frequencies variable
    min_frequency = min(frequencies)
    
    # Create time array using minimum frequency
    time = np.arange(len(X)) / min_frequency
    

    # Perform regression
    model = make_pipeline(
        StandardScaler(),
        LinearRegression()
    )
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Plot regression results
    plt.figure(figsize=(12, 6))
    plt.plot(time, y, label='Target')
    plt.plot(time, y_pred, label='Predicted')
    plt.xlabel('Time (s)')
    plt.ylabel('Percentage (%)')
    plt.title('Linear Regression Results')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Create bar plot of coefficients with named features
    coefficients = model.named_steps['linearregression'].coef_  # Shape (11,)
    

    
    # Plot diagnostics
    if plot:
        plt.figure(figsize=(15, 10))

        # 1. Prediction vs Actual
        plt.subplot(3, 1, 1)
        plt.plot(time, y, label='True', color='blue')
        plt.plot(time, y_pred, label='Predicted', color='orange')
        plt.title('Temporal Alignment')
        plt.ylabel('Progression (%)')
        plt.xlabel('Time (s)')
        plt.legend()
        plt.grid(True)

        # 2. Feature Importance
        plt.subplot(3, 1, 2)
        importance = np.abs(coefficients)
        sorted_idx = np.argsort(importance)[-10:]  # Top 10 features
        feature_names = X.columns[sorted_idx]
        plt.barh(range(len(sorted_idx)), importance[sorted_idx], color='skyblue')
        plt.yticks(range(len(sorted_idx)), feature_names)
        plt.title('Top 10 Predictive Features')
        plt.xlabel('Coefficient Magnitude')
        plt.grid(axis='x')

        # 3. Residual Analysis
        plt.subplot(3, 1, 3)
        residuals = y - y_pred
        plt.scatter(y_pred, residuals, alpha=0.5, color='purple')
        plt.axhline(0, color='red', linestyle='--')
        plt.title('Residual Analysis')
        plt.xlabel('Predicted Progression (%)')
        plt.ylabel('Residual Error (%)')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    # # Calculate and print R-squared score
    # r2_score = model.score(X, y)
    # print(f"R-squared score: {r2_score:.4f}")
    # print(f"Using minimum frequency: {min_frequency} Hz")
    
    return model, y_pred, time



def test_regression(model, gyro_test_interp, acc_test_interp, orientation_test_interp, frequencies_test):
    # Use minimum frequency from the frequencies variable
    min_frequency_test = min(frequencies_test)
    
    # Combine all data
    X_new = pd.concat([gyro_test_interp, acc_test_interp.iloc[:, :3], orientation_test_interp], axis=1)
    
    # Create time array using minimum frequency
    time_new = np.arange(len(X_new)) / min_frequency_test
    
    # Create target (0 to 100%)
    y = np.linspace(0, 100, len(X_new))

     # Make predictions
    y_pred_new = model.predict(X_new)
    
    # Plot regression results
    plt.figure(figsize=(12, 6))
    plt.plot(time_new, y, label='Target')
    plt.plot(time_new, y_pred_new, label='Predicted')
    plt.xlabel('Time (s)')
    plt.ylabel('Percentage (%)')
    plt.title('Linear Regression Results')
    plt.legend()
    plt.grid(True)
    plt.show()

    return y_pred_new, time_new

def calculate_time_array(gyro_interp, acc_interp, or_interp, frequencies):
    """
    Calculate and return the time array used in the regression.
    This extracts the time calculation logic from the perform_regression function.
    """
    # Use minimum frequency from the frequencies variable
    min_frequency = min(frequencies)
    
    # Combine all data
    combined_data = pd.concat([gyro_interp, acc_interp.iloc[:, :3], or_interp], axis=1)
    
    # Create time array using minimum frequency
    time = np.arange(len(combined_data)) / min_frequency

    return time
