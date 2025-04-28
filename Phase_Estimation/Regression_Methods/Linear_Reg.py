import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

def linear_regression(X, y, frequencies, feature_names, plot=True):
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
    
    mse = mean_squared_error(y, y_pred)

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
        sorted_idx = np.argsort(importance)[-11:]  # Top 10 features
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
    print("Regreession performed successfully.")
    return model, y_pred, time, mse



def test_regression(model, X_new, frequencies_test, Plot_flag=True):
    # Use minimum frequency from the frequencies variable
    min_frequency_test = min(frequencies_test)
    
    
    # Create time array using minimum frequency
    time_new = np.arange(len(X_new)) / min_frequency_test
    
    # Create target (0 to 100%)
    y = np.linspace(0, 1, len(X_new))

     # Make predictions
    y_pred_new = model.predict(X_new)
    
        # Calculate metrics
    mse = mean_squared_error(y, y_pred_new)
    r2 = r2_score(y, y_pred_new)

    if Plot_flag:
        # Plot regression results
        plt.figure(figsize=(12, 6))
        plt.plot(time_new, y, label='Target')
        plt.plot(time_new, y_pred_new, label='Predicted')
        plt.xlabel('Time (s)')
        plt.ylabel('Percentage (%)')
        plt.title(f'Linear Regression Results\nTest MSE: {mse:.4e}, Test R² Score: {r2:.4f}')
        plt.legend()
        plt.grid(True)
        plt.show()

    return y_pred_new, y, mse

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
