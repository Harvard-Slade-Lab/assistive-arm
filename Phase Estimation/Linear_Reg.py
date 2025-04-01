import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def perform_regression(gyro_interp, acc_interp, orientation_interp, frequencies):
    # Use minimum frequency from the frequencies variable
    min_frequency = min(frequencies)
    
    # Combine all data
    combined_data = pd.concat([gyro_interp, acc_interp.iloc[:, :3], orientation_interp], axis=1)
    
    # Create time array using minimum frequency
    time = np.arange(len(combined_data)) / min_frequency
    
    # Create target (0 to 100%)
    y = np.linspace(0, 100, len(combined_data))
    
    # Prepare data for regression
    X = combined_data.values
    
    # Perform regression
    model = LinearRegression()
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
    coefficients = model.coef_
    feature_names = ['Intercept', 'Gyro X', 'Gyro Y', 'Gyro Z', 'Acc X', 'Acc Y', 'Acc Z', 'Orientation W', 'Orientation X', 'Orientation Y', 'Orientation Z']

    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(feature_names, coefficients, color='skyblue')
    plt.xlabel('Features')
    plt.ylabel('Coefficient Value')
    plt.title('Bar Plot of Coefficients Associated with Different Features')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom')
    
    plt.grid(axis='y')
    plt.show()
    
    # Calculate and print R-squared score
    r2_score = model.score(X, y)
    print(f"R-squared score: {r2_score:.4f}")
    print(f"Using minimum frequency: {min_frequency} Hz")
    
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
