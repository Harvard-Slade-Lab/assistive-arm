# main.py
import BiasAndSegmentation
import matplotlib.pyplot as plt
import numpy as np
from Interpolation import interpolate_and_visualize
from RidgeRegressionCV import enhanced_ridge_regression, test_ridge
from LassoRegressionCV import enhanced_lasso_regression, test_lasso
from Linear_Reg import perform_regression, test_regression, calculate_time_array

def run_ridge_regression(gyro_interp, acc_interp, or_interp, frequencies):
    """Perform Ridge regression."""
    print("\nPerforming Ridge Regression...")
    ridge_result, y_ridge = enhanced_ridge_regression(
        gyro_interp,
        acc_interp,
        or_interp,
        alpha_range=(-7, 7, 40),
        cv=None,
        frequencies=frequencies,                 
    )
    return ridge_result, y_ridge

def run_lasso_regression(gyro_interp, acc_interp, or_interp, frequencies):
    """Perform Lasso regression."""
    print("\nPerforming Lasso Regression...")
    lasso_result, y_lasso = enhanced_lasso_regression(
        gyro_interp,
        acc_interp,
        or_interp,
        alpha_range=(-7, 7, 40),
        cv=None,
        frequencies=frequencies,                 
    )
    return lasso_result, y_lasso

def run_linear_regression(gyro_interp, acc_interp, or_interp, frequencies):
    """Perform Linear regression."""
    print("\nPerforming Linear Regression...")
    linear_model, y_linear, _ = perform_regression(
        gyro_interp, 
        acc_interp, 
        or_interp, 
        frequencies
    )
    return linear_model, y_linear

def prepare_test_data(frequencies):
    """Prepare fresh test data for validation"""
    print("\nPreparing test data...")
    gyro_test, acc_test, orientation_test, *_ = BiasAndSegmentation.segmentation_and_bias(frequencies=frequencies, plot_flag=False)
    return interpolate_and_visualize(gyro_test, acc_test, orientation_test, frequencies=frequencies, plot_flag=False)

def execute_test(choice, model, frequencies):
    """Execute the test for the selected regression type"""
    gyro_test_interp, acc_test_interp, or_test_interp = prepare_test_data(frequencies)
    
    if choice == '1':
        test_ridge(model, gyro_test_interp, acc_test_interp, or_test_interp, frequencies)
    elif choice == '2':
        test_lasso(model, gyro_test_interp, acc_test_interp, or_test_interp, frequencies)
    elif choice == '3':
        test_regression(model, gyro_test_interp, acc_test_interp, or_test_interp, frequencies)

def handle_test_decision(choice, model, frequencies):
    """Handle user decision about testing"""
    if choice != '4':
        test_decision = input("\nDo you want to perform the test? (yes/no): ").lower()
        if test_decision == 'yes':
            execute_test(choice, model, frequencies)

def main():
    # Initial data processing
    frequencies = BiasAndSegmentation.sensors_frequencies()
    gyro, acc, orientation, *_ = BiasAndSegmentation.segmentation_and_bias(frequencies=frequencies, plot_flag=True)
    gyro_interp, acc_interp, or_interp = interpolate_and_visualize(gyro, acc, orientation, frequencies=frequencies, plot_flag=False)
    time_array = calculate_time_array(gyro_interp, acc_interp, or_interp, frequencies)
    
    # User interaction
    print("\nRegression Options:")
    print("1. Ridge Regression\n2. Lasso Regression\n3. Linear Regression\n4. All three regressions")
    choice = input("Enter your choice (1-4): ")
    
    # Initialize variables
    models = {'ridge': None, 'lasso': None, 'linear': None}
    results = {'ridge': None, 'lasso': None, 'linear': None}
    
    # Perform selected regression(s)
    if choice == '1':
        models['ridge'], results['ridge'] = run_ridge_regression(gyro_interp, acc_interp, or_interp, frequencies)
    elif choice == '2':
        models['lasso'], results['lasso'] = run_lasso_regression(gyro_interp, acc_interp, or_interp, frequencies)
    elif choice == '3':
        models['linear'], results['linear'] = run_linear_regression(gyro_interp, acc_interp, or_interp, frequencies)
    elif choice == '4':
        models['ridge'], results['ridge'] = run_ridge_regression(gyro_interp, acc_interp, or_interp, frequencies)
        models['lasso'], results['lasso'] = run_lasso_regression(gyro_interp, acc_interp, or_interp, frequencies)
        models['linear'], results['linear'] = run_linear_regression(gyro_interp, acc_interp, or_interp, frequencies)
    else:
        print("Invalid choice")
        return
    
    # Handle testing
    if choice == '4':
        print("\nPlotting comparison...")
        plt.figure(figsize=(12, 6))
        plt.plot(time_array, results['linear']/100, label='Linear', color='blue')
        plt.plot(time_array, results['ridge'], label='Ridge', color='orange')
        plt.plot(time_array, results['lasso'], label='Lasso', color='green')
        plt.plot(time_array, np.linspace(0, 1, len(time_array)), label='Target', color='red', linestyle='--')
        plt.xlabel('Time (s)')
        plt.ylabel('Percentage (%)')
        plt.title('Regression Comparison')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        current_model = models['ridge'] if choice == '1' else models['lasso'] if choice == '2' else models['linear']
        handle_test_decision(choice, current_model['model'] if choice in ['1', '2'] else current_model, frequencies)

if __name__ == "__main__":
    main()

plt.show(block=True)