import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score

def enhanced_ridge_regression(X,y, feature_names, alpha_range=None, cv=None, plot=True, frequencies=None):
    """
    Applies Ridge Regression with optimized alpha selection (CV)
    
    Parameters:
    -----------
    gyro_interp : pandas.DataFrame
        Gyroscope data.
    acc_interp : pandas.DataFrame
        Accelerometer data.
    orientation_interp : pandas.DataFrame
        Orientation data.
    alphas : list, optional
        List of alpha values to try. If None, defaults to [0.1, 1.0, 10.0, 100.0].
    plot : bool, optional
        Whether to plot the results. Default is True.
    alpha_range : tuple (start, stop, num)
        Logarithmic range for alpha values. Default: (-5, 5, 20)
    cv : int
        Number of cross-validation folds. Default: 5
    frequencies : list
        List of sensor frequencies. Default: None
    """

    # Extract minimum frequency for plotting:
    min_frequency = np.min(frequencies) if frequencies is not None else print("No frequencies provided.")
    if min_frequency is None:
        raise ValueError("Frequencies must be provided for plotting.")

    
    # Create target progression (0-100% over time)
    n_samples = len(X)
    
    
    # Configure alpha values
    if alpha_range is None:
        alpha_range = (-5, 5, 20)
    alphas = np.logspace(*alpha_range)
    
    # Apply RidgeRegression with cross-validation
    model = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=alphas, cv=cv, store_cv_results=True)
        )
        
    
    # Fit model with cross-validation
    # y_pred = cross_val_predict(model, X, y, cv=cv)
    model.fit(X, y)
    
    # Get final predictions
    final_pred = model.predict(X)
    

    # Extract RidgeCV component
    ridge_cv = model.named_steps['ridgecv']
    
    # Calculate metrics
    mse = mean_squared_error(y, final_pred)
    r2 = r2_score(y, final_pred)
    
    # Diagnostic outputs
    print(f"Optimal alpha: {ridge_cv.alpha_:.2e}")
    print(f"Cross-validated MSE: {mse:.4e}")
    print(f"R² Score: {r2:.4f}")
    residuals = (y - final_pred) * 100

    # Plot diagnostics
    if plot:
        plt.figure(figsize=(15, 10))
        
        # 1. Alpha Selection Visualization
        plt.subplot(2, 2, 1)
        plt.semilogx(alphas, np.mean(ridge_cv.cv_results_, axis=0))
        plt.axvline(ridge_cv.alpha_, color='r', linestyle='--')
        plt.title('Alpha Selection Analysis')
        plt.xlabel('Alpha (log scale)')
        plt.ylabel('Validation MSE')
        
        # 2. Prediction vs Actual
        plt.subplot(2, 2, 2)
        time_seconds = np.arange(n_samples)/min_frequency
        plt.plot(time_seconds, y*100, label='True')
        plt.plot(time_seconds, final_pred*100, label='Predicted')
        plt.title('Temporal Alignment')
        plt.ylabel('Progression (%)')
        plt.legend()
        
        # 3. Feature Importance
        plt.subplot(2, 2, 3)
        coefficients = ridge_cv.coef_
        importance = np.abs(coefficients)
        sorted_idx = np.argsort(importance)[-15:]
        plt.barh(range(len(sorted_idx)), importance[sorted_idx])
        plt.yticks(range(len(sorted_idx)), feature_names)
        plt.title('Top 15 Predictive Features')
        
        # 4. Residual Analysis
        plt.subplot(2, 2, 4)

        plt.scatter(final_pred*100, residuals, alpha=0.5)
        plt.axhline(0, color='r', linestyle='--')
        plt.title('Residual Analysis')
        plt.xlabel('Predicted Progression (%)')
        plt.ylabel('Residual Error (%)')
        
        plt.tight_layout()
        plt.show()

        # Print regression equation
        print_regression_equation(model.named_steps['ridgecv'], feature_names)

    return {
        'model': model,
        'optimal_alpha': ridge_cv.alpha_,
        'cv_values': ridge_cv.cv_results_,
        'feature_importance': pd.Series(ridge_cv.coef_, index=feature_names),
        'residuals': residuals,
        'mse': mse
        
    }, final_pred

def print_regression_equation(model, feature_names):
    """
    Prints the regression equation from a fitted Ridge regression model.
    
    Parameters:
    -----------
    model : sklearn.linear_model.Ridge or sklearn.pipeline.Pipeline
        The fitted Ridge regression model.
    feature_names : list of str
        Names of the features used in the model.
    """
    # If using a pipeline, extract the Ridge model from it
    if hasattr(model, 'named_steps'):
        ridge_model = model.named_steps['ridge']
    else:
        ridge_model = model
    
    # Extract coefficients and intercept
    coefficients = ridge_model.coef_
    intercept = ridge_model.intercept_
    
    # Build the equation as a string
    equation = f"y = {intercept:.3f}"
    for coef, feature in zip(coefficients, feature_names):
        equation += f" + ({coef:.3f}) * {feature}"
    
    print("Regression Equation:")
    print(equation)

def test_ridge(model, X_new, frequencies, Plot_flag=True):
    """
    Test the Ridge regression model with new data.
    
    Parameters:
    -----------
    model : sklearn.linear_model.Ridge or sklearn.pipeline.Pipeline
        The fitted Ridge regression model.
    gyro_interp : pandas.DataFrame
        New gyroscope data.
    acc_interp : pandas.DataFrame
        New accelerometer data.
    orientation_interp : pandas.DataFrame
        New orientation data.
    frequencies : list
        List of sensor frequencies.
    
    Returns:
    --------
    predictions : numpy.ndarray
        Predicted values from the model.
    """
    

    min_frequency = np.min(frequencies) if frequencies is not None else print("No frequencies provided.")
    if min_frequency is None:
        raise ValueError("Frequencies must be provided for plotting.")
    
    # Predict using the model
    y_new = model.predict(X_new)

    # Create target progression (0-100% over time)
    n_samples = len(X_new)
    y = np.linspace(0, 1, n_samples)

    # Calculate metrics
    mse = mean_squared_error(y, y_new)
    r2 = r2_score(y, y_new)

    if Plot_flag:
        # 2. Prediction vs Actual
        plt.figure(figsize=(15, 5))
        time_seconds = np.arange(n_samples)/min_frequency
        plt.plot(time_seconds, y*100, label='True')
        plt.plot(time_seconds, y_new*100, label='Predicted')
        plt.title(f'Temporal Alignment\nTest MSE: {mse:.4e}, Test R² Score: {r2:.4f}')
        plt.ylabel('Progression (%)')
        plt.legend()
        plt.grid(True)
    return y_new, mse