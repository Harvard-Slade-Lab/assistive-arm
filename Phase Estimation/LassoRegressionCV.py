import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score

def enhanced_lasso_regression(gyro_interp, acc_interp, orientation_interp, 
                             alpha_range=None, cv=None, plot=True, frequencies=None):
    """
    Applies Lasso Regression with optimized alpha selection (CV)
    
    Parameters:
    -----------
    gyro_interp : pandas.DataFrame
        Gyroscope data.
    acc_interp : pandas.DataFrame
        Accelerometer data.
    orientation_interp : pandas.DataFrame
        Orientation data.
    alpha_range : tuple (start, stop, num)
        Logarithmic range for alpha values. Default: (-5, 5, 20)
    cv : int
        Number of cross-validation folds. Default: 5
    plot : bool, optional
        Whether to plot the results. Default: True
    frequencies : list
        List of sensor frequencies. Default: None
    """
    
    # Combine and clean data
    X = pd.concat([gyro_interp, acc_interp.iloc[:, :3], orientation_interp], axis=1)
    
    # Extract minimum frequency for plotting
    min_frequency = np.min(frequencies) if frequencies is not None else None
    if min_frequency is None:
        raise ValueError("Frequencies must be provided for plotting.")
    
    # Create target progression (0-100% over time)
    n_samples = len(X)
    y = np.linspace(0, 1, n_samples)
    
    # Configure alpha values
    if alpha_range is None:
        alpha_range = (-5, 5, 20)
    alphas = np.logspace(*alpha_range)
    
    # Create and fit LassoCV model
    model = make_pipeline(
        StandardScaler(),
        LassoCV(alphas=alphas, cv=cv, random_state=42)
    )
    
    model.fit(X, y)
    
    # Get final predictions
    final_pred = model.predict(X)
    
    # Extract LassoCV component
    lasso_cv = model.named_steps['lassocv']
    
    # Calculate metrics
    mse = mean_squared_error(y, final_pred)
    r2 = r2_score(y, final_pred)
    
    # Diagnostic outputs
    print(f"Optimal alpha: {lasso_cv.alpha_:.2e}")
    print(f"Number of selected features: {np.sum(lasso_cv.coef_ != 0)}")
    print(f"Cross-validated MSE: {mse:.4e}")
    print(f"R² Score: {r2:.4f}")
    
    # Plot diagnostics
    if plot:
        plt.figure(figsize=(15, 10))
        
        # 1. Alpha Selection Visualization
        plt.subplot(2, 2, 1)
        plt.semilogx(lasso_cv.alphas_, np.mean(lasso_cv.mse_path_, axis=1))
        plt.axvline(lasso_cv.alpha_, color='r', linestyle='--')
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
        coefficients = lasso_cv.coef_
        importance = np.abs(coefficients)
        sorted_idx = np.argsort(importance)[-15:]
        plt.barh(range(len(sorted_idx)), importance[sorted_idx])
        plt.yticks(range(len(sorted_idx)), X.columns[sorted_idx])
        plt.title('Top 15 Predictive Features')
        
        # 4. Residual Analysis
        plt.subplot(2, 2, 4)
        residuals = (y - final_pred) * 100
        plt.scatter(final_pred*100, residuals, alpha=0.5)
        plt.axhline(0, color='r', linestyle='--')
        plt.title('Residual Analysis')
        plt.xlabel('Predicted Progression (%)')
        plt.ylabel('Residual Error (%)')
        
        plt.tight_layout()
        plt.show()

        # Print regression equation
        print_regression_equation(model.named_steps['lassocv'], X.columns)

    return {
        'model': model,
        'optimal_alpha': lasso_cv.alpha_,
        'selected_features': X.columns[lasso_cv.coef_ != 0],
        'feature_importance': pd.Series(lasso_cv.coef_, index=X.columns),
        'residuals': residuals
    }, final_pred

def print_regression_equation(model, feature_names):
    """
    Prints the regression equation from a fitted Lasso model.
    
    Parameters:
    -----------
    model : sklearn.linear_model.Lasso or sklearn.pipeline.Pipeline
        The fitted Lasso regression model.
    feature_names : list of str
        Names of the features used in the model.
    """
    # If using a pipeline, extract the Lasso model from it
    if hasattr(model, 'named_steps'):
        lasso_model = model.named_steps['lassocv']
    else:
        lasso_model = model
    
    # Extract coefficients and intercept
    coefficients = lasso_model.coef_
    intercept = lasso_model.intercept_
    
    # Build the equation as a string
    equation = f"y = {intercept:.3f}"
    for coef, feature in zip(coefficients, feature_names):
        if coef != 0:  # Only show non-zero coefficients
            equation += f" + ({coef:.3f}) * {feature}"
    
    print("Regression Equation:")
    print(equation)

def test_lasso(model, gyro_interp, acc_interp, orientation_interp, frequencies):
    """
    Test the Lasso regression model with new data.
    
    Parameters:
    -----------
    model : sklearn.pipeline.Pipeline
        The fitted Lasso regression pipeline.
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
    
    # Combine and clean new data
    X_new = pd.concat([gyro_interp, acc_interp.iloc[:, :3], orientation_interp], axis=1)
    
    min_frequency = np.min(frequencies)
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

    # Prediction vs Actual plot
    plt.figure(figsize=(15, 5))
    time_seconds = np.arange(n_samples)/min_frequency
    plt.plot(time_seconds, y*100, label='True')
    plt.plot(time_seconds, y_new*100, label='Predicted')
    plt.title('Temporal Alignment (Test Data)')
    plt.ylabel('Progression (%)')
    plt.legend()
    plt.grid(True)

    return y_new
