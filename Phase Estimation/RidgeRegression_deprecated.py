import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import learning_curve

def ridge_regression(gyro_interp, acc_interp, orientation_interp, alphas=None, plot=True):
    """
    Perform Ridge regression on the given sensor data.
    
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
    
    Returns:
    --------
    dict
        A dictionary containing model, predictions, and performance metrics.
    """
    
    # Default alphas
    if alphas is None:
        alphas = [0.1, 1.0, 10.0, 100.0]
    
    # Combine all features
    X = pd.concat([gyro_interp, acc_interp, orientation_interp], axis=1)
    
    # Remove columns with any NaN values
    X = X.dropna(axis=1)
    
    # Create target variable: linear progression from 0 to 100%
    n_samples = len(X)
    y = np.linspace(0, 1, n_samples)  # 0 to 1 represents 0% to 100%
    
    # Create and train RidgeCV model
    model = RidgeCV(alphas=alphas, cv=None, store_cv_results=True)
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # Print results
    print(f"Selected alpha: {model.alpha_}")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    
    # Plot results if requested
    if plot:
        plot_regression_results(model, X, y, y_pred, alphas, n_samples)
    
    # Return the model, predictions, target, metrics, and duration
    return {
        'model': model,
        'y_pred': y_pred,
        'y_true': y,
        'mse': mse,
        'r2': r2,
        'duration_seconds': n_samples / 519,
        'selected_alpha': model.alpha_
    }

def plot_regression_results(model, X, y, y_pred, alphas, n_samples):
    """Plot the main regression results and additional analysis."""
    time_seconds = np.arange(n_samples) / 519  # Convert to seconds
    
    # Plot 1: True vs Predicted Progression
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time_seconds, y * 100, 'b-', label='True')  # Convert to 0-100%
    plt.plot(time_seconds, y_pred * 100, 'r-', label='Predicted')  # Convert to 0-100%
    plt.xlabel('Time (seconds)')
    plt.ylabel('Progression (%)')
    plt.title('True vs Predicted Progression')
    plt.legend()
    
    # Plot 2: Residuals
    plt.subplot(2, 1, 2)
    residuals = y - y_pred
    plt.plot(time_seconds, residuals * 100, 'g-')  # Convert to 0-100%
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Residuals (%)')
    plt.title('Residuals')
    
    plt.tight_layout()
    plt.show()
    
    # Additional analysis plots
    plot_feature_importance(model, X.columns)
    plot_cv_scores(model, alphas)
    plot_learning_curve(model, X, y)

def plot_feature_importance(model, feature_names):
    """Plot the feature importances based on model coefficients."""
    coefficients = model.coef_
    
    # Sort feature names by absolute coefficient values (descending order)
    importance = np.abs(coefficients)
    sorted_idx = np.flip(np.argsort(importance))
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(sorted_idx)), importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('Absolute Coefficient Value')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()

def plot_cv_scores(model, alphas):
    """Plot cross-validation scores for different alpha values."""
    plt.figure(figsize=(10, 6))
    cv_scores = np.mean(model.cv_values_, axis=0)
    plt.semilogx(alphas, cv_scores, marker='o')
    plt.axvline(x=model.alpha_, color='r', linestyle='--', 
                label=f'Selected alpha: {model.alpha_}')
    plt.xlabel('Alpha')
    plt.ylabel('Mean Squared Error (CV)')
    plt.title('Cross-validation Scores for Different Alphas')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_learning_curve(model, X, y):
    """Plot the learning curve to analyze training data requirements."""
    # Create a Ridge model with the selected alpha
    ridge_model = Ridge(alpha=model.alpha_)
    
    train_sizes, train_scores, test_scores = learning_curve(
        ridge_model, X, y, cv=None, scoring='neg_mean_squared_error', 
        train_sizes=np.linspace(0.1, 1.0, 5)
    )
    
    train_scores_mean = -np.mean(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', 
             label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', 
             label='Cross-validation score')
    plt.xlabel('Training Examples')
    plt.ylabel('Mean Squared Error')
    plt.title('Learning Curve')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
