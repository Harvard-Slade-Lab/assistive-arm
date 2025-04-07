import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

def enhanced_svr_regression(X, y, kernel='rbf', param_grid=None, cv=5, plot=True, frequencies=None):
    """
    Applies Support Vector Regression with optimized parameter selection via GridSearchCV
    
    Parameters:
    -----------
    X : numpy.ndarray or pandas.DataFrame
        Feature matrix.
    y : numpy.ndarray
        Target values.
    feature_names : list
        Names of the features in X.
    kernel : str, optional
        Kernel type to be used in SVR ('linear', 'poly', 'rbf', 'sigmoid'). Default is 'rbf'.
    param_grid : dict, optional
        Dictionary with parameter names as keys and lists of parameter settings to try.
        If None, a default grid will be used based on the chosen kernel.
    cv : int, optional
        Number of cross-validation folds. Default is 5.
    plot : bool, optional
        Whether to plot the results. Default is True.
    frequencies : list, optional
        List of sensor frequencies. Default is None.
    """
    
    # Extract minimum frequency for plotting:
    min_frequency = np.min(frequencies) if frequencies is not None else None
    if frequencies is not None and min_frequency is None:
        raise ValueError("Frequencies must be provided for plotting.")
    
    # Setup default parameter grid based on kernel
    if param_grid is None:
        if kernel == 'linear':
            param_grid = {
                'svr__C': np.logspace(-3, 3, 7),
                'svr__epsilon': np.logspace(-3, 0, 4)
            }
        else:  # for non-linear kernels
            param_grid = {
                'svr__C': np.logspace(-3, 3, 7),
                'svr__epsilon': np.logspace(-3, 0, 4),
                'svr__gamma': np.logspace(-4, 1, 6)
            }
    
    # Create SVR pipeline with StandardScaler
    pipeline = make_pipeline(
        StandardScaler(),
        SVR(kernel=kernel)
    )
    
    # Set up GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1,
        return_train_score=True
    )
    
    # Fit the model
    grid_search.fit(X, y)
    
    # Get the best estimator and its predictions
    best_model = grid_search.best_estimator_
    final_pred = best_model.predict(X)
    
    # Calculate metrics
    mse = mean_squared_error(y, final_pred)
    r2 = r2_score(y, final_pred)
    
    # Extract SVR parameters from the best model
    svr_model = best_model.named_steps['svr']
    
    # Print results
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Mean Squared Error: {mse:.4e}")
    print(f"R² Score: {r2:.4f}")
    
    # Get residuals for plotting
    residuals = (y - final_pred) * 100
    
    # Plot diagnostics
    if plot:
        n_samples = len(X)
        plt.figure(figsize=(15, 10))
        
        # 1. Parameter Selection Visualization
        plt.subplot(2, 2, 1)
        results = pd.DataFrame(grid_search.cv_results_)
        
        # Plot the mean test scores
        plt.plot(range(len(results)), -results['mean_test_score'])
        plt.xlabel('Parameter Combination Index')
        plt.ylabel('Mean Squared Error')
        plt.axvline(results['mean_test_score'].argmax(), color='r', linestyle='--')
        plt.title('Cross-Validation Performance')
        
        # 2. Prediction vs Actual
        plt.subplot(2, 2, 2)
        if min_frequency:
            time_seconds = np.arange(n_samples)/min_frequency
            plt.plot(time_seconds, y*100, label='True')
            plt.plot(time_seconds, final_pred*100, label='Predicted')
            plt.title('Temporal Alignment')
            plt.xlabel('Time (s)')
        else:
            plt.plot(y*100, label='True')
            plt.plot(final_pred*100, label='Predicted')
            plt.title('Prediction vs Actual')
            plt.xlabel('Sample Index')
        plt.ylabel('Progression (%)')
        plt.legend()
        
        # 3. Support Vectors Visualization
        plt.subplot(2, 2, 3)
        if len(svr_model.support_) < 50:  # Only show if not too many support vectors
            plt.scatter(np.arange(len(X)), y*100, c='b', label='Data Points')
            plt.scatter(svr_model.support_, y[svr_model.support_]*100, c='r', label='Support Vectors')
            plt.title(f'Support Vectors ({len(svr_model.support_)} of {len(X)})')
            plt.xlabel('Sample Index')
            plt.ylabel('Progression (%)')
            plt.legend()
        else:
            plt.text(0.5, 0.5, f"{len(svr_model.support_)} support vectors\nout of {len(X)} samples", 
                    horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            plt.title('Support Vectors Information')
        
        # 4. Residual Analysis
        plt.subplot(2, 2, 4)
        plt.scatter(final_pred*100, residuals, alpha=0.5)
        plt.axhline(0, color='r', linestyle='--')
        plt.title('Residual Analysis')
        plt.xlabel('Predicted Progression (%)')
        plt.ylabel('Residual Error (%)')
        
        plt.tight_layout()
        plt.show()
        
        # Analyze and print SVR model details
        analyze_svr_model(best_model)
    
    # Prepare return values
    return {
        'model': best_model,
        'best_params': grid_search.best_params_,
        'cv_results': grid_search.cv_results_,
        'residuals': residuals,
        'mse': mse
    }, final_pred


def analyze_svr_model(model):
    """
    Analyzes the SVR model and prints its key parameters.
    
    Parameters:
    -----------
    model : sklearn.pipeline.Pipeline
        The fitted SVR model pipeline.
    """
    # Extract the SVR component from the pipeline
    svr_model = model.named_steps['svr']
    
    # Print model parameters
    print("\nSVR Model Analysis:")
    print(f"Kernel: {svr_model.kernel}")
    print(f"C: {svr_model.C:.4e}")
    print(f"Epsilon: {svr_model.epsilon:.4e}")
    
    if svr_model.kernel != 'linear':
        print(f"Gamma: {svr_model.gamma if isinstance(svr_model.gamma, (int, float)) else svr_model.gamma}")
    
    if svr_model.kernel == 'poly':
        print(f"Degree: {svr_model.degree}")
        print(f"Coef0: {svr_model.coef0}")
    
    print(f"Number of support vectors: {len(svr_model.support_)}")
    if len(svr_model.support_) > 5:
        print(f"First 5 support vector indices: {svr_model.support_[:5]}...")
    else:
        print(f"Support vector indices: {svr_model.support_}")

def test_svr(model, X_new, frequencies=None, Plot_flag=True):
    """
    Test the SVR model with new data.
    
    Parameters:
    -----------
    model : sklearn.pipeline.Pipeline
        The fitted SVR model pipeline.
    X_new : numpy.ndarray or pandas.DataFrame
        New feature matrix.
    frequencies : list, optional
        List of sensor frequencies. Default is None.
    
    Returns:
    --------
    predictions : numpy.ndarray
        Predicted values from the model.
    """
    
    min_frequency = np.min(frequencies) if frequencies is not None else None
    
    # Predict using the model
    y_new = model.predict(X_new)

    # Create expected linear progression (0-100% over time)
    n_samples = len(X_new)
    y_expected = np.linspace(0, 1, n_samples)

    # Calculate metrics
    mse = mean_squared_error(y_expected, y_new)
    r2 = r2_score(y_expected, y_new)
    
    print(f"Test Mean Squared Error: {mse:.4e}")
    print(f"Test R² Score: {r2:.4f}")

    if Plot_flag:
        # Prediction vs Expected
        plt.figure(figsize=(15, 5))
        if min_frequency:
            time_seconds = np.arange(n_samples)/min_frequency
            plt.plot(time_seconds, y_expected*100, label='Expected')
            plt.plot(time_seconds, y_new*100, label='Predicted')
            plt.xlabel('Time (s)')
        else:
            plt.plot(y_expected*100, label='Expected')
            plt.plot(y_new*100, label='Predicted')
            plt.xlabel('Sample Index')
        plt.title('Model Predictions on New Data')
        plt.ylabel('Progression (%)')
        plt.legend()
        plt.grid(True)
        plt.show()

    return y_new, mse
