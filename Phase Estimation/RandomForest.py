import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

def enhanced_random_forest_regression(X, y, feature_names, param_grid=None, cv=None, plot=True, frequencies=None):
    """
    Applies Random Forest Regression with optimized hyperparameter selection (GridSearchCV)
    
    Parameters:
    -----------
    X : array-like
        Feature data.
    y : array-like
        Target variable.
    feature_names : list
        Names of the features.
    param_grid : dict
        Parameter grid for GridSearchCV. Default: None (uses default parameters)
    cv : int
        Number of cross-validation folds. Default: None (uses 5-fold CV)
    plot : bool, optional
        Whether to plot the results. Default: True
    frequencies : list
        List of sensor frequencies. Default: None
    """
    
    # Extract minimum frequency for plotting
    min_frequency = np.min(frequencies) if frequencies is not None else None
    if min_frequency is None:
        raise ValueError("Frequencies must be provided for plotting.")
    
    # Set default CV
    if cv is None:
        cv = 5
        
    # Create target progression (0-100% over time)
    n_samples = len(X)
    
    # Configure default parameter grid if not provided
    if param_grid is None:
        param_grid = {
            'randomforestregressor__n_estimators': [50, 100, 200],
            'randomforestregressor__max_depth': [None, 10, 20, 30],
            'randomforestregressor__min_samples_split': [2, 5, 10]
        }
    
    # Create and fit Random Forest model with GridSearchCV
    base_model = make_pipeline(
        StandardScaler(),
        RandomForestRegressor(random_state=42)
    )
    
    grid_search = GridSearchCV(
        base_model, 
        param_grid=param_grid, 
        cv=cv, 
        scoring='neg_mean_squared_error',
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Get final predictions
    final_pred = best_model.predict(X)
    
    # Extract RandomForestRegressor component
    rf_model = best_model.named_steps['randomforestregressor']
    
    # Calculate metrics
    mse = mean_squared_error(y, final_pred)
    r2 = r2_score(y, final_pred)
    
    # Diagnostic outputs
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Cross-validated MSE: {mse:.4e}")
    print(f"R² Score: {r2:.4f}")
    
    # Calculate feature importance
    feature_importance = rf_model.feature_importances_
    
    # Plot diagnostics
    if plot:
        plt.figure(figsize=(15, 10))
        
        # 1. Hyperparameter Tuning Results
        plt.subplot(2, 2, 1)
        cv_results = grid_search.cv_results_
        plt.errorbar(range(len(cv_results['mean_test_score'])), 
                     -cv_results['mean_test_score'], 
                     yerr=cv_results['std_test_score'])
        plt.axvline(np.argmax(cv_results['mean_test_score']), color='r', linestyle='--')
        plt.title('Hyperparameter Tuning Results')
        plt.xlabel('Parameter Combination Index')
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
        sorted_idx = np.argsort(feature_importance)[-15:]
        plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
        plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
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

        # Print feature importance information
        print_feature_importance(rf_model, feature_names)

    return {
        'model': best_model,
        'best_params': grid_search.best_params_,
        'selected_features': feature_names,
        'feature_importance': pd.Series(feature_importance, index=feature_names),
        'residuals': (y - final_pred) * 100,
        'mse': mse
    }, final_pred

def print_feature_importance(model, feature_names):
    """
    Prints feature importance information from a fitted Random Forest model.
    
    Parameters:
    -----------
    model : sklearn.ensemble.RandomForestRegressor
        The fitted Random Forest regression model.
    feature_names : list of str
        Names of the features used in the model.
    """
    # If using a pipeline, extract the RandomForest model from it
    if hasattr(model, 'named_steps'):
        rf_model = model.named_steps['randomforestregressor']
    else:
        rf_model = model
    
    # Extract feature importances
    importances = rf_model.feature_importances_
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    
    # Print feature ranking
    print("Feature Importance Ranking:")
    for i, idx in enumerate(indices[:15]):  # Show top 15 features
        print(f"{i+1}. {feature_names[idx]} ({importances[idx]:.4f})")

def test_random_forest(model, X_new, frequencies, Plot_flag=True):
    """
    Test the Random Forest regression model with new data.
    
    Parameters:
    -----------
    model : sklearn.pipeline.Pipeline
        The fitted Random Forest regression pipeline.
    X_new : array-like
        New feature data for testing.
    frequencies : list
        List of sensor frequencies.
    Plot_flag : bool
        Whether to plot the results. Default: True
    
    Returns:
    --------
    predictions : numpy.ndarray
        Predicted values from the model.
    mse : float
        Mean squared error on test data.
    """
    
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

    if Plot_flag:
        # Prediction vs Actual
        plt.figure(figsize=(15, 5))
        time_seconds = np.arange(n_samples)/min_frequency
        plt.plot(time_seconds, y*100, label='True')
        plt.plot(time_seconds, y_new*100, label='Predicted')
        plt.title(f'Temporal Alignment\nTest MSE: {mse:.4e}, Test R² Score: {r2:.4f}')
        plt.ylabel('Progression (%)')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    return y_new, mse
