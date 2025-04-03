import matplotlib.pyplot as plt
from tkinter.filedialog import askdirectory
from Interpolation import interpolate_and_visualize
import DataLoader
import MatrixCreator
import RidgeRegressionCV
import LassoRegressionCV
import Linear_Reg
import TestManager
import SVR_Reg

# Select folder
folder_path = DataLoader.select_folder()

if not folder_path:
    print("No folder selected. Exiting...")
    
print(f"Selected folder: {folder_path}")

try:
    # Load and process files
    acc_data, gyro_data, or_data, acc_files, gyro_files, or_files = DataLoader.load_and_process_files(folder_path)
    print(f"Loaded {len(acc_files)} ACC files, {len(gyro_files)} GYRO files, and {len(or_files)} OR files")
    
    # Group files by timestamp
    grouped_indices = DataLoader.group_files_by_timestamp(acc_files, gyro_files, or_files)
    print(f"Found {len(grouped_indices)} complete data sets")
    if not grouped_indices:
        print("No complete data sets found. Exiting...")
        
    # Create X and Y matrices
    X, Y, timestamps, segment_lengths, feature_names, frequencies = MatrixCreator.create_matrices(acc_data, gyro_data, or_data, grouped_indices, biasPlot_flag=False, interpPlot_flag=False)
    print(f"Created X matrix with shape {X.shape} and Y matrix with length {len(Y)}")
    
    # Print column information
    print("\nColumn information:")
    for i, name in enumerate(feature_names):
        print(f"Column {i}: {name}")
    
    # Visualize matrices
    MatrixCreator.visualize_matrices(X, Y, timestamps, segment_lengths, feature_names)
    
    # User interaction
    print("\nRegression Options:")
    print("1. Ridge Regression\n2. Lasso Regression\n3. Linear Regression\n4. All regression\n5. SVR Regression")
    choice = input("Enter your choice (1-5): ")
    
    # Initialize variables
    models = {'ridge': None, 'lasso': None, 'linear': None}
    results = {'ridge': None, 'lasso': None, 'linear': None}
    
    # Perform selected regression(s)
    if choice == '1':
        ridge_result, y_ridge = RidgeRegressionCV.enhanced_ridge_regression(X,Y,feature_names,alpha_range=(-7, 7, 40), cv=None, plot=True, frequencies=frequencies)
    elif choice == '2':
        lasso_result, y_lasso = LassoRegressionCV.enhanced_lasso_regression(X,Y,feature_names,alpha_range=(-7, 7, 40), cv=None, plot=True, frequencies=frequencies)
    elif choice == '3':
        linear_model, y_linear, _ = Linear_Reg.linear_regression(X,Y, frequencies, feature_names=feature_names,  plot=True)
    elif choice == '4':
        ridge_result, y_ridge = RidgeRegressionCV.enhanced_ridge_regression(X,Y,feature_names,alpha_range=(-7, 7, 40), cv=None, plot=True, frequencies=frequencies)
        lasso_result, y_lasso = LassoRegressionCV.enhanced_lasso_regression(X,Y,feature_names,alpha_range=(-7, 7, 40), cv=None, plot=True, frequencies=frequencies)
        linear_model, y_linear, _ = Linear_Reg.linear_regression(X,Y, frequencies, feature_names=feature_names, plot=True)
    elif choice == '5':
        svr_model, y_svr = SVR_Reg.enhanced_svr_regression(X,Y, kernel='rbf',  plot=True, frequencies=frequencies)
    else:
        print("Invalid choice")
    
        # Handle testing
    if choice != '4':
        current_model = ridge_result['model'] if choice == '1' else lasso_result['model'] if choice == '2' else svr_model['model'] if choice == '5' else linear_model
        TestManager.handle_test_decision(choice, current_model, frequencies)
   
   
    plt.show(block=True)
    
except Exception as e:
    print(f"An error occurred: {e}")
