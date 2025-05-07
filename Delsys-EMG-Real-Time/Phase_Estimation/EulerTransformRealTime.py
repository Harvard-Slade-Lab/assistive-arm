import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import warnings

# Function to convert quaternions to Euler angles
def quaternion_to_euler(quat_vec):

    if quat_vec == {}:
        return None
    
    # Extract quaternion components
    w = quat_vec[1]['W']
    x = quat_vec[1]['X']
    y = quat_vec[1]['Y']
    z = quat_vec[1]['Z']

        # Ensure all components are non-empty and of the same length
    if not (len(w) == len(x) == len(y) == len(z) > 0):
        return None

    # Convert quaternion components to a numpy array
    quat_array = np.array([x, y, z, w]).T  # Transpose to get shape (n, 4)
    
    # Calculate quaternion magnitudes
    magnitudes = np.linalg.norm(quat_array, axis=1)
    
    # Check for invalid quaternions (magnitude close to zero)
    if np.any(magnitudes < 1e-10):
        warnings.warn("Some quaternions have near-zero magnitude. Returning zeros for these cases.")
        return np.zeros((len(w), 3))  # Return zeros for invalid quaternions
    
    # Normalize the quaternions
    quat_norm = quat_array / magnitudes[:, np.newaxis]
    
    # Create rotation objects
    rotations = R.from_quat(quat_norm)
    
    # Convert to Euler angles (roll, pitch, yaw) in degrees
    euler_angles = rotations.as_euler('xyz', degrees=True)
    
    return euler_angles
