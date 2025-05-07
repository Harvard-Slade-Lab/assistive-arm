import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import warnings

# Function to convert quaternions to Euler angles
def quaternion_to_euler(quat_df, frequency):
    # Create a list to store the Euler angles
    euler_angles = []
    
    # Process each row in the DataFrame
    for index, row in quat_df.iterrows():
        # Extract quaternion values (x, y, z, w) - scipy order
        quat = [
            row['ORIENTATION Sensor 1 X'],
            row['ORIENTATION Sensor 1 Y'],
            row['ORIENTATION Sensor 1 Z'],
            row['ORIENTATION Sensor 1 W']
        ]
        
        # Calculate quaternion magnitude
        magnitude = np.sqrt(sum(np.array(quat) ** 2))
        
        # Check if quaternion is valid (not zero)
        if magnitude < 1e-10:  # Very close to zero
            # Add zeros for invalid quaternions
            euler_angles.append([0.0, 0.0, 0.0])
            
        else:
            # Normalize the quaternion
            quat_norm = np.array(quat) / magnitude
            
            # Create a rotation object
            rot = R.from_quat(quat_norm)
            
            # Convert to Euler angles (roll, pitch, yaw) in degrees
            euler = rot.as_euler('xyz', degrees=True)
            
            # Append to list
            euler_angles.append(euler)
    
    # Convert the list to a numpy array
    euler_angles = np.array(euler_angles)
    
    # Create a DataFrame for Euler angles
    euler_df = pd.DataFrame({
        'Roll (X)': euler_angles[:, 0],
        'Pitch (Y)': euler_angles[:, 1],
        'Yaw (Z)': euler_angles[:, 2]
    }, index=quat_df.index)
    
    plotFlag = False
    if plotFlag:
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Calculate time based on frequency
        time = quat_df.index / frequency

        # Plot quaternions
        for col in ['ORIENTATION Sensor 1 W', 'ORIENTATION Sensor 1 X', 'ORIENTATION Sensor 1 Y', 'ORIENTATION Sensor 1 Z']:
            ax1.plot(time, quat_df[col], label=col.split()[-1])
        ax1.set_title('Quaternion Values')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True)

        # Plot Euler angles
        for col in ['Roll (X)', 'Pitch (Y)', 'Yaw (Z)']:
            ax2.plot(time, euler_df[col], label=col)
        ax2.set_title('Euler Angles')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Degrees')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()
    return euler_df