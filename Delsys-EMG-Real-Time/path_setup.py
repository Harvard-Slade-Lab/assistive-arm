import sys
import os

# Get the absolute path to this script's directory
project_root = os.path.dirname(os.path.abspath(__file__))

# Define the path to Phase_Estimation relative to the project root
phase_estimation_path = os.path.join(project_root, "Phase_Estimation")

# Add to sys.path if not already included
if phase_estimation_path not in sys.path:
    sys.path.insert(0, phase_estimation_path)
