
# Assistive Arm Simulation

This project is designed to simulate movements and interactions with an assistive arm. It utilizes the OpenSim API to model and analyze different biomechanical scenarios.

**Last Updated:** February 26th, 2025  
**Editor:** Haedo Cho

## Installation Guide

### Setting Up the Python Scripting Environment

The quickest way to start using OpenSim for Python scripting is through the pre-built Conda packages. This method is recommended, particularly for those new to Python.

1. **Install Conda**: Follow the [Conda user guide](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html) to install Conda and create a Python environment.

2. **Create and Activate a Conda Environment**: 
   
   - Open a terminal (or Anaconda prompt) and run:

     ```bash
     conda create -n assistive_arm python=3.11 numpy
     conda activate assistive_arm
     ```

   - Note: The environment name `assistive_arm` can be changed to any name you prefer.

3. **For Mac Systems (Arm64 processors like M1, M2, etc.)**

   - Run the following command to install the latest version of `libcxx`, which is needed for OpenSim installation:

     ```bash
     conda install conda-forge::libcxx
     ```

4. **Install OpenSim**

   - Install the latest compatible OpenSim version with:

     ```bash
     conda install -c opensim-org opensim
     ```

   - To install a specific version, visit [Anaconda.org](https://anaconda.org) and find the package with the desired version. For example, to install OpenSim 4.4 in a Python 3.10 environment, use:

     ```bash
     conda install -c opensim-org opensim=4.4=py310np121
     ```

5. **Install OpenSim with Moco (Windows only)**

   - OpenSim Conda packages with Moco are available for versions 4.4, 4.4.1, and 4.5. Install these packages using:

     ```bash
     conda install -c opensim-org opensim-moco
     ```

### Recommendation

For the best experience and access to the latest OpenSim features, use the Conda packages for the latest OpenSim version, which includes Moco and other updates.

## Code Workflow

This Python script is designed to simulate biomechanical movements using OpenSim. It outlines several critical steps for setting up the simulation environment. Below is a detailed breakdown of the script's components:

### Imports and Setup

- **Imports**: The script begins by importing core libraries such as `os`, `yaml`, `sys`, `datetime`, and `Path`. Additionally, it imports specific helper functions from a module named `moco_helpers`. These functions assist with model setup, problem configuration, and simulation tasks.
- **Path Configuration**: The script extends the system path to include the parent directory, enabling the import of modules from higher directory levels.

### Main Function

The main function organizes the simulation setup process into several key steps:

1. **Session Parameters**: Defines session-specific parameters such as `subject` (participant ID), `task` (e.g., standing or stairs), date, assistance level, trial ID, and model type.
   
2. **Assistive Force Parameters**:
   - Determines if the simulation involves assistive forces.
   - Configures external forces and reserve pelvis settings based on these choices.

3. **Simulation Parameters**:
   - Defines the time resolution (`mesh_interval`) for the tracking problem.
   - Configures the computation and use of ground reaction forces (GRF).

4. **Effort and Tracking Parameters**:
   - Configures weights and values for tracking and control optimization, affecting how the simulation prioritizes different movements and forces.

5. **Configuration Paths**:
   - Establishes file paths for necessary data such as models, GRF data, and session metadata.
   - Constructs names and paths for output files based on current configurations and timing.

6. **Loading and Checking Data**:
   - Loads simulation metadata from stored YAML configuration files.
   - Verifies that simulation settings align with expected configurations to ensure consistency and accuracy.

7. **Model Setup**:
   - Initializes a biomechanical model using provided parameters and the `get_model` helper function.
   - Sets up the tracking problem to define the dynamics being simulated, specifying the movements and interactions to model.

8. **Simulation Execution**:
   - Utilizes helper functions like `set_moco_problem_weights` and `set_moco_problem_solver_settings` to configure weights and solver settings.
   - Saves the settings used for this simulation in a configuration YAML file.
   - Solves the configured simulation to produce results, tackling any issues if the solution isn't immediately successful.

9. **Output and Visualization**:
   - Outputs the simulation results to files for further analysis.
   - Launches visualization tools to interactively explore the simulation results.

### Execution

- **Main Execution Block**: The script checks if it's being executed as the main module and, if so, calls the `main()` function to begin the process.

This code is particularly useful for researchers and developers in biomechanics who need to set up, execute, and analyze complex simulations of human movement with assistive technologies. Designed for flexibility, it allows researchers to easily modify configurations to test various hypotheses or experiment conditions.