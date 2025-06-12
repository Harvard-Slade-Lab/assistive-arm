# OpenSim Moco: 2D Walking Simulation

- This repository provides a Python-based implementation of an optimal control problem for simulating 2D walking using OpenSim Moco. 
- The original source code, written in MATLAB, can be found [here](https://opensim-org.github.io/opensim-moco-site/docs/1.1.0/html_user/example2DWalking_8m-example.html). 
- The script includes two key optimal control scenarios: a tracking simulation and a predictive simulation. 
- This repository serves as a great introduction to understanding the capabilities of OpenSim Moco.

## Overview

This Python script:
- Executes a 2D walking simulation using OpenSim Moco.
- Handles two distinct optimal control problems:
  1. **Tracking Simulation**: Aims to simulate walking by minimizing the difference between provided and simulated coordinates and speeds, while also minimizing the effort.
  2. **Predictive Simulation**: Predicts walking dynamics by prioritizing effort minimization over distance traveled.

The approach is inspired by the work of Falisse A, Serrancoli G, Dembia C, Gillis J, and De Groote F in their 2019 publication, "Algorithmic differentiation improves the computational efficiency of OpenSim-based trajectory optimization of human movement" in PLOS One.

## Model

The model employed in this simulation is a modified version of `gait10dof18musc.osim` available in OpenSim:
- Converted the knee flexion axis from moving to fixed.
- Replaced Millard2012EquilibriumMuscles with DeGrooteFregly2016Muscles.
- Added two `SmoothSphereHalfSpaceForces` per foot to represent foot-ground contact.

**Note:** This model is not suitable for research purposes due to inaccuracies in the gastroc muscle path—it incorrectly fails to cross the knee joint.

## Data

The reference coordinate data originates from predictive simulations conducted by Falisse et al. (2019) and may slightly differ from standard experimental gait data.

## How to Use

### Prerequisites

Ensure OpenSim and any necessary dependencies are installed in your environment.

### Running the Script

1. **Tracking Simulation**: Run `gaitTracking()` to simulate walking using tracking. The objective is to minimize discrepancies in coordinates, speeds, and ground reaction forces (GRFs), alongside the effort cost.

2. **Predictive Simulation**: After obtaining a tracking solution, run `gaitPrediction()` to forecast future walking dynamics by minimizing effort across the traveled distance.

Both simulations incorporate limb and muscle activation symmetry to produce a realistic periodic gait cycle.

### Files

- `2D_gait.osim`: The OpenSim model utilized in the simulations.
- `referenceCoordinates.sto`: Reference coordinate data for the tracking problem.
- `referenceGRF.xml`: Reference ground reaction force data for optional GRF tracking.
- `gaitTracking_solution_fullcycle.sto`: Output file for the full gait cycle solution from the tracking simulation.
- `gaitTracking_solutionGRF_fullcycle.sto`: Ground reaction forces from the full cycle solution.
- `trackingSolution.sto`: Intermediate file for the tracking solution, used in predictive simulation.
- `gaitPrediction_solution_fullcycle_[identifier].sto`: Output file for the full gait cycle solution from the predictive simulation.
- `gaitPrediction_solutionGRF_fullcycle_[identifier].sto`: Ground reaction forces from the predictive solution.

### Visualization

The script includes commented-out statements for enabling visualization via OpenSim. To visually examine the simulation, uncomment `study.visualize(full)`.

## Code Structure

- **Step 1: Problem Definition**: Articulate the optimal control problems.
- **Step 2: Goal Setting**: Define tracking and symmetry goals.
- **Step 3: Set Bounds**: Specify bounds for joint angles and state variables.
- **Step 4: Solver Configuration**: Set solver parameters, including tolerances and optimization settings.
- **Step 5: Problem Solving**: Address both tracking and prediction problems.
- **Step 6: Extraction**: Retrieve and store ground reaction forces.
- **Step 7: Visualization**: (Optional) Visualize solutions.

## License

This code is distributed as-is without warranties. Users assume responsibility for any results or research conducted with this simulation code.

## Acknowledgments

This simulation setup is inspired by Falisse A et al.'s work, highlighting the benefits of algorithmic differentiation in OpenSim-based trajectory optimization. Special thanks to the OpenSim community for providing a robust platform for simulating complex human movements.