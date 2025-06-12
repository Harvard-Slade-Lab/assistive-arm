# Author: Haedo Cho
# Purpose: Run simulation using OpenSim Moco for activities such as sit-to-stand or object lifting.

import os
import sys
import numpy as np
from datetime import datetime

import opensim as osim

from utils_simulation import load_params
from moco_helper import (
    get_model,
    get_tracking_problem,
    set_state_bounds,
    set_control_input_weights,
    set_control_input_bound,
    set_state_tracking_weights,
    set_moco_problem_solver_settings,
    set_optimal_forces,
    extract_osim_GRF,
    print_optimal_force_values,
    print_control_effort_weight
)

def main():
    # Load parameters from config
    params = load_params()
    subj = params['subject']
    motion = params['motion']

    # Print note of current simulation
    print(f"#----------- Notes of current sim: {params['note']} -----------#\n")

    # Set up output directory
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    moco_output_path = os.path.join(params['moco_path'], f"{subj}_{motion}_{current_time}")
    os.makedirs(moco_output_path)
    params['cur_moco_path'] = moco_output_path

    # Define kinematics and marker data paths
    subject_dir = os.path.join(params['opencap_data_path'], subj)
    params['marker_data_path'] = os.path.join(subject_dir, 'MarkerData', f'{motion}.trc')
    params['kinematic_data_path'] = os.path.join(subject_dir, 'OpenSimData', 'Kinematics', f'{motion}.mot')
    osim_model_path = os.path.join(subject_dir, params['osim_model_path'])

    # Load and initialize the model
    model = get_model(subject_name=subj, model_path=osim_model_path, params=params)
    model.initSystem()

    # Print out all the optimal force values
    print_optimal_force_values(model= model)

    # Optional: model check only, skip optimization
    if params.get('check_model', False):
        print("Model check enabled: exiting before tracking optimization.")
        sys.exit(0)

    # Set up tracking problem from kinematics
    tracking_problem = get_tracking_problem(model=model, kinematics_path=params['kinematic_data_path'], params=params)
    study = tracking_problem.initialize()

    # Configure problem weights and bounds
    set_state_bounds(moco_study=study, params=params)
    set_state_tracking_weights(moco_study=study, params=params) # this is for tracking kinematics
    set_control_input_weights(model=model, moco_study=study, params=params) # this is for control effort
    set_control_input_bound(model=model, moco_study=study, params=params)
    set_moco_problem_solver_settings(moco_study=study, params=params, nb_max_iterations=params['max_iterations'])

    # Solve the problem
    solution = study.solve()

    if not solution.success():
        print("Moco solution was not successful (max iterations reached).")
    else:
        print("Moco problem solved successfully.")

    # Save the solution regardless of success (unsealed state supported)
    solution.unseal()
    solution_path = os.path.join(params['cur_moco_path'], f"{params['osim_model_name']}.sto")
    solution.write(solution_path)
    print(f"Solution written to: {solution_path}")

    # Extract GRFs if specified
    if params.get("add_contact_dynamics", False):
        extract_osim_GRF(model, solution, params)

if __name__ == '__main__':
    main()