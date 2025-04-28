# Author: Haedo Cho
# Purpose: This code runs a simulation using the Moco toolkit in OpenSim.
# The simulation involves a participant performing various activities, such as sit-to-stand or lifting objects.

import numpy as np
import os
import sys
from pathlib import Path
from utils_simulation import load_params
import opensim as osim
from datetime import datetime

from moco_helper_refactor import (
    get_model_refactor,  # Temporary replacement for get_model
    get_tracking_problem,
    set_state_bounds,
    set_moco_problem_weights,
    set_state_tracking_weights,
    set_moco_problem_solver_settings,
    check_parameters_refactor,
    extract_osim_GRF
)

def main():
    # Load configuration parameters from the config file
    params = load_params()
    target_subj = params['subject']
    target_motion = params['motion']
    
    # Create a path for saving results
    params['cur_moco_path'] = os.path.join(params['moco_path'], f"{target_subj}_{target_motion}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(params['cur_moco_path'])
    
    # Define paths for marker and kinematic data
    params['marker_data_path'] = os.path.join(params['opencap_data_path'], target_subj, 'MarkerData', target_motion + '.trc')
    params['kinematic_data_path'] = os.path.join(params['opencap_data_path'], target_subj, 'OpenSimData', 'Kinematics', target_motion + '.mot')
    osim_model_path = os.path.join(params['opencap_data_path'], target_subj, params['osim_model_path'])  # Path to the baseline model
    
    # Check model and assistance parameters
    # check_parameters_refactor(params)

    # Update and initialize the model
    osim_model = get_model_refactor(subject_name=target_subj, model_path=osim_model_path, params=params)
    osim_model.initSystem()

    # Check model option: terminate program if check_model is True
    if params['check_model']:
        sys.exit(0)  # Stops the program after designing the model before tracking optimization

    # Define the tracking problem with the given parameters
    tracking_problem = get_tracking_problem(
        model=osim_model,
        kinematics_path=str(params['kinematic_data_path']),
        t_0=params['t0'],
        t_f=params['tf'],
        mesh_interval=params['mesh_interval'],
        tracking_weight=params['tracking_weight'],
        control_effort_weight=params['control_effort_weight'],
    )

    # Initialize the study for the tracking problem
    study = tracking_problem.initialize()

    # Set weights for different states in the tracking problem
    set_state_tracking_weights(moco_study=study, params=params)
    # set_marker_tracking_weights(moco_study=study, params=params)
    set_moco_problem_weights(model=osim_model, moco_study=study, params=params)
    set_state_bounds(moco_study=study, params=params)
    
    # Set solver settings with a maximum number of iterations
    set_moco_problem_solver_settings(moco_study=study, nb_max_iterations= params['max_iterations'])
    # Access the problem to iterate over goals, using alternative methods if available
    problem = study.updProblem()
    
    # Run the simulation and solve the problem
    solution = study.solve()

    # Unseal and write the solution to the specified path
    solution.unseal()
    solution.write(os.path.join(params['cur_moco_path'], params['osim_model_name'] + '.sto'))        

    if params["osim_GRF"]: # if the moco simulation includes GRF estimation
        extract_osim_GRF(osim_model, solution, params)

    # Check if the solution is successful
    if not solution.success():
        print("Maximum iterations reached:", "solution is unsuccessful but unsealed")
        solution.unseal()

if __name__ == '__main__':
    main()
