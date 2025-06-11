import numpy as np
import opensim as osim
import xml.etree.ElementTree as ET
import sys
from scipy import signal
from pathlib import Path
import os
import pandas as pd
from data_preprocessing import prepare_opencap_markers
from datetime import datetime
import json

def get_model(subject_name, model_path, params):

    osim_model, subject_name = getMuscleDrivenModel(subject_name = subject_name, 
                                                    model_path = model_path, 
                                                    params = params)

    if params["add_contact_dynamics"] and 'sit' in params["motion"].lower():
        generate_model_with_contacts(model_name=subject_name, model=osim_model, params=params, contact_side='all')

    # If robot assistive force
    if params["robot_assistance"]:
        add_assistive_force(
            target = params["robot_assistance_target_body"],
            offset= None,
            model=osim_model,
            name="assistive_force_x",
            direction=osim.Vec3(1, 0, 0),
            magnitude=params["opt_force_robot_assistance"],
        )
        add_assistive_force(
            target = params["robot_assistance_target_body"],
            offset = None,
            model=osim_model,
            name="assistive_force_y",
            direction=osim.Vec3(0, 1, 0),
            magnitude=params["opt_force_robot_assistance"],
        )

    model_name = f"{subject_name}_{params['motion']}"
    osim_model.setName(model_name)
    osim_model.finalizeConnections()

    # Save the model under "saved_osim_model_path"
    params['osim_model_name'] = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    saved_osim_model_path = os.path.join(params['cur_moco_path'], params['osim_model_name'] + '.osim')
    osim_model.printToXML(saved_osim_model_path)

    # Save params dictionary
    dictionary_path = os.path.join(params['cur_moco_path'], params['osim_model_name'] + '.json')
    with open(dictionary_path, 'w') as file:
        json.dump(params, file, indent=4, cls=NumpyEncoder)    

    return osim_model

def getMuscleDrivenModel(subject_name, model_path, params):

    # Load the OpenSim model from file
    osim_model = osim.Model(model_path)

    # Remove specified muscles/forces based on parameters
    simplify_model(osim_model, params)

    # Set optimal force for all coordinate actuators (e.g., upper limb actuators)
    actuators = osim_model.getActuators()
    for i in range(actuators.getSize()):
        actuator = actuators.get(i)
        if actuator.getConcreteClassName() == 'CoordinateActuator':
            coord_actuator = osim.CoordinateActuator.safeDownCast(actuator)
            coord_actuator.setOptimalForce(params["opt_force_coord_actuator"])
            
    # Build model processing pipeline
    model_processor = osim.ModelProcessor(osim_model)
    model_processor.append(osim.ModOpIgnoreTendonCompliance())                        # Remove tendon compliance
    model_processor.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())          # Use De Groote-Fregly 2016 muscle model
    model_processor.append(osim.ModOpIgnorePassiveFiberForcesDGF())                   # Remove passive fiber forces in DeGroote-Fregly muscles
    model_processor.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5))         # Scale width of active fiber force curve
    model_processor.append(osim.ModOpAddReserves(params["opt_force_reserve_actuator"])) # Add reserve actuators

    # Process and return
    processed_model = model_processor.process()
    return processed_model, subject_name

def simplify_model(model: osim.Model, params):
    # Simplify model by reducing actuators and muscles    
    force_set = model.upd_ForceSet()
    muscles_to_remove = params['muscles_to_remove']

    for cur_muscle in muscles_to_remove:
        cur_muscle_l = cur_muscle + '_l'
        cur_muscle_r = cur_muscle + '_r'
        force_set.remove(force_set.getIndex(cur_muscle_l))
        force_set.remove(force_set.getIndex(cur_muscle_r))

def generate_model_with_contacts(model_name, model, params, contact_side=None):
    
    floor_contact_loc = get_floor_ref_markers(markers_path = params["marker_data_path"], params = params)
    chair_contact_loc = get_chair_ref_markers(markers_path = params["marker_data_path"], params = params)
    chair_contact_loc[1] = floor_contact_loc[1] + params['chair_height'] # y-coordinate # floor loc + chair height

    # np.pi/2 rotation is for matching XY plane of contact half space to the ground
    reference_contact_half_space = {"name": "floor", 
                                    "location": floor_contact_loc,
                                    "orientation": np.array([0, 0, -np.pi/2]), 
                                    "frame": "ground"}

    reference_contact_half_space_chair = {"name": "chair", 
                                          "location": chair_contact_loc,
                                          "orientation": np.array([0, 0, -np.pi/2]), 
                                          "frame": "ground"}

    contact_half_space_frame = model.get_ground()
    contact_half_space_floor = add_contact_half_space(model, reference_contact_half_space, contact_half_space_frame)
    contact_half_space_chair = add_contact_half_space(model, reference_contact_half_space_chair, contact_half_space_frame)

    ## --- add foot contact spheres --- ##
    foot_contact_spheres_dict = params['foot_contact_spheres']
    for cur_contact_spheres in foot_contact_spheres_dict:
        cur_contact_spheres_info = foot_contact_spheres_dict[cur_contact_spheres]
        add_contact_spheres_and_forces(model = model, 
                                       sphere_name = cur_contact_spheres, 
                                       radius = cur_contact_spheres_info["radius"],
                                       relative_distance = cur_contact_spheres_info["location"], 
                                       reference_socketframe = cur_contact_spheres_info["socket_frame"], 
                                       reference_contact_space = contact_half_space_floor, 
                                       params = params)

    ## --- add hip contact spheres --- ##
    hip_contact_spheres_dict = params['hip_contact_spheres']
    for cur_contact_spheres in hip_contact_spheres_dict:
        cur_contact_spheres_info = hip_contact_spheres_dict[cur_contact_spheres]
        add_contact_spheres_and_forces(model = model, 
                                       sphere_name = cur_contact_spheres, 
                                       radius = cur_contact_spheres_info["radius"],
                                       relative_distance = cur_contact_spheres_info["location"], 
                                       reference_socketframe = cur_contact_spheres_info["socket_frame"], 
                                       reference_contact_space = contact_half_space_chair, 
                                       params = params)

def add_contact_spheres_and_forces(model, sphere_name, radius, relative_distance, reference_socketframe, reference_contact_space, params):

    bodySet = model.get_BodySet() # Retrieves all bodies in the model, which will be used to attach the contact geometries.
    body = bodySet.get(reference_socketframe)

    # Create and attach the ContactSphere
    c_contactSphere = osim.ContactSphere(radius,
                                         osim.Vec3(relative_distance), 
                                         body, 
                                         sphere_name)
    c_contactSphere.connectSocket_frame(body)
    contact_geometry_set = model.getContactGeometrySet()
    model.addContactGeometry(c_contactSphere)

    # Create and configure the SmoothSphereHalfSpaceForce
    smooth_force = osim.SmoothSphereHalfSpaceForce("SmoothSphereHalfSpaceForce_" + sphere_name + "_" + reference_contact_space.getName(), 
                                                    c_contactSphere, 
                                                    reference_contact_space)
    smooth_force.set_stiffness(params['stiffness'])
    smooth_force.set_dissipation(params['dissipation'])
    smooth_force.set_static_friction(params['static_friction'])
    smooth_force.set_dynamic_friction(params['dynamic_friction'])
    smooth_force.set_viscous_friction(params['viscous_friction'])
    smooth_force.set_transition_velocity(params['transition_velocity'])
    
    # Connect the force's sockets
    smooth_force.connectSocket_half_space(reference_contact_space)
    smooth_force.connectSocket_sphere(c_contactSphere)
    
    # Add the force to the model
    model.addForce(smooth_force)

def add_contact_half_space(model, contact_def, frame):
    contact_half_space = osim.ContactHalfSpace(
        osim.Vec3(contact_def["location"]),
        osim.Vec3(contact_def["orientation"]),
        frame, 
        contact_def["name"]
    )
    contact_half_space.connectSocket_frame(frame)
    model.addContactGeometry(contact_half_space)

    return contact_half_space

def add_assistive_force(target, offset, model, name, direction, magnitude):
    assistActuator = osim.PointActuator(target)
    if offset is not None:
        assistActuator.set_point(offset)
    assistActuator.setName(name)
    assistActuator.set_force_is_global(True)
    assistActuator.set_direction(direction)
    assistActuator.setOptimalForce(magnitude) 
    model.addForce(assistActuator)

# def generate_model_with_contacts_incline(model_name, model, params, contact_side=None):
    
#     floor_contact_loc = get_floor_ref_markers(markers_path = params["marker_data_path"], params = params)
#     contact_half_space_frame = model.get_ground()

#     #     # np.pi/2 rotation is for matching XY plane of contact half space to the ground
#     reference_contact_half_space = {"name": f"incline_floor", 
#                             "location": floor_contact_loc,
#                             "orientation": np.array([0, 0, -np.pi/2 + np.deg2rad(params['incline_deg'])]), 
#                             "frame": "ground"}
#     contact_half_space_floor = add_contact_half_space(model, reference_contact_half_space, contact_half_space_frame)

#     ## --- add foot contact spheres --- ##
#     foot_contact_spheres_dict = params['foot_contact_spheres']
#     for cur_contact_spheres in foot_contact_spheres_dict:
#         cur_contact_spheres_info = foot_contact_spheres_dict[cur_contact_spheres]
#         add_contact_spheres_and_forces_refactor(model = model, 
#                                                 sphere_name = cur_contact_spheres, 
#                                                 radius = cur_contact_spheres_info["radius"],
#                                                 relative_distance = cur_contact_spheres_info["location"], 
#                                                 reference_socketframe = cur_contact_spheres_info["socket_frame"], 
#                                                 reference_contact_space = contact_half_space_floor, 
#                                                 params = params)

# def generate_model_with_contacts_stair(model_name, model, params, contact_side=None):
    
#     floor_contact_loc = get_floor_ref_markers(markers_path = params["marker_data_path"], params = params)
#     # print(floor_contact_loc)
#     # exit()
#     contact_half_space_frame = model.get_ground()

#     for idx in range(params['num_of_steps']):
#         floor_contact_loc[0] += params['stair_width']
#         floor_contact_loc[1] += params['stair_height'] # y-axis (Opensim space) equals to gravity direction in real space

#         # np.pi/2 rotation is for matching XY plane of contact half space to the ground
#         reference_contact_half_space = {"name": f"stair_{idx}", 
#                                 "location": floor_contact_loc,
#                                 "orientation": np.array([0, 0, -np.pi/2]), 
#                                 "frame": "ground"}
#         contact_half_space_floor = add_contact_half_space(model, reference_contact_half_space, contact_half_space_frame)

#     ## --- add foot contact spheres --- ##
#     foot_contact_spheres_dict = params['foot_contact_spheres']
#     for cur_contact_spheres in foot_contact_spheres_dict:
#         cur_contact_spheres_info = foot_contact_spheres_dict[cur_contact_spheres]
#         add_contact_spheres_and_forces_refactor(model = model, 
#                                                 sphere_name = cur_contact_spheres, 
#                                                 radius = cur_contact_spheres_info["radius"],
#                                                 relative_distance = cur_contact_spheres_info["location"], 
#                                                 reference_socketframe = cur_contact_spheres_info["socket_frame"], 
#                                                 reference_contact_space = contact_half_space_floor, 
#                                                 params = params)

def get_tracking_problem(model, kinematics_path, params):

    tracking = osim.MocoTrack()
    tracking.setModel(osim.ModelProcessor(model))
    tracking.setName("tracking_problem")
    tableProcessor = osim.TableProcessor(kinematics_path)
    tableProcessor.append(osim.TabOpUseAbsoluteStateNames())    
    tableProcessor.append(osim.TabOpLowPassFilter(6.0))

    tracking.setStatesReference(tableProcessor)

    tracking.set_allow_unused_references(True)
    tracking.set_track_reference_position_derivatives(True) # Enable tracking of the derivatives of the reference positions

    tracking.set_states_global_tracking_weight(params['tracking_weight_global_state'])
    tracking.set_control_effort_weight(params['control_effort_weight_global'])

    tracking.set_initial_time(params['t0'])
    tracking.set_final_time(params['tf'])
    tracking.set_mesh_interval(params['mesh_interval'])

    return tracking

def set_state_tracking_weights(moco_study, params):
    problem = moco_study.updProblem()
    stateTrackingGoal = osim.MocoStateTrackingGoal.safeDownCast(problem.updGoal('state_tracking'))

    # (joint, coordinate, [state_types])
    state_specs = [
        ('ground_pelvis', 'pelvis_tx',    ['value']),
        ('ground_pelvis', 'pelvis_ty',    ['value']),
        ('ground_pelvis', 'pelvis_tilt',  ['value', 'speed']),
        ('back',          'lumbar_extension', ['value', 'speed']),
        ('hip_r',         'hip_flexion_r',   ['value', 'speed']),
        ('hip_l',         'hip_flexion_l',   ['value', 'speed']),
        ('walker_knee_r', 'knee_angle_r',    ['value', 'speed']),
        ('walker_knee_l', 'knee_angle_l',    ['value', 'speed']),
        ('ankle_r',       'ankle_angle_r',   ['value', 'speed']),
        ('ankle_l',       'ankle_angle_l',   ['value', 'speed']),
    ]

    weight = params['tracking_weight_individual_state']

    for joint, coord, types in state_specs:
        for stype in types:
            path = f'/jointset/{joint}/{coord}/{stype}'
            stateTrackingGoal.setWeightForState(path, weight)

def set_state_bounds(moco_study, params):
    
    problem = moco_study.updProblem()
    
    fraction_extra_bound_size = params['state_extra_bound_size']

    tracked_states_file = "tracking_problem_tracked_states.sto"
    tracked_states_table = osim.TimeSeriesTable(tracked_states_file)
    tracked_states_table.trimFrom(params["t0"])
    tracked_states_table.trimTo(params["tf"])
    col_labels = tracked_states_table.getColumnLabels()
    state_paths = [col for col in col_labels]

    for state_path in state_paths:
        state_column = tracked_states_table.getDependentColumn(state_path).to_numpy()
        col_min = np.min(state_column)
        col_max = np.max(state_column)
        col_range = col_max - col_min
        extra_bound_size = col_range * fraction_extra_bound_size
        bounds = osim.MocoBounds(col_min - extra_bound_size, col_max + extra_bound_size)
        problem.setStateInfo(state_path, bounds)

def set_control_input_bound(model, moco_study, params):
    """
    Set control bounds for reserves, assistive, and coordinate actuators using MocoControlInfo.
    """
    problem = moco_study.updProblem()

    # Set reserve actuator control bounds
    forceSet = model.getForceSet()
    for i in range(forceSet.getSize()):
        force = forceSet.get(i)
        path = force.getAbsolutePathString()

        # Reserve actuators (e.g., 'reserve_')
        if 'reserve_' in path:
            minval, maxval = params['reserve_actuator_control_bound']
            problem.setControlInfo(path, (minval, maxval))
        
        # Assistive actuators (e.g., 'assistive_force')
        if 'assistive_force' in path:
            minval, maxval = params['robot_assistance_control_bound']
            problem.setControlInfo(path, (minval, maxval))

    # Set coordinate actuator control bounds for all CoordinateActuators
    actuators = model.getActuators()
    for i in range(actuators.getSize()):
        act = actuators.get(i)
        if act.getConcreteClassName() == 'CoordinateActuator':
            path = act.getAbsolutePathString()
            minval, maxval = params['coord_actuator_control_bound']
            problem.setControlInfo(path, (minval, maxval))

def set_control_input_weights(model, moco_study, params):
    
    problem = moco_study.updProblem() # problem setup
    problem.addGoal(osim.MocoInitialActivationGoal("activation")) # Muscle 

    # Get a reference to the MocoControlGoal that is added to every MocoTrack problem by default and change the weights
    effort_goal = osim.MocoControlGoal.safeDownCast(problem.updGoal("control_effort"))

    actuators = model.getActuators()
    for i in range(actuators.getSize()):
        actuator = actuators.get(i)
        if actuator.getConcreteClassName() == 'CoordinateActuator':
            effort_goal.setWeightForControl(actuator.getAbsolutePathString(), params["control_effort_weight_coord_actuators"])

    forceSet = model.getForceSet()
    for i in range(forceSet.getSize()):
        forcePath = forceSet.get(i).getAbsolutePathString()
        if "reserve_" in str(forcePath):
            effort_goal.setWeightForControl(forcePath, params["control_effort_weight_reserve_actuator"])

    for side in ['r', 'l']:
        effort_goal.setWeightForControl(f"/forceset/bflh_{side}", params["control_effort_weight_muscles"])
        effort_goal.setWeightForControl(f"/forceset/bfsh_{side}", params["control_effort_weight_muscles"])
        effort_goal.setWeightForControl(f"/forceset/semimem_{side}", params["control_effort_weight_muscles"])
        effort_goal.setWeightForControl(f"/forceset/gasmed_{side}", params["control_effort_weight_muscles"])
        effort_goal.setWeightForControl(f"/forceset/glmax1_{side}", params["control_effort_weight_muscles"])
        effort_goal.setWeightForControl(f"/forceset/glmax2_{side}", params["control_effort_weight_muscles"])
        effort_goal.setWeightForControl(f"/forceset/glmax3_{side}", params["control_effort_weight_muscles"])
        effort_goal.setWeightForControl(f"/forceset/glmed1_{side}", params["control_effort_weight_muscles"])
        effort_goal.setWeightForControl(f"/forceset/glmed2_{side}", params["control_effort_weight_muscles"])
        effort_goal.setWeightForControl(f"/forceset/glmed3_{side}", params["control_effort_weight_muscles"])
        effort_goal.setWeightForControl(f"/forceset/psoas_{side}", params["control_effort_weight_muscles"])
        effort_goal.setWeightForControl(f"/forceset/recfem_{side}", params["control_effort_weight_muscles"])
        effort_goal.setWeightForControl(f"/forceset/soleus_{side}", params["control_effort_weight_muscles"])
        effort_goal.setWeightForControl(f"/forceset/tibant_{side}", params["control_effort_weight_muscles"])
        effort_goal.setWeightForControl(f"/forceset/vaslat_{side}", params["control_effort_weight_muscles"])
        effort_goal.setWeightForControl(f"/forceset/vasmed_{side}", params["control_effort_weight_muscles"])

    if params["robot_assistance"]: # unconstrained
        effort_goal.setWeightForControl("/forceset/assistive_force_x", params["control_effort_weight_robot_assistance"])
        effort_goal.setWeightForControl("/forceset/assistive_force_y", params["control_effort_weight_robot_assistance"])

def set_moco_problem_solver_settings(moco_study, params, nb_max_iterations):
    # Downcast to CasADiSolver to access full solver features
    solver = osim.MocoCasADiSolver.safeDownCast(moco_study.updSolver())

    # Access the underlying optimization problem
    problem = moco_study.updProblem()
    # Ensure the solver has the latest version of the problem (useful if modified after initialize)
    solver.resetProblem(problem)

    # Set the maximum allowed solver iterations, if specified
    if nb_max_iterations:
        solver.set_optim_max_iterations(nb_max_iterations)
    
    # Set convergence and constraint tolerances (smaller = stricter, but slower/harder)
    solver.set_optim_convergence_tolerance(params['convergence_tolerance'])
    solver.set_optim_constraint_tolerance(params['constraint_tolerance'])
    
    # If an initial guess is requested:
    if params.get('initial_guess'):
        # Create a template guess that's compatible with this problem
        guess = solver.createGuess()

        # Load the previous solution as a MocoTrajectory
        prev_solution = osim.MocoTrajectory(params['previous_solution_path'])

        # Export tables for states and controls from the previous solution
        prev_states_table = prev_solution.exportToStatesTable()
        prev_controls_table = prev_solution.exportToControlsTable()
        # # Insert only the matching states & controls into the guess
        guess.insertStatesTrajectory(prev_states_table, True)
        guess.insertControlsTrajectory(prev_controls_table, True)
        # Set the completed guess as the initial guess for the solver
        solver.setGuess(guess)

def set_optimal_forces(model, params):
    force_set = model.getForceSet()
    for force_name, optimal_force in params['optimal_forces'].items():
        found = False
        for i in range(force_set.getSize()):
            force = force_set.get(i)
            path = force.getAbsolutePathString()
            if force_name == path:
                ca = osim.CoordinateActuator.safeDownCast(force)
                if ca:
                    ca.setOptimalForce(optimal_force)
                    # print(f"Set {force_name} optimal force to {optimal_force}")
                else:
                    print(f"{force_name} found but is not a CoordinateActuator (skipped).")
                found = True
                break
        if not found:
            print(f"Force {force_name} (path) not found in model (skipped).")
        # if "reserve_" in str(forcePath):
            # effort_goal.setWeightForControl(forcePath, params["control_effort_weight_reserve_actuator"])

    #define

def modify_datafile_path(xml_file_path: str, new_datafile_path: str) -> None:
    """
    Modify the file path specified in the datafile element of an XML file.

    Parameters:
    xml_file_path (str): Path to the XML file.
    new_datafile_path (str): New file path to be set in the datafile element.
    """
    try:
        # Parse the XML file
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        # Find the datafile element and update its text
        datafile_element = root.find('.//datafile')
        if datafile_element is not None:
            datafile_element.text = new_datafile_path
            print(f'Updated GRF datafile path to: {datafile_element.text}')
        else:
            print('GRF datafile element not found!')

        # Save the modified XML back to file
        tree.write(xml_file_path, encoding='utf-8', xml_declaration=True)
        print('GRF XML file has been updated successfully.')

    except Exception as e:
        print(f'An error occurred: {e}')

def get_floor_ref_markers(markers_path: str, params):   
    opencap_markers = pd.read_csv(markers_path, delimiter="\t", skiprows=3).tail(-1)
    filtered_markers = prepare_opencap_markers(opencap_markers)

    filtered_markers = filtered_markers[filtered_markers["Time"].values == params['t0']]

    column_interest = []
    for cur_marker in params['floor_ref_marker']:
        for cur_col in filtered_markers.columns:
            if cur_marker in cur_col[0]:
                column_interest.append(cur_col)
    df_floor_mark = filtered_markers[column_interest]

    r_foot_columns = [col for col in df_floor_mark.columns if col[0].startswith('r')]
    l_foot_columns = [col for col in df_floor_mark.columns if col[0].startswith('L')]

    r_foot_marker_set = df_floor_mark[r_foot_columns]
    r_foot_x_markers = [col for col in r_foot_marker_set.columns if col[1] in ['X']]
    r_foot_y_markers = [col for col in r_foot_marker_set.columns if col[1] in ['Y']]
    r_foot_z_markers = [col for col in r_foot_marker_set.columns if col[1] in ['Z']]

    r_foot_x = np.mean(r_foot_marker_set[r_foot_x_markers].values)
    r_foot_y = np.mean(r_foot_marker_set[r_foot_y_markers].values)
    r_foot_z = np.mean(r_foot_marker_set[r_foot_z_markers].values)

    l_foot_marker_set = df_floor_mark[l_foot_columns]
    l_foot_x_markers = [col for col in l_foot_marker_set.columns if col[1] in ['X']]
    l_foot_y_markers = [col for col in l_foot_marker_set.columns if col[1] in ['Y']]
    l_foot_z_markers = [col for col in l_foot_marker_set.columns if col[1] in ['Z']]

    l_foot_x = np.mean(l_foot_marker_set[l_foot_x_markers].values)
    l_foot_y = np.mean(l_foot_marker_set[l_foot_y_markers].values)
    l_foot_z = np.mean(l_foot_marker_set[l_foot_z_markers].values)

    floor_coord_x = np.mean([r_foot_x, l_foot_x])
    floor_coord_y = np.mean([r_foot_y, l_foot_y])
    floor_coord_z = np.mean([r_foot_z, l_foot_z])

    floor_coord_y = floor_coord_y - params['floor_offset'] # y-axis is equal to z-axis before rotation

    return np.array([floor_coord_x, floor_coord_y, floor_coord_z])

def get_chair_ref_markers(markers_path: str, params):    
    opencap_markers = pd.read_csv(markers_path, delimiter="\t", skiprows=3).tail(-1)
    filtered_markers = prepare_opencap_markers(opencap_markers)

    filtered_markers = filtered_markers[filtered_markers["Time"].values == params['t0']]

    column_interest = []
    for cur_marker in params['chair_ref_marker']:
        for cur_col in filtered_markers.columns:
            if cur_marker in cur_col[0]:
                column_interest.append(cur_col)
    df_chair_mark = filtered_markers[column_interest]

    x_columns = [col for col in df_chair_mark.columns if col[1] in ['X']]
    y_columns = [col for col in df_chair_mark.columns if col[1] in ['Y']]
    z_columns = [col for col in df_chair_mark.columns if 'thigh' in col[0] and col[1] == 'Z']

    chair_coord_x = np.mean(df_chair_mark[x_columns].values)
    chair_coord_y = np.mean(df_chair_mark[y_columns].values)
    chair_coord_z = np.mean(df_chair_mark[z_columns].values)

    params['chair_coord_x'] = chair_coord_x
    params['chair_coord_y'] = chair_coord_y
    params['chair_coord_z'] = chair_coord_z

    return np.array([chair_coord_x, chair_coord_y, chair_coord_z])

def get_reference_markers(markers_path: str): 
    
    # load marker data
    opencap_markers = pd.read_csv(markers_path, delimiter="\t", skiprows=3).tail(-1)
    filtered_markers = prepare_opencap_markers(opencap_markers)

    # load only foot markers
    feet_markers_right=filtered_markers[["Time","r_ankle_study","r_calc_study","r_5meta_study","r_toe_study"]].copy()
    feet_markers_left=filtered_markers[["Time","L_ankle_study","L_calc_study","L_5meta_study","L_toe_study"]].copy()
    feet_markers=pd.concat([feet_markers_right,feet_markers_left], axis=1).round(3)
    feet_markers = feet_markers.loc[:, ~feet_markers.columns.duplicated()]

    reference_feet_markers=feet_markers.loc[[1,feet_markers.shape[0]],:]
    reference_feet_markers.index = ["t0", "tf"]

    mean_ankle_Y = np.mean([reference_feet_markers.r_ankle_study.Y.t0, reference_feet_markers.L_ankle_study.Y.t0]).round(3)
    mean_calc_Y = np.mean([reference_feet_markers.r_calc_study.Y.t0, reference_feet_markers.L_calc_study.Y.t0]).round(3)
    mean_5meta_Y = np.mean([reference_feet_markers.r_5meta_study.Y.t0, reference_feet_markers.L_5meta_study.Y.t0]).round(3)
    mean_toe_Y = np.mean([reference_feet_markers.r_toe_study.Y.t0, reference_feet_markers.L_toe_study.Y.t0]).round(3)

    mean_ankle_X = np.mean([reference_feet_markers.r_ankle_study.X.t0, reference_feet_markers.L_ankle_study.X.t0]).round(3)
    mean_calc_X = np.mean([reference_feet_markers.r_calc_study.X.t0, reference_feet_markers.L_calc_study.X.t0]).round(3)
    mean_5meta_X = np.mean([reference_feet_markers.r_5meta_study.X.t0, reference_feet_markers.L_5meta_study.X.t0]).round(3)
    mean_toe_X = np.mean([reference_feet_markers.r_toe_study.X.t0, reference_feet_markers.L_toe_study.X.t0]).round(3)

    initial_markers = {
        "mean_ankle_Y": mean_ankle_Y,
        "mean_calc_Y": mean_calc_Y,
        "mean_5meta_Y": mean_5meta_Y,
        "mean_toe_Y": mean_toe_Y,
        "mean_ankle_X": mean_ankle_X,
        "mean_calc_X": mean_calc_X,
        "mean_5meta_X": mean_5meta_X,
        "mean_toe_X": mean_toe_X
    }

    mean_ankle_Y = np.mean([reference_feet_markers.r_ankle_study.Y.tf, reference_feet_markers.L_ankle_study.Y.tf]).round(3)
    mean_calc_Y = np.mean([reference_feet_markers.r_calc_study.Y.tf, reference_feet_markers.L_calc_study.Y.tf]).round(3)
    mean_5meta_Y = np.mean([reference_feet_markers.r_5meta_study.Y.tf, reference_feet_markers.L_5meta_study.Y.tf]).round(3)
    mean_toe_Y = np.mean([reference_feet_markers.r_toe_study.Y.tf, reference_feet_markers.L_toe_study.Y.tf]).round(3)

    mean_ankle_X = np.mean([reference_feet_markers.r_ankle_study.X.tf, reference_feet_markers.L_ankle_study.X.tf]).round(3)
    mean_calc_X = np.mean([reference_feet_markers.r_calc_study.X.tf, reference_feet_markers.L_calc_study.X.tf]).round(3)
    mean_5meta_X = np.mean([reference_feet_markers.r_5meta_study.X.tf, reference_feet_markers.L_5meta_study.X.tf]).round(3)
    mean_toe_X = np.mean([reference_feet_markers.r_toe_study.X.tf, reference_feet_markers.L_toe_study.X.tf]).round(3)
    
    final_markers = {
        "mean_ankle_Y": mean_ankle_Y,
        "mean_calc_Y": mean_calc_Y,
        "mean_5meta_Y": mean_5meta_Y,
        "mean_toe_Y": mean_toe_Y,
        "mean_ankle_X": mean_ankle_X,
        "mean_calc_X": mean_calc_X,
        "mean_5meta_X": mean_5meta_X,
        "mean_toe_X": mean_toe_X
    }
    #print("final markers: ", final_markers)

    return initial_markers, final_markers

def extract_osim_GRF(model, solution, params):

    if 'sit' in params['motion'].lower():
        #Extract ground reaction forces for downstream analysis. Add the contact force elements to vectors, then use Moco's
        # createExternalLoadsTableForGait() function.
        contact_r = osim.StdVectorString()
        contact_l = osim.StdVectorString()

        contact_chair_r = osim.StdVectorString()
        contact_chair_l = osim.StdVectorString()

        # Feet GRFs
        contact_r.append('/forceset/SmoothSphereHalfSpaceForce_s1_r_floor')
        contact_r.append('/forceset/SmoothSphereHalfSpaceForce_s2_r_floor')
        contact_r.append('/forceset/SmoothSphereHalfSpaceForce_s3_r_floor')
        contact_r.append('/forceset/SmoothSphereHalfSpaceForce_s4_r_floor')
        contact_r.append('/forceset/SmoothSphereHalfSpaceForce_s5_r_floor')
        contact_r.append('/forceset/SmoothSphereHalfSpaceForce_s6_r_floor')

        contact_l.append('/forceset/SmoothSphereHalfSpaceForce_s1_l_floor')
        contact_l.append('/forceset/SmoothSphereHalfSpaceForce_s2_l_floor')
        contact_l.append('/forceset/SmoothSphereHalfSpaceForce_s3_l_floor')
        contact_l.append('/forceset/SmoothSphereHalfSpaceForce_s4_l_floor')
        contact_l.append('/forceset/SmoothSphereHalfSpaceForce_s5_l_floor')
        contact_l.append('/forceset/SmoothSphereHalfSpaceForce_s6_l_floor')
        
        ## save if the forceset element inclues 
        contact_chair_r.append('/forceset/SmoothSphereHalfSpaceForce_s8_r_chair')
        contact_chair_l.append('/forceset/SmoothSphereHalfSpaceForce_s8_l_chair')
        
        model_processor = osim.ModelProcessor(model)
        model = model_processor.process()
        externalForcesTableFlat_feet = osim.createExternalLoadsTableForGait(model, solution.exportToStatesTrajectory(model), contact_r, contact_l)
        externalForcesTableFlat_chair = osim.createExternalLoadsTableForGait(model, solution.exportToStatesTrajectory(model), contact_chair_r, contact_chair_l)

        save_path_floor = os.path.join(params['cur_moco_path'],f"{params['osim_model_name']}_grfs_osim_floor.sto")
        save_path_chair = os.path.join(params['cur_moco_path'],f"{params['osim_model_name']}_grfs_osim_chair.sto")

        osim.STOFileAdapter.write(externalForcesTableFlat_feet, save_path_floor)
        osim.STOFileAdapter.write(externalForcesTableFlat_chair, save_path_chair)

    elif 'incline' in params['motion'].lower():
        contact_r = osim.StdVectorString()
        contact_l = osim.StdVectorString()

        # Feet GRFs
        contact_r.append('/forceset/SmoothSphereHalfSpaceForce_s1_r_incline_floor')
        contact_r.append('/forceset/SmoothSphereHalfSpaceForce_s2_r_incline_floor')
        contact_r.append('/forceset/SmoothSphereHalfSpaceForce_s3_r_incline_floor')
        contact_r.append('/forceset/SmoothSphereHalfSpaceForce_s4_r_incline_floor')
        contact_r.append('/forceset/SmoothSphereHalfSpaceForce_s5_r_incline_floor')
        contact_r.append('/forceset/SmoothSphereHalfSpaceForce_s6_r_incline_floor')

        contact_l.append('/forceset/SmoothSphereHalfSpaceForce_s1_l_incline_floor')
        contact_l.append('/forceset/SmoothSphereHalfSpaceForce_s2_l_incline_floor')
        contact_l.append('/forceset/SmoothSphereHalfSpaceForce_s3_l_incline_floor')
        contact_l.append('/forceset/SmoothSphereHalfSpaceForce_s4_l_incline_floor')
        contact_l.append('/forceset/SmoothSphereHalfSpaceForce_s5_l_incline_floor')
        contact_l.append('/forceset/SmoothSphereHalfSpaceForce_s6_l_incline_floor')
        
        model_processor = osim.ModelProcessor(model)
        model = model_processor.process()
        externalForcesTableFlat_feet = osim.createExternalLoadsTableForGait(model, solution.exportToStatesTrajectory(model), contact_r, contact_l)
        save_path_floor = os.path.join(params['cur_moco_path'],f"{params['osim_model_name']}_grfs_osim_floor.sto")
        osim.STOFileAdapter.write(externalForcesTableFlat_feet, save_path_floor)

def check_simulation_parameters(simulation_session_Metadata, expected_values):
    mismatches = []
    
    for param, expected_value in expected_values.items():
        actual_value = simulation_session_Metadata.get(param)
        if actual_value != expected_value:
            mismatches.append(f"{param} (expected: {expected_value}, actual: {actual_value})")
    
    if mismatches:
        print("Error: The following GRFs simulation parameters differ from this session's parameters:")
        for mismatch in mismatches:
            print(f"- {mismatch}")
        sys.exit(1)
    
    print("All simulation parameters match the session's parameters.")


def read_sto_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find the index of the 'endheader' line
    header_end = next(i for i, line in enumerate(lines) if line.strip() == 'endheader')
    
    # The next line after 'endheader' contains the column names
    column_names = lines[header_end + 1].strip().split('\t')
    
    # Read the data, starting from the line after the column names
    df = pd.read_csv(file_path, 
                     delimiter='\t', 
                     skiprows=header_end + 2,  # Skip header + column names row
                     names=column_names)  # Use the extracted column names
    
    return df


def write_sto_file(df, original_file_path, suffix="_modified"):
    # Generate new file name
    file_dir, file_name = os.path.split(original_file_path)
    name, ext = os.path.splitext(file_name)
    new_file_name = f"{name}{suffix}{ext}"
    new_file_path = os.path.join(file_dir, new_file_name)

    # Read the original file to get the header
    with open(original_file_path, 'r') as f:
        header_lines = []
        for line in f:
            header_lines.append(line)
            if line.strip() == 'endheader':
                break

    # Write the new file
    with open(new_file_path, 'w') as f:
        # Write the original header
        f.writelines(header_lines)
        
        # Write the column names
        f.write('\t'.join(df.columns) + '\n')
        
        # Write the data without the column names
        df.to_csv(f, sep='\t', index=False, float_format='%.6f', lineterminator='\n', header=False)
    
    print(f"New file written: {new_file_path}")
    return new_file_path


def compute_ground_reactions_forces(model, solution, grf_solution_path) -> None:
        
    modelProcessor = osim.ModelProcessor(model)
    
    # Define contact forces for right and left sides
    contact_r = osim.StdVectorString()
    contact_l = osim.StdVectorString()

    contact_r.append('/forceset/SmoothSphereHalfSpaceForce_s1_r')
    contact_r.append('/forceset/SmoothSphereHalfSpaceForce_s2_r')
    contact_r.append('/forceset/SmoothSphereHalfSpaceForce_s3_r')
    contact_r.append('/forceset/SmoothSphereHalfSpaceForce_s4_r')
    contact_r.append('/forceset/SmoothSphereHalfSpaceForce_s5_r')
    contact_r.append('/forceset/SmoothSphereHalfSpaceForce_s6_r')

    contact_l.append('/forceset/SmoothSphereHalfSpaceForce_s1_l')
    contact_l.append('/forceset/SmoothSphereHalfSpaceForce_s2_l')
    contact_l.append('/forceset/SmoothSphereHalfSpaceForce_s3_l')
    contact_l.append('/forceset/SmoothSphereHalfSpaceForce_s4_l')
    contact_l.append('/forceset/SmoothSphereHalfSpaceForce_s5_l')
    contact_l.append('/forceset/SmoothSphereHalfSpaceForce_s6_l')

    # Process the model using modelProcessor
    model = modelProcessor.process()

    # Create external loads table for gait
    externalForcesTableFlat = osim.createExternalLoadsTableForGait(
        model,
        solution.exportToStatesTrajectory(model),
        contact_r,
        contact_l
    )

    # Optionally use externalForcesTable as needed
    print("External loads table created:", externalForcesTableFlat)

    osim.STOFileAdapter.write(externalForcesTableFlat, grf_solution_path)
    
def lowPassFilter(time, data, lowpass_cutoff_frequency, order=4):
    
    fs = 1/np.round(np.mean(np.diff(time)),16)
    wn = lowpass_cutoff_frequency/(fs/2)
    sos = signal.butter(order/2, wn, btype='low', output='sos')
    dataFilt = signal.sosfiltfilt(sos, data, axis=0)

    return dataFilt

def check_assistance_parameters(assist_with_force_ext, force_ext_opt_value, 
    ):

    """
    Validates assistance parameters to ensure consistency in assistance configuration.

    This function checks if exactly one of the two possible assistance methods is selected.
    It will print an error message and terminate the program if:
    - Both assistance modes are selected.
    - External force assistance is selected but `force_ext_opt_value` is not provided.
    - Both assistance modes are deselected.

    Parameters:
    assist_with_force_ext (bool): True if external force assistance is enabled.
    force_ext_opt_value (float): There is a value set for external force; None otherwise.
    assist_with_reserve_pelvis_txy (bool): True if pelvis residual assistance is enabled.

    Terminates the program with an error message if the configuration is invalid.
    """
    
    if assist_with_force_ext and assist_with_reserve_pelvis_txy:
        print("Error: Select either external force assistance or pelvis residual assistance")
        sys.exit(1)
    if assist_with_force_ext and (not force_ext_opt_value):
        print("Error: External force value missing")
        sys.exit(1)
    if (not assist_with_force_ext) and force_ext_opt_value:
        print(f"Error: External force set to {force_ext_opt_value} but supposed to be None")
        sys.exit(1)
    if (not assist_with_force_ext) and (not assist_with_reserve_pelvis_txy):
        print("Error: Must choose an assistance mode (external force or pelvis residual)")
        sys.exit(1)


import sys

def print_optimal_force_values(model):

    print("\n===== print optimal force values =====\n")
    forceSet = model.getForceSet()

    for i in range(forceSet.getSize()):
        forcePath = forceSet.get(i).getAbsolutePathString()
        this_actuator = forceSet.get(i)
        this_actuator = osim.CoordinateActuator.safeDownCast(this_actuator)

        try:
            print(f"{this_actuator.getName()}\t optimum force value: {this_actuator.getOptimalForce()}N")
        except:
            pass

def print_control_effort_weight(moco_study):
    # study = osim.MocoStudy()
    problem = moco_study.getProblem()
    goal = problem.updGoal("control_effort")
    control_goal = osim.MocoControlGoal.safeDownCast(goal)
    print(control_goal.get_control_effort_weight())
    # for i in range(control_goal.getNumWeightSetEntries()):
    #     entry = control_goal.getWeightSet().get(i)
    #     print(f"Actuator: {entry.getName()}, Weight: {entry.getWeight()}")

def check_parameters_refactor(params):
    """
    Validates assistance parameters to ensure consistency in assistance configuration.

    This function checks if exactly one of the two possible assistance methods is selected.
    It will print an error message and terminate the program if:
    - Both assistance modes are selected.
    - External force assistance is selected but `force_ext_opt_value` is not provided.
    - Both assistance modes are deselected.

    Parameters:
    assist_with_force_ext (bool): True if external force assistance is enabled.
    force_ext_opt_value (float): There is a value set for external force; None otherwise.
    assist_with_reserve_pelvis_txy (bool): True if pelvis residual assistance is enabled.

    Terminates the program with an error message if the configuration is invalid.
    """
    
    # Check for both or neither assistance modes being active
    if params['assist_with_force_ext'] == params['assist_with_reserve_pelvis_txy']:
        if params['assist_with_force_ext']:
            print("Error: Select either external force assistance or pelvis residual assistance")
        else:
            print("Error: Must choose an assistance mode (external force or pelvis residual)")
        sys.exit(1)

    # Check for incorrect external force configuration
    if params['assist_with_force_ext'] and params['force_ext_opt_value'] is None:
        print("Error: External force value missing")
        sys.exit(1)
    if not params['assist_with_force_ext'] and params['force_ext_opt_value'] is not None:
        print(f"Error: External force set to {params['force_ext_opt_value']} but supposed to be None")
        sys.exit(1)

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj) # Let the base class default method raise the TypeError