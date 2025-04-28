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

def add_contact_spheres_and_forces_refactor(model, sphere_name, radius, relative_distance, reference_socketframe, reference_contact_space, params):

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

def get_model_refactor(subject_name, model_path, params) -> osim.Model:

    osim_model, subject_name = getMuscleDrivenModel_refactor(subject_name = subject_name, 
                                                            model_path = model_path, 
                                                            params = params)

    if params["osim_GRF"]:
        generate_model_with_contacts(model_name=subject_name, model=osim_model, params=params, contact_side='all')
    
    # Add external assistive force
    if params["assist_with_force_ext"]:
        add_assistive_force(
            coordName="pelvis_tx",
            model=osim_model,
            name="assistive_force_x",
            direction=osim.Vec3(1, 0, 0),
            magnitude=params["force_ext_opt_value"],
        )
        add_assistive_force(
            coordName="pelvis_ty",
            model=osim_model,
            name="assistive_force_y",
            direction=osim.Vec3(0, 1, 0),
            magnitude=params["force_ext_opt_value"],
        )

    if params["assist_with_reserve_pelvis_txy"]:
        forceSet = osim_model.getForceSet()

        for i in range(forceSet.getSize()):
            forcePath = forceSet.get(i).getAbsolutePathString()
            this_actuator = forceSet.get(i)
            #default values: actu.setMinControl(-np.inf), actu.setMaxControl(np.inf) 
            if "reserve_jointset_ground_pelvis_pelvis_tx" in str(forcePath):
                print("set reserve pelvis tx optimum force to ", params["reserve_pelvis_tx_ty_opt_value"])
                this_coordinate_actuator = osim.CoordinateActuator.safeDownCast(this_actuator)
                this_coordinate_actuator.setOptimalForce(params["reserve_pelvis_tx_ty_opt_value"])

            if "reserve_jointset_ground_pelvis_pelvis_ty" in str(forcePath):
                print("set reserve pelvis ty optimum force to ", params["reserve_pelvis_tx_ty_opt_value"])
                this_coordinate_actuator = osim.CoordinateActuator.safeDownCast(this_actuator)
                this_coordinate_actuator.setOptimalForce(params["reserve_pelvis_tx_ty_opt_value"])
                if params["pelvis_ty_positive"]:
                    this_coordinate_actuator.setMinControl(0)
    
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

def getMuscleDrivenModel_refactor(subject_name, model_path, params):
    """
    Create and process a muscle-driven OpenSim model by applying several modifications 
    to make it suitable for optimal control problems, particularly focusing on muscle 
    simplifications and enhancements for better simulation stability and accuracy.

    Parameters:
    - subject_name (str): Identifier for the subject or simulation.
    - model_path (str): File path to the baseline OpenSim model.
    - params (dict): Dictionary containing parameters such as reserve actuator values.

    Returns:
    - osim_model: Processed OpenSim Model object.
    - subject_name: Identifier for the subject.
    """

    # Load the OpenSim model from the given file path
    osim_model = osim.Model(model_path)

    # Simplify the model by reducing unnecessary muscle groups based on parameters
    simplify_model_refactor(osim_model, params)

    # Initialize actuator retrieval and model processing
    actuators_in_forceSet = osim_model.getActuators()

    for i in range(actuators_in_forceSet.getSize()):
        this_actuator = actuators_in_forceSet.get(i)
        this_actuator_path = actuators_in_forceSet.get(i).getAbsolutePathString()
        
        # Set same value for all upper limb coordinate actuators
        if this_actuator.getConcreteClassName() == 'CoordinateActuator':
            this_coordinate_actuator = osim.CoordinateActuator.safeDownCast(this_actuator)
            this_coordinate_actuator.setOptimalForce(params["coord_actuator_opt_value"]) # The maximum generalized force produced by this actuator.
        
    modelProcessor = osim.ModelProcessor(osim_model)
    modelProcessor.append(osim.ModOpIgnoreTendonCompliance())   # Disable tendon compliance in all muscles
    modelProcessor.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())     # Replace existing muscles with De Groote-Fregly 2016 model for improved computational efficiency\
    modelProcessor.append(osim.ModOpIgnorePassiveFiberForcesDGF())    # Disable passive fiber forces in DeGroote-Fregly 2016 muscles
    modelProcessor.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5))    # Scale the width of the active fiber force curve for these muscles

    # Add reserve coordinate actuators to all model degrees-of-freedom, excluding 
    # those for the upper limbs and pelvis which already have them
    modelProcessor.append(osim.ModOpAddReserves(params["reserve_actuator_opt_value"])) # Add reserve actuators to the model

    # Process the model with the specified modifications
    osim_model = modelProcessor.process()

    return osim_model, subject_name

def simplify_model_refactor(model: osim.Model, params):
    """ Simplify model by reducing actuators and muscles

    Args:
        model (osim.Model): model object
    """
    force_set = model.upd_ForceSet()
    muscles_to_remove = params['muscles_to_remove_v2']

    for cur_muscle in muscles_to_remove:
        cur_muscle_l = cur_muscle + '_l'
        cur_muscle_r = cur_muscle + '_r'
        force_set.remove(force_set.getIndex(cur_muscle_l))
        force_set.remove(force_set.getIndex(cur_muscle_r))

def generate_model_with_contacts(model_name, model, params, contact_side=None) -> str:
    
    floor_contact_loc = get_floor_ref_markers(markers_path = params["marker_data_path"], params = params)
    chair_contact_loc = get_chair_ref_markers(markers_path = params["marker_data_path"], params = params)
    chair_contact_loc[1] = floor_contact_loc[1] + params['chair_height'] # y-coordinate

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
        add_contact_spheres_and_forces_refactor(model = model, 
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
        add_contact_spheres_and_forces_refactor(model = model, 
                                                sphere_name = cur_contact_spheres, 
                                                radius = cur_contact_spheres_info["radius"],
                                                relative_distance = cur_contact_spheres_info["location"], 
                                                reference_socketframe = cur_contact_spheres_info["socket_frame"], 
                                                reference_contact_space = contact_half_space_chair, 
                                                params = params)

def get_tracking_problem(
    model: osim.Model,
    kinematics_path: str,
    t_0: float,
    t_f: float,
    mesh_interval: float,
    tracking_weight: float,
    control_effort_weight: float,
) -> osim.MocoStudy:
    
    # Create a MocoTrack problem
    tracking = osim.MocoTrack()
    tracking.setModel(osim.ModelProcessor(model))
    tracking.setName("tracking_problem")
    tableProcessor = osim.TableProcessor(kinematics_path)
    tableProcessor.append(osim.TabOpLowPassFilter(6.0))
    tableProcessor.append(osim.TabOpUseAbsoluteStateNames())    
    tracking.setStatesReference(tableProcessor)

    tracking.set_allow_unused_references(True)
    tracking.set_track_reference_position_derivatives(True) # Enable tracking of the derivatives of the reference positions
    tracking.set_states_global_tracking_weight(tracking_weight)
    tracking.set_initial_time(t_0)
    tracking.set_final_time(t_f)
    tracking.set_mesh_interval(mesh_interval)
    tracking.set_control_effort_weight(control_effort_weight)

    return tracking

def set_state_tracking_weights(moco_study: osim.MocoStudy, params):
    
    problem = moco_study.updProblem()
    
    stateTrackingGoal = osim.MocoStateTrackingGoal.safeDownCast(problem.updGoal('state_tracking'))
    
    stateTrackingGoal.setWeightForState('/jointset/ground_pelvis/pelvis_tilt/value', params['kinematic_state_tracking_weight'])
    stateTrackingGoal.setWeightForState('/jointset/ground_pelvis/pelvis_tilt/speed', params['kinematic_state_tracking_weight'])
    stateTrackingGoal.setWeightForState('/jointset/back/lumbar_extension/value', params['kinematic_state_tracking_weight'])
    stateTrackingGoal.setWeightForState('/jointset/back/lumbar_extension/speed', params['kinematic_state_tracking_weight'])
    stateTrackingGoal.setWeightForState('/jointset/hip_r/hip_flexion_r/value', params['kinematic_state_tracking_weight'])
    stateTrackingGoal.setWeightForState('/jointset/hip_r/hip_flexion_r/speed', params['kinematic_state_tracking_weight'])
    stateTrackingGoal.setWeightForState('/jointset/hip_l/hip_flexion_l/value', params['kinematic_state_tracking_weight'])
    stateTrackingGoal.setWeightForState('/jointset/hip_l/hip_flexion_l/speed', params['kinematic_state_tracking_weight'])
    stateTrackingGoal.setWeightForState('/jointset/walker_knee_r/knee_angle_r/value', params['kinematic_state_tracking_weight'])
    stateTrackingGoal.setWeightForState('/jointset/walker_knee_r/knee_angle_r/speed', params['kinematic_state_tracking_weight'])
    stateTrackingGoal.setWeightForState('/jointset/walker_knee_l/knee_angle_l/value', params['kinematic_state_tracking_weight'])
    stateTrackingGoal.setWeightForState('/jointset/walker_knee_l/knee_angle_l/speed', params['kinematic_state_tracking_weight'])
    stateTrackingGoal.setWeightForState('/jointset/ankle_r/ankle_angle_r/speed', params['kinematic_state_tracking_weight'])
    stateTrackingGoal.setWeightForState('/jointset/ankle_r/ankle_angle_r/value', params['kinematic_state_tracking_weight'])
    stateTrackingGoal.setWeightForState('/jointset/ankle_l/ankle_angle_l/speed', params['kinematic_state_tracking_weight'])
    stateTrackingGoal.setWeightForState('/jointset/ankle_l/ankle_angle_l/value', params['kinematic_state_tracking_weight'])
    # stateTrackingGoal.setWeightForState('/jointset/ground_pelvis/pelvis_tx/value', params["kinematic_state_tracking_weight"])
    # stateTrackingGoal.setWeightForState('/jointset/ground_pelvis/pelvis_ty/value', params["kinematic_state_tracking_weight"])
    
    if params["tracking_weight_pelvis_txy"] != 1:
        print("set pelvis tx and ty tracking weights to: ", params["tracking_weight_pelvis_txy"])
        stateTrackingGoal.setWeightForState('/jointset/ground_pelvis/pelvis_tx/value', params["tracking_weight_pelvis_txy"])
        stateTrackingGoal.setWeightForState('/jointset/ground_pelvis/pelvis_ty/value', params["tracking_weight_pelvis_txy"])

def set_state_bounds(moco_study: osim.MocoStudy, params):
    
    problem = moco_study.updProblem()
    
    fraction_extra_bound_size = 0.25
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
        # Lower bound: (minimum value) - fractionExtraBoundSize * (range of value)
        # Upper bound: (maximum value) + fractionExtraBoundSize * (range of value)
        bounds = osim.MocoBounds(col_min - extra_bound_size, col_max + extra_bound_size)
        problem.setStateInfo(state_path, bounds)
                                                  
def set_moco_problem_weights(model: osim.Model, moco_study: osim.MocoStudy, params):
    
    problem = moco_study.updProblem() # problem setup
    problem.addGoal(osim.MocoInitialActivationGoal("activation")) # encouraging starting with low initial muscle activations.

    # Get a reference to the MocoControlGoal that is added to every MocoTrack problem by default and change the weights
    effort_goal = osim.MocoControlGoal.safeDownCast(problem.updGoal("control_effort"))

    forceSet = model.getForceSet()
    for i in range(forceSet.getSize()):
        forcePath = forceSet.get(i).getAbsolutePathString()
        
        if "reserve_jointset_ground_pelvis" in str(forcePath):
            effort_goal.setWeightForControl(forcePath, params["reserve_pelvis_weight"])
        
        if params["assist_with_reserve_pelvis_txy"]:
            if "reserve_jointset_ground_pelvis_pelvis_tx" in str(forcePath):
                # print("set reserve pelvis tx control weight to ", params["reserve_pelvis_weight_txy"])
                effort_goal.setWeightForControl(forcePath, params["reserve_pelvis_weight_txy"])
            if "reserve_jointset_ground_pelvis_pelvis_ty" in str(forcePath):
                # print("set reserve pelvis ty control weight to ", params["reserve_pelvis_weight_txy"])
                effort_goal.setWeightForControl(forcePath, params["reserve_pelvis_weight_txy"])
        
        if "reserve_jointset_hip" in str(forcePath):
            effort_goal.setWeightForControl(forcePath, params["reserve_hip_weight"])
        if "reserve_jointset_walker_knee" in str(forcePath):
            effort_goal.setWeightForControl(forcePath, params["reserve_knee_weight"])
        if "reserve_jointset_ankle" in str(forcePath):
            effort_goal.setWeightForControl(forcePath, params["reserve_feet_weight"])
        if "reserve_jointset_subtalar" in str(forcePath):
            effort_goal.setWeightForControl(forcePath, params["reserve_feet_weight"])
        if "reserve_jointset_mtp" in str(forcePath):
            effort_goal.setWeightForControl(forcePath, params["reserve_feet_weight"])

    # if params["control_effort_muscles"] !=1:
        for side in ['r', 'l']:
            effort_goal.setWeightForControl(f"/forceset/bflh_{side}", params["control_effort_muscles"])
            effort_goal.setWeightForControl(f"/forceset/bfsh_{side}", params["control_effort_muscles"])
            effort_goal.setWeightForControl(f"/forceset/gasmed_{side}", params["control_effort_muscles"])
            effort_goal.setWeightForControl(f"/forceset/glmax1_{side}", params["control_effort_muscles"])
            effort_goal.setWeightForControl(f"/forceset/glmax2_{side}", params["control_effort_muscles"])
            effort_goal.setWeightForControl(f"/forceset/glmax3_{side}", params["control_effort_muscles"])
            effort_goal.setWeightForControl(f"/forceset/glmed1_{side}", params["control_effort_muscles"])
            effort_goal.setWeightForControl(f"/forceset/glmed2_{side}", params["control_effort_muscles"])
            effort_goal.setWeightForControl(f"/forceset/glmed3_{side}", params["control_effort_muscles"])
            effort_goal.setWeightForControl(f"/forceset/psoas_{side}", params["control_effort_muscles"])
            effort_goal.setWeightForControl(f"/forceset/recfem_{side}", params["control_effort_muscles"])
            effort_goal.setWeightForControl(f"/forceset/soleus_{side}", params["control_effort_muscles"])
            effort_goal.setWeightForControl(f"/forceset/tibant_{side}", params["control_effort_muscles"])
            effort_goal.setWeightForControl(f"/forceset/vaslat_{side}", params["control_effort_muscles"])
            effort_goal.setWeightForControl(f"/forceset/vasmed_{side}", params["control_effort_muscles"])

    if params["assist_with_force_ext"]: # unconstrained
        effort_goal.setWeightForControl("/forceset/assistive_force_x", 0)
        effort_goal.setWeightForControl("/forceset/assistive_force_y", 0)

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


# def simplify_grf(df_grf_osim):
#     df_grf_osim_simplified = df_grf_osim.copy()
    
#     # Set x and z components of forces to zero
#     for side in ['r', 'l']:
#         df_grf_osim_simplified[f'ground_force_{side}_vx'] = 0
#         df_grf_osim_simplified[f'ground_force_{side}_vz'] = 0
    
#     # Set x and y components of torques to zero
#     for side in ['r', 'l']:
#         df_grf_osim_simplified[f'ground_torque_{side}_x'] = 0
#         df_grf_osim_simplified[f'ground_torque_{side}_y'] = 0
    
#     # Calculate average y-component of force
#     avg_force = (df_grf_osim_simplified['ground_force_r_vy'] + 
#                  df_grf_osim_simplified['ground_force_l_vy']) / 2
    
#     # Calculate average z-component of torque
#     avg_torque = (df_grf_osim_simplified['ground_torque_r_z'] + 
#                   df_grf_osim_simplified['ground_torque_l_z']) / 2
    
#     # Set average force and torque for both sides
#     for side in ['r', 'l']:
#         df_grf_osim_simplified[f'ground_force_{side}_vy'] = avg_force
#         df_grf_osim_simplified[f'ground_torque_{side}_z'] = avg_torque
    
#     return df_grf_osim_simplified


# def filter_grf(df: pd.DataFrame, cutoff: float, order: int):
    
#     fs = 1 / np.mean(np.diff(df['time']))  # Sampling frequency
#     nyq = 0.5 * fs
#     normal_cutoff = cutoff / nyq
#     b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)

#     df['unfiltered_ground_force_l_vy'] = df['ground_force_l_vy']
#     df['ground_force_l_vy'] = signal.filtfilt(b, a, df['ground_force_l_vy'])
#     df['unfiltered_ground_force_r_vy'] = df['ground_force_r_vy']
#     df['ground_force_r_vy'] = signal.filtfilt(b, a, df['ground_force_r_vy'])
#     """
#     df['unfiltered_ground_force_l_px'] = df['ground_force_l_px']
#     df['ground_force_l_px'] = signal.filtfilt(b, a, df['ground_force_l_px'])	
#     df['unfiltered_ground_force_l_pz'] = df['ground_force_l_pz']
#     df['ground_force_l_pz'] = signal.filtfilt(b, a, df['ground_force_l_pz'])	
    
#     df['unfiltered_ground_force_r_px'] = df['ground_force_r_px']
#     df['ground_force_r_px'] = signal.filtfilt(b, a, df['ground_force_r_px'])	
#     df['unfiltered_ground_force_r_pz'] = df['ground_force_r_pz']
#     df['ground_force_r_pz'] = signal.filtfilt(b, a, df['ground_force_r_pz'])
#     """	
#     df['unfiltered_ground_torque_l_z'] =  df['ground_torque_l_z']
#     df['ground_torque_l_z'] = signal.filtfilt(b, a, df['ground_torque_l_z'])
#     df['unfiltered_ground_torque_r_z'] =  df['ground_torque_r_z']
#     df['ground_torque_r_z'] = signal.filtfilt(b, a, df['ground_torque_r_z'])

#     return df

def get_floor_ref_markers(markers_path: str, params):   
    opencap_markers = pd.read_csv(markers_path, delimiter="\t", skiprows=3).tail(-1)
    filtered_markers = prepare_opencap_markers(opencap_markers)

    # filtered_markers = filtered_markers[filtered_markers["Time"].values == params['t0']]

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

def set_moco_problem_solver_settings(moco_study: osim.MocoStudy, nb_max_iterations: int = None):
    # Modify the solver settings: 
    solver = osim.MocoCasADiSolver.safeDownCast(moco_study.updSolver())
    if nb_max_iterations:
        solver.set_optim_max_iterations(nb_max_iterations) # Stop optimization after 500 iterations => solution failed
    
    # solver.set_output_interval(10) # Monitor solver progress by writing every 10th iterate to file, cf. doc
    solver.set_optim_convergence_tolerance(1e-3) # Loosen the convergence tolerances, cf. doc
    solver.set_optim_constraint_tolerance(1e-3) # Loosen the constraint tolerances, cf. doc

def add_assistive_force(
    coordName: str,
    model: osim.Model,
    name: str,
    direction: osim.Vec3,
    magnitude: int = 100,
) -> None:
    """Add an assistive force to the model.

    Args:
        model (osim.Model): model to add assistive force to
        name (str): name of assistive force
        direction (osim.Vec3): direction of assistive force
        magnitude (int, optional): How strong the force is. Defaults to 100.
    """
    # coordSet = model.updCoordinateSet()

    # actu = osim.CoordinateActuator()
    # actu.setName(name)
    # actu.setCoordinate(coordSet.get(coordName))
    # actu.setOptimalForce(magnitude)
    # actu.setMinControl(-1)
    # actu.setMaxControl(1)
    # model.addComponent(actu)

    assistActuator = osim.PointActuator("pelvis")
    assistActuator.setName(name)
    assistActuator.set_force_is_global(True)
    assistActuator.set_direction(direction)

    if coordName == "pelvis_tx":
        print("control bounds assistive force x: [-np.inf, np.inf]")
        assistActuator.setMinControl(-np.inf) #manage the bounds on ScalarActuator's control
        assistActuator.setMaxControl(np.inf)  #manage the bounds on ScalarActuator's control
    if coordName == "pelvis_ty":
        print("control bounds assistive force y: [0,inf]")
        assistActuator.setMinControl(0) #manage the bounds on ScalarActuator's control
        assistActuator.setMaxControl(np.inf) #manage the bounds on ScalarActuator's control

    assistActuator.setOptimalForce(magnitude) #250 in exampleWalking.m
    model.addForce(assistActuator)

def extract_osim_GRF(model, solution, params):
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

# def extract_GRF(model, solution, solution_path_assistive_arm, solution_path_local, config):
#     #Extract ground reaction forces for downstream analysis. Add the contact force elements to vectors, then use Moco's
#     # createExternalLoadsTableForGait() function.
#     contact_r = osim.StdVectorString()
#     contact_l = osim.StdVectorString()

#     contact_chair_r = osim.StdVectorString()
#     contact_chair_l = osim.StdVectorString()

#     # Feet GRFs
#     contact_r.append('/forceset/SmoothSphereHalfSpaceForce_s1_r')
#     contact_r.append('/forceset/SmoothSphereHalfSpaceForce_s2_r')
#     contact_r.append('/forceset/SmoothSphereHalfSpaceForce_s3_r')
#     contact_r.append('/forceset/SmoothSphereHalfSpaceForce_s4_r')
#     contact_r.append('/forceset/SmoothSphereHalfSpaceForce_s5_r')
#     contact_r.append('/forceset/SmoothSphereHalfSpaceForce_s6_r')

#     contact_l.append('/forceset/SmoothSphereHalfSpaceForce_s1_l')
#     contact_l.append('/forceset/SmoothSphereHalfSpaceForce_s2_l')
#     contact_l.append('/forceset/SmoothSphereHalfSpaceForce_s3_l')
#     contact_l.append('/forceset/SmoothSphereHalfSpaceForce_s4_l')
#     contact_l.append('/forceset/SmoothSphereHalfSpaceForce_s5_l')
#     contact_l.append('/forceset/SmoothSphereHalfSpaceForce_s6_l')
    
#     # Chair GRFs
#     if config["osim_GRF_type"] == "s7":
#         contact_chair_r.append('/forceset/SmoothSphereHalfSpaceForce_s7_r')
#         contact_chair_l.append('/forceset/SmoothSphereHalfSpaceForce_s7_l')
#     if config["osim_GRF_type"] == "s8":
#         contact_chair_r.append('/forceset/SmoothSphereHalfSpaceForce_s8_r')
#         contact_chair_l.append('/forceset/SmoothSphereHalfSpaceForce_s8_l')
#     if config["osim_GRF_type"] == "s9":
#         contact_chair_r.append('/forceset/SmoothSphereHalfSpaceForce_s9_r')
#         contact_chair_l.append('/forceset/SmoothSphereHalfSpaceForce_s9_l')
    
#     model_processor = osim.ModelProcessor(model)
#     model = model_processor.process()
#     externalForcesTableFlat_feet = osim.createExternalLoadsTableForGait(model, solution.exportToStatesTrajectory(model), contact_r, contact_l)
#     externalForcesTableFlat_chair = osim.createExternalLoadsTableForGait(model, solution.exportToStatesTrajectory(model), contact_chair_r, contact_chair_l)

#     osim.STOFileAdapter.write(externalForcesTableFlat_feet, f"{solution_path_assistive_arm}_grfs_osim_feet.sto")
#     osim.STOFileAdapter.write(externalForcesTableFlat_feet, f"{solution_path_local}_grfs_osim_feet.sto")
#     osim.STOFileAdapter.write(externalForcesTableFlat_chair, f"{solution_path_assistive_arm}_grfs_osim_chair.sto")
#     osim.STOFileAdapter.write(externalForcesTableFlat_chair, f"{solution_path_local}_grfs_osim_chair.sto")


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

def check_assistance_parameters(assist_with_force_ext, force_ext_opt_value, assist_with_reserve_pelvis_txy):

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