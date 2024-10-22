import numpy as np
import opensim as osim
import xml.etree.ElementTree as ET
import sys
from scipy import signal

from pathlib import Path

import os
import pandas as pd

from assistive_arm.utils.data_preprocessing import prepare_opencap_markers

def simplify_model(model_name: str, model: osim.Model) -> None:
    """ Simplify model by reducing actuators and muscles

    Args:
        model_name (str): model name
        model (osim.Model): model object
    """
    muscles_to_remove = []

    # Remove unnecessary actuators (even if model full)
    actuators_to_remove = [
        # Actuators from LaiUlrich2022: lumbar, shoulder r/l, elbow r/l, pro_sup r/l
        # CoordinateActuators 
        #"shoulder_flex_r",
        #"shoulder_add_r",
        #"shoulder_rot_r",
        #"elbow_flex_r",
        #"pro_sup_r", # proximal radioulnar joint _ pronation and supination of the arm
        #"shoulder_flex_l",
        #"shoulder_add_l",
        #"shoulder_rot_l",
        #"elbow_flex_l",
        #"pro_sup_l",
        # Remaining actuators: lumbar_ext, lumbar_bend, lumbar_rot (extension, bending, rotation)
    ]

    right_muscles = [
        "addlong_r",
        "addbrev_r",
        "addmagDist_r",
        "addmagIsch_r",
        "addmagProx_r",
        "addmagMid_r",
        "fdl_r",
        "ehl_r",  
        "fhl_r",  
        "gaslat_r",  
        #"glmax1_r",
        #"glmax2_r",
        #"glmax3_r",
        #"glmed1_r",
        #"glmed2_r",
        #"glmed3_r",
        "glmin1_r",
        "glmin2_r",
        "glmin3_r",
        "perlong_r",
        "edl_r",
        "grac_r", 
        "iliacus_r",  
        "perbrev_r",
        "piri_r",
        "sart_r",
        "semimem_r",
        "semiten_r", # Consider keeping ??
        "tfl_r",
        "tibpost_r",
        "vasint_r",  # Consider keeping ??
        #"vaslat_r", 
        #"psoas_r" 
    ]

    left_muscles = [muscle[:-1] + "l" for muscle in right_muscles]

    lumbar_muscles = [
        #"lumbar_bend",
        #"lumbar_ext",
        #"lumbar_rot",
    ]

    # remaining muscles: bflh_r, bfsh_r, gasmed_r, glmax_r, psoas_r, recfem_r, soleus_r, tibant_r, vasmed_r
    # psoas_r not in paper   

    force_set = model.upd_ForceSet()

    if "simple" in model_name:
        muscles_to_remove = right_muscles + left_muscles + lumbar_muscles

    for i in actuators_to_remove + muscles_to_remove:
        force_set.remove(force_set.getIndex(i))


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


def simplify_grf(df_grf_osim):
    df_grf_osim_simplified = df_grf_osim.copy()
    
    # Set x and z components of forces to zero
    for side in ['r', 'l']:
        df_grf_osim_simplified[f'ground_force_{side}_vx'] = 0
        df_grf_osim_simplified[f'ground_force_{side}_vz'] = 0
    
    # Set x and y components of torques to zero
    for side in ['r', 'l']:
        df_grf_osim_simplified[f'ground_torque_{side}_x'] = 0
        df_grf_osim_simplified[f'ground_torque_{side}_y'] = 0
    
    # Calculate average y-component of force
    avg_force = (df_grf_osim_simplified['ground_force_r_vy'] + 
                 df_grf_osim_simplified['ground_force_l_vy']) / 2
    
    # Calculate average z-component of torque
    avg_torque = (df_grf_osim_simplified['ground_torque_r_z'] + 
                  df_grf_osim_simplified['ground_torque_l_z']) / 2
    
    # Set average force and torque for both sides
    for side in ['r', 'l']:
        df_grf_osim_simplified[f'ground_force_{side}_vy'] = avg_force
        df_grf_osim_simplified[f'ground_torque_{side}_z'] = avg_torque
    
    return df_grf_osim_simplified


def filter_grf(df: pd.DataFrame, cutoff: float, order: int):
    
    fs = 1 / np.mean(np.diff(df['time']))  # Sampling frequency
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)

    df['unfiltered_ground_force_l_vy'] = df['ground_force_l_vy']
    df['ground_force_l_vy'] = signal.filtfilt(b, a, df['ground_force_l_vy'])
    df['unfiltered_ground_force_r_vy'] = df['ground_force_r_vy']
    df['ground_force_r_vy'] = signal.filtfilt(b, a, df['ground_force_r_vy'])
    """
    df['unfiltered_ground_force_l_px'] = df['ground_force_l_px']
    df['ground_force_l_px'] = signal.filtfilt(b, a, df['ground_force_l_px'])	
    df['unfiltered_ground_force_l_pz'] = df['ground_force_l_pz']
    df['ground_force_l_pz'] = signal.filtfilt(b, a, df['ground_force_l_pz'])	
    
    df['unfiltered_ground_force_r_px'] = df['ground_force_r_px']
    df['ground_force_r_px'] = signal.filtfilt(b, a, df['ground_force_r_px'])	
    df['unfiltered_ground_force_r_pz'] = df['ground_force_r_pz']
    df['ground_force_r_pz'] = signal.filtfilt(b, a, df['ground_force_r_pz'])
    """	
    df['unfiltered_ground_torque_l_z'] =  df['ground_torque_l_z']
    df['ground_torque_l_z'] = signal.filtfilt(b, a, df['ground_torque_l_z'])
    df['unfiltered_ground_torque_r_z'] =  df['ground_torque_r_z']
    df['ground_torque_r_z'] = signal.filtfilt(b, a, df['ground_torque_r_z'])

    return df


def getMuscleDrivenModel(subject_name: str, model_path: Path, config: dict = None):
    
    model = osim.Model(str(model_path))

    simplify_model(model_name=subject_name, model=model) 

    actuators_in_forceSet = model.getActuators()
    
    for i in range(actuators_in_forceSet.getSize()):
        this_actuator = actuators_in_forceSet.get(i)
        this_actuator_path = actuators_in_forceSet.get(i).getAbsolutePathString()
        
        # Set same value for all upper limb coordinate actuators
        if this_actuator.getConcreteClassName() == 'CoordinateActuator':
            this_coordinate_actuator = osim.CoordinateActuator.safeDownCast(this_actuator)
            this_coordinate_actuator.setOptimalForce(config["coord_actuator_opt_value"])
        
        # Set individual values for each upper limb coordinate actuator
        #if "lumbar" in str(this_actuator_path):
        #    this_coordinate_actuator.setOptimalForce(xxx)
        #if "shoulder" in str(this_actuator_path):
        #    this_coordinate_actuator.setOptimalForce(xxx)
        #if "elbow" in str(this_actuator_path):
        #    this_coordinate_actuator.setOptimalForce(xxx)
        #if "pro_sup" in str(this_actuator_path):
        #    this_coordinate_actuator.setOptimalForce(xxx)
        
    
    modelProcessor = osim.ModelProcessor(model)

    modelProcessor.append(osim.ModOpIgnoreTendonCompliance()) # Turn off tendon compliance for all muscles
    modelProcessor.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016()) # Replace muscles with De Groote-Fregly 2016 muscles (optimization friendly model) for improved force-length and force-velocity characteristics
    modelProcessor.append(osim.ModOpIgnorePassiveFiberForcesDGF()) # Turn off passive fiber forces for all DeGrooteFregly2016Muscles
    modelProcessor.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5)) # Scale the active fiber force curve width for all DeGrooteFregly2016Muscles

    # Add residual CoordinateActuators to the model degrees-of-freedom. This ignores the upper limb and pelvis coordinates which already have residual CoordinateActuators.
    modelProcessor.append(osim.ModOpAddReserves(config["actuator_magnitude"])) 

    if config["add_mocap_GRF"]:
        grf_xml_path = './moco/forces/grf_sit_stand_camille.xml'
        grf_data_path = config["grf_path"]
        modify_datafile_path(grf_xml_path, grf_data_path)
        modelProcessor.append(osim.ModOpAddExternalLoads(grf_xml_path))

    if config["add_osim_GRF_precomputed"]:
        grf_feet_xml_path = '/Users/camilleguillaume/Documents/MasterThesis/assistive-arm/moco/forces/grf_osim_feet.xml'
        grf_feet_data_path = config["grf_osim_feet_precomputed_path"]
        
        df_grf_osim_feet_modified = read_sto_file(grf_feet_data_path)
        df_grf_osim_feet_modified = simplify_grf(df_grf_osim_feet_modified)
        cutoff = 2  # Cutoff frequency in Hz, adjust as needed
        order = 3  # Filter order
        #df_grf_osim_feet_modified = filter_grf(df_grf_osim_feet_modified, cutoff, order)

        new_file_path = write_sto_file(df_grf_osim_feet_modified, grf_feet_data_path, suffix="_new_version")
        modify_datafile_path(grf_feet_xml_path, new_file_path)
        config["grf_osim_feet_precomputed_path"] = new_file_path
        modelProcessor.append(osim.ModOpAddExternalLoads(grf_feet_xml_path))

        grf_chair_xml_path = '/Users/camilleguillaume/Documents/MasterThesis/assistive-arm/moco/forces/grf_osim_chair.xml'
        grf_chair_data_path = config["grf_osim_chair_precomputed_path"]
        
        df_grf_osim_chair_modified = read_sto_file(grf_chair_data_path)
        df_grf_osim_chair_modified.fillna(value=0, inplace=True)
        df_grf_osim_chair_modified = simplify_grf(df_grf_osim_chair_modified)
        cutoff = 4  # Cutoff frequency in Hz, adjust as needed
        order = 3  # Filter orde
        #df_grf_osim_chair_modified = filter_grf(df_grf_osim_chair_modified, cutoff, order)

        new_file_path = write_sto_file(df_grf_osim_chair_modified, grf_chair_data_path, suffix="_new_version")
        modify_datafile_path(grf_chair_xml_path, new_file_path)
        config["grf_osim_chair_precomputed_path"] = new_file_path
        modelProcessor.append(osim.ModOpAddExternalLoads(grf_chair_xml_path))

    model = modelProcessor.process()

    return model, subject_name


def generate_model_with_contacts(model_name: str, model: osim.Model, config: dict, contact_side=None) -> str:
    
    # Return error is side is not None, 'right', or 'left'.
    if contact_side not in ['all', 'right', 'left']:
        raise ValueError('side must be "all", "right", or "left"')
    
    if contact_side == 'all':
        model_name = model_name + "_contacts"
    else:
        model_name = model_name + "_contacts_" + contact_side
    
    #print('Add foot-ground contacts.')
    
    # %% Add contact spheres to the scaled model.
    # The parameters of the foot-ground contacts are based on previous work. We
    # scale the contact sphere locations based on foot dimensions.
    reference_contact_spheres = {
        "s1_r": {"radius": 0.032, "location": np.array([0.0019011578840796601,   -0.01,  -0.00382630379623308]), "orientation": np.array([0, 0, 0]), "socket_frame": "calcn_r"}, # heel
        "s2_r": {"radius": 0.032, "location": np.array([0.14838639994206301,     -0.01,  -0.028713422052654002]), "orientation": np.array([0, 0, 0]), "socket_frame": "calcn_r"}, # interior side
        "s3_r": {"radius": 0.032, "location": np.array([0.13300117060705099,     -0.01,  0.051636247344956601]), "orientation": np.array([0, 0, 0]), "socket_frame": "calcn_r"}, # exterior side
        "s4_r": {"radius": 0.032, "location": np.array([0.066234666199163503,    -0.01,  0.026364160674169801]), "orientation": np.array([0, 0, 0]), "socket_frame": "calcn_r"}, # exterior between heel and side
        "s5_r": {"radius": 0.032, "location": np.array([0.059999999999999998,    -0.01,  -0.018760308461917698]), "orientation": np.array([0, 0, 0]), "socket_frame": "toes_r" }, # toe interior
        "s6_r": {"radius": 0.032, "location": np.array([0.044999999999999998,    -0.01,  0.061856956754965199]), "orientation": np.array([0, 0, 0]), "socket_frame": "toes_r" }, # toe exterior
        "s1_l": {"radius": 0.032, "location": np.array([0.0019011578840796601,   -0.01,  0.00382630379623308]), "orientation": np.array([0, 0, 0]), "socket_frame": "calcn_l"},
        "s2_l": {"radius": 0.032, "location": np.array([0.14838639994206301,     -0.01,  0.028713422052654002]), "orientation": np.array([0, 0, 0]), "socket_frame": "calcn_l"},
        "s3_l": {"radius": 0.032, "location": np.array([0.13300117060705099,     -0.01,  -0.051636247344956601]), "orientation": np.array([0, 0, 0]), "socket_frame": "calcn_l"},
        "s4_l": {"radius": 0.032, "location": np.array([0.066234666199163503,    -0.01,  -0.026364160674169801]), "orientation": np.array([0, 0, 0]), "socket_frame": "calcn_l"},
        "s5_l": {"radius": 0.032, "location": np.array([0.059999999999999998,    -0.01,  0.018760308461917698]), "orientation": np.array([0, 0, 0]), "socket_frame": "toes_l" },
        "s6_l": {"radius": 0.032, "location": np.array([0.044999999999999998,    -0.01,  -0.061856956754965199]), "orientation": np.array([0, 0, 0]), "socket_frame": "toes_l" }
        }      
    reference_scale_factors = {"calcn_r": np.array([0.91392399999999996, 0.91392399999999996, 0.91392399999999996]),
                               "toes_r":  np.array([0.91392399999999996, 0.91392399999999996, 0.91392399999999996]),
                               "calcn_l": np.array([0.91392399999999996, 0.91392399999999996, 0.91392399999999996]),
                               "toes_l":  np.array([0.91392399999999996, 0.91392399999999996, 0.91392399999999996])}
    reference_contact_half_space = {"name": "floor", "location": np.array([0, 0, 0]),"orientation": np.array([0, 0, -np.pi/2]), "frame": "ground"}
    
    if "sts" in config["trial_name"]:
        if "s7" == config["osim_GRF_type"]:
            reference_contact_spheres_chair = {
                "s7_l": {"radius": 0.032, "location": np.array([-0.1, -0.15, -0.10]), "orientation": np.array([0, 0, 0]), "socket_frame": "pelvis"},
                "s7_r": {"radius": 0.032, "location": np.array([-0.1, -0.15, 0.10]), "orientation": np.array([0, 0, 0]), "socket_frame": "pelvis"}}
        elif "s8" == config["osim_GRF_type"]:
            reference_contact_spheres_chair = {
                "s8_l": {"radius": 0.1, "location": np.array([-0.15, -0.08, -0.07]), "orientation": np.array([0, 0, 0]), "socket_frame": "pelvis"},
                "s8_r": {"radius": 0.1, "location": np.array([-0.15, -0.08, 0.07]), "orientation": np.array([0, 0, 0]), "socket_frame": "pelvis"}}
        elif "s9" == config["osim_GRF_type"]:
            reference_contact_spheres_chair = {
                "s9": {"radius": 0.032, "location": np.array([-0.1, -0.15, 0]), "orientation": np.array([0, 0, 0]), "socket_frame": "pelvis"}}
        else:
            print("Error in osim_GRF_type syntax: must be either s7, s8 or s9")
            sys.exit(1)

        initial_feet_markers, final_feet_markers = get_reference_markers(config["trial_markers_path"])
        #print(initial_feet_markers)
        #print(final_feet_markers)
        
        if ("s7_l" in reference_contact_spheres_chair) or ("s9" in reference_contact_spheres_chair):
            print("s7 or s9")
            initial_step_x = initial_feet_markers["mean_ankle_X"]+0.4
            initial_step_y = initial_feet_markers["mean_ankle_Y"]-0.09
            chair_height = 0.45
            chair_distance = 0.5
        if "s8_l" in reference_contact_spheres_chair:
            print("s8")
            initial_step_x = initial_feet_markers["mean_ankle_X"]+0.4
            initial_step_y = initial_feet_markers["mean_ankle_Y"]-0.09
            chair_height = 0.425
            chair_distance = 0.5  

        reference_contact_half_space_chair = {"name": "chair", "location": np.array([initial_step_x-chair_distance, chair_height, 0]),"orientation": np.array([0, 0, -np.pi/2]), "frame": "ground"}
        reference_contact_half_space = {"name": "floor_sts", "location": np.array([initial_step_x, initial_step_y, 0]),"orientation": np.array([0, 0, -np.pi/2]), "frame": "ground"}


    if "stairs" in config["trial_name"]:

        initial_feet_markers, final_feet_markers = get_reference_markers(config["trial_markers_path"])
        # "mean_ankle_Y", "mean_calc_Y", "mean_5meta_Y", "mean_toe_Y", "mean_ankle_X", "mean_calc_X", "mean_5meta_X", "mean_toe_X"
        #print(initial_feet_markers)
        #print(final_feet_markers)

        initial_step_x = initial_feet_markers["mean_ankle_X"]+0.4
        initial_step_y = initial_feet_markers["mean_ankle_Y"]-0.1
        step_height=0.172
        first_step=0.50
        step_width=0.28

        reference_contact_half_space_step1 = {"name": "step1", "location": np.array([initial_step_x, initial_step_y, 0]),"orientation": np.array([0, 0, -np.pi/2]), "frame": "ground"}
        reference_contact_half_space_step2 = {"name": "step2", "location": np.array([initial_step_x+first_step, initial_step_y+step_height, 0]),"orientation": np.array([0, 0, -np.pi/2]), "frame": "ground"}
        reference_contact_half_space_step3 = {"name": "step3", "location": np.array([initial_step_x+first_step+step_width, initial_step_y+step_height*2, 0]),"orientation": np.array([0, 0, -np.pi/2]), "frame": "ground"}
        reference_contact_half_space_step4 = {"name": "step4", "location": np.array([initial_step_x+first_step+step_width*2, initial_step_y+step_height*3, 0]),"orientation": np.array([0, 0, -np.pi/2]), "frame": "ground"}
        reference_contact_half_space_step5 = {"name": "step5", "location": np.array([initial_step_x+first_step+step_width*3, initial_step_y+step_height*4, 0]),"orientation": np.array([0, 0, -np.pi/2]), "frame": "ground"}


    stiffness = 1000000
    dissipation = 2.0
    static_friction = 0.8
    dynamic_friction = 0.8
    viscous_friction = 0.5
    transition_velocity = 0.2

    # Add contact spheres and SmoothSphereHalfSpaceForces.
    #osim.Logger.setLevelString('error')  
    bodySet = model.get_BodySet()
    
    # ContactHalfSpaces.
    if reference_contact_half_space["frame"] == "ground":
        contact_half_space_frame = model.get_ground()
    else:
        raise ValueError('Not yet supported.')
        
    contactHalfSpace = osim.ContactHalfSpace(
        osim.Vec3(reference_contact_half_space["location"]),
        osim.Vec3(reference_contact_half_space["orientation"]),
        contact_half_space_frame, reference_contact_half_space["name"])
    contactHalfSpace.connectSocket_frame(contact_half_space_frame)
    model.addContactGeometry(contactHalfSpace)
    
    if "stairs" in config["trial_name"]:
        contactHalfSpace_step1 = osim.ContactHalfSpace(
            osim.Vec3(reference_contact_half_space_step1["location"]),
            osim.Vec3(reference_contact_half_space_step1["orientation"]),
            contact_half_space_frame, reference_contact_half_space_step1["name"])
        contactHalfSpace_step1.connectSocket_frame(contact_half_space_frame)
        model.addContactGeometry(contactHalfSpace_step1)

        contactHalfSpace_step2 = osim.ContactHalfSpace(
            osim.Vec3(reference_contact_half_space_step2["location"]),
            osim.Vec3(reference_contact_half_space_step2["orientation"]),
            contact_half_space_frame, reference_contact_half_space_step2["name"])
        contactHalfSpace_step2.connectSocket_frame(contact_half_space_frame)
        model.addContactGeometry(contactHalfSpace_step2)

        contactHalfSpace_step3 = osim.ContactHalfSpace(
            osim.Vec3(reference_contact_half_space_step3["location"]),
            osim.Vec3(reference_contact_half_space_step3["orientation"]),
            contact_half_space_frame, reference_contact_half_space_step3["name"])
        contactHalfSpace_step3.connectSocket_frame(contact_half_space_frame)
        model.addContactGeometry(contactHalfSpace_step3)

        contactHalfSpace_step4 = osim.ContactHalfSpace(
            osim.Vec3(reference_contact_half_space_step4["location"]),
            osim.Vec3(reference_contact_half_space_step4["orientation"]),
            contact_half_space_frame, reference_contact_half_space_step4["name"])
        contactHalfSpace_step4.connectSocket_frame(contact_half_space_frame)
        model.addContactGeometry(contactHalfSpace_step4)

        contactHalfSpace_step5 = osim.ContactHalfSpace(
            osim.Vec3(reference_contact_half_space_step5["location"]),
            osim.Vec3(reference_contact_half_space_step5["orientation"]),
            contact_half_space_frame, reference_contact_half_space_step5["name"])
        contactHalfSpace_step5.connectSocket_frame(contact_half_space_frame)
        model.addContactGeometry(contactHalfSpace_step5)

    if "sts" in config["trial_name"]:
        contactHalfSpace_chair = osim.ContactHalfSpace(
            osim.Vec3(reference_contact_half_space_chair["location"]),
            osim.Vec3(reference_contact_half_space_chair["orientation"]),
            contact_half_space_frame, reference_contact_half_space_chair["name"])
        contactHalfSpace_chair.connectSocket_frame(contact_half_space_frame)
        model.addContactGeometry(contactHalfSpace_chair)

    # ContactSpheres and SmoothSphereHalfSpaceForces.
    for ref_contact_sphere in reference_contact_spheres:

        if contact_side == 'right' and '_l' in ref_contact_sphere:
            continue
        if contact_side == 'left' and '_r' in ref_contact_sphere:
            continue

        # ContactSpheres.
        body = bodySet.get(reference_contact_spheres[ref_contact_sphere]["socket_frame"])
        # Scale location based on attached_geometry scale_factors.      
        # We don't scale the y_position.
        attached_geometry = body.get_attached_geometry(0)
        c_scale_factors = attached_geometry.get_scale_factors().to_numpy() 
        c_ref_scale_factors = reference_scale_factors[reference_contact_spheres[ref_contact_sphere]["socket_frame"]]
        scale_factors = c_ref_scale_factors / c_scale_factors        
        scale_factors[1] = 1        
        scaled_location = reference_contact_spheres[ref_contact_sphere]["location"] / scale_factors
        c_contactSphere = osim.ContactSphere(
            reference_contact_spheres[ref_contact_sphere]["radius"],
            osim.Vec3(scaled_location), body, ref_contact_sphere)
        c_contactSphere.connectSocket_frame(body)
        model.addContactGeometry(c_contactSphere)
        
        # SmoothSphereHalfSpaceForces.
        SmoothSphereHalfSpaceForce = osim.SmoothSphereHalfSpaceForce(
            "SmoothSphereHalfSpaceForce_" + ref_contact_sphere, 
            c_contactSphere, contactHalfSpace)
        SmoothSphereHalfSpaceForce.set_stiffness(stiffness)
        SmoothSphereHalfSpaceForce.set_dissipation(dissipation)
        SmoothSphereHalfSpaceForce.set_static_friction(static_friction)
        SmoothSphereHalfSpaceForce.set_dynamic_friction(dynamic_friction)
        SmoothSphereHalfSpaceForce.set_viscous_friction(viscous_friction)
        SmoothSphereHalfSpaceForce.set_transition_velocity(transition_velocity)        
        SmoothSphereHalfSpaceForce.connectSocket_half_space(contactHalfSpace)
        SmoothSphereHalfSpaceForce.connectSocket_sphere(c_contactSphere)
        model.addForce(SmoothSphereHalfSpaceForce)
    
        if "stairs" in config["trial_name"]:
            # SmoothSphereHalfSpaceForces associated to stairs
            SmoothSphereHalfSpaceForce = osim.SmoothSphereHalfSpaceForce(
                "SmoothSphereHalfSpaceForce_" + ref_contact_sphere, 
                c_contactSphere, contactHalfSpace_step1)
            SmoothSphereHalfSpaceForce.set_stiffness(stiffness)
            SmoothSphereHalfSpaceForce.set_dissipation(dissipation)
            SmoothSphereHalfSpaceForce.set_static_friction(static_friction)
            SmoothSphereHalfSpaceForce.set_dynamic_friction(dynamic_friction)
            SmoothSphereHalfSpaceForce.set_viscous_friction(viscous_friction)
            SmoothSphereHalfSpaceForce.set_transition_velocity(transition_velocity)        
            SmoothSphereHalfSpaceForce.connectSocket_half_space(contactHalfSpace_step1)
            SmoothSphereHalfSpaceForce.connectSocket_sphere(c_contactSphere)
            model.addForce(SmoothSphereHalfSpaceForce)

            SmoothSphereHalfSpaceForce = osim.SmoothSphereHalfSpaceForce(
                "SmoothSphereHalfSpaceForce_" + ref_contact_sphere, 
                c_contactSphere, contactHalfSpace_step2)
            SmoothSphereHalfSpaceForce.set_stiffness(stiffness)
            SmoothSphereHalfSpaceForce.set_dissipation(dissipation)
            SmoothSphereHalfSpaceForce.set_static_friction(static_friction)
            SmoothSphereHalfSpaceForce.set_dynamic_friction(dynamic_friction)
            SmoothSphereHalfSpaceForce.set_viscous_friction(viscous_friction)
            SmoothSphereHalfSpaceForce.set_transition_velocity(transition_velocity)        
            SmoothSphereHalfSpaceForce.connectSocket_half_space(contactHalfSpace_step2)
            SmoothSphereHalfSpaceForce.connectSocket_sphere(c_contactSphere)
            model.addForce(SmoothSphereHalfSpaceForce)

            SmoothSphereHalfSpaceForce = osim.SmoothSphereHalfSpaceForce(
                "SmoothSphereHalfSpaceForce_" + ref_contact_sphere, 
                c_contactSphere, contactHalfSpace_step3)
            SmoothSphereHalfSpaceForce.set_stiffness(stiffness)
            SmoothSphereHalfSpaceForce.set_dissipation(dissipation)
            SmoothSphereHalfSpaceForce.set_static_friction(static_friction)
            SmoothSphereHalfSpaceForce.set_dynamic_friction(dynamic_friction)
            SmoothSphereHalfSpaceForce.set_viscous_friction(viscous_friction)
            SmoothSphereHalfSpaceForce.set_transition_velocity(transition_velocity)        
            SmoothSphereHalfSpaceForce.connectSocket_half_space(contactHalfSpace_step3)
            SmoothSphereHalfSpaceForce.connectSocket_sphere(c_contactSphere)
            model.addForce(SmoothSphereHalfSpaceForce)

            SmoothSphereHalfSpaceForce = osim.SmoothSphereHalfSpaceForce(
                "SmoothSphereHalfSpaceForce_" + ref_contact_sphere, 
                c_contactSphere, contactHalfSpace_step4)
            SmoothSphereHalfSpaceForce.set_stiffness(stiffness)
            SmoothSphereHalfSpaceForce.set_dissipation(dissipation)
            SmoothSphereHalfSpaceForce.set_static_friction(static_friction)
            SmoothSphereHalfSpaceForce.set_dynamic_friction(dynamic_friction)
            SmoothSphereHalfSpaceForce.set_viscous_friction(viscous_friction)
            SmoothSphereHalfSpaceForce.set_transition_velocity(transition_velocity)        
            SmoothSphereHalfSpaceForce.connectSocket_half_space(contactHalfSpace_step4)
            SmoothSphereHalfSpaceForce.connectSocket_sphere(c_contactSphere)
            model.addForce(SmoothSphereHalfSpaceForce)

            SmoothSphereHalfSpaceForce = osim.SmoothSphereHalfSpaceForce(
                "SmoothSphereHalfSpaceForce_" + ref_contact_sphere, 
                c_contactSphere, contactHalfSpace_step5)
            SmoothSphereHalfSpaceForce.set_stiffness(stiffness)
            SmoothSphereHalfSpaceForce.set_dissipation(dissipation)
            SmoothSphereHalfSpaceForce.set_static_friction(static_friction)
            SmoothSphereHalfSpaceForce.set_dynamic_friction(dynamic_friction)
            SmoothSphereHalfSpaceForce.set_viscous_friction(viscous_friction)
            SmoothSphereHalfSpaceForce.set_transition_velocity(transition_velocity)        
            SmoothSphereHalfSpaceForce.connectSocket_half_space(contactHalfSpace_step5)
            SmoothSphereHalfSpaceForce.connectSocket_sphere(c_contactSphere)
            model.addForce(SmoothSphereHalfSpaceForce)
            
    if "sts" in config["trial_name"]:
        for ref_contact_sphere in reference_contact_spheres_chair:
            # ContactSpheres.
            body = bodySet.get(reference_contact_spheres_chair[ref_contact_sphere]["socket_frame"])
            c_contactSphere = osim.ContactSphere(
                reference_contact_spheres_chair[ref_contact_sphere]["radius"],
                osim.Vec3(reference_contact_spheres_chair[ref_contact_sphere]["location"]), body, ref_contact_sphere)
            c_contactSphere.connectSocket_frame(body)
            model.addContactGeometry(c_contactSphere)

            # SmoothSphereHalfSpaceForces.
            SmoothSphereHalfSpaceForce = osim.SmoothSphereHalfSpaceForce(
                "SmoothSphereHalfSpaceForce_" + ref_contact_sphere, 
                c_contactSphere, contactHalfSpace_chair)
            SmoothSphereHalfSpaceForce.set_stiffness(stiffness)
            SmoothSphereHalfSpaceForce.set_dissipation(dissipation)
            SmoothSphereHalfSpaceForce.set_static_friction(static_friction)
            SmoothSphereHalfSpaceForce.set_dynamic_friction(dynamic_friction)
            SmoothSphereHalfSpaceForce.set_viscous_friction(viscous_friction)
            SmoothSphereHalfSpaceForce.set_transition_velocity(transition_velocity)        
            SmoothSphereHalfSpaceForce.connectSocket_half_space(contactHalfSpace_chair)
            SmoothSphereHalfSpaceForce.connectSocket_sphere(c_contactSphere)
            model.addForce(SmoothSphereHalfSpaceForce)


    return model_name


def get_reference_markers(markers_path: str): 
    
    # Set directories
    opencap_markers = pd.read_csv(markers_path, delimiter="\t", skiprows=3).tail(-1)
    filtered_markers = prepare_opencap_markers(opencap_markers)

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
    #print("initial markers: ", initial_markers)

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


def get_model(subject_name: str, model_path: Path, target_path: Path, config: dict) -> osim.Model:

    config["target_path"] = str(target_path) # path to store updated model

    model, subject_name = getMuscleDrivenModel(
        subject_name = subject_name,
        model_path = str(model_path),
        add_mocap_GRF = config["add_mocap_GRF"],
        add_osim_GRF_precomputed = config["add_osim_GRF_precomputed"],
        config = config,
    )

    model_name = f"{subject_name}_{model_path.stem}"
    model.setName(model_name)

    # Add external assistive force
    if config["assist_with_force_ext"]:
        add_assistive_force(
            coordName="pelvis_tx",
            model=model,
            name="assistive_force_x",
            direction=osim.Vec3(1, 0, 0),
            magnitude=config["force_ext_opt_value"],
        )
        add_assistive_force(
            coordName="pelvis_ty",
            model=model,
            name="assistive_force_y",
            direction=osim.Vec3(0, 1, 0),
            magnitude=config["force_ext_opt_value"],
        )

    coordSet = model.updCoordinateSet()

    if config["add_osim_GRF_online"]:
        generate_model_with_contacts(model_name=subject_name, model=model, config=config, contact_side='all')

    
    if config["assist_with_reserve_pelvis_txy"]:
        forceSet = model.getForceSet()
        
        for i in range(forceSet.getSize()):
            forcePath = forceSet.get(i).getAbsolutePathString()
            this_actuator = forceSet.get(i)
            #default values: actu.setMinControl(-np.inf), actu.setMaxControl(np.inf) 

            if "reserve_jointset_ground_pelvis_pelvis_tx" in str(forcePath):
                print("set reserve pelvis tx optimum force to ",config["reserve_pelvis_opt_value"])
                this_coordinate_actuator = osim.CoordinateActuator.safeDownCast(this_actuator)
                this_coordinate_actuator.setOptimalForce(config["reserve_pelvis_opt_value"])
                
            if "reserve_jointset_ground_pelvis_pelvis_ty" in str(forcePath):
                print("set reserve pelvis ty optimum force to ",config["reserve_pelvis_opt_value"])
                this_coordinate_actuator = osim.CoordinateActuator.safeDownCast(this_actuator)
                this_coordinate_actuator.setOptimalForce(config["reserve_pelvis_opt_value"])
                if config["pelvis_ty_positive"]:
                    this_coordinate_actuator.setMinControl(0)

    model.finalizeConnections()

    model.printToXML(f"./moco/models/{model_name}.osim")
    model.printToXML(str(target_path / f"{model_name}.osim"))

    config["xml_path"] = f"./moco/models/{model_name}.osim"

    return model


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
    tableProcessor.append(osim.TabOpLowPassFilter(6))
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


def set_state_tracking_weights(moco_study: osim.MocoStudy, config: dict):
    
    problem = moco_study.updProblem()
    
    stateTrackingGoal = osim.MocoStateTrackingGoal.safeDownCast(problem.updGoal('state_tracking'))
    
    stateTrackingGoal.setWeightForState('/jointset/ground_pelvis/pelvis_tilt/value', 10.0)
    stateTrackingGoal.setWeightForState('/jointset/ground_pelvis/pelvis_tilt/speed', 10.0)
    stateTrackingGoal.setWeightForState('/jointset/back/lumbar_extension/value', 10.0)
    stateTrackingGoal.setWeightForState('/jointset/back/lumbar_extension/speed', 10.0)
    stateTrackingGoal.setWeightForState('/jointset/hip_r/hip_flexion_r/value', 10.0)
    stateTrackingGoal.setWeightForState('/jointset/hip_r/hip_flexion_r/speed', 10.0)
    stateTrackingGoal.setWeightForState('/jointset/hip_l/hip_flexion_l/value', 10.0)
    stateTrackingGoal.setWeightForState('/jointset/hip_l/hip_flexion_l/speed', 10.0)
    stateTrackingGoal.setWeightForState('/jointset/walker_knee_r/knee_angle_r/value', 10.0)
    stateTrackingGoal.setWeightForState('/jointset/walker_knee_r/knee_angle_r/speed', 10.0)
    stateTrackingGoal.setWeightForState('/jointset/walker_knee_l/knee_angle_l/value', 10.0)
    stateTrackingGoal.setWeightForState('/jointset/walker_knee_l/knee_angle_l/speed', 10.0)
    stateTrackingGoal.setWeightForState('/jointset/ankle_r/ankle_angle_r/speed', 10.0)
    stateTrackingGoal.setWeightForState('/jointset/ankle_r/ankle_angle_r/value', 10.0)
    stateTrackingGoal.setWeightForState('/jointset/ankle_l/ankle_angle_l/speed', 10.0)
    stateTrackingGoal.setWeightForState('/jointset/ankle_l/ankle_angle_l/value', 10.0)

    if config["tracking_weight_pelvis_txy"] != 1:
        print("set pelvis tx and ty tracking weights to: ", config["tracking_weight_pelvis_txy"])
        stateTrackingGoal.setWeightForState('/jointset/ground_pelvis/pelvis_tx/value', config["tracking_weight_pelvis_txy"])
        stateTrackingGoal.setWeightForState('/jointset/ground_pelvis/pelvis_ty/value', config["tracking_weight_pelvis_txy"])



def set_state_bounds(moco_study: osim.MocoStudy, config: dict):
    
    problem = moco_study.updProblem()
    
    fraction_extra_bound_size = 0.25
    tracked_states_file = "tracking_problem_tracked_states.sto"
    tracked_states_table = osim.TimeSeriesTable(tracked_states_file)
    tracked_states_table.trimFrom(config["t_0"])
    tracked_states_table.trimTo(config["t_f"])
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
                                                  


def set_moco_problem_weights(model: osim.Model, moco_study: osim.MocoStudy, config: dict):
    
    problem = moco_study.updProblem()
    problem.addGoal(osim.MocoInitialActivationGoal("activation"))

    # Get a reference to the MocoControlGoal that is added to every MocoTrack problem by default and change the weights
    effort_goal = osim.MocoControlGoal.safeDownCast(problem.updGoal("control_effort"))

    forceSet = model.getForceSet()
    for i in range(forceSet.getSize()):
        forcePath = forceSet.get(i).getAbsolutePathString()
        
        if "reserve_jointset_ground_pelvis" in str(forcePath):
            effort_goal.setWeightForControl(forcePath, config["reserve_pelvis_weight"])
        
        if config["assist_with_reserve_pelvis_txy"]:
            if "reserve_jointset_ground_pelvis_pelvis_tx" in str(forcePath):
                print("set reserve pelvis tx control weight to ", config["reserve_pelvis_weight_txy"])
                effort_goal.setWeightForControl(forcePath, config["reserve_pelvis_weight_txy"])
            if "reserve_jointset_ground_pelvis_pelvis_ty" in str(forcePath):
                print("set reserve pelvis ty control weight to ", config["reserve_pelvis_weight_txy"])
                effort_goal.setWeightForControl(forcePath, config["reserve_pelvis_weight_txy"])
        
        if "reserve_jointset_hip" in str(forcePath):
            effort_goal.setWeightForControl(forcePath, config["reserve_hip_weight"])
        if "reserve_jointset_walker_knee" in str(forcePath):
            effort_goal.setWeightForControl(forcePath, config["reserve_knee_weight"])
        if "reserve_jointset_ankle" in str(forcePath):
            effort_goal.setWeightForControl(forcePath, config["reserve_feet_weight"])
        if "reserve_jointset_subtalar" in str(forcePath):
            effort_goal.setWeightForControl(forcePath, config["reserve_feet_weight"])
        if "reserve_jointset_mtp" in str(forcePath):
            effort_goal.setWeightForControl(forcePath, config["reserve_feet_weight"])

            
    if config["control_effort_muscles"] !=1:
        for side in ['r', 'l']:
            effort_goal.setWeightForControl(f"/forceset/bflh_{side}", config["control_effort_muscles"])
            effort_goal.setWeightForControl(f"/forceset/bfsh_{side}", config["control_effort_muscles"])
            effort_goal.setWeightForControl(f"/forceset/gasmed_{side}", config["control_effort_muscles"])
            effort_goal.setWeightForControl(f"/forceset/glmax1_{side}", config["control_effort_muscles"])
            effort_goal.setWeightForControl(f"/forceset/glmax2_{side}", config["control_effort_muscles"])
            effort_goal.setWeightForControl(f"/forceset/glmax3_{side}", config["control_effort_muscles"])
            effort_goal.setWeightForControl(f"/forceset/glmed1_{side}", config["control_effort_muscles"])
            effort_goal.setWeightForControl(f"/forceset/glmed2_{side}", config["control_effort_muscles"])
            effort_goal.setWeightForControl(f"/forceset/glmed3_{side}", config["control_effort_muscles"])
            effort_goal.setWeightForControl(f"/forceset/psoas_{side}", config["control_effort_muscles"])
            effort_goal.setWeightForControl(f"/forceset/recfem_{side}", config["control_effort_muscles"])
            effort_goal.setWeightForControl(f"/forceset/soleus_{side}", config["control_effort_muscles"])
            effort_goal.setWeightForControl(f"/forceset/tibant_{side}", config["control_effort_muscles"])
            effort_goal.setWeightForControl(f"/forceset/vaslat_{side}", config["control_effort_muscles"])
            effort_goal.setWeightForControl(f"/forceset/vasmed_{side}", config["control_effort_muscles"])

    if config["assist_with_force_ext"]:
        effort_goal.setWeightForControl("/forceset/assistive_force_x", 0)
        effort_goal.setWeightForControl("/forceset/assistive_force_y", 0)


def set_moco_problem_solver_settings(moco_study: osim.MocoStudy, nb_max_iterations: int = None):
    # Modify the solver settings: 
    solver = osim.MocoCasADiSolver.safeDownCast(moco_study.updSolver())
    if nb_max_iterations:
        solver.set_optim_max_iterations(nb_max_iterations) # Stop optimization after 500 iterations => solution failed
    
    #solver.set_output_interval(10) # Monitor solver progress by writing every 10th iterate to file, cf. doc
    #solver.set_optim_convergence_tolerance(1e-3) # Loosen the convergence tolerances, cf. doc
    #solver.set_optim_constraint_tolerance(1e-3) # Loosen the constraint tolerances, cf. doc



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
        print("control bounds assistive force x: [-1,1]")
        assistActuator.setMinControl(-np.inf) #manage the bounds on ScalarActuator's control
        assistActuator.setMaxControl(np.inf)  #manage the bounds on ScalarActuator's control
    if coordName == "pelvis_ty":
        print("control bounds assistive force y: [0,inf]")
        assistActuator.setMinControl(0) #manage the bounds on ScalarActuator's control
        assistActuator.setMaxControl(np.inf) #manage the bounds on ScalarActuator's control

    assistActuator.setOptimalForce(magnitude) #250 in exampleWalking.m

    model.addForce(assistActuator)


def extract_GRF(model, solution, solution_path_assistive_arm, solution_path_local, config):
    #Extract ground reaction forces for downstream analysis. Add the contact force elements to vectors, then use Moco's
    # createExternalLoadsTableForGait() function.
    contact_r = osim.StdVectorString()
    contact_l = osim.StdVectorString()

    contact_chair_r = osim.StdVectorString()
    contact_chair_l = osim.StdVectorString()

    # Feet GRFs
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
    
    # Chair GRFs
    if config["osim_GRF_type"] == "s7":
        contact_chair_r.append('/forceset/SmoothSphereHalfSpaceForce_s7_r')
        contact_chair_l.append('/forceset/SmoothSphereHalfSpaceForce_s7_l')
    if config["osim_GRF_type"] == "s8":
        contact_chair_r.append('/forceset/SmoothSphereHalfSpaceForce_s8_r')
        contact_chair_l.append('/forceset/SmoothSphereHalfSpaceForce_s8_l')
    if config["osim_GRF_type"] == "s9":
        contact_chair_r.append('/forceset/SmoothSphereHalfSpaceForce_s9_r')
        contact_chair_l.append('/forceset/SmoothSphereHalfSpaceForce_s9_l')
    
    model_processor = osim.ModelProcessor(model)
    model = model_processor.process()
    externalForcesTableFlat_feet = osim.createExternalLoadsTableForGait(model, solution.exportToStatesTrajectory(model), contact_r, contact_l)
    externalForcesTableFlat_chair = osim.createExternalLoadsTableForGait(model, solution.exportToStatesTrajectory(model), contact_chair_r, contact_chair_l)

    osim.STOFileAdapter.write(externalForcesTableFlat_feet, f"{solution_path_assistive_arm}_grfs_osim_feet.sto")
    osim.STOFileAdapter.write(externalForcesTableFlat_feet, f"{solution_path_local}_grfs_osim_feet.sto")
    osim.STOFileAdapter.write(externalForcesTableFlat_chair, f"{solution_path_assistive_arm}_grfs_osim_chair.sto")
    osim.STOFileAdapter.write(externalForcesTableFlat_chair, f"{solution_path_local}_grfs_osim_chair.sto")


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