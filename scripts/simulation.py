#import numpy as np
import os
import yaml
import sys
from datetime import datetime
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parent.parent))
from assistive_arm.moco_helpers import (
    get_model,
    get_tracking_problem,
    set_moco_problem_weights,
    set_state_tracking_weights,
    set_moco_problem_solver_settings,
    extract_GRF,
    set_state_bounds,
    check_simulation_parameters,
    check_assistance_parameters,
)

def main():
    # Session paramaters
    subject = "CG" # CG / ZK
    task = "sts" # sts / reaching / stairs
    date = "20240503"
    assistance = "no_arm" # no_arm / assistive_arm_off / assistive_arm_on
    trial_ID = "1" # "1","2","3" for sts / "LRLR","LRLR2","LRLR3" for stairs
    model_type = "simple" # simple / full

    # Assistive force parameters
    assistive_force = True
    if not assistive_force:
        assist_with_force_ext = False   
        force_ext_opt_value = None
        assist_with_reserve_pelvis_txy = False
    else:
        assist_with_force_ext = False   
        force_ext_opt_value = None # 700 (N) or None, previous param name "assistive_force"
        assist_with_reserve_pelvis_txy = True
        check_assistance_parameters(assist_with_force_ext, force_ext_opt_value, assist_with_reserve_pelvis_txy)

    # Simulation parameters
    mesh_interval = 0.05 # time resolution in tracking problem -> default value of 0.08 seconds

    # Ground Reaction Forces parameters
    add_mocap_GRF = False # add GRF computed from mocap force plates as inputs
    add_osim_GRF_online = True # compute GRF during opensim simulation
    add_osim_GRF_precomputed = False # add GRF computed from previous opensim simulation as inputs
    osim_GRF_type = None
    if task == "sts":
        osim_GRF_type = "s8" #s7, s8, s9, type of chair contact model (s7: two small spheres, s8: two large spheres, s9: one small sphere)
    
    # Tracking parameters
    tracking_weight = 1
    tracking_weight_pelvis_txy = 1 #default 1
    
    # Effort parameters
    control_effort_weight = 0.001
    control_effort_muscles = 20 
    coord_actuator_opt_value = 250 # coordinate acturators: lumbar, arms (Nm)
    reserve_actuator_opt_value = 10 # reserve actuators: pelvis, legs, feet (Nm)
    reserve_pelvis_weight = 15 
    reserve_hip_weight = 5 
    reserve_knee_weight = 5 
    reserve_feet_weight = 1 
    if assist_with_reserve_pelvis_txy:
        reserve_pelvis_weight_txy = 0
        reserve_pelvis_opt_value = 700
        pelvis_ty_positive = True
    else:
        reserve_pelvis_weight_txy = None
        reserve_pelvis_opt_value = None
        pelvis_ty_positive = False

    # Session datas
    session = Path(f"/Users/camilleguillaume/Documents/MasterThesis/Opencap_data/{date}_opencap_{task}_{subject}")
    trial_name = f"{task}_{assistance}_{trial_ID}"
    model_name = f"{subject}_{model_type}_{trial_name}_osimGRF-{add_osim_GRF_online}"  
    if task == "sts":
        model_name = f"{subject}_{model_type}_{trial_name}_osimGRF-{add_osim_GRF_online}-{osim_GRF_type}" 
    model_path = session / "OpenSimData" / "Model" / "LaiUhlrich2022_scaled.osim"
    yaml_path = session / "sessionMetadata.yaml"

    # GRF datas
    if assist_with_force_ext and add_osim_GRF_precomputed:
        date_simulation_grf_osim = "2024-07-30"
        time_simulation_grf_osim = "16-28"
        #Sanity check
        simulation_yaml_path = Path(f"./moco/control_solutions/{trial_name}/{subject}_{model_type}_{trial_name}_assistance_none_{date_simulation_grf_osim}_{time_simulation_grf_osim}.yaml")
        with open(simulation_yaml_path, 'r') as f:
            simulation_session_Metadata = yaml.load(f, Loader=yaml.FullLoader)
        expected_values = {
            "osim_GRF_type": osim_GRF_type,
            "mesh_interval": mesh_interval,
            "actuator_magnitude": reserve_actuator_opt_value,
            "coord_actuator_opt_value": coord_actuator_opt_value,
            "reserve_pelvis_weight": reserve_pelvis_weight,
            "reserve_hip_weight": reserve_hip_weight,
            "reserve_knee_weight": reserve_knee_weight,
            "reserve_feet_weight": reserve_feet_weight
        }
        check_simulation_parameters(simulation_session_Metadata, expected_values)
        #Load GRFs datas from this simulation
        grf_osim_feet_precomputed_path = str(Path(f"/Users/camilleguillaume/Documents/MasterThesis/assistive-arm/moco/control_solutions/{trial_name}/{subject}_{model_type}_{trial_name}_assistance_none_{date_simulation_grf_osim}_{time_simulation_grf_osim}_grfs_osim_feet.sto"))
        grf_osim_chair_precomputed_path = str(Path(f"/Users/camilleguillaume/Documents/MasterThesis/assistive-arm/moco/control_solutions/{trial_name}/{subject}_{model_type}_{trial_name}_assistance_none_{date_simulation_grf_osim}_{time_simulation_grf_osim}_grfs_osim_chair.sto"))
        print("GRFs loaded from previous OpenSim simulation: ",date_simulation_grf_osim," ",time_simulation_grf_osim)
    
    # Solution
    solution_folder_local = session / "control_solutions" / trial_name
    solution_folder_CurDir = Path(f"./moco/control_solutions/{trial_name}")
    if not os.path.exists(solution_folder_local):
        os.makedirs(solution_folder_local)
    if not os.path.exists(solution_folder_CurDir):
        os.makedirs(solution_folder_CurDir)
    cur_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    solution_name = f"{subject}_{model_type}_{trial_name}_assistance_{str(assist_with_force_ext).lower()}_{cur_time}"
    solution_path_local = solution_folder_local / f"{solution_name}.sto"
    solution_path_CurDir = solution_folder_CurDir / f"{solution_name}.sto"

    # Time parameters
    with open(yaml_path, 'r') as f:
        session_Metadata = yaml.load(f, Loader=yaml.FullLoader)
    if task == "sts":
        t_0 = session_Metadata[f"{trial_name}"][0] #Subject standing, before sitting
        t_i = session_Metadata[f"{trial_name}"][1] #Subject sitting, before sts
        t_f = session_Metadata[f"{trial_name}"][2] #Subject standing, after sts

    if task == "stairs": 
        if subject == "CG" and trial_name == "stairs_no_arm_LRLR2":
            # first 2 steps
            #t_0=3
            #t_f=4.6
            # last 2 steps
            t_0=4.72
            t_f=7.1
        if subject == "ZK" and trial_name == "stairs_no_arm_LRLR":
            t_0 = 3.8 
            t_f = 5.7 
        if subject == "ZK" and trial_name == "stairs_no_arm_LRLR2":
            t_0 = 3.5
            t_f = 5.5
        if subject == "ZK" and trial_name == "stairs_assistive_arm_off_LRLR":
            t_0 = 3.6 
            t_f = 5.5 
        if subject == "ZK" and trial_name == "stairs_assistive_arm_off_LRLR2":
            t_0 = 3.9 
            t_f = 5.5 
        if subject == "ZK" and trial_name == "stairs_assistive_arm_off_LRLR3":
            t_0 = 4.4 
            t_f = 6.3

    print("\nSubject: ",subject)
    print("Trial: ",trial_name)
    print("t_0: ",t_0)

    if task == "sts":
        print("t_i: ",t_i)
        t_0=t_i
     
    print("t_f: ",t_f)
    print("mesh interval: ",mesh_interval)

    # Dict for yaml file
    config_file = dict()
    config_file["subject"] = subject
    config_file["trial_name"] = trial_name
    config_file["trial_kinematics_path"] = str(session / "OpenSimData" / "Kinematics" / f"{trial_name}.mot")
    config_file["trial_markers_path"] = str(session / "MarkerData" / f"{trial_name}.trc")
    
    config_file["assistive_force"] = assistive_force
    config_file["assist_with_force_ext"] = assist_with_force_ext
    config_file["force_ext_opt_value"] = force_ext_opt_value
    config_file["assist_with_reserve_pelvis_txy"] = assist_with_reserve_pelvis_txy

    config_file["reserve_pelvis_weight"] = reserve_pelvis_weight
    config_file["reserve_hip_weight"] = reserve_hip_weight
    config_file["reserve_knee_weight"] = reserve_knee_weight
    config_file["reserve_feet_weight"] = reserve_feet_weight 

    config_file["t_0"] = t_0
    config_file["t_f"] = t_f
    config_file["mesh_interval"] = mesh_interval

    config_file["add_mocap_GRF"] = add_mocap_GRF
    config_file["add_osim_GRF_online"] = add_osim_GRF_online
    config_file["add_osim_GRF_precomputed"] = add_osim_GRF_precomputed
    config_file["osim_GRF_type"] = osim_GRF_type

    config_file["actuator_magnitude"] = reserve_actuator_opt_value
    config_file["coord_actuator_opt_value"] = coord_actuator_opt_value
    config_file["tracking_weight"] = tracking_weight
    config_file["control_effort_weight"] = control_effort_weight
    config_file["control_effort_muscles"] = control_effort_muscles
    config_file["tracking_weight_pelvis_txy"] = tracking_weight_pelvis_txy
    config_file["reserve_pelvis_opt_value"] = reserve_pelvis_opt_value
    config_file["reserve_pelvis_weight_txy"] = reserve_pelvis_weight_txy
    config_file["pelvis_ty_positive"] = pelvis_ty_positive

    if add_mocap_GRF:
        config_file["grf_path"] = str(session / "grf_filtered.mot")
    if assist_with_force_ext and add_osim_GRF_precomputed:
        config_file["grf_osim_feet_precomputed_path"] = grf_osim_feet_precomputed_path
        config_file["grf_osim_chair_precomputed_path"] = grf_osim_chair_precomputed_path

    print("model name: ", model_name)
    print("model path: ", model_path)
    print("target path: ", model_path.parent)

    model = get_model(
        subject_name = model_name,
        model_path = model_path,
        target_path = model_path.parent,
        assistive_force = assistive_force,
        assist_with_force_ext = assist_with_force_ext,
        force_ext_opt_value = force_ext_opt_value, 
        add_mocap_GRF = add_mocap_GRF, 
        add_osim_GRF_online = add_osim_GRF_online,
        add_osim_GRF_precomputed = add_osim_GRF_precomputed,
        config = config_file,
    )

    model.initSystem()

    #sys.exit(0) # stop the programm after designing the model before tracking optimization

    tracking_problem = get_tracking_problem(
        model=model,
        kinematics_path=str(session / "OpenSimData" / "Kinematics" / f"{trial_name}.mot"),
        t_0=t_0,
        t_f=t_f,
        mesh_interval=mesh_interval,
        tracking_weight = tracking_weight,
        control_effort_weight = control_effort_weight
    )

    study = tracking_problem.initialize()

    set_state_tracking_weights(moco_study=study, config=config_file)
    set_moco_problem_weights(model=model, moco_study=study, config=config_file)
    #set_state_bounds(moco_study=study, config=config_file)

    max_iterations = None # if set to None: no limit will be set
    set_moco_problem_solver_settings(moco_study = study, nb_max_iterations = max_iterations)

    config_file["solution_name"] = solution_name
    config_file["solution_path"] = str(solution_path_local)

    with open(str(solution_folder_CurDir/f"{solution_name}.yaml"), "w") as f:
        yaml.dump(config_file, f)
    
    solution = study.solve()

    if not solution.success():
        print("Maximum iterations reached: ","solution is unsuccessful but unsealed")
        solution.unseal()

    solution.write(str(solution_path_CurDir))
    solution.write(str(solution_path_local))

    print("Start Visualization")
    print("Press Esc to stop the visualization")
    study.visualize(solution)

    if add_osim_GRF_online:
        if task == "sts":
            extract_GRF(model, solution, str(solution_path_CurDir.with_suffix('')), str(solution_path_local.with_suffix('')), config_file)

if __name__ == "__main__":
    main()
