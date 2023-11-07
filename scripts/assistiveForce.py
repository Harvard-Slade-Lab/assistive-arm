import numpy as np
import os
import opensim as osim
import yaml

from datetime import datetime
from pathlib import Path

from assistive_arm.moco_helpers import (
    get_model,
    get_tracking_problem,
)
from assistive_arm.utils.file_modification import modify_force_xml


def main():
    enable_assist = False

    # Dict for yaml file
    config_file = dict()

    subject = "subject_4"
    subject_data = Path("/Users/xabieririzar/Desktop/Life/Studium/TUM/M.Sc Robotics/Masterarbeit Harvard/Thesis/Subject testing/Subject data/") / subject
    trial = subject_data / "trial_4"

    model_name = f"{subject}_full"  # simple / full
    model_path = subject_data / "model" / "LaiUhlrich2022_scaled.osim"

    config_file["subject"] = subject
    config_file["trial"] = str(trial)

    t_0 = 5.7
    t_f = 7.7
    mesh_interval = 0.05
    
    config_file["t_0"] = t_0
    config_file["t_f"] = t_f
    config_file["mesh_interval"] = mesh_interval

    modify_force_xml('./moco/forces/grf_sit_stand.xml', str(trial / "grf_filtered.mot"))

    config_file["grf_path"] = str(trial / "grf_filtered.mot")


    model = get_model(
        subject_name=model_name,
        model_path=model_path,
        target_path=model_path.parent,
        enable_assist=enable_assist,
        ground_forces=True,
        config=config_file,
    )
    model.initSystem()


    tracking_problem = get_tracking_problem(
        model=model, markers_path=str(trial / "opencap_tracker.trc"), t_0=t_0, t_f=t_f, mesh_interval=mesh_interval
    )
    study = tracking_problem.initialize()

    problem = study.updProblem()
    problem.addGoal(osim.MocoInitialActivationGoal("activation"))

    effort_goal = osim.MocoControlGoal.safeDownCast(problem.updGoal("control_effort"))

    # Increase weight of pelvis coordinate
    pelvis_weight = 15
    config_file["reserve_pelvis_weight"] = pelvis_weight

    # TODO Should  be packed in a function
    effort_goal.setWeightForControl("/reserve_pelvis_tilt", pelvis_weight)
    effort_goal.setWeightForControl("/reserve_pelvis_rotation", pelvis_weight)
    effort_goal.setWeightForControl("/reserve_pelvis_list", pelvis_weight)
    effort_goal.setWeightForControl("/reserve_pelvis_tx", pelvis_weight)
    effort_goal.setWeightForControl("/reserve_pelvis_ty", pelvis_weight)
    effort_goal.setWeightForControl("/reserve_pelvis_tz", pelvis_weight)

    forceSet = model.getForceSet()  
    for i in range(forceSet.getSize()):
        forcePath = forceSet.get(i).getAbsolutePathString()
        if 'pelvis' in str(forcePath):
            print("Adjusting residual force weight")
            effort_goal.setWeightForControl(forcePath, pelvis_weight)

    # Set weights for muscles
    effort_goal.setWeightForControl("/forceset/recfem_r", 1)
    effort_goal.setWeightForControl("/forceset/vasmed_r", 1)
    effort_goal.setWeightForControl("/forceset/recfem_l", 1)
    effort_goal.setWeightForControl("/forceset/vasmed_l", 1)
    effort_goal.setWeightForControl("/forceset/soleus_r", 1)
    effort_goal.setWeightForControl("/forceset/soleus_l", 1)
    effort_goal.setWeightForControl("/forceset/tibant_r", 1)
    effort_goal.setWeightForControl("/forceset/tibant_l", 1)


    if enable_assist:
        effort_goal.setWeightForControl("/forceset/assistive_force_y", 0)
        effort_goal.setWeightForControl("/forceset/assistive_force_x", 0)

    cur_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    solution_name = f"{model_name}_{trial.stem}_assistance_{str(enable_assist).lower()}_{cur_time}"

    config_file["solution_name"] = solution_name
    config_file["solution_path"] = f"./moco/control_solutions/{solution_name}.sto"

    with open(f"./moco/control_solutions/{solution_name}.yaml", "w") as f:
        yaml.dump(config_file, f)

    solution = study.solve()
    solution.write(f"./moco/control_solutions/{solution_name}.sto")
    study.visualize(solution)


if __name__ == "__main__":
    main()
