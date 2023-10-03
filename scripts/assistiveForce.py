import numpy as np
import os
import opensim as osim

from datetime import datetime
from pathlib import Path

from assistive_arm.moco_helpers import (
    get_model,
    get_tracking_problem,
)
from assistive_arm.utils.file_modification import modify_force_xml


def main():
    enable_assist = False

    subject = "subject_1"
    subject_data = Path("/Users/xabieririzar/Desktop/Life/Studium/TUM/M.Sc Robotics/Masterarbeit Harvard/Thesis/Subject testing/Subject data/") / subject

    model_name = f"{subject}_simple"  # simple / full
    model_path = subject_data / "model" / "LaiUhlrich2022_scaled.osim"

    trial = subject_data / "trial_2"

    modify_force_xml('./moco/forces/grf_sit_stand.xml', str(trial / "grf_filtered.mot"))


    model = get_model(
        subject_name=model_name,
        model_path=model_path,
        target_path=trial,
        enable_assist=enable_assist,
        ground_forces=True,
    )
    model.initSystem()


    tracking_problem = get_tracking_problem(
        model=model, markers_path=str(trial / "opencap_tracker.trc"), t_0=1.8, t_f=3.2, mesh_interval=0.05
    )
    study = tracking_problem.initialize()

    problem = study.updProblem()
    problem.addGoal(osim.MocoInitialActivationGoal("activation"))

    effort_goal = osim.MocoControlGoal.safeDownCast(problem.updGoal("control_effort"))

    # Increase weight of pelvis coordinate, 
    # TODO Should  be packed in a function
    
    pelvis_weight = 15

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
    effort_goal.setWeightForControl("/forceset/recfem_r", 5)
    effort_goal.setWeightForControl("/forceset/vasmed_r", 5)
    effort_goal.setWeightForControl("/forceset/recfem_l", 5)
    effort_goal.setWeightForControl("/forceset/vasmed_l", 5)
    effort_goal.setWeightForControl("/forceset/soleus_r", 0.5)
    effort_goal.setWeightForControl("/forceset/soleus_l", 0.5)
    effort_goal.setWeightForControl("/forceset/tibant_r", 0.5)
    effort_goal.setWeightForControl("/forceset/tibant_l", 0.5)


    if enable_assist:
        effort_goal.setWeightForControl("/forceset/assistive_force_y", 0)
        effort_goal.setWeightForControl("/forceset/assistive_force_x", 0)

    cur_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    solution_name = f"{subject}_{trial.stem}_{model_name}_assistance_{str(enable_assist).lower()}_{cur_time}"

    solution = study.solve()
    solution.write(f"./moco/control_solutions/{solution_name}.sto")
    study.visualize(solution)


if __name__ == "__main__":
    main()
