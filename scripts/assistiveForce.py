import numpy as np
import os
import opensim as osim

from datetime import datetime
from pathlib import Path

from assistive_arm.moco_helpers import (
    get_model,
    get_tracking_problem,
)


def main():
    enable_assist = True

    subject_name = "opencap"  # opencap / simple

    base_model_path = Path("./moco/models/base/LaiUhlrich2022_scaled.osim")
    marker_data = "./moco/marker_data/sit_stand_2.trc"

    model = get_model(
        subject_name=subject_name,
        scaled_model_path=base_model_path,
        enable_assist=enable_assist,
        ground_forces=True,
    )
    model.initSystem()

    tracking_problem = get_tracking_problem(
        model=model, markers_path=marker_data, t_0=1.8, t_f=3.2, mesh_interval=0.08
    )
    study = tracking_problem.initialize()
    problem = study.updProblem()
    problem.setStateInfo("/jointset/hip_r/hip_flexion_r/value", [], 0.9287312561625947 ,  0.02591920578828876)
    problem.setStateInfo("/jointset/hip_l/hip_flexion_l/value", [], 0.9104126773960669 ,  -0.007427588272934308)
    problem.setStateInfo("/jointset/walker_knee_r/knee_angle_r/value", [], 1.5939609408603523 ,  0.053242318473099175)
    problem.setStateInfo("/jointset/walker_knee_l/knee_angle_l/value", [], 1.5254744304516052 ,  0.0342592930348702)
    problem.setStateInfo("/jointset/ankle_r/ankle_angle_r/value", [], 0.034762444929215534 ,  -0.00792284359961192)
    problem.setStateInfo("/jointset/ankle_l/ankle_angle_l/value", [], -0.05069420338956732 ,  0.06299835329364592)

    # Create a control cost
    problem.addGoal(osim.MocoInitialActivationGoal("activation"))

    effort = osim.MocoControlGoal.safeDownCast(problem.updGoal("control_effort"))
    effort.setWeight(0.001)
    # effort.setWeightForControl("/forceset/vasmed_r", 0.7)
    # effort.setWeightForControl("/forceset/vasmed_l", 0.7)
    if enable_assist:
        effort.setWeightForControl("/forceset/assistive_force_y", 0)
        effort.setWeightForControl("/forceset/assistive_force_x", 0)
        effort.setWeightForControl("/forceset/assistive_force_z", 0)

    cur_time = datetime.now().strftime("%Y-%m-%d_%H-%M")

    solution_name = f"{subject_name}_assistance_{str(enable_assist).lower()}_{cur_time}"

    solution = study.solve()
    solution.write(f"./moco/control_solutions/{solution_name}.sto")
    study.visualize(solution)


if __name__ == "__main__":
    main()
