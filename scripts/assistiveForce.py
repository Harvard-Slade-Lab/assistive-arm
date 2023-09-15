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
    enable_assist = False

    subject_name = "opencap_simple"  # opencap_simple / opencap_full / simple

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
        model=model, markers_path=marker_data, t_0=1.8, t_f=3.2, mesh_interval=0.05
    )
    study = tracking_problem.initialize()
    problem = study.updProblem()

    # Create a control cost
    problem.addGoal(osim.MocoInitialActivationGoal("activation"))

    effort = osim.MocoControlGoal.safeDownCast(problem.updGoal("control_effort"))
    forceSet = model.getForceSet()

    # Increase weight of pelvis coordinate
    for i in range(forceSet.getSize()):
        forcePath = forceSet.get(i).getAbsolutePathString()
        if 'pelvis' in str(forcePath):
            print("Adjusting residual force weight")
            effort.setWeightForControl(forcePath, 10)
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
