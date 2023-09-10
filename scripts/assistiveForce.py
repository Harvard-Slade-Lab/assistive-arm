import numpy as np
import os
import opensim as osim

from datetime import datetime
from pathlib import Path

from assistive_arm.moco_helpers import get_model, set_marker_tracking_weights, get_tracking_problem


def main():
    enable_assist = True

    base_model_path = Path("./moco/models/base/LaiUhlrich2022_scaled.osim")
    marker_data = "./moco/marker_data/sit_stand_2_w_grf.trc"

    model = get_model(scaled_model_path=base_model_path, enable_assist=enable_assist, ground_forces=False)
    model.initSystem()
    
    # Use MocoTrack to do the tracking

    tracking_problem = get_tracking_problem(
        model=model, markers_path=marker_data, t_0=1.4, t_f=3
    )
    study = tracking_problem.initialize()
    problem = study.updProblem()
    # problem.setStateInfoPattern("/jointset/.*/speed", [], 0, 0)
    # problem.setStateInfoPattern("/forceset/*/activation", [], 1, 1)

    # Create a control cost
    effort = osim.MocoControlGoal.safeDownCast(problem.updGoal("control_effort"))
    # effort.setWeightForControl("/forceset/vasmed_r", 0.7)
    # effort.setWeightForControl("/forceset/vasmed_l", 0.7)
    if enable_assist:
        effort.setWeightForControl("/forceset/assistive_force_y", 0)
        effort.setWeightForControl("/forceset/assistive_force_x", 0)

    cur_time = datetime.now().strftime("%Y-%m-%d_%H-%M")

    solution_name = f"solution_assistance_{str(enable_assist).lower()}_{cur_time}"

    solution = study.solve()
    solution.write(f"./moco/control_solutions/{solution_name}.sto")
    study.visualize(solution)


if __name__ == "__main__":
    main()
