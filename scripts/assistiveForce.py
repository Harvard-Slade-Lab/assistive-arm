import numpy as np
import os
import opensim as osim

from datetime import datetime
from pathlib import Path

from assistive_arm.moco_helpers import (
    getMuscleDrivenModel,
    getTorqueDrivenModel,
    add_assistive_force,
)


def get_model(model_type: str, enable_assist: bool) -> osim.Model:
    model_path = Path("./moco/models/base/squatToStand_3dof9musc.osim")

    if model_type == "muscle":
        model = getMuscleDrivenModel(model_path=model_path)
    else:
        model = getTorqueDrivenModel(model_path=model_path)
    model_name = f"{model_type}_driven_{model_path.stem}"
    model.setName(model_name)

    # Add assistive actuators
    if enable_assist:
        add_assistive_force(model, "assistive_force_y", osim.Vec3(0, 1, 0), 250)
        add_assistive_force(model, "assistive_force_x", osim.Vec3(1, 0, 0), 250)

    model.initSystem()
    model.printToXML(f"./moco/models/{model_name}.osim")

    return model


def main():
    enable_assist = True

    model_type = "torque"
    model = get_model(model_type=model_type, enable_assist=enable_assist)

    # Create a MocoStudy.
    study = osim.MocoStudy()
    study.setName("testForceHuman")

    # Define the optimal control problem.
    problem = study.updProblem()
    problem.setModel(model)

    problem.setTimeBounds(0, 1)
    problem.setStateInfo("/jointset/hip_r/hip_flexion_r/value", [-2, 0.5], -2, 0)
    problem.setStateInfo("/jointset/knee_r/knee_angle_r/value", [-2, 0], -2, 0)
    problem.setStateInfo("/jointset/ankle_r/ankle_angle_r/value", [-0.5, 0.7], -0.5, 0)

    problem.setStateInfoPattern("/jointset/.*/speed", [], 0, 0)

    # Add a control cost
    controlCost = osim.MocoControlGoal("reduce_effort", 1)
    controlCost.setWeightForControlPattern("/forceset/assistive*", 0)
    problem.addGoal(controlCost)

    # Configure the solver.
    solver = study.initCasADiSolver()
    solver.set_num_mesh_intervals(25)
    solver.set_optim_convergence_tolerance(1e-4)
    solver.set_optim_constraint_tolerance(1e-4)

    cur_time = datetime.now().strftime("%Y-%m-%d_%H-%M")

    solution_name = (
        f"{model_type}_solution_assistance_{str(enable_assist).lower()}_{cur_time}"
    )

    print(model.getWorkingState())

    # Part 1f: Solve! Write the solution to file, and visualize.
    predictSolution = study.solve()
    predictSolution.write(f"./moco/control_solutions/{solution_name}.sto")
    study.visualize(predictSolution)


if __name__ == "__main__":
    main()
