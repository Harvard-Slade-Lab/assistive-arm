import os
import opensim as osim
import numpy as np

from assistive_arm.helpers import (
    getMuscleDrivenModel,
    getTorqueDrivenModel,
    add_assistive_force,
)


def get_model(model_type: str, assistive_forces: bool) -> osim.Model:
    if model_type == "muscle":
        model = getMuscleDrivenModel()
    else:
        model = getTorqueDrivenModel()
    model_name = f"{model_type}_driven_model"
    model.setName(model_name)

    # Add assistive actuators
    if assistive_forces:
        add_assistive_force(model, "assistive_force_y", osim.Vec3(0, 1, 0), 250)
        add_assistive_force(model, "assistive_force_x", osim.Vec3(1, 0, 0), 250)

    model.finalizeConnections()
    model.printToXML(f"./moco/models/{model_name}.osim")

    return model


def main():
    model_type = "muscle"
    assistive_forces = True

    model = get_model(model_type=model_type, assistive_forces=assistive_forces)

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

    solution_name = (
        f"{model_type}_driven_solution_assistance_{str(assistive_forces).lower()}.sto"
    )

    if not os.path.isfile(f"./moco/control_solutions/{solution_name}"):
        # Part 1f: Solve! Write the solution to file, and visualize.
        predictSolution = study.solve()
        predictSolution.write(f"./moco/control_solutions/{solution_name}")
        study.visualize(predictSolution)


if __name__ == "__main__":
    main()
