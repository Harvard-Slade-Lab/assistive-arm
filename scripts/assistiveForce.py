import numpy as np
import os
import opensim as osim

from datetime import datetime
from pathlib import Path

from assistive_arm.moco_helpers import get_model



def main(): 
    enable_assist = True

    model_type = "muscle"
    model = get_model(model_type=model_type, enable_assist=enable_assist)

    # Create a MocoStudy.
    study = osim.MocoStudy()
    study.setName("testForceHuman")

    # Define the optimal control problem.
    problem = study.updProblem()
    problem.setModel(model)

    problem.setTimeBounds(0, 1)

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


    # Part 1f: Solve! Write the solution to file, and visualize.
    predictSolution = study.solve()
    predictSolution.write(f"./moco/control_solutions/{solution_name}.sto")
    study.visualize(predictSolution)


if __name__ == "__main__":
    main()
