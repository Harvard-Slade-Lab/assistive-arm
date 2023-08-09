import os
import opensim as osim
import numpy as np

from assistive_arm.helpers import getMuscleDrivenModel, getTorqueDrivenModel

muscleDrivenModel = getMuscleDrivenModel()
torqueDrivenModel = getTorqueDrivenModel()
muscleDrivenModel.setName("human_body")


# Add assistive actuator
assistActuator_y = osim.PointActuator('torso')
assistActuator_y.setName("assistive_force_y")
assistActuator_y.set_force_is_global(True)
assistActuator_y.set_direction(osim.Vec3(0, 1, 0))
assistActuator_y.setMinControl(-1)
assistActuator_y.setMaxControl(1)
assistActuator_y.setOptimalForce(250)

assistActuator_x = osim.PointActuator('torso')
assistActuator_x.setName("assistive_force_x")
assistActuator_x.set_force_is_global(True)
assistActuator_x.set_direction(osim.Vec3(1, 0, 0))
assistActuator_x.setMinControl(-1)
assistActuator_x.setMaxControl(1)
assistActuator_x.setOptimalForce(250)

torqueDrivenModel.addForce(assistActuator_y)
torqueDrivenModel.addForce(assistActuator_x)
torqueDrivenModel.finalizeConnections()

torqueDrivenModel.printToXML("./moco/models/torqueDrivenModel.osim")

# Create a MocoStudy.
study = osim.MocoStudy()
study.setName("testForceHuman")

# Define the optimal control problem.
problem = study.updProblem()
problem.setModel(torqueDrivenModel)

problem.setTimeBounds(0, 1)
problem.setStateInfo("/jointset/hip_r/hip_flexion_r/value", [-2, 0.5], -2, 0)
problem.setStateInfo("/jointset/knee_r/knee_angle_r/value", [-2, 0], -2, 0)
problem.setStateInfo("/jointset/ankle_r/ankle_angle_r/value", [-0.5, 0.7], -0.5, 0)
problem.setStateInfoPattern("/jointset/.*/speed", [], 0, 0)

# Add a control cost
controlCost = osim.MocoControlGoal("reduce_effort", 1)
controlCost.setWeightForControl("/forceset/assistive_force_y", 0)
controlCost.setWeightForControl("/forceset/assistive_force_x", 0)
problem.addGoal(controlCost)


# Configure the solver.
solver = study.initCasADiSolver()
solver.set_num_mesh_intervals(25)
solver.set_optim_convergence_tolerance(1e-4)
solver.set_optim_constraint_tolerance(1e-4)

if not os.path.isfile("predictSolution.sto"):
    # Part 1f: Solve! Write the solution to file, and visualize.
    predictSolution = study.solve()
    predictSolution.write("./moco/results/testForceHuman_solution_yx_weight_0.sto")
    study.visualize(predictSolution)