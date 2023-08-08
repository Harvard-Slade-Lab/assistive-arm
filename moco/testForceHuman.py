import os
import opensim as osim
import numpy as np

from moco.helpers import getMuscleDrivenModel, getTorqueDrivenModel

muscleDrivenModel = getMuscleDrivenModel()
torqueDrivenModel = getTorqueDrivenModel()
muscleDrivenModel.setName("human_body")
#model.set_gravity(osim.Vec3(0, 0, -9.80665))


# actu = osim.CoordinateActuator()
# actu.setCoordinate(muscleDrivenModel.getCoordinateSet().get(0))
# actu.setName('external_force')
# actu.setOptimalForce(1.0)

# muscleDrivenModel.addComponent(actu)
# muscleDrivenModel.finalizeConnections()

# Create a MocoStudy.
study = osim.MocoStudy()
study.setName('testForceHuman')

# Define the optimal control problem.
problem = study.updProblem()
problem.setModel(torqueDrivenModel)

problem.setTimeBounds(0, 1)
problem.setStateInfo('/jointset/hip_r/hip_flexion_r/value', 
    [-2, 0.5], -2, 0)
problem.setStateInfo('/jointset/knee_r/knee_angle_r/value', 
    [-2, 0], -2, 0)
problem.setStateInfo('/jointset/ankle_r/ankle_angle_r/value', 
    [-0.5, 0.7], -0.5, 0)

# Set the speed bounds to zero for all joints that match the pattern
problem.setStateInfoPattern("/jointset/.*/speed", [], 0, 0)

# Add a control cost
problem.addGoal(osim.MocoControlGoal("myeffort"))


# Applied force must be between -100 and 100.
#problem.setControlInfo("/external_force", osim.MocoBounds(-100, 100))

# Configure the solver.
solver = study.initCasADiSolver()
solver.set_num_mesh_intervals(25)
solver.set_optim_convergence_tolerance(1e-4)
solver.set_optim_constraint_tolerance(1e-4)

if not os.path.isfile('predictSolution.sto'):
    # Part 1f: Solve! Write the solution to file, and visualize.
    predictSolution = study.solve()
    predictSolution.write('testForceHuman_solution.sto')
    study.visualize(predictSolution)