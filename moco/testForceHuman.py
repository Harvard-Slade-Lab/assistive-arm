import os
import opensim as osim
import numpy as np

def getMuscleDrivenModel():

    # Load the base model.
    model = osim.Model('./moco/squatToStand_3dof9musc.osim')
    model.finalizeConnections()

    # Replace the muscles in the model with muscles from DeGroote, Fregly,
    # et al. 2016, "Evaluation of Direct Collocation Optimal Control Problem
    # Formulations for Solving the Muscle Redundancy Problem". These muscles
    # have the same properties as the original muscles but their characteristic
    # curves are optimized for direct collocation (i.e. no discontinuities,
    # twice differentiable, etc).
    osim.DeGrooteFregly2016Muscle().replaceMuscles(model)

    # Make problems easier to solve by strengthening the model and widening the
    # active force-length curve.
    for m in np.arange(model.getMuscles().getSize()):
        musc = model.updMuscles().get(int(m))
        musc.setMinControl(0.0)
        musc.set_ignore_activation_dynamics(False)
        musc.set_ignore_tendon_compliance(False)
        musc.set_max_isometric_force(2.0 * musc.get_max_isometric_force())
        dgf = osim.DeGrooteFregly2016Muscle.safeDownCast(musc)
        dgf.set_active_force_width_scale(1.5)
        dgf.set_tendon_compliance_dynamics_mode('implicit')
        if str(musc.getName()) == 'soleus_r':
            # Soleus has a very long tendon, so modeling its tendon as rigid
            # causes the fiber to be unrealistically long and generate
            # excessive passive fiber force.
            dgf.set_ignore_passive_fiber_force(True)

    return model

def addCoordinateActuator(model, coordName, optForce):
    coordSet = model.updCoordinateSet()
    actu = osim.CoordinateActuator()
    actu.setName('tau_' + coordName)
    actu.setCoordinate(coordSet.get(coordName))
    actu.setOptimalForce(optForce)
    actu.setMinControl(-1)
    actu.setMaxControl(1)
    model.addComponent(actu)

def getTorqueDrivenModel():
    # Load the base model.
    model = osim.Model('./moco/squatToStand_3dof9musc.osim')

    # Remove the muscles in the model.
    model.updForceSet().clearAndDestroy()
    model.initSystem()

    # Add CoordinateActuators to the model degrees-of-freedom.
    addCoordinateActuator(model, 'hip_flexion_r', 150)
    addCoordinateActuator(model, 'knee_angle_r', 300)
    addCoordinateActuator(model, 'ankle_angle_r', 150)

    return model

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