# OpenSim Moco: Python2Dwalking.py

# This script implements a Python-based optimal control problem for simulating 
# 2D walking using OpenSim Moco. The example includes two distinct optimal 
# control problems:
# 1. A tracking simulation of walking.
# 2. A predictive simulation of walking.

# The inspiration for this code is based on the work by Falisse A, Serrancoli G, 
# Dembia C, Gillis J, De Groote F titled "Algorithmic differentiation improves the 
# computational efficiency of OpenSim-based trajectory optimization of human movement," 
# published in PLOS One in 2019.

# Model
# -----
# The model specified in '2D_gait.osim' is a modified version of 'gait10dof18musc.osim', 
# which is a part of OpenSim. Modifications include:
# - Transitioning from a moving to a fixed knee flexion axis.
# - Replacing Millard2012EquilibriumMuscles with DeGrooteFregly2016Muscles.
# - Incorporating SmoothSphereHalfSpaceForces (two contacts per foot) to simulate 
#   foot-ground contact interactions.
# Note: Do not use this model for research because the gastroc muscle path is not 
# accurate—it doesn’t pass over the knee joint as it should.

# Data
# ----
# The coordinate reference data included in 'referenceCoordinates.sto' was derived 
# from previous predictive simulations conducted by Falisse et al. 2019. These data 
# might slightly differ from typical experimental gait data.

import opensim as osim
import os
import math

# -----------------------------------------------------------------------------
# Function: gaitTracking
# This function sets up and solves a coordinate tracking problem. The objective is 
# to minimize the difference between provided and simulated coordinates, speeds, and 
# ground reaction forces (GRFs) while minimizing the effort (squared controls). It considers 
# half a gait cycle and imposes endpoint constraints for periodicity on coordinates 
# (excluding pelvis translation in x) and speeds, along with coordinate actuator controls 
# and muscle activations.

def gaitTracking(controlEffortWeight=10.0, stateTrackingWeight=1.0, GRFTrackingWeight=1.0):
    # Configure weights for the objective function terms. The default values were 
    # determined through trial and error.
    # - Setting GRFTrackingWeight to 0 disables GRF tracking.
    # - A GRFTrackingWeight value of 1 balances the tracking error (states + GRF) with 
    #   control effort in the final objective value.

    print("2D gait simulation in Python")

    # Step 1: Define the optimal control problem
    # -----------------------------
    track = osim.MocoTrack()
    track.setName("gaitTracking")

    # Specify reference data for the tracking problem
    ref = osim.TableProcessor("referenceCoordinates.sto")
    # Apply a low-pass filter to the reference data to smoothen it
    ref.append_operators(osim.TabOpLowPassFilter(6.0))

    # Define the model for the optimal control problem
    modelProcessor = osim.ModelProcessor("2D_gait.osim")
    track.setModel(modelProcessor)
    track.setStatesReference(ref)
    track.set_states_global_tracking_weight(stateTrackingWeight)
    track.set_allow_unused_references(True)
    track.set_track_reference_position_derivatives(True)
    track.set_apply_tracked_states_to_guess(True)
    track.set_initial_time(0.0)
    track.set_final_time(0.47008941)

    # Initialize the study and problem
    study = track.initialize()
    problem = study.updProblem()

    # Step 2: Define goals
    # --------------------
    # Symmetry goal to simulate a single left-right symmetric step
    # This can double to form a complete gait cycle
    symmetryGoal = osim.MocoPeriodicityGoal("symmetryGoal")
    problem.addGoal(symmetryGoal)

    model = modelProcessor.process()
    model.initSystem()

    # Symmetric coordinate values and speeds (excluding pelvis_tx).
    # Enforce final coordinate values of one leg to match the initial value of 
    # the other leg (or be the same for pelvis_tx).

    # Enforce symmetry for states and speeds for each coordinate
    listCoord = model.getCoordinateSet()

    for i in range(0, listCoord.getSize()):
        # Handle symmetry between left and right limbs
        if str(listCoord.get(i).getName()).endswith("_r"):
            variable = listCoord.get(i).getStateVariableNames().get(0)
            symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(variable, str(variable).replace("_r", "_l")))
            variable = listCoord.get(i).getStateVariableNames().get(1)
            symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(variable, str(variable).replace("_r", "_l")))
        if str(listCoord.get(i).getName()).endswith("_l"):
            variable = listCoord.get(i).getStateVariableNames().get(0)
            symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(variable, str(variable).replace("_l", "_r")))
            variable = listCoord.get(i).getStateVariableNames().get(1)
            symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(variable, str(variable).replace("_l", "_r")))

        # Handle coordinates for symmetry with themselves
        if not (str(listCoord.get(i).getName()).endswith("_r")) and \
           not (str(listCoord.get(i).getName()).endswith("_l")) and \
           not (str(listCoord.get(i).getName()).endswith("_tx")):

            variable = listCoord.get(i).getStateVariableNames().get(0)
            symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(variable, variable))
            variable = listCoord.get(i).getStateVariableNames().get(1)
            symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(variable, variable))

    # Add symmetric control goals
    symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair("/jointset/groundPelvis/pelvis_tx/speed"))
    # Lumbar coordinate actuator control is symmetric
    symmetryGoal.addControlPair(osim.MocoPeriodicityGoalPair("/lumbarAct"))

    # Add symmetric goals for muscle activations
    # Constrain final muscle activation values of one leg to the initial values of the other leg.
    print("\nMuscles\n")
    for muscle in model.getComponentsList():
        if str(muscle.getClassName()).find("Muscle") >= 0:
            if str(muscle.getName()).endswith("_r"):
                variable = muscle.getStateVariableNames().get(0)
                symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(variable, str(variable).replace("_r", "_l")))
                print(variable)
                print(str(variable).replace("_r", "_l"))
            if str(muscle.getName()).endsWith("_l"):
                variable = muscle.getStateVariableNames().get(0)
                symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(variable, str(variable).replace("_l", "_r")))
                print(variable)
                print(str(variable).replace("_l", "_r"))

    # Access the MocoControlGoal in every MocoTrack
    effort = osim.MocoControlGoal.safeDownCast(problem.updGoal("control_effort"))
    effort.setWeight(controlEffortWeight)

    # Optionally, add a contact tracking goal for ground reaction forces (GRFs)
    if GRFTrackingWeight != 0:
        contactTracking = osim.MocoContactTrackingGoal("contact", GRFTrackingWeight)
        contactTracking.setExternalLoadsFile("referenceGRF.xml")

        # Define right foot contact forces
        forceNamesRightFoot = osim.StdVectorString()
        forceNamesRightFoot.append("contactHeel_r")
        forceNamesRightFoot.append("contactFront_r")
        trackRightGRF = osim.MocoContactTrackingGoalGroup(forceNamesRightFoot, "Right_GRF")
        contactTracking.addContactGroup(trackRightGRF)

        # Define left foot contact forces
        forceNamesLeftFoot = osim.StdVectorString()
        forceNamesLeftFoot.append("contactHeel_l")
        forceNamesLeftFoot.append("contactFront_l")
        trackLeftGRF = osim.MocoContactTrackingGoalGroup(forceNamesLeftFoot, "Left_GRF")
        contactTracking.addContactGroup(trackLeftGRF)

        # Configure projection settings
        contactTracking.setProjection("plane")
        contactTracking.setProjectionVector(osim.Vec3(0, 0, 1))

        problem.addGoal(contactTracking)

    # Step 3: Set bounds
    # ------------------
    problem.setStateInfo("/jointset/groundPelvis/pelvis_tilt/value", [-20 * math.pi / 180, -10 * math.pi / 180])
    problem.setStateInfo("/jointset/groundPelvis/pelvis_tx/value", [0, 1])
    problem.setStateInfo("/jointset/groundPelvis/pelvis_ty/value", [0.75, 1.25])
    problem.setStateInfo("/jointset/hip_l/hip_flexion_l/value", [-10 * math.pi / 180, 60 * math.pi / 180])
    problem.setStateInfo("/jointset/hip_r/hip_flexion_r/value", [-10 * math.pi / 180, 60 * math.pi / 180])
    problem.setStateInfo("/jointset/knee_l/knee_angle_l/value", [-50 * math.pi / 180, 0])
    problem.setStateInfo("/jointset/knee_r/knee_angle_r/value", [-50 * math.pi / 180, 0])
    problem.setStateInfo("/jointset/ankle_l/ankle_angle_l/value", [-15 * math.pi / 180, 25 * math.pi / 180])
    problem.setStateInfo("/jointset/ankle_r/ankle_angle_r/value", [-15 * math.pi / 180, 25 * math.pi / 180])
    problem.setStateInfo("/jointset/lumbar/lumbar/value", [0, 20 * math.pi / 180])

    # Step 4: Configure solver
    # ------------------------
    solver = osim.MocoCasADiSolver.safeDownCast(study.updSolver())
    solver.set_num_mesh_intervals(50)
    solver.set_verbosity(2)
    solver.set_optim_solver("ipopt")
    solver.set_optim_convergence_tolerance(1e-4)
    solver.set_optim_constraint_tolerance(1e-4)
    solver.set_optim_max_iterations(1000)

    # Step 5: Solve the problem
    # -------------------------
    solution = study.solve()
    # Extend the periodic single-step solution to a full stride.
    # Refer to the Doxygen documentation for createPeriodicTrajectory() for more information.
    full = osim.createPeriodicTrajectory(solution)
    full.write("gaitTracking_solution_fullcycle.sto")

    # Step 6: Extract and write ground reaction forces
    # ------------------------------------------------
    contact_r = osim.StdVectorString()
    contact_l = osim.StdVectorString()

    contact_r.append("contactHeel_r")
    contact_r.append("contactFront_r")
    contact_l.append("contactHeel_l")
    contact_l.append("contactFront_l")

    externalForcesTableFlat = osim.createExternalLoadsTableForGait(model, full, contact_r, contact_l)
    osim.STOFileAdapter().write(externalForcesTableFlat, "gaitTracking_solutionGRF_fullcycle.sto")

    # Step 7: Visualize solution
    # --------------------------
    # study.visualize(full)
    solution.write("trackingSolution.sto")

    return solution

    print("\nExiting ... Tracking")


# Function: gaitPrediction
# This function sets up and solves a gait prediction problem using a solution from tracking.
# The goal is to minimize effort (squared controls) over distance traveled while enforcing symmetry.
def gaitPrediction(guessFile, identifier=''):

    study = osim.MocoStudy()
    study.setName("gaitPrediction")

    # Define the prediction problem
    problem = study.updProblem()
    modelProcessor = osim.ModelProcessor("2D_gait.osim")
    problem.setModelProcessor(modelProcessor)

    # Add symmetry goals
    symmetryGoal = osim.MocoPeriodicityGoal("symmetryGoal")
    problem.addGoal(symmetryGoal)

    # Process the model and enforce symmetric conditions as in tracking
    model = modelProcessor.process()
    model.initSystem()
    listCoord = model.getCoordinateSet()

    for i in range(0, listCoord.getSize()):
        if str(listCoord.get(i).getName()).endswith("_r"):
            variable = listCoord.get(i).getStateVariableNames().get(0)
            symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(variable, str(variable).replace("_r", "_l")))
            variable = listCoord.get(i).getStateVariableNames().get(1)
            symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(variable, str(variable).replace("_r", "_l")))
        if str(listCoord.get(i).getName()).endswith("_l"):
            variable = listCoord.get(i).getStateVariableNames().get(0)
            symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(variable, str(variable).replace("_l", "_r")))
            variable = listCoord.get(i).getStateVariableNames().get(1)
            symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(variable, str(variable).replace("_l", "_r")))

        if not (str(listCoord.get(i).getName()).endswith("_r")) and \
           not (str(listCoord.get(i).getName()).endswith("_l")) and \
           not (str(listCoord.get(i).getName()).endswith("_tx")):
            variable = listCoord.get(i).getStateVariableNames().get(0)
            symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(variable, variable))
            variable = listCoord.get(i).getStateVariableNames().get(1)
            symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(variable, variable))

    symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair("/jointset/groundPelvis/pelvis_tx/speed"))
    symmetryGoal.addControlPair(osim.MocoPeriodicityGoalPair("/lumbarAct"))

    # Enforce muscle activation symmetry
    print("\nMuscles\n")
    for muscle in model.getComponentsList():
        if str(muscle.getClassName()).find("Muscle") >= 0:
            if str(muscle.getName()).endswith("_r"):
                variable = muscle.getStateVariableNames().get(0)
                symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(variable, str(variable).replace("_r", "_l")))
                print(variable)
                print(str(variable).replace("_r", "_l"))
            if str(muscle.getName()).endswith("_l"):
                variable = muscle.getStateVariableNames().get(0)
                symmetryGoal.addStatePair(osim.MocoPeriodicityGoalPair(variable, str(variable).replace("_l", "_r")))
                print(variable)
                print(str(variable).replace("_l", "_r"))

    # Define average speed goal
    speedGoal = osim.MocoAverageSpeedGoal("speed")
    speedGoal.set_desired_average_speed(1.2)
    problem.addGoal(speedGoal)

    # Define effort goal, normalized by distance
    effortGoal = osim.MocoControlGoal("effort", 10)
    effortGoal.setExponent(3)
    effortGoal.setDivideByDisplacement(True)
    problem.addGoal(effortGoal)

    # Time and state bounds
    problem.setTimeBounds(0, [0.4, 0.6])
    problem.setStateInfo("/jointset/groundPelvis/pelvis_tilt/value", [-20 * math.pi / 180, -10 * math.pi / 180])
    problem.setStateInfo("/jointset/groundPelvis/pelvis_tx/value", [0, 1])
    problem.setStateInfo("/jointset/groundPelvis/pelvis_ty/value", [0.75, 1.25])
    problem.setStateInfo("/jointset/hip_l/hip_flexion_l/value", [-10 * math.pi / 180, 60 * math.pi / 180])
    problem.setStateInfo("/jointset/hip_r/hip_flexion_r/value", [-10 * math.pi / 180, 60 * math.pi / 180])
    problem.setStateInfo("/jointset/knee_l/knee_angle_l/value", [-50 * math.pi / 180, 0])
    problem.setStateInfo("/jointset/knee_r/knee_angle_r/value", [-50 * math.pi / 180, 0])
    problem.setStateInfo("/jointset/ankle_l/ankle_angle_l/value", [-15 * math.pi / 180, 25 * math.pi / 180])
    problem.setStateInfo("/jointset/ankle_r/ankle_angle_r/value", [-15 * math.pi / 180, 25 * math.pi / 180])
    problem.setStateInfo("/jointset/lumbar/lumbar/value", [0, 20 * math.pi / 180])

    # Configure the solver
    solver = study.initCasADiSolver()
    solver.set_num_mesh_intervals(10)
    solver.set_verbosity(2)
    solver.set_optim_solver("ipopt")
    solver.set_optim_convergence_tolerance(1e-4)
    solver.set_optim_constraint_tolerance(1e-4)
    solver.set_optim_max_iterations(1000)
    solver.setGuessFile(guessFile)

    # Solve the prediction problem
    solution = study.solve()
    full = osim.createPeriodicTrajectory(solution)
    full.write("gaitPrediction_solution_fullcycle" + identifier + ".sto")

    # Extract ground reaction forces
    contact_r = osim.StdVectorString()
    contact_l = osim.StdVectorString()

    contact_r.append("contactHeel_r")
    contact_r.append("contactFront_r")
    contact_l.append("contactHeel_l")
    contact_l.append("contactFront_l")

    externalForcesTableFlat = osim.createExternalLoadsTableForGait(model, full, contact_r, contact_l)
    osim.STOFileAdapter().write(externalForcesTableFlat, "gaitPrediction_solutionGRF_fullcycle" + identifier + ".sto")

    # Visualize prediction results
    # study.visualize(full)

    print("\nExiting .... Prediction")

# Run the gait tracking and prediction simulations
gaitTrackingSolution = gaitTracking()
gaitPrediction("trackingSolution.sto", "_meshIntervals10_")