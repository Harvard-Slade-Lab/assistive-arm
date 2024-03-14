import numpy as np
import opensim as osim

from pathlib import Path


def simplify_model(model_name: str, model: osim.Model) -> None:
    muscles_to_remove = []

    # Remove unnecessary actuators
    actuators = [
        "shoulder_flex_r",
        "shoulder_add_r",
        "shoulder_rot_r",
        "elbow_flex_r",
        "pro_sup_r",
        "shoulder_flex_l",
        "shoulder_add_l",
        "shoulder_rot_l",
        "elbow_flex_l",
        "pro_sup_l",
    ]

    right_muscles = [
        "addlong_r",
        "addbrev_r",
        "addmagDist_r",
        "addmagIsch_r",
        "addmagProx_r",
        "addmagMid_r",
        "fdl_r",
        "ehl_r",  # Consider deleting
        "fhl_r",  # Consider deleting
        "gaslat_r",  # Consider deleting
        "glmax2_r",
        "glmax3_r",
        "glmed1_r",
        "glmed2_r",
        "glmed3_r",
        "glmin1_r",
        "glmin2_r",
        "glmin3_r",
        "perlong_r",
        "edl_r",
        "grac_r",  # Consider deleting
        "iliacus_r",  # Consider deleting
        "perbrev_r",  # Consider deleting
        "piri_r",
        "sart_r",
        "semimem_r",
        "semiten_r",
        "tfl_r",
        "tibpost_r",
        "vasint_r",  # Consider deleting
        "vaslat_r",  # Consider deleting
    ]

    left_muscles = [muscle[:-1] + "l" for muscle in right_muscles]

    lumbar_muscles = [
        "lumbar_bend",
        "lumbar_ext",
        "lumbar_rot",
    ]

    force_set = model.upd_ForceSet()

    if "simple" in model_name:
        muscles_to_remove = right_muscles + left_muscles + lumbar_muscles

    for i in actuators + muscles_to_remove:
        force_set.remove(force_set.getIndex(i))


def set_marker_tracking_weights(track: osim.MocoTrack) -> None:
    markerWeights = osim.MocoWeightSet()

    markerWeights.cloneAndAppend(osim.MocoWeight("r.ASIS_study", 20))
    markerWeights.cloneAndAppend(osim.MocoWeight("r.PSIS_study", 20))
    markerWeights.cloneAndAppend(osim.MocoWeight("r_knee_study", 10))
    markerWeights.cloneAndAppend(osim.MocoWeight("r_ankle_study", 10))
    markerWeights.cloneAndAppend(osim.MocoWeight("r_calc_study", 10))
    markerWeights.cloneAndAppend(osim.MocoWeight("r_5meta_study", 5))
    markerWeights.cloneAndAppend(osim.MocoWeight("r_toe_study", 2))

    markerWeights.cloneAndAppend(osim.MocoWeight("l.ASIS_study", 20))
    markerWeights.cloneAndAppend(osim.MocoWeight("l.PSIS_study", 20))
    markerWeights.cloneAndAppend(osim.MocoWeight("l_knee_study", 10))
    markerWeights.cloneAndAppend(osim.MocoWeight("l_ankle_study", 10))
    markerWeights.cloneAndAppend(osim.MocoWeight("l_calc_study", 10))
    markerWeights.cloneAndAppend(osim.MocoWeight("l_5meta_study", 5))
    markerWeights.cloneAndAppend(osim.MocoWeight("l_toe_study", 2))

    track.set_markers_weight_set(markerWeights)


def getMuscleDrivenModel(
    subject_name: str,
    model_path: Path,
    ground_forces: bool,
    minimal_actuators: bool,
    config: dict = None,
) -> osim.Model:
    # Load the base model.
    model = osim.Model(str(model_path))

    simplify_model(model_name=subject_name, model=model)

    modelProcessor = osim.ModelProcessor(model)

    if "simple" not in subject_name:
        jointNames = osim.StdVectorString()
        jointNames.append("acromial_r")
        jointNames.append("elbow_r")
        jointNames.append("radioulnar_r")
        jointNames.append("radius_hand_r")
        jointNames.append("acromial_l")
        jointNames.append("elbow_l")
        jointNames.append("radioulnar_l")
        jointNames.append("radius_hand_l")

        modelProcessor.append(osim.ModOpReplaceJointsWithWelds(jointNames))

    modelProcessor.append(osim.ModOpIgnoreTendonCompliance())
    modelProcessor.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
    modelProcessor.append(osim.ModOpIgnorePassiveFiberForcesDGF())
    modelProcessor.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5))

    if not minimal_actuators:
        modelProcessor.append(osim.ModOpAddReserves(config["actuator_magnitude"]))

    if ground_forces:
        modelProcessor.append(
            osim.ModOpAddExternalLoads("./moco/forces/grf_sit_stand.xml")
        )

    model = modelProcessor.process()

    return model


def get_model(
    subject_name: str,
    model_path: Path,
    target_path: Path,
    assistive_force: int = None,
    ground_forces: bool = False,
    minimal_actuators: bool = False,
    config: dict = None,
) -> osim.Model:
    """Get a model with assistive forces.

    Args:
        scaled_model_path (Path): path to scaled model
        enable_assist (bool): enable assistive forces
        output_model (bool, optional): whether to output the model. Defaults to True.
    Returns:
        osim.Model: model with assistive forces
    """

    config["actuator_magnitude"] = 700
    config["target_path"] = str(target_path)
    config["assistive_force"] = assistive_force
    config["ground_forces"] = ground_forces
    config["minimal_actuators"] = minimal_actuators
    if minimal_actuators:
        config["minimal_actuator_names"] = ["tilt", "rotation", "list", "tx", "ty", "tz"]

    model = getMuscleDrivenModel(
        subject_name=subject_name,
        model_path=str(model_path),
        ground_forces=ground_forces,
        minimal_actuators=minimal_actuators,
        config=config,
    )
    model_name = f"{subject_name}_{model_path.stem}"
    model.setName(model_name)

    # Add assistive force
    if assistive_force:
        add_assistive_force(
            coordName="pelvis_tx",
            model=model,
            name="assistive_force_x",
            direction=osim.Vec3(1, 0, 0),
            magnitude=assistive_force,
        )
        add_assistive_force(
            coordName="pelvis_ty",
            model=model,
            name="assistive_force_y",
            direction=osim.Vec3(0, 1, 0),
            magnitude=assistive_force,
        )
        config["assistive_force"] = assistive_force

    coordSet = model.updCoordinateSet()

    if minimal_actuators:
        for actuator in config["minimal_actuator_names"]:
            actu = osim.CoordinateActuator()
            actu.setName(f"reserve_pelvis_{actuator}")
            actu.setCoordinate(coordSet.get(f"pelvis_{actuator}"))
            actu.setOptimalForce(config["actuator_magnitude"])
            actu.setMinControl(-np.inf)
            actu.setMaxControl(np.inf)
            model.addComponent(actu)

    model.finalizeConnections()

    model.printToXML(f"./moco/models/{model_name}.osim")
    model.printToXML(str(target_path / f"{model_name}.osim"))

    config["xml_path"] = f"./moco/models/{model_name}.osim"

    return model


def get_tracking_problem(
    model: osim.Model,
    markers_path: str,
    t_0: float,
    t_f: float,
    mesh_interval: float = 0.08,
) -> osim.MocoStudy:
    # Create a MocoTrack problem
    tracking = osim.MocoTrack()
    tracking.setModel(osim.ModelProcessor(model))
    tracking.setName("tracking_problem")

    tracking.setMarkersReferenceFromTRC(markers_path)
    tracking.set_allow_unused_references(True)
    tracking.set_track_reference_position_derivatives(True)
    tracking.set_markers_global_tracking_weight(10)
    tracking.set_initial_time(t_0)
    tracking.set_final_time(t_f)
    tracking.set_mesh_interval(mesh_interval)

    set_marker_tracking_weights(track=tracking)

    return tracking


def set_moco_problem_weights(
    model: osim.Model, moco_study: osim.MocoStudy, config: dict
):
    problem = moco_study.updProblem()
    problem.addGoal(osim.MocoInitialActivationGoal("activation"))

    effort_goal = osim.MocoControlGoal.safeDownCast(problem.updGoal("control_effort"))

    # Only 6 reserve actuators
    if config["minimal_actuators"]:
        for actu_name in config["minimal_actuator_names"]:
            effort_goal.setWeightForControl(f"/reserve_pelvis_{actu_name}", config["reserve_pelvis_weight"])
    # For the case where we have a full set of reserve actuators
    else:
        forceSet = model.getForceSet()
        for i in range(forceSet.getSize()):
            forcePath = forceSet.get(i).getAbsolutePathString()
            if "pelvis" in str(forcePath):
                effort_goal.setWeightForControl(
                    forcePath, config["reserve_pelvis_weight"]
                )
    effort_goal.setWeightForControl("/forceset/recfem_r", 1)
    effort_goal.setWeightForControl("/forceset/vasmed_r", 1)
    effort_goal.setWeightForControl("/forceset/bflh_r", 1)
    effort_goal.setWeightForControl("/forceset/bfsh_r", 1)
    effort_goal.setWeightForControl("/forceset/recfem_l", 1)
    effort_goal.setWeightForControl("/forceset/vasmed_l", 1)
    effort_goal.setWeightForControl("/forceset/bflh_l", 1)
    effort_goal.setWeightForControl("/forceset/bfsh_l", 1)
    effort_goal.setWeightForControl("/forceset/soleus_r", 1)
    effort_goal.setWeightForControl("/forceset/soleus_l", 1)
    effort_goal.setWeightForControl("/forceset/tibant_r", 1)
    effort_goal.setWeightForControl("/forceset/tibant_l", 1)

    if config["assistive_force"]:
        effort_goal.setWeightForControl("/forceset/assistive_force_y", 0)
        effort_goal.setWeightForControl("/forceset/assistive_force_x", 0)


def add_assistive_force(
    coordName: str,
    model: osim.Model,
    name: str,
    direction: osim.Vec3,
    location: str = "torso",
    magnitude: int = 100,
) -> None:
    """Add an assistive force to the model.

    Args:
        model (osim.Model): model to add assistive force to
        name (str): name of assistive force
        direction (osim.Vec3): direction of assistive force
        location (str, optional): where the force will act. Defaults to "torso".
        magnitude (int, optional): How strong the force is. Defaults to 100.
    """
    # coordSet = model.updCoordinateSet()

    # actu = osim.CoordinateActuator()
    # actu.setName(name)
    # actu.setCoordinate(coordSet.get(coordName))
    # actu.setOptimalForce(magnitude)
    # actu.setMinControl(-1)
    # actu.setMaxControl(1)
    # model.addComponent(actu)

    assistActuator = osim.PointActuator("pelvis")
    assistActuator.setName(name)
    assistActuator.set_force_is_global(True)
    assistActuator.set_direction(direction)
    assistActuator.setMinControl(-1)
    assistActuator.setMaxControl(1)
    assistActuator.setOptimalForce(magnitude)

    model.addForce(assistActuator)
