import numpy as np
import opensim as osim

from pathlib import Path


def simplify_model(name: str, model: osim.Model) -> None:
    # bodies = [
    #     "humerus_r",
    #     "humerus_l",
    #     "ulna_r",
    #     "ulna_l",
    #     "radius_r",
    #     "radius_l",
    #     "hand_r",
    #     "hand_l",
    # ]

    # body_set = model.updBodySet()

    # for body in bodies:
    #     body_set.remove(body_set.getIndex(body))

    # Remove unnecessary joints
    # joints = [
    #     "acromial_l",
    #     "radius_hand_l",
    #     "radioulnar_l",
    #     "elbow_l",
    #     "acromial_r",
    #     "radius_hand_r",
    #     "radioulnar_r",
    #     "elbow_r",
    # ]
    # joint_set = model.upd_JointSet()

    # for joint in joints:
    #     joint_set.remove(joint_set.getIndex(joint))

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
        # "recfem_r", # Consider deleting
        "sart_r",
        "semimem_r",
        "semiten_r",
        "tfl_r",
        "tibpost_r",
        "vasint_r",  # Consider deleting
        "vaslat_r",  # Consider deleting
        # "vasmed_r", # Consider deleting
    ]

    left_muscles = [muscle[:-1] + "l" for muscle in right_muscles]

    lumbar_muscles = [
        "lumbar_bend",
        "lumbar_ext",
        "lumbar_rot",
    ]

    force_set = model.upd_ForceSet()


    if "simple" in name:
        print("Removing most muscles")
        muscles_to_remove = right_muscles + left_muscles + lumbar_muscles
    else:
        print("Only removing actuators")
        muscles_to_remove = []

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


def getMuscleDrivenModel(subject_name: str, model_path: Path, ground_forces: bool) -> osim.Model:
    # Load the base model.
    if "opencap" in subject_name:
        print("Reading opencap model...")
        model = osim.Model(str(model_path))
    else:
        print("Reading simplified model...")
        model = osim.Model("./simplified_model.osim")
    
    simplify_model(name=subject_name, model=model)

    modelProcessor = osim.ModelProcessor(model)
    modelProcessor.append(osim.ModOpIgnoreTendonCompliance())
    modelProcessor.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
    modelProcessor.append(osim.ModOpIgnorePassiveFiberForcesDGF())
    modelProcessor.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5))
    modelProcessor.append(osim.ModOpAddReserves(700))


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

    if ground_forces:
        modelProcessor.append(
            osim.ModOpAddExternalLoads("./moco/forces/grf_sit_stand.xml")
        )

    model = modelProcessor.process()
    model.finalizeConnections()

    return model


def get_model(
    subject_name: str,
    scaled_model_path: Path,
    enable_assist: bool,
    ground_forces: bool = False,
    output_model: bool = True,
) -> osim.Model:
    """Get a model with assistive forces.

    Args:
        scaled_model_path (Path): path to scaled model
        enable_assist (bool): enable assistive forces
        output_model (bool, optional): whether to output the model. Defaults to True.
    Returns:
        osim.Model: model with assistive forces
    """
    model = getMuscleDrivenModel(
        subject_name=subject_name,
        model_path=str(scaled_model_path),
        ground_forces=ground_forces,
    )
    model_name = f"{subject_name}_{scaled_model_path.stem}" 
    model.setName(model_name)

    # Add assistive force
    if enable_assist:
        add_assistive_force(coordName="pelvis_ty", model=model, name="assistive_force_y", direction=osim.Vec3(0, 1, 0), magnitude=300)
        add_assistive_force(coordName="pelvis_tx", model=model, name="assistive_force_x", direction=osim.Vec3(1, 0, 0), magnitude=200)
        add_assistive_force(coordName="pelvis_tz", model=model, name="assistive_force_z", direction=osim.Vec3(0, 0, 1), magnitude=50)

    if output_model:
        model.printToXML(f"./moco/models/{model_name}.osim")

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

    if model.getName().startswith("opencap"):
        tracking.setMarkersReferenceFromTRC(markers_path)
    else:
        tracking.setMarkersReferenceFromTRC(markers_path)
        # tracking.setStatesReference(osim.TableProcessor("./filtered.mot"))
    # tracking.set_states_global_tracking_weight(10)
    tracking.set_allow_unused_references(True)
    tracking.set_track_reference_position_derivatives(True)
    tracking.set_markers_global_tracking_weight(10)
    tracking.set_initial_time(t_0)
    tracking.set_final_time(t_f)
    tracking.set_mesh_interval(mesh_interval)

    # set_marker_tracking_weights(track=tracking)

    return tracking


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
    coordSet = model.updCoordinateSet()

    actu = osim.CoordinateActuator()
    actu.setName(name)
    actu.setCoordinate(coordSet.get(coordName))
    actu.setOptimalForce(magnitude)
    actu.setMinControl(-1)
    actu.setMaxControl(1)
    model.addComponent(actu)

    # assistActuator = osim.PointActuator("torso")
    # assistActuator.setName(name)
    # assistActuator.set_force_is_global(True)
    # assistActuator.set_direction(direction)
    # assistActuator.setMinControl(-1)
    # assistActuator.setMaxControl(1)
    # assistActuator.setOptimalForce(magnitude)

    # model.addForce(assistActuator)
