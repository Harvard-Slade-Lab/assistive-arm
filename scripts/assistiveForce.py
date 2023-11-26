import numpy as np
import os
import opensim as osim
import yaml

from datetime import datetime
from pathlib import Path

from assistive_arm.moco_helpers import (
    get_model,
    get_tracking_problem,
    set_moco_problem_weights,
)
from assistive_arm.utils.file_modification import modify_force_xml


def main():
    # Dict for yaml file
    config_file = dict()

    subject = "subject_4"
    subject_data = (
        Path(
            "/Users/xabieririzar/Desktop/Life/Studium/TUM/M.Sc Robotics/Harvard/Thesis/Subject testing/Subject data/"
        )
        / subject
    )
    model_name = f"{subject}_simple"  # simple / full
    model_path = subject_data / "model" / "LaiUhlrich2022_scaled.osim"

    trial = subject_data / "trial_4"

    if not os.path.exists(trial / "control_solutions"):
        os.makedirs(trial / "control_solutions")

    config_file["subject"] = subject
    config_file["trial"] = str(trial)
    config_file["reserve_pelvis_weight"] = 15

    assistive_force = None  # 700N or None

    t_0 = 5.7
    t_f = 8.5  # 7.25s for subject4 trial3
    mesh_interval = 0.05

    config_file["t_0"] = t_0
    config_file["t_f"] = t_f
    config_file["mesh_interval"] = mesh_interval
    config_file["grf_path"] = str(trial / "grf_filtered.mot")

    modify_force_xml(
        xml_file="./moco/forces/grf_sit_stand.xml",
        new_datafile=str(trial / "grf_filtered.mot"),
    )

    model = get_model(
        subject_name=model_name,
        model_path=model_path,
        target_path=model_path.parent,
        assistive_force=assistive_force,
        ground_forces=True,
        minimal_actuators=False,
        config=config_file,
    )
    model.initSystem()

    tracking_problem = get_tracking_problem(
        model=model,
        markers_path=str(trial / "opencap_tracker.trc"),
        t_0=t_0,
        t_f=t_f,
        mesh_interval=mesh_interval,
    )
    study = tracking_problem.initialize()

    set_moco_problem_weights(model=model, moco_study=study, config=config_file)

    cur_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    solution_name = f"{model_name}_{trial.stem}_assistance_{str(assistive_force).lower()}_{cur_time}"

    config_file["solution_name"] = solution_name
    config_file["solution_path"] = str(
        trial / f"/control_solutions/{solution_name}.sto"
    )

    with open(f"./moco/control_solutions/{solution_name}.yaml", "w") as f:
        yaml.dump(config_file, f)

    solution = study.solve()
    solution.write(f"./moco/control_solutions/{solution_name}.sto")
    solution.write(str(trial / f"{solution_name}.sto"))
    study.visualize(solution)


if __name__ == "__main__":
    main()
