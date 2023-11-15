import csv
import numpy as np
import pandas as pd

from typing import List, Tuple
from pathlib import Path


def export_filtered_force(force_data: pd.DataFrame, filename: Path) -> None:
    force_data.to_csv(filename, sep="\t", index=False)
    # Add header containing version, nRows, nColumns, inDegrees
    with open(filename, "r") as f:
        contents = f.readlines()

    header = [
        f"{filename.stem}\n",
        "version=1\n",
        f"nRows={force_data.shape[0]}\n",
        f"nColumns={force_data.shape[1]}\n",
        "inDegrees=no\n",
        "endheader\n",
    ]

    print("Writing to ", filename)

    with open(filename, "w") as f:
        f.writelines(header + contents)


def read_headers(file_path: Path, rows: int) -> list:
    """Read the first rows of a csv file and return them as a list.

    Args:
        file_path (Path): Path to the csv file.
        rows (int): Number of rows to read.

    Returns:
        list: List of the first rows of the csv file.
    """
    header_lines = []
    with open(file_path, "r") as f:
        reader = csv.reader(f, delimiter="\t")  # replace ';' by your delimiter
        for i, line in enumerate(reader):
            if i < rows:
                header_lines.append(line)
            else:
                break
    return header_lines


def prepare_opencap_markers(df: pd.DataFrame) -> pd.DataFrame:
    df.drop(columns=["Frame#", "Unnamed: 191"], axis=1, inplace=True)
    df = df.astype(np.float64)

    new_cols = []

    for column in df.columns:
        if not column.startswith("Unnamed"):
            if column == "Time":
                new_cols.append("Time")
            else:
                new_cols.extend(3 * [column])

    df.columns = new_cols
    coord = ["X", "Y", "Z"] * (len(df.columns) // 3)
    coord.insert(0, "t")

    df.columns = pd.MultiIndex.from_tuples(zip(df.columns, coord))

    return df


def prepare_mocap_data(df: pd.DataFrame, marker_names: List[str]) -> pd.DataFrame:
    """ Prepare dataframe containing mocap data for further processing.
    Rotate 90 degrees around x-axis to align with opencap coordinate system

    Args:
        df (pd.DataFrame): dataframe
        marker_names (List[str]): marker names

    Returns:
        pd.DataFrame: dataframe
    """
    frequency = 60 #Hz
    new_cols = ["Time"]

    for marker in marker_names:
        new_cols.extend(3 * [marker])

    coord = ["X", "Y", "Z"] * (len(df.columns) // 3)
    coord.insert(0, "t")
    df.reset_index(inplace=True)
    df.columns = pd.MultiIndex.from_tuples(zip(new_cols, coord))

    # Rotate around X by 90 degrees
    df["Time"] /= frequency

    rotate_x_90 = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]])
    
    df.iloc[:, 1:4] = df.iloc[:, 1:4] @ rotate_x_90
    df.iloc[:, 4:] = df.iloc[:, 4:] @ rotate_x_90

    # Mocap data is in mm, convert to m
    for marker in marker_names:
        df[marker] = df[marker] / 1000

    return df


def transform_force_coordinates(force_trial: pd.DataFrame, new_origin: pd.Series, plates: dict):
    # OpenCap origin in Mocap coordinates
    # Rotate 90 degrees around the x-axis, and translate by Origin (already in OpenCap frame)
    T_W_OC = np.array([[1, 0, 0, new_origin.X],
                       [0, 0, -1, -new_origin.Z], 
                       [0, 1, 0, 0], # We ignore the Z coordinate because we only care the about the translation on the XY plane
                       [0, 0, 0, 1]])
    
    T_OC_W = np.eye(4)
    T_OC_W[:3, :3] = T_W_OC[:3, :3].T
    T_OC_W[:3, 3] = -T_OC_W[:3, :3] @ T_W_OC[:3, 3]

    force_trial_tf = force_trial.copy(deep=True)

    # Convert force plate coordinates to MoCap coordinates
    for side in plates.keys():
        cop_cols = [f"ground_force_{side}_px", f"ground_force_{side}_py", f"ground_force_{side}_pz"]
        df_pxyz = force_trial_tf.loc[:, cop_cols]
        df_pxyz["W"] = 1

        force_cols = [f"ground_force_{side}_vx", f"ground_force_{side}_vy", f"ground_force_{side}_vz"]
        df_vxyz = force_trial_tf.loc[:, force_cols]
        df_vxyz["W"] = 1

        # Transform force vector
        transformed_vxyz = (T_OC_W @ df_vxyz.T).T
        transformed_vxyz.drop(transformed_vxyz.columns[-1], axis=1, inplace=True)
        transformed_vxyz.columns = force_cols
        
        transformed_xyz = (T_OC_W @ df_pxyz.T).T
        transformed_xyz.drop(transformed_xyz.columns[-1], axis=1, inplace=True)
        transformed_xyz.columns = cop_cols

        force_trial_tf.loc[:, cop_cols] = transformed_xyz
        force_trial_tf.loc[:, force_cols] = transformed_vxyz

        # Flip y-axis
        force_trial_tf.loc[:, f"ground_force_{side}_vy"] *= -1

    return force_trial_tf


def sync_mocap_with_opencap(mocap_data: pd.DataFrame, force_data: pd.DataFrame, opencap_data: pd.DataFrame) -> tuple:
    # Cut the mocap data such that it perfectly overlaps with opencap

    # using zero-crossings
    # gradient_opencap = opencap_data.LKnee.Y.diff().rolling(window=14).mean()
    # opencap_min = np.where(np.diff(np.sign(gradient_opencap + 0.001)) > 0)[0][0]
    
    # gradient_mocap = mocap_data.Knee.Y.diff().rolling(window=14).mean()
    # mocap_min = np.where(np.diff(np.sign(gradient_mocap + 0.001)) > 0)[0][0]
    
    # Using argmax
    mocap_min = mocap_data.Knee.Y.argmax()
    opencap_min = opencap_data.LKnee.Y.argmax()

    lag = int(opencap_min - mocap_min)
    print("Lag: ", lag)

    # Shift the data by lag
    mocap_data_synced = mocap_data.copy(deep=True)
    mocap_data_synced = mocap_data_synced.iloc[-lag:-lag + opencap_data.shape[0]].reset_index(drop=True)
    force_data = force_data.iloc[-lag*10:-lag*10 + opencap_data.shape[0]*10].reset_index(drop=True)
    
    force_data["time"] = force_data["time"] - force_data["time"][0]
    mocap_data_synced["Time"] = mocap_data_synced["Time"] - mocap_data_synced["Time"].t[0]

    opencap_synced = opencap_data.copy(deep=True)
    opencap_synced.reset_index(drop=True, inplace=True)

    # Filter data after user stands up
    threshold = 5

    standup_index = force_data['ground_force_chair_vy'].lt(threshold).idxmax()

    # Check if the series actually contains any values below 5 to prevent setting the whole column to 0 if there is none
    if force_data['ground_force_chair_vy'][standup_index] < threshold:
        # Set every element after this index to 0
        force_data.loc[standup_index:, 'ground_force_chair_vx'] = 0
        force_data.loc[standup_index:, 'ground_force_chair_vy'] = 0
        force_data.loc[standup_index:, 'ground_force_chair_vz'] = 0
        force_data.loc[standup_index:, 'ground_torque_chair_x'] = 0
        force_data.loc[standup_index:, 'ground_torque_chair_y'] = 0
        force_data.loc[standup_index:, 'ground_torque_chair_z'] = 0

    return mocap_data_synced, force_data, opencap_synced, lag, standup_index


def prepare_mocap_force_df(
    force_plate_data: dict,
    forces_in_world: bool=True,
) -> pd.DataFrame:

    # Represent center of pressure in world coordinates
    for side in force_plate_data.keys():
        force_plate_data[side]["data"].drop(columns=["nan"], inplace=True)
        if not forces_in_world:
            coords = [float(coord[1]) for coord in force_plate_data[side]["headers"]]
            plate_origin_x = np.mean(coords[0::3])
            plate_origin_y = np.mean(coords[1::3])
            plate_origin_z = np.mean(coords[2::3])

            force_plate_data[side]["data"][f"ground_force_{side}_px"] = plate_origin_x + force_plate_data[side]["data"][f"ground_force_{side}_px"]
            force_plate_data[side]["data"][f"ground_force_{side}_py"] = plate_origin_y + force_plate_data[side]["data"][f"ground_force_{side}_py"]
            force_plate_data[side]["data"][f"ground_force_{side}_pz"] = plate_origin_z + force_plate_data[side]["data"][f"ground_force_{side}_pz"]


    # Divide frame by sampling rate
    
    if "chair" in force_plate_data.keys():
        df_merged = pd.concat([force_plate_data["r"]["data"], force_plate_data["l"]["data"], force_plate_data["chair"]["data"]], axis=1)
    else:
        df_merged = pd.concat([force_plate_data["r"]["data"], force_plate_data["l"]["data"]], axis=1)

    df_merged.reset_index(names="time", inplace=True)
    df_merged["time"] = df_merged["time"].apply(lambda x: x / 600)

    # Convert from mm to m
    for side in force_plate_data.keys():
        df_merged[f"ground_force_{side}_pz"] = 0
        df_merged[f"ground_force_{side}_px"] = df_merged[f"ground_force_{side}_px"] / 1000
        df_merged[f"ground_force_{side}_py"] = df_merged[f"ground_force_{side}_py"] / 1000
        df_merged[f"ground_force_{side}_pz"] = df_merged[f"ground_force_{side}_pz"] / 1000

    return df_merged
