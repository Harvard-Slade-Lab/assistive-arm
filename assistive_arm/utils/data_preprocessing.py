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


def prepare_opencap_data(df: pd.DataFrame) -> pd.DataFrame:
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

    df["Time"] = df["Time"] / frequency

    # Mocap data is in mm, convert to m
    for marker in marker_names:
        df[marker] = df[marker] / 1000

    return df


def transform_force_coordinates(force_trial: pd.DataFrame, new_origin: pd.Series):
    # OpenCap origin in Mocap coordinates, rotated 90 degrees around the x-axis
    T_W_OC = np.array([[1, 0, 0, new_origin.X],
                       [0, 0, -1, new_origin.Y], 
                       [0, 1, 0, 0], # We ignore the Z coordinate because we only care the about the translation on the XY plane
                       [0, 0, 0, 1]])
    
    T_OC_W = np.eye(4)
    T_OC_W[:3, :3] = T_W_OC[:3, :3].T
    T_OC_W[:3, 3] = -T_OC_W[:3, :3] @ T_W_OC[:3, 3]

    # Convert force plate coordinates to MoCap coordinates
    for side in ["r", "l"]:
        cop_cols = [f"ground_force_{side}_px", f"ground_force_{side}_py", f"ground_force_{side}_pz"]
        df_xyz = force_trial.loc[:, cop_cols]
        df_xyz["W"] = 1

        force_cols = [f"ground_force_{side}_vx", f"ground_force_{side}_vy", f"ground_force_{side}_vz"]
        df_vxyz = force_trial.loc[:, force_cols]
        df_vxyz["W"] = 1

        # Transform force vector
        transformed_vxyz = (T_OC_W @ df_vxyz.T).T
        transformed_vxyz.drop(transformed_vxyz.columns[-1], axis=1, inplace=True)
        transformed_vxyz.columns = force_cols
        
        transformed_xyz = (T_OC_W @ df_xyz.T).T
        transformed_xyz.drop(transformed_xyz.columns[-1], axis=1, inplace=True)
        transformed_xyz.columns = cop_cols

        force_trial.loc[:, cop_cols] = transformed_xyz
        force_trial.loc[:, force_cols] = transformed_vxyz

        # Flip y-axis
        force_trial.loc[:, f"ground_force_{side}_vy"] *= -1

    return force_trial


def sync_mocap_with_opencap(marker_data: pd.DataFrame, force_data: pd.DataFrame, opencap_data: pd.DataFrame) -> pd.DataFrame:
    # Cut the mocap data such that it perfectly overlaps with opencap
    mocap_min = marker_data.Knee.X.argmin()
    opencap_min = opencap_data.LKnee.X.argmin()

    lag = opencap_min - mocap_min

    # Shift the data by lag
    marker_data = marker_data.iloc[-lag:-lag + opencap_data.shape[0]].reset_index(drop=True)
    force_data = force_data.iloc[-lag*10:-lag*10 + opencap_data.shape[0]*10].reset_index(drop=True)
    
    force_data["time"] = force_data["time"] - force_data["time"][0]
    marker_data["Time"] = marker_data["Time"] - marker_data["Time"].t[0]

    opencap_data = opencap_data.reset_index(drop=True)

    return marker_data, force_data, opencap_data


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
    df_merged = pd.concat([force_plate_data["r"]["data"], force_plate_data["l"]["data"]], axis=1)
    df_merged.reset_index(names="time", inplace=True)
    df_merged["time"] = df_merged["time"].apply(lambda x: x / 600)

    # Convert from mm to m
    for side in ["r", "l"]:
        df_merged[f"ground_force_{side}_pz"] = 0
        df_merged[f"ground_force_{side}_px"] = df_merged[f"ground_force_{side}_px"] / 1000
        df_merged[f"ground_force_{side}_py"] = df_merged[f"ground_force_{side}_py"] / 1000
        df_merged[f"ground_force_{side}_pz"] = df_merged[f"ground_force_{side}_pz"] / 1000

    return df_merged
