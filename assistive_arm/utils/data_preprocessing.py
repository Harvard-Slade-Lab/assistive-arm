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
    frequency = 60 #Hz

    new_cols = ["Time"]

    for marker in marker_names:
        new_cols.extend(3 * [marker])

    coord = ["X", "Y", "Z"] * (len(df.columns) // 3)
    coord.insert(0, "time")

    df.reset_index(inplace=True)
    df.columns = pd.MultiIndex.from_tuples(zip(new_cols, coord))

    df["Time"] = df["Time"] / frequency

    # Mocap data is in mm, convert to m
    for marker in marker_names:
        df[marker] = df[marker] / 1000

    return df

def xcorr_and_shift(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, int]:
    """ Cross-correlate two signals and return the lag

    Args:
        x (np.ndarray): first signal
        y (np.ndarray): second signal

    Returns:
        Tuple[np.ndarray, int]: tuple containing the cross-correlation and the lag
    """
    # Pad the shorter signal with NaN
    size_diff = len(x) - len(y)

    if size_diff > 0:
        y = np.pad(y, (0, size_diff), mode='constant', constant_values=np.nan)
    elif size_diff < 0:
        x = np.pad(x, (0, -size_diff), mode='constant', constant_values=np.nan)

    # Compute cross-correlation and lag
    correlation = np.correlate(x, y, 'full')
    lag = round((np.argmax(correlation) - len(x) + 1)/4) - 19 # TODO Hard-coded shifting
    # Shift the 'y' signal based on the calculated lag
    # y_shifted = np.roll(y, -lag)

    return correlation, lag

def sync_mocap_with_opencap(marker_data: pd.DataFrame, force_data: pd.DataFrame, opencap_data: pd.DataFrame) -> pd.DataFrame:
    # Cut the mocap data such that it perfectly overlaps with opencap

    _, lag = xcorr_and_shift(marker_data.Knee.X, opencap_data.LKnee.X)

    # Shift the data by lag
    marker_data = marker_data.iloc[-lag:-lag + opencap_data.shape[0]].reset_index(drop=True)
    force_data = force_data.iloc[-lag*10:-lag*10 + opencap_data.shape[0]*10].reset_index(drop=True)
    
    force_data["time"] = force_data["time"] - force_data["time"][0]
    marker_data["Time"] = marker_data["Time"] - marker_data["Time"].time[0]
    
    opencap_data = opencap_data.reset_index(drop=True)

    return marker_data, force_data, opencap_data


def prepare_mocap_force_data(
    df_right: pd.DataFrame,
    df_left: pd.DataFrame,
    forces_in_world: bool=True,
    right_coordinates: List[str]=None,
    left_coordinates: List[str]=None,
) -> pd.DataFrame:
    
    df_right.drop(columns=["nan"], inplace=True)
    df_left.drop(columns=["nan"], inplace=True)

    if not forces_in_world:
        # Represent center of pressure in world coordinates
        for df, edge_coordinates, side in zip([df_right, df_left], [right_coordinates, left_coordinates], ['r', 'l']):
            coords = [float(coord[1]) for coord in edge_coordinates]
            plate_origin_x = np.mean(coords[0::3])
            plate_origin_y = np.mean(coords[1::3])
            plate_origin_z = np.mean(coords[2::3])
            
            df[f"ground_force_{side}_px"] = plate_origin_x + df[f"ground_force_{side}_px"]
            df[f"ground_force_{side}_py"] = plate_origin_y + df[f"ground_force_{side}_py"]
            df[f"ground_force_{side}_pz"] = plate_origin_z + df[f"ground_force_{side}_pz"]

    # Divide frame by sampling rate
    df_merged = pd.concat([df_right, df_left], axis=1)
    df_merged.reset_index(names="time", inplace=True)
    df_merged["time"] = df_merged["time"].apply(lambda x: x / 600)

    # Convert from mm to m
    for side in ["r", "l"]:
        df_merged[f"ground_force_{side}_pz"] = 0
        df_merged[f"ground_force_{side}_px"] = df_merged[f"ground_force_{side}_px"] / 1000
        df_merged[f"ground_force_{side}_py"] = df_merged[f"ground_force_{side}_py"] / 1000
        df_merged[f"ground_force_{side}_pz"] = df_merged[f"ground_force_{side}_pz"] / 1000

    return df_merged
