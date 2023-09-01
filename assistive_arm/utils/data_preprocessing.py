import csv
import numpy as np
import pandas as pd

from pathlib import Path


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


def prepare_mocap_data(df: pd.DataFrame) -> pd.DataFrame:
    df.reset_index(names="timestamp", inplace=True)
    df["timestamp"] = df["timestamp"].apply(lambda x: x / 60)

    # Mocap data is in mm, convert to m
    df["X"] = df["X"] / 1000
    df["Y"] = df["Y"] / 1000
    df["Z"] = df["Z"] / 1000
    return df


def prepare_mocap_force_data(df: pd.DataFrame) -> pd.DataFrame:
    df.reset_index(names="timestamp", inplace=True)
    df.drop(columns=["CoPz", "nan"], inplace=True)
    df["timestamp"] = df["timestamp"].apply(lambda x: x / 600)

    return df
