import pandas as pd
import numpy as np

from pathlib import Path
from typing import Tuple
import typer

import matplotlib.pyplot as plt
from assistive_arm.utils.data_preprocessing import (
    read_headers,
    prepare_opencap_data,
    prepare_mocap_data,
    prepare_mocap_force_data,
)


def main(
    right_force_path: Path = typer.Argument(..., help="Path to right force .tsv file"),
    left_force_path: Path = typer.Argument(..., help="Path to left force .tsv file"),
    mocap_marker_path: Path = typer.Argument(
        ..., help="Path to mocap marker .tsv file"
    ),
    opencap_marker_path: Path = typer.Argument(
        ..., help="Path to opencap marker .trc file"
    ),
):
    right_force = pd.read_csv(
        right_force_path,
        delimiter="\t",
        skiprows=26,
        names=["FPx", "FPy", "FPz", "Mx", "My", "Mz", "CoPx", "CoPy", "CoPz", "nan"],
    )
    left_force = pd.read_csv(
        left_force_path,
        delimiter="\t",
        skiprows=26,
        names=["FPx", "FPy", "FPz", "Mx", "My", "Mz", "CoPx", "CoPy", "CoPz", "nan"],
    )
    mocap_marker = pd.read_csv(
        mocap_marker_path, delimiter="\t", skiprows=11, names=["X", "Y", "Z"]
    )

    opencap_marker = pd.read_csv(opencap_marker_path, delimiter="\t", skiprows=3).tail(
        -1
    )

    opencap_headers = read_headers(opencap_marker_path, 3)


if __name__ == "__main__":
    typer.run(main)
