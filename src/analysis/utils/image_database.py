import numpy as np
import pandas as pd


def filter_exposures_with_both_light_and_dark(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows whose exposure_times have both 'light' and 'dark' frames present.
    """
    # For each exposure_time, figure out the set of frame_types
    grouped = df.groupby("exposure_time")["frame_type"].apply(set)

    # We only want those exposure_times that contain both 'light' and 'dark'
    valid_exposures = grouped[
        grouped.apply(lambda s: {"light", "dark"}.issubset(s))
    ].index

    # Filter the original df to keep only those exposure times
    return df[df["exposure_time"].isin(valid_exposures)].copy()


def get_light_frames_for_exposure(df: pd.DataFrame, exposure: float) -> pd.DataFrame:
    """
    Return all rows (as a DataFrame) from df for a given exposure_time
    with frame_type == 'light'. Includes 'filepath' for opening the file.
    """
    return df[
        (np.abs(df["exposure_time"] - exposure) < 1e-6) & (df["frame_type"] == "light")
    ]


def get_dark_frames_for_exposure(df: pd.DataFrame, exposure: float) -> pd.DataFrame:
    """
    Return all rows (as a DataFrame) from df for a given exposure_time
    with frame_type == 'dark'. Includes 'filepath' for opening the file.
    """
    return df[
        (np.abs(df["exposure_time"] - exposure) < 1e-6) & (df["frame_type"] == "dark")
    ]


def get_unique_exposures_in_ascending_order(df: pd.DataFrame) -> list:
    """
    Return a list of all unique exposure times in ascending order.
    """
    exposures = [float(x) for x in sorted(df["exposure_time"].unique())]
    return exposures
