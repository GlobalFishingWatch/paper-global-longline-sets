import logging
from typing import List

import h5py
import numpy as np
import pandas as pd


def hdf_to_dataframe(hdf: h5py.File) -> pd.DataFrame:
    """Convert HDF data to Pandas DataFrame"""
    df = pd.DataFrame({k: v[:] for (k, v) in hdf.items()})
    df["mmsi"] = df.mmsi.str.decode("utf-8")
    df["timestamp"] = np.round(hdf["timestamp"][:] * 1e6).astype("datetime64[us]")
    return df


def randomly_sample_track(
    track: pd.DataFrame,
    n_samples: int,
    sample_len: int,
    sample_interval: int,
    sample_labels: bool = True,
) -> List[pd.DataFrame]:
    """Create n_sample random subtracks from a track"""
    eff_len = (sample_len - 1) * sample_interval + 1
    valid_len = len(track) - eff_len + 1
    assert valid_len > 0
    if valid_len < n_samples:
        logging.info("short track; sampling with replacement")
        replace = True
    else:
        replace = False

    indices = np.arange(0, valid_len)
    lbl_loc = sample_len // 2 * sample_interval
    labels = track["label"].iloc[lbl_loc : valid_len + lbl_loc].values
    if sample_labels:
        # Only sample points with labels
        if len(labels.shape) == 1:
            mask = labels != -1
        else:
            mask = np.ones_like(labels, dtype=bool)
    else:
        mask = np.ones_like(labels, dtype=bool)

    if not mask.sum():
        logging.info("No samples, return empty list")
        return []

    sample_locs = np.random.choice(indices[mask], size=n_samples, replace=replace)
    return [track[i : i + eff_len : sample_interval] for i in sample_locs]


def sequentially_sample_track(
    track: pd.DataFrame, sample_len: int, sample_interval: int
) -> List[pd.DataFrame]:
    """Create a subtrack at every position along a track"""
    eff_len = (sample_len - 1) * sample_interval + 1
    valid_len = len(track) - eff_len + 1
    assert valid_len > 0

    indices = np.arange(0, valid_len)

    return [track[i : i + eff_len : sample_interval] for i in indices]
