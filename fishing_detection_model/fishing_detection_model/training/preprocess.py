"""Preprocess data


Note: DK labeler -> David Kroodsma's labeling tool
"""
import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from . import train


def load_dk_labeller_data(path):
    """Load data from DK labeler as DataFrame

    Parameters
    ----------
    path : str

    Returns
    -------
    Pandas DataFrame
    """
    with open(path) as f:
        obj = json.load(f)
    data = {
        "mmsi": [obj["mmsi"]] * len(obj["timestamps"]),
        "timestamp": obj["timestamps"],
        "lat": obj["lats"],
        "lon": obj["lons"],
        "speed": obj["sogs"],
        "course": obj["courses"],
        "label": obj["fishing"],
    }
    return pd.DataFrame(data)


def load_gfw_labeller_data(path, annotator=None):
    """Load data from GFW labeler as DataFrame

    Parameters
    ----------
    path : str
    annotator: str

    Returns
    -------
    Pandas DataFrame
    """
    with open(path) as f:
        obj = json.load(f)
    data = {
        "mmsi": [obj["properties"]["vessel"]["mmsi"]]
        * len(obj["features"][0]["properties"]["coordinateProperties"]["times"]),
        "timestamp": pd.to_datetime(
            obj["features"][0]["properties"]["coordinateProperties"]["times"], unit="ms"
        ),
        "lat": [coord[1] for coord in obj["features"][0]["geometry"]["coordinates"]],
        "lon": [coord[0] for coord in obj["features"][0]["geometry"]["coordinates"]],
        "speed": obj["features"][0]["properties"]["coordinateProperties"]["speed"],
        "course": obj["features"][0]["properties"]["coordinateProperties"]["course"],
        "elevation": obj["features"][0]["properties"]["coordinateProperties"]["elevation"],
        "label": obj["features"][0]["properties"]["coordinateProperties"]["labels_id"],
        "annotator": annotator,
    }
    return pd.DataFrame(data)


def load_gfw_unlabelled_data(path, annotator=None):
    """Load data from DK labeler as DataFrame

    Parameters
    ----------
    path : str

    Returns
    -------
    Pandas DataFrame
    """
    with open(path) as f:
        obj = json.load(f)

    data = {
        "mmsi": [obj["properties"]["vessel"]["mmsi"]]
        * len(obj["features"][0]["properties"]["coordinateProperties"]["times"]),
        "timestamp": pd.to_datetime(
            obj["features"][0]["properties"]["coordinateProperties"]["times"], unit="ms"
        ),
        "lat": [coord[1] for coord in obj["features"][0]["geometry"]["coordinates"]],
        "lon": [coord[0] for coord in obj["features"][0]["geometry"]["coordinates"]],
        "speed": obj["features"][0]["properties"]["coordinateProperties"]["speed"],
        "course": obj["features"][0]["properties"]["coordinateProperties"]["course"],
        "label": -np.ones(
            len(obj["features"][0]["properties"]["coordinateProperties"]["times"])
        ),
        "annotator": annotator,
    }
    return pd.DataFrame(data)


def _check_consistency(group, columns):
    """assure values are consistent across DataFrames that we are merging"""
    for c in columns:
        example = group[0][c].values
        for x in group[1:]:
            assert np.alltrue(x[c].values == example)


def _merge_labels(labels, undefined=-1):
    """Merge multiple labels at a single time point"""
    filtered = [x for x in labels if x != undefined]
    n_valid_labels = len(filtered)
    if n_valid_labels > 0:
        [(val, cnt)] = Counter(filtered).most_common(1)
        if cnt > n_valid_labels / 2:
            return val
    return undefined


def _merge_labels_score(labels, undefined=-1):
    """Merge multiple labels at a single time point"""
    filtered = [x for x in labels if x != undefined]
    if len(filtered) <= 1:
        filtered = list(labels)
    n_valid_labels = len(filtered)
    if filtered:
        [(val, cnt)] = Counter(filtered).most_common(1)
        n_unique_labels = len(np.unique([str(x) for x in filtered]))
        cnt2 = 0
        if n_unique_labels > 1:
            (val2, cnt2) = Counter(filtered).most_common()[1]
        if (cnt > (n_valid_labels / 2)) or (
            (cnt >= n_valid_labels / 2) and (cnt > 1) and (cnt > cnt2)
        ):
            return (
                val,
                float(cnt / len(filtered)),
            )  # Add cnt / num_labels and num_filters / num_labels
    return (undefined, 0.0)


def _check_that_mmsi_match(sequence):
    #     sequence = sequence.astype(str)
    if not np.alltrue(sequence.values[0] == sequence.values):
        print(sequence.values[0])
        print(np.unique(sequence.values))
        raise ValueError("mmsi are not all equal")


def _check_timestamps_in_order(sequence):
    if not np.alltrue(sequence.values[1:] >= sequence.values[:-1]):
        raise ValueError("timestamps are not ordered")


def _drop_duplicate_timestamps(df):
    duplicates = df.timestamp.values[1:] == df.timestamp.values[:-1]
    duplicates = np.concatenate([[False], duplicates])
    return df[~duplicates]


def _check_consistent_delta(sequence):
    delta = sequence.values[1] - sequence.values[0]
    if not np.alltrue((sequence.values[1:] - sequence.values[:-1]) == delta):
        raise ValueError("timestamps have different delta")


def merge_gfw_labeller_data(datasets, undefined=-1):
    """Merge data sets where appropriate.

    If there are multiple datasets related to the same
    time period and MMSI merge them, combining annotations
    appropriately.

    Uses the opinion of the majority of annotators, treating
    ties as undefined

    Parameters
    ----------
    datasets : list of DataFrames

    Return
    list of DataFrames
    """
    grouped_by_key = defaultdict(list)
    for df in datasets:
        mmsi = df.mmsi[0]
        start = df.timestamp[0]
        grouped_by_key[(mmsi, start)].append(df)

    merged = []
    for group in grouped_by_key.values():
        _check_that_mmsi_match(group[0].mmsi)
        _check_consistency(
            group, ("mmsi", "timestamp", "lat", "lon", "speed", "course")
        )
        df = group[0].copy()
        labelsets = [x.label for x in group]
        df["label"] = [
            _merge_labels_score(lbls_at_t, undefined=undefined)
            for lbls_at_t in zip(*labelsets)
        ]
        if "annotator" in df:
            annsets = [x.annotator for x in group]
            df["annotator"] = [ann_at_t for ann_at_t in zip(*annsets)]
            df["annotator"] = df["annotator"].astype(str)
        df.sort_values(by="timestamp", inplace=True)
        _check_timestamps_in_order(group[0].timestamp)
        df = _drop_duplicate_timestamps(df)
        df.reset_index(drop=True, inplace=True)
        merged.append(df)

    merged.sort(key=lambda x: x.timestamp[0])

    return merged


def merge_dk_labeller_data(datasets, undefined=-1):
    """Merge data sets where appropriate.

    If there are multiple datasets related to the same
    time period and MMSI merge them, combining annotations
    appropriately.

    Uses the opinion of the majority of annotators, treating
    ties as undefined

    Parameters
    ----------
    datasets : list of DataFrames

    Return
    list of DataFrames
    """
    grouped_by_key = defaultdict(list)
    for df in datasets:
        mmsi = df.mmsi[0]
        start = df.timestamp[0]
        grouped_by_key[(mmsi, start)].append(df)

    merged = []
    for group in grouped_by_key.values():
        _check_that_mmsi_match(group[0].mmsi)
        _check_consistency(
            group, ("mmsi", "timestamp", "lat", "lon", "speed", "course")
        )
        df = group[0].copy()
        labelsets = [x.label for x in group]
        df["label"] = [
            _merge_labels(lbls_at_t, undefined=undefined)
            for lbls_at_t in zip(*labelsets)
        ]
        df.sort_values(by="timestamp", inplace=True)
        _check_timestamps_in_order(group[0].timestamp)
        df = _drop_duplicate_timestamps(df)
        df.reset_index(drop=True, inplace=True)
        merged.append(df)

    merged.sort(key=lambda x: x.timestamp[0])

    return merged


def _nn_interpolate(x, xp, fp):
    """Nearest Neighbor Interpolation"""
    indices = np.searchsorted(xp, x, side="left")
    indices = np.clip(indices, 0, len(xp) - 1)
    return fp[indices]


def interpolate(
    df,
    dt,
    time_column="timestamp",
    label_columns=("label",),
    degree_columns=("lon", "course"),
    constant_columns=("mmsi",),
):
    """Interpolate, treating angles correctly.

    Parameters
    ----------
    df : DataFrame
    dt : float
    time_column : str, optional
    label_columns : sequence of str, optional
        Interpolated using nearest neighbor interpolation
    angle_columns : sequence of str, optional
    constant_columns :
        Assumed to be constant.

    Returns
    -------
    DataFrame
    """
    df = df.reset_index(drop=True)
    label_columns = set(label_columns)
    degree_columns = set(degree_columns)
    constant_columns = set(constant_columns)
    old_t = df[time_column].values
    new_t = np.arange(old_t[0], old_t[-1], dt)
    tp = (old_t - old_t[0]) / np.timedelta64(1, "ns")
    t = (new_t - old_t[0]) / np.timedelta64(1, "ns")
    new_columns = {time_column: new_t}
    for c in df.columns:
        fp = df[c]
        if c == time_column:
            continue
        if c in constant_columns:
            _check_that_mmsi_match(fp)
            new_columns[c] = [fp[0]] * len(t)
        elif c in label_columns:
            new_columns[c] = _nn_interpolate(t, tp, fp)
        elif c in degree_columns:
            cos_p = np.cos(np.deg2rad(fp))
            sin_p = np.sin(np.deg2rad(fp))
            cos = np.interp(t, tp, cos_p)
            sin = np.interp(t, tp, sin_p)
            new_columns[c] = np.rad2deg(np.arctan2(sin, cos))
        #         elif c in concat_columns:

        else:
            new_columns[c] = np.interp(t, tp, fp)
    return pd.DataFrame(new_columns)


def add_weights(df):
    df["weights"] = 0
    for ndx in df.ndx.unique():
        weights = []
        for segment in train.point2segs(df[df.ndx == ndx]).values:
            weight = compute_segment_weights(segment)
            weights.extend(weight)
        df.loc[(df.ndx == ndx), "weights"] = weights


def preprocess_data(data, mdl, weights=False, train=True):
    # TODO: document me
    if train:
        angle = "random"
        sigma = 0.1
    else:
        angle = 0
        sigma = 0
    label_idx = int(np.floor(mdl.metadata["sample_length"] / 2))
    # Non numeric columns break things downstream so filter out here.
    columns = [
        cl for (cl, dt) in zip(data[0].columns, data[0].dtypes) if dt in (int, float)
    ]
    values = [x[columns].values for x in data]
    features = mdl.preprocessor.process_set(columns, values, angle=angle, sigma=sigma)
    s_weights = np.array([x.weights.iloc[label_idx] for x in data]) if weights else None
    labels = np.array([x.label.iloc[label_idx] for x in data])
    return features, labels, s_weights


# assumes consistent time intervals
def compute_segment_weights(segment, scale=2):  # 6 is half an hour
    # TODO: document me
    startIdx = segment[8]
    endIdx = segment[9]
    length = endIdx - startIdx
    crds = np.arange(0, length) + 1
    edge = np.minimum(crds, crds[::-1])
    return 1 - np.exp(-edge)
