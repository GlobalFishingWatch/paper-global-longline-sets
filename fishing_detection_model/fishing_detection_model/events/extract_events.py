import datetime
import logging
from collections import namedtuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from skimage import morphology

SegmentInfo = namedtuple(
    "SegmentInfo", ["id", "start_ndx", "end_ndx", "label", "start_time", "end_time"]
)


def create_SegmentInfo(first_row_of_seg, last_row_of_seg):
    """Create a SegmentInfo from its first and last points"""
    assert last_row_of_seg.category == first_row_of_seg.category

    return SegmentInfo(
        id=str(first_row_of_seg.id),
        start_ndx=first_row_of_seg.ndx,
        end_ndx=last_row_of_seg.ndx + 1,
        label=first_row_of_seg.category,
        start_time=first_row_of_seg.timestamp
        - datetime.timedelta(minutes=2, seconds=30),
        end_time=last_row_of_seg.timestamp + datetime.timedelta(minutes=2, seconds=30),
    )


def create_segments(track):
    """Convert a track into series of segments with uniform labels

    Parameters
    ----------
    track : Dataframe
        Must have `timestamp` and `category` columns.

    Yields
    ------
    SegmentInfo
    """

    track = track.reset_index(drop=True)
    track["ndx"] = range(len(track))
    if not len(track) >= 2:
        logging.info("track too short, returning")
        return
    delta = track.iloc[1].timestamp - track.iloc[0].timestamp
    assert delta >= np.timedelta64(5, "m"), delta
    first_row_of_seg = last_row_of_seg = track.iloc[0]
    for row in track.iloc[1:].itertuples():
        if row.category != first_row_of_seg.category:
            logging.debug(
                f"Yielding seg between {first_row_of_seg} and {last_row_of_seg}"
            )
            yield create_SegmentInfo(first_row_of_seg, last_row_of_seg)
            first_row_of_seg = row
        last_row_of_seg = row
    logging.debug(f"Yielding seg between {first_row_of_seg} and {last_row_of_seg}")
    yield create_SegmentInfo(first_row_of_seg, last_row_of_seg)


def scores_to_classes(predictions, categories, lbl_map, sigma=0.0, closing=0):
    """Turn array of class scores into predicted classes

    A combination of Gaussian smoothing and morphological
    closing is used to reduce unrealistic islands of distinct
    class scores.

    Parameters
    ----------
    predictions: N x n_classes array of float
    categories: list of strings
        in sequence for binary_closing
    lbl_map: Dict
        maps string labels to integers
    sigma: float, optional
        Controls the width of Gaussian smoothing
    closing: int, optional
        Width of morphological closing to apply

    Returns
    -------
    array of int
    """
    if sigma > 0.0:
        smoothed = gaussian_filter1d(predictions, sigma, mode="nearest", axis=0)
    else:
        smoothed = predictions
    category_nos = np.argmax(smoothed, axis=1)

    if closing > 0:
        for cat in categories:
            cat_no = lbl_map[cat]
            mask = category_nos == cat_no
            mask = morphology.binary_closing(
                mask[np.newaxis, :], morphology.rectangle(1, closing)
            )[0]
            category_nos[mask] = cat_no
    return category_nos


def inferred_to_score_array(inferred, categories, lbl_map):
    """Convert inferred values from BQ to 2D array

    Parameters
    ----------
    inferred : DataFrame
        Must have timestamp, category, and score attributes
    categories : list of strings
    lbl_map : Dict
        maps string labels to integers

    Yields
    -------
    str
        Id of vessel
    2D array of float
        Array of scores where rows are by timestamp and cols are by category
    DataFrame
        A dataframe that matches the scores array. Used so that we can attach
        class labels to lon, lat, etc.
    """

    ids = inferred.id.unique()
    for id_ in ids:
        df = inferred[inferred.id == id_]
        times = df.timestamp.unique()
        time_map = {t: i for (i, t) in enumerate(times)}
        array = np.zeros([len(times), len(categories)], dtype=float)
        for x in df.itertuples():
            i = time_map[x.timestamp]
            j = lbl_map[x.category]
            array[i, j] = x.score
        yield id_, array, df


def add_labels(inferred, categories, sigma=None, closing=None):
    """Add class labels to inferred data.

    Parameters
    ----------
    inferred : Dataframe
    categories : list of strings
    sigma: float, optional
        Controls the width of Gaussian smoothing
    closing: int, optional
        Width of morphological closing to apply

    Yields
    ------
    str
        Id of vessel
    DataFrame
        Matches inferred values for that vessel except that scores
        are collapsed into `category` (str) and `category_no` (int).
    """

    lbl_map = {c: i for (i, c) in enumerate(categories)}
    inv_lbl_map = {v: k for (k, v) in lbl_map.items()}

    keep_columns = ["id", "timestamp"]
    for id_, scores, df in inferred_to_score_array(inferred, categories, lbl_map):
        logging.debug(f"processing {len(df)} values in add_labels")
        cats = scores_to_classes(
            scores, categories, lbl_map, sigma=sigma, closing=closing
        )
        df = df.drop_duplicates(subset=keep_columns).copy()
        df["category_no"] = cats
        df["category"] = [inv_lbl_map[x] for x in cats]
        yield id_, df


def add_dummy_class(data, categories):
    orig_columns = data.columns
    assert len(categories) in [1, 2]
    row_list = []
    df_categories = data.category.unique()
    if len(categories) == 2:
        [dummy_name] = set(categories) - set(df_categories)
    else:
        dummy_name = "other"

    for row in data.itertuples(index=False):
        row2 = row._replace(category=dummy_name, score=1 - row.score)
        row_list.append(row)
        row_list.append(row2)

    data = pd.DataFrame(row_list)
    assert all(data.columns == orig_columns)
    return data


def label_and_segment(data, categories, sigma=0.0, closing=0):
    """Add class labels to inferred data and create segments

    note: contents of data are ['id', 'timestamp', 'category', 'score']

    Parameters
    ----------
    data : Dataframe
    categories :  list of strings
        In sequence to apply binary_closing
    sigma: float, optional
        Controls the width of Gaussian smoothing
    closing: int, optional
        Width of morphological closing to apply

    Yields
    ------
    SegmentInfo
        See create_SegmentInfo for structure
    """
    df_categories = data.category.unique()

    if len(df_categories) == 1:
        data = add_dummy_class(data, categories)

    predictions = add_labels(data, categories, sigma=sigma, closing=closing)
    for _, preds in predictions:
        logging.debug(f"processing {len(preds)} preds")
        for seg in create_segments(preds):
            seg = seg._asdict()
            seg.pop("start_ndx")
            seg.pop("end_ndx")
            yield seg
