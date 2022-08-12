"""Functions to help set up model training
"""
from typing import Generator, List, Tuple

import numpy as np
import pandas as pd
from skimage import morphology


def extract_valid_segments(
    track: pd.DataFrame, max_stop_hr: float = 24, min_segment_len: int = 97
) -> Generator[pd.DataFrame, None, None]:
    """Yield valid segments of track

    Sections of track are considered invalid if they:
    * Have implied speed > 30 knots
    * Jump over the dateline
    * Have dt != 5 minutes
    * Are stopped (< 0.2 knots) for over max_stop_hr.
    * are shorter than min_segment_len

    Args:
        track: set of features from a contiguous track.
        max_stop_hr (optional): break track if there is a stop greater than this.
        min_segment_len (optional): segments shorter than this length are dropped.

    Yields:
        valid segment as defined above.
    """
    raw_dlons = track.lon.values[1:] - track.lon.values[:-1]
    dlons = (raw_dlons + 180) % 360 - 180
    vx = np.cos(np.radians(track.lat.values[1:])) * dlons * 60 * 12
    dlats = track.lat.values[1:] - track.lat.values[:-1]
    vy = dlats * 60 * 12
    deltas = np.hypot(vx, vy)
    # raw_breaks indicate the first point in a two point run that should
    # be excluded. This results from both deltas and raw_dlons being compute
    # from pairs of points and both the first and second should be included.
    raw_breaks = False
    # Add points where vessel is traveling over 30 knots
    raw_breaks |= deltas >= 30
    # Add points where vessel is crossing the dateline.
    # We may already handle this correctly elsewhere, but remove to be safe
    raw_breaks |= abs(raw_dlons) > 10
    # Add points where dt is not 5 minutes
    dt = track.timestamp.values[1:] - track.timestamp.values[:-1]
    raw_breaks |= dt != np.timedelta64(5, "m")
    # Convert `raw_breaks` to `breaks` by expanding each point one to the right.
    breaks = np.zeros(len(track), dtype=bool)
    breaks[:-1] |= raw_breaks
    breaks[1:] |= raw_breaks
    # Add points where stopped for over an hour
    stopped = track.speed_knots.values < 0.2
    mask = morphology.square(max_stop_hr)
    # Keep max_hours / 2 stopped points at each end
    stopped = morphology.binary_erosion(stopped[None, :], mask)[0]
    breaks |= stopped
    # Get a list of break points, and use that to create list of ranges, then return
    # longest range.
    [breaks_ndxs] = np.where(breaks)
    last = 0
    ranges = []
    for ndx in breaks_ndxs:
        ranges.append((last, ndx))
        last = ndx + 1
    ranges.append((last, len(track) + 1))
    ranges.sort(key=lambda x: (x[1] - x[0]), reverse=True)
    for i, (start, end) in enumerate(ranges):
        if i >= 1000:
            break
        if end - start >= min_segment_len:
            chunk = track.iloc[start:end].copy()
            chunk.ndx = chunk.ndx + i / 1000
            yield chunk


def extract_valid_tracks(track_data: pd.DataFrame) -> List[pd.DataFrame]:
    """Make list of valid tracks"""
    tracks: List[pd.DataFrame] = []
    for ndx in track_data.ndx.unique():
        track = track_data[track_data.ndx == ndx]
        tracks.extend(extract_valid_segments(track))
    return tracks


def point2segs(track: pd.DataFrame, column: str = "label") -> List[Tuple]:
    track.reset_index(inplace=True, drop=True)
    lastIdxs = np.where(track[column].shift(-1) != track[column])[0]
    startIdx = 0
    track_start_time = track.timestamp[startIdx]
    result = []
    track_end = len(track)
    timeDelta = track.timestamp.iloc[1] - track.timestamp.iloc[0]
    for i, lastIdx in enumerate(lastIdxs):

        seg_end = lastIdx + 1
        best_end = min(track_end, seg_end)
        seg_label = np.unique(track[column].iloc[startIdx:best_end])[0]
        seg_start_time = track.timestamp.iloc[startIdx]
        relative_start_time = (seg_start_time - track_start_time) / np.timedelta64(
            1, "h"
        )
        seg_end_time = track.timestamp.iloc[lastIdx] + timeDelta
        relative_end_time = (seg_end_time - track_start_time) / np.timedelta64(1, "h")
        seg_duration = (seg_end_time - seg_start_time) / np.timedelta64(1, "h")
        # mmsi, track idx, label/prediction, Start timestamp, End timestamp, Duration, Start index, End index
        seg_info = (
            track.mmsi.iloc[0],
            track.ndx.iloc[0],
            seg_label,
            seg_start_time,
            seg_end_time,
            seg_duration,
            relative_start_time,
            relative_end_time,
            startIdx,
            seg_end,
        )

        result.append(seg_info)
        startIdx = seg_end

    columns = [
        "mmsi",
        "track_idx",
        "label",
        "start_time",
        "end_time",
        "duration",
        "relative_start_time",
        "relative_end_time",
        "start_idx",
        "end_idx",
    ]

    return pd.DataFrame(result, columns=columns)
