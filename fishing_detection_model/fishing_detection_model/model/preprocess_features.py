"""Process records into features for fishing models
"""
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Deque, Generator, List, Literal, Sequence, Tuple, Union

import numpy as np


class Preprocessor:
    """Turn sequences of records into features for fishing models

    Args:
        outputs: names of the output features.
        n_samples: number of temporal samples per feature set.
        delta: spacing to use between records when generating features.
        raw_ndx: locations in each set of records where the label is taken.

    Note: currently only shift == 1 is supported.

    Available outputs:
    - timestamp
    - sin_local_time, cos_local_time
    - speed_knots
    - elevation_m
    - course_degrees
    - sin_course_degrees, cos_course_degrees
    - x, y : Vector in nm from center from track segment
    - proj_dx, proj_dy : projected distance after one hour at current speed
    - distance_from_shore_km
    """

    def __init__(
        self,
        outputs: List[str],
        n_samples: int,
        delta: int,
        raw_ndx: Sequence[int],
    ):
        self.outputs = outputs
        if raw_ndx is None:
            raw_ndx = [n_samples // 2]
        self.n_samples = n_samples
        self.delta = delta
        self.raw_ndx = raw_ndx

    def _output_to_inputs(self, key: str) -> List[str]:
        """Find which input key(s) are needed to supply the given output"""
        if key in ("x", "y"):
            return ["lon", "lat"]
        if key in ("proj_dx", "proj_dy"):
            return ["speed_knots", "course_degrees"]
        if key in ("sin_course_degrees", "cos_course_degrees"):
            return ["course_degrees"]
        if key in ("depth",):
            return ["elevation_m"]
        if key in ("cos_local_time", "sin_local_time"):
            return ["lon", "timestamp"]
        return [key]

    def find_input_keys(self) -> List[str]:
        """Find which input keys are needed to to use this preprocessor

        Returns:
            list of keys
        """
        keys = set()
        for k in self.outputs:
            keys |= set(self._output_to_inputs(k))
        return sorted(keys)

    @staticmethod
    def _compute_xy(
        lon: np.ndarray,
        lat: np.ndarray,
        angle: Union[float, np.ndarray],
        speed_noise: np.ndarray,
    ):
        """Compute x and y features from lon and lat"""
        n = lon.shape[-2] // 2
        dlon = (lon[..., 1:, :] - lon[..., :-1, :] + 180) % 360 - 180
        dlat = lat[..., 1:, :] - lat[..., :-1, :]
        lat = np.asarray(0.5 * (lat[..., 1:, :] + lat[..., :-1, :]), dtype=float)
        position_noise = 0.5 * (speed_noise[..., 1:, :] + speed_noise[..., :-1, :])
        dx = np.cos(np.radians(lat)) * dlon * 60.0 * position_noise
        dy = dlat * 60 * position_noise
        rdx = np.cos(angle) * dx - np.sin(angle) * dy
        rdy = np.cos(angle) * dy + np.sin(angle) * dx
        x = np.zeros_like(lon)
        x[..., 1:, :] = np.cumsum(rdx, axis=-2)
        y = np.zeros_like(lon)
        y[..., 1:, :] = np.cumsum(rdy, axis=-2)
        x -= x[..., n : n + 1, :]
        y -= y[..., n : n + 1, :]
        return x, y

    def _compute_feature(  # noqa
        self, key, t, speed, x, y, theta, elevation, dist_from_shore, lt
    ):
        """Compute the feature associated with key"""
        if key == "timestamp":
            return t
        if key == "cos_local_time":
            return np.cos(np.pi / 12 * lt)
        if key == "sin_local_time":
            return np.sin(np.pi / 12 * lt)
        if key == "speed_knots":
            return speed
        if key == "x":
            return x
        if key == "y":
            return y
        if key == "proj_dx":
            return np.cos(theta) * speed
        if key == "proj_dy":
            return np.sin(theta) * speed
        if key == "course_degrees":
            return 180 - 180 / np.pi * theta
        if key == "sin_course_degrees":
            return np.sin(np.pi / 2 - theta)
        if key == "cos_course_degrees":
            return np.cos(np.pi / 2 - theta)
        if key == "elevation_m":
            return elevation
        if key == "depth":
            return np.maximum(-elevation, 0)
        if key == "distance_from_shore_km":
            return dist_from_shore
        raise KeyError(key)

    def process_set(self, keys, values, angle=0, sigma=0):
        """Process a set of values into features

        Parameters
        ----------
        keys : list of str
        values : array of float
            The values in keys correspond to the last axis of the array.
            The array can either have shape n_sets × n_samples × n_features
            or n_samples × n_features.
        angle : float or "random", optional
            All spatial quantities are rotated by this amount. This helps reduce
            over fitting when using, e.g., `x` and `y` as features.
        sigma : float, optional
            How much randomness to inject into features.

        Returns
        -------
        array of float
        """
        values = np.asarray(values)
        input_keys = self.find_input_keys()

        def col(name):
            return values[..., keys.index(name)][..., None]

        def need(*names):
            return bool(set(names) & (set(input_keys) | set(self.outputs)))

        shape = values[..., 0:1].shape

        # Noise associated with speed, which we also use for jittering locations
        speed_noise = np.random.lognormal(0, sigma, size=shape)

        if angle == "random":
            if len(values.shape) == 3:
                angle = np.random.uniform(0, 2 * np.pi, size=(len(values), 1, 1))
            else:
                angle = np.random.uniform(0, 2 * np.pi)

        speed = col("speed_knots") * speed_noise if need("speed_knots") else None
        theta = (
            np.radians(90 - np.asarray(col("course_degrees"), dtype=float)) + angle
            if need("course_degrees")
            else None
        )
        x, y = (
            self._compute_xy(col("lon"), col("lat"), angle, speed_noise)
            if need("x", "y")
            else (None, None)
        )
        need_elev = need("elevation_m") or need("depth")
        elevation = col("elevation_m") if need_elev else None
        t = col("timestamp") if need("timestamp") else None
        if need("cos_local_time", "sin_local_time"):
            dt = []
            times = col("timestamp")
            lons = col("lon")
            for val, ln in zip(times.flatten(), lons.flatten()):
                dtx = datetime.fromtimestamp(val, timezone.utc)
                dtx += timedelta(hours=ln * 24 / 360)
                dt.append(dtx.hour + dtx.minute / 60)
            lt = np.array(dt).reshape(*times.shape)
        else:
            lt = None

        dist_from_shore = (
            col("distance_from_shore_km") if need("distance_from_shore_km") else None
        )

        features = [
            self._compute_feature(
                k, t, speed, x, y, theta, elevation, dist_from_shore, lt
            )
            for k in self.outputs
        ]
        return np.concatenate(features, axis=-1)

    def _extract_dq_vals(self, dq, keys, angle, sigma):
        valset = [vals for (raw, vals) in dq][:: self.delta]
        raw_raw = [r for (r, vals) in dq][:: self.delta]
        raw = [raw_raw[i] for i in self.raw_ndx]
        return raw, self.process_set(keys, valset, angle=angle, sigma=sigma)

    def feature_sets(
        self,
        input_records: Sequence[dict],
        angle: Union[float, Literal["random"]] = 0,
        sigma: float = 0,
    ) -> Generator[Tuple[List[dict], np.array], None, None]:
        """Yield sets of features suitable for fishing models.

        This yields `n_outputs = len(inputs_records) - maxlen` outputs
        where `maxlen = (n_samples - 1) * delta  + 1`

        Args:
            input_records : sequence of dict
                The dict should contain the keys returned by `find_input_keys`
            angle : float or "random", optional
                All spatial quantities are rotated by this amount. This helps
                reduce overfitting when using, e.g., `x` and `y` as features.
            sigma : float, optional
                How much randomness to inject into features.

        Yields:
            output_records: the records colocated with the predictions. This
                allows values such as `timestamp`, 'lon', and 'lat' to be
                extracted for use with the predictions. The list has a length of
                `n_outputs` x len(raw_ndx).
            features : n_outputs × n_samples x len(outputs) array of float
                Features suitable for feeding to fishing model.
        """
        maxlen = (self.n_samples - 1) * self.delta + 1
        dq: Deque[Tuple[dict, list]] = deque(maxlen=maxlen)
        keys = self.find_input_keys()
        # This will need to updated when supporting shift != 1
        shift = len(self.raw_ndx)

        # Prime deque by loading maxlen records
        ir_iter = iter(input_records)
        for i in range(maxlen):
            try:
                x = next(ir_iter)
            except StopIteration:
                # There were not enough records to do any inference
                return
            dq.append((x, [x[k] for k in keys]))

        while True:
            yield self._extract_dq_vals(dq, keys, angle, sigma)
            i = -1
            for i in range(shift):
                try:
                    x = next(ir_iter)
                except StopIteration:
                    if i + 1 > 0:
                        yield self._extract_dq_vals(dq, keys, angle, sigma)
                    return
                dq.append((x, [x[k] for k in keys]))
