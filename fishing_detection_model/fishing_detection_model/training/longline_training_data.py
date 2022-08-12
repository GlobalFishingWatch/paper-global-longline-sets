from dataclasses import dataclass
from typing import List, Set

import tensorflow as tf

import h5py
import numpy as np
import pandas as pd

from ..model.generic_fishing_model import GenericFishingModel
from . import preprocess, train, training_data


@dataclass
class TrainingData:
    mmsi: Set[str]
    tracks: List[pd.DataFrame]  # list of valid tracks
    data: pd.DataFrame  # valid tracks concatenated into single DataFrame
    validation_mmsi: List[Set[str]]


# We keep a consistent set of validation folds so results don't shift when we update
# the training data
validation_mmsi = [
    {
        "225319000",
        "225374000",
        "367161340",
        "368027840",
        "412328757",
        "412328761",
        "412331499",
        "412465075",
    },
    {
        "412465076",
        "412465077",
        "412465078",
        "412467334",
        "412467335",
        "413322620",
        "416002616",
        "416002659",
    },
    {
        "416004453",
        "416085700",
        "416232900",
        "416233500",
        "416768000",
        "416826000",
        "431154000",
    },
    {
        "431704220",
        "432298000",
        "432881000",
        "440236000",
        "441057000",
        "510052000",
        "510053000",
    },
    {
        "576594000",
        "576660000",
        "576678000",
        "601061600",
        "601274700",
        "659283000",
        "664104000",
    },
]


def load(path: str) -> TrainingData:
    """Load training data from path"""

    with h5py.File(path, "r") as hdf:
        raw_df_labeled = training_data.hdf_to_dataframe(hdf)

    tracks = train.extract_valid_tracks(raw_df_labeled)

    for track in tracks:
        # Map "between_setting_and_hauling" (2) to "other" (0)
        track.loc[(track.label == 2), "label"] = 0
        # Now map "hauling" (previously 3) to the unoccupied 2
        track.loc[(track.label == 3), "label"] = 2
        track["orig_label"] = track["label"]

    df = pd.concat(tracks)
    mmsi = set(df.mmsi)

    return TrainingData(
        mmsi=mmsi, tracks=tracks, data=df, validation_mmsi=validation_mmsi
    )


def is_labeled(track: pd.DataFrame) -> bool:
    """Determine if track is labeled"""
    # By convention, unlabeled tracks have indices >= 1000
    return track.ndx.iloc[0] < 1000


class DataGenerator(tf.keras.utils.Sequence):
    """Generate training data for longline fishing model

    Args:
        tracks: features and labels along a contiguous vessel track
        model: model we are generating data for
        batch_size: size of batches to pass to the model
        samples_per_labeled_track: create this many samples per track
        resample_interval: how often (in epochs) to resample the track
            and obtain new samples.
        shuffle: if True, shuffle the samples every epoch

    Attributes:
        norm_samples: number of training examples to use when estimating
            normalization values.
        means: these values are subtracted from features before passing
            to the model. They may not be actual means.
        stds: features are divided by these values before being passed
            to the model. The may not be actual standard deviations.
    """

    norm_samples: int = 100
    means: np.ndarray
    stds: np.ndarray

    def __init__(
        self,
        tracks: List[pd.DataFrame],
        model: GenericFishingModel,
        batch_size: int,
        samples_per_labeled_track: int,
        resample_interval: int = 5,
        shuffle: bool = True,
    ):
        self.tracks = tracks
        self.features = None
        self.batch_size = batch_size
        self.model = model
        self.samples_per_labeled_track = samples_per_labeled_track
        self.shuffle = shuffle
        self.set_norm()
        self.sample()
        self.indices = np.arange(len(self.data[0]))
        if self.shuffle:
            np.random.shuffle(self.indices)
        if "feature_means" not in self.model.metadata:
            self.model.metadata["feature_means"] = self.means
            self.model.metadata["features_stds"] = self.stds
        self.n_features = self.means.shape[2]
        self.epoch = 0
        self.resample_interval = resample_interval

    def compute_n_indices(self):
        n_indices = 0
        for t in self.tracks:
            assert is_labeled(t)
            n_indices += self.samples_per_labeled_track
        return n_indices

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, index):
        batch = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        return self.__get_data(batch)

    def sample(self):
        samples = []
        for track in self.tracks:
            n_samples = self.samples_per_labeled_track
            interval = self.model.metadata["sample_interval"]
            samples.extend(
                training_data.randomly_sample_track(
                    track,
                    n_samples=n_samples,
                    sample_len=self.model.metadata["sample_length"],
                    sample_interval=interval,
                )
            )
        features, labels, weights = preprocess.preprocess_data(
            samples, self.model, weights=True, train=True
        )

        mask = np.asarray(labels) == -2
        assert mask.sum() == 0
        if len(labels.shape) == 1:
            labels = tf.one_hot(labels, 3)
        labels = np.asarray(labels)
        self.data = ((features - self.means) / self.stds, labels, weights)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        if self.epoch % self.resample_interval == 0:
            self.sample()
        self.epoch += 1

    def set_norm(self):
        features = []
        for track in self.tracks:
            samples = training_data.randomly_sample_track(
                track,
                n_samples=self.norm_samples,
                sample_len=self.model.metadata["sample_length"],
                sample_interval=self.model.metadata["sample_interval"],
            )
            if not samples:
                continue
            ftrs, _, _ = preprocess.preprocess_data(
                samples, self.model, weights=True, train=True
            )
            features.extend(ftrs)
        self.means = np.mean(features, axis=(0, 1), keepdims=True)
        # Don't offset xy, we want zero to be at label.
        self.means[:, :, :] = 0
        self.stds = np.std(features, axis=(0, 1), keepdims=True)
        # Use features as is
        self.stds[:, :, :] = 1

    def __get_data(self, batch):
        features, labels, weights = self.data
        batch_features = []
        for i in batch:
            batch_features.append(features[i])
        return (
            np.array(batch_features),
            np.array([labels[i] for i in batch]),
            np.array([weights[i] for i in batch]),
        )
