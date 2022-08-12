from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

from tensorflow.keras.callbacks import History

import numpy as np
import pandas as pd

from ..model.generic_fishing_model import GenericFishingModel
from . import longline_training_data, preprocess, training_data


@dataclass
class TrainerResult:
    trainer: "LonglineTrainer"
    model: GenericFishingModel
    history: History


class LonglineTrainer:
    """Train a longline model

    Args:
        model_factory: function that creates a model
        model_args: keyword args for model_factory
        data: DataFrame used to construct training data
        training_mmsi: set of mmsi to train on
        val_mmsi: set of mmsi to compute validation metrics on
        allow_overlap: allow training and validation mmsi to overlap.
            Used during final training when validation is there just
            to make sure things don't go completely off the rails.

    Attributes:
        train_samples: samples per track to use in training set
        val_samples: samples per track to use in validation set
        step: epochs to train for
        batch_size: examples per minibatch
    """

    train_samples: int = 100
    val_samples: int = 100
    steps: int = 100
    batch_size: int = 32

    def __init__(
        self,
        model_factory: Type[GenericFishingModel],
        model_args: Dict[str, Any],
        data: pd.DataFrame,
        training_mmsi: Iterable[str],
        val_mmsi: Iterable[str],
        allow_overlap: bool = False,
    ):
        self.model_factory = model_factory
        self.model_args = model_args
        self.data = data
        if not allow_overlap:
            assert len(np.intersect1d(training_mmsi, val_mmsi)) == 0
        self.training_mmsi = training_mmsi
        self.val_mmsi = val_mmsi

    def extract_tracks(self) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
        """Extract valid tracks and split into training and test"""
        training = []
        validation = []
        val_mmsi = self.val_mmsi
        training_mmsi = self.training_mmsi

        for track in self.data.tracks:
            if len(track) < 97 * 3:
                continue
            track = track.copy()
            labeled = longline_training_data.is_labeled(track)

            if labeled:
                track["weights"] = 1.0
                track["label"] = track["orig_label"]
            else:
                continue

            preprocess._check_timestamps_in_order(track["timestamp"])
            preprocess._check_that_mmsi_match(track["mmsi"])

            is_training = track.mmsi.iloc[0] in training_mmsi
            if is_training:
                training.append(track)
            is_validation = track.mmsi.iloc[0] in val_mmsi
            if is_validation:
                validation.append(track)
        return training, validation

    def extract_samples(self, tracks: List[pd.DataFrame], sample: int, seed: int = 88):
        """Extract samples from a set of tracks"""
        samples = []
        np.random.seed(seed)
        for track in tracks:
            samples.extend(
                training_data.randomly_sample_track(
                    track,
                    n_samples=sample,
                    sample_len=self.model_args["sample_length"],
                    sample_interval=self.model_args["sample_interval"],
                )
            )
        return samples

    def create_generator(
        self, model: GenericFishingModel, tracks: List[pd.DataFrame], samples: int
    ):
        """Create a generator from tracks"""
        return longline_training_data.DataGenerator(
            tracks,
            model,
            batch_size=self.batch_size,
            samples_per_labeled_track=self.train_samples,
        )

    def fit(
        self, verbose: bool = True, seed: int = 88
    ) -> Tuple[GenericFishingModel, History]:
        """Create a model and train it"""
        model = self.model_factory(**self.model_args)
        np.random.seed(seed)
        training_tracks, valid_tracks = self.extract_tracks()
        validation = self.extract_samples(valid_tracks, self.val_samples)
        training_generator = self.create_generator(
            model, training_tracks, self.train_samples
        )

        np.random.seed(seed + 1)
        self.val_features, self.val_labels, _ = preprocess.preprocess_data(
            validation, model, train=False
        )

        if verbose:
            print("Training Tracks: ", len(training_tracks))
            print("Validation Tracks: ", len(valid_tracks))

        hist = model.fit(
            training_generator,
            steps=self.steps,
            validation_data=(self.val_features, self.val_labels),
        )
        return model, hist

    @classmethod
    def train(
        cls,
        model_factory: Type[GenericFishingModel],
        data: pd.DataFrame,
        fold: Optional[int],
    ) -> TrainerResult:
        """Create a LonglineTrainer and launch training

        Args:
            model_factory: GenericModel to instantiate and train
            data: raw training data packed up as a DataFrame
            fold: folds to train or None to train on all
                available data

        Returns
            TrainerResult
        """
        if fold is None:
            val_mmsi = data.validation_mmsi[0]
            training_mmsi = set(data.mmsi)
        else:
            val_mmsi = data.validation_mmsi[fold]
            training_mmsi = set([x for x in data.mmsi if x not in val_mmsi])
        print()
        print(f"Fold #{fold}")
        print(f"Test MMSI: {val_mmsi}")
        model_args = dict(
            inputs=[
                "x",
                "y",
                "cos_course_degrees",
                "sin_course_degrees",
                "speed_knots",
            ],
            outputs=["other", "setting", "hauling"],
            sample_length=97,
            sample_interval=3,
        )
        trainer = cls(
            model_factory,
            model_args,
            data,
            training_mmsi,
            val_mmsi,
            allow_overlap=fold is None,
        )
        model, history = trainer.fit()
        return TrainerResult(trainer, model, history)
