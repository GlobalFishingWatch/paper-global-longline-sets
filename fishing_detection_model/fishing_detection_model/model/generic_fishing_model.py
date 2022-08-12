"""Generic model for predicting fishing behavior

"""
import json
from typing import List, Optional, Tuple, Union

import tensorflow as tf
from tensorflow.python.keras.saving import hdf5_format

import h5py
import numpy as np

from . import preprocess_features
from .custom_layers import custom_layers

NumpyData = Tuple[np.ndarray, np.ndarray]
TrainingData = Union[tf.keras.utils.Sequence, NumpyData]


class GenericFishingModel:
    """A fishing model metadata that can be loaded and saved.

    This must be subclassed to create a trainable model, but
    the generic model is sufficient to load a model that was
    trained using a subclass.

    Args:
        inputs: the input names are used to determine the number of
            channels in the input layer and are used by the preprocessor
            to determine which features to extract.
        outputs: the output names determine the number of output channels
            and document which channel corresponds to which behavior.
        sample_length: the number of time-points at the input to the net.
        sample_interval : how frequently to sample the underlying data.
            For instance if the base data is samples every 5-minutes, setting
            this to 6 would result in 30 minute resolution data being passed
            to the net.

    Note: The following terminology is used in method docstring
        - `n_inputs`: equivalent to `len(inputs)`
        - `n_outputs`: equivalent to `len(outputs)`
    """

    output_length = 1

    def __init__(
        self,
        inputs: List[str],
        outputs: List[str],
        sample_length: int,
        sample_interval: int,
    ):
        self.metadata = dict(
            inputs=inputs,
            outputs=outputs,
            sample_length=sample_length,
            sample_interval=sample_interval,
            output_length=self.output_length or sample_length,
        )
        self.preprocessor = self.create_preprocessor()
        self.keras_model = self.build_keras_model()

    def build_keras_model(self):
        """Build, compile and return the underlying keras model

        Returns:
            keras.Model
        """
        raise NotImplementedError()

    def fit(
        self,
        features: TrainingData,
        weights: Optional[np.ndarray] = None,
        validation_data: NumpyData = None,
        steps: int = 100,
        batch_size: int = 16,
    ):
        """Train the underlying Keras Model.

        Args:
            features: either a data generator or a tuple of (features, labels)
            weights (optional): weights per sample
            validation_data (optional): tuple of (features, labels) used to
                compute validation metrics during training
            steps (optional): number of epochs to train for after warmup
            batch_size (optional)

        Returns:
            Keras History object
        """
        raise NotImplementedError()

    def set_norm(self, features: np.ndarray):
        """Set the mean and stddev of the features for normalizing inputs

        This should be called only once, at the beginning of `fit`.
        """
        features = np.asarray(features)
        self.metadata["feature_means"] = features.mean(
            axis=(0, 1), keepdims=True
        ).tolist()
        self.metadata["features_stds"] = features.std(
            axis=(0, 1), keepdims=True
        ).tolist()

    def apply_norm(self, features: np.ndarray):
        """Normalize features to zero mean, unit variance

        Args:
            features: samples × sample_length × n_inputs array of float

        Returns:
            samples × sample_length × n_inputs array of float
                The array has ZMUV for each feature (input)
        """
        features = np.asarray(features)
        means = self.metadata["feature_means"]
        stds = self.metadata["features_stds"]
        return (features - means) / stds

    def evaluate(self, features: np.ndarray, labels: np.ndarray):
        """Run the underlying keras model's evaluate on normed data

        Args:
            features: samples × sample_length × n_inputs array of float
            labels: length `samples` array of int

        Returns:
            float or list of float
                List is returned if the model defines more than one metric.
        """
        features = self.apply_norm(features)
        labels = np.asarray(labels)
        return self.keras_model.evaluate(features, labels, verbose=0)

    def predict(self, features):
        """Predict using the underlying Keras model after applying norm

        Args:
            features: samples × sample_length × n_inputs array of float

        Returns:
            samples × n_outputs array of float
        """
        features = self.apply_norm(features)
        return self.keras_model.predict(features)

    def save_model(self, path):
        """Save the model, including metadata, to an HDF5 file

        Args:
            path
        """
        if self.keras_model is None:
            raise ValueError("model not initialized")
        with h5py.File(path, mode="w") as f:
            hdf5_format.save_model_to_hdf5(self.keras_model, f)
            for k, v in self.metadata.items():
                f.attrs[f"metadata_{k}"] = json.dumps(v)

    @classmethod
    def load_model(cls, path):
        """Load a model, including metadata, from an HDF5 file

        Args:
            path
        """
        self = cls.__new__(cls)
        with h5py.File(path, mode="r") as f:
            # Default to output_length of 1 for backward compatibility,
            # but allow override.
            # TODO: should figure out how to deprecate this.
            self.metadata = {"output_length": 1}
            for k, v in f.attrs.items():
                if k.startswith("metadata_"):
                    _, k = k.split("_", 1)
                    self.metadata[k] = json.loads(v)
            self.keras_model = tf.keras.models.load_model(
                f, custom_objects=custom_layers
            )
        self.preprocessor = self.create_preprocessor()
        return self

    def create_preprocessor(self):
        """Create a Preprocessor based on metadata

        The preprocessor is in charge of turning records,
        such as those our feature table, into features
        specific to the model

        Returns:
            preprocess_features.Preprocessor
        """
        output_length = self.metadata["output_length"]
        offset = (self.metadata["sample_length"] - output_length) // 2
        raw_ndx = [i + offset for i in range(output_length)]
        return preprocess_features.Preprocessor(
            self.metadata["inputs"],
            self.metadata["sample_length"],
            self.metadata["sample_interval"],
            raw_ndx=raw_ndx,
        )

    def predict_from_records(self, records, chunksize=1024):
        """Predict from a sequence of records

        This is the main entry point for predicting from within
        Beam.

        Note that the output is shorter than the input
        because the model has a certain amount leading and
        trailing data that it need to make a prediction.

        Args:
            records : sequence of dict
                The dictionaries map raw feature names to values
            chunksize : int, optional
                The number of predictions to make at once. Too small
                a value results in large slowdowns. You are unlikely
                to need to change this value.

        Returns
        -------
        records : list of dict
            These are the records colocated with the predictions. This
            allows values such as `timestamp`, 'lon', and 'lat' to be
            extracted for use with the predictions.
        predictions : len(records) × n_outputs array of float
            The predicted scores for each output class.
        """
        assert chunksize >= 1
        raw = []
        predictions = []
        features = []
        for r, f in self.preprocessor.feature_sets(records):
            raw.extend(r)
            features.append(f)
            if len(features) == chunksize:
                predictions.extend(self.predict(features))
                features = []
        if features:
            predictions.extend(self.predict(features))
        predictions = np.asarray(predictions)
        n_outputs = len(self.metadata["outputs"])
        predictions = predictions.reshape(-1, n_outputs)
        # TODO: use dict to merge predictions
        mask = []
        seen = set()
        for x in raw:
            mask.append(x["timestamp"] not in seen)
            seen.add(x["timestamp"])
        raw = [x for (x, m) in zip(raw, mask) if m]
        predictions = np.array([x for (x, m) in zip(predictions, mask) if m])
        return raw, predictions
