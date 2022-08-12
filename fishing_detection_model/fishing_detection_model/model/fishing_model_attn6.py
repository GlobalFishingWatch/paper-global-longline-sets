"""First cut at implementing a new Advanced Fishing Model

See generic_fishing_model.GenericFishingModel for docs
"""
from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import callbacks
from tensorflow.keras import layers as kl
from tensorflow.keras.activations import gelu
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.optimizers import Adam

from .cosine_annealing import CosineAnnealingScheduler
from .custom_layers import SinusoidalEmbedding
from .generic_fishing_model import GenericFishingModel

NumpyData = Tuple[np.ndarray, np.ndarray]
TrainingData = Union[tf.keras.utils.Sequence, NumpyData]


def add_transformer_block(
    y: kl.Layer,
    n_heads: int,
    key_dim: int,
    n_filters: int,
    dropout: float,
):
    """Add a transformer block to y

    Args:
        y: input layer
        n_heads, key_dim, n_filters: values passed to kl.MultiHeadedAttention
        dropout: used both for kl.MultiHeadedAttention and before Dense layer

    Returns
        new kl.Layer
    """
    skip = y
    y = gelu(kl.BatchNormalization()(y))
    y = kl.MultiHeadAttention(
        num_heads=n_heads,
        key_dim=key_dim,
        value_dim=n_filters,
        dropout=dropout,
        output_shape=n_filters,
    )(y, y)
    y = skip = y + skip
    y = gelu(kl.BatchNormalization()(y))
    y = kl.Dense(4 * n_filters, use_bias=False)(y)
    y = gelu(kl.BatchNormalization()(y))
    y = kl.Dropout(dropout)(y)
    y = kl.Dense(n_filters, use_bias=None)(y)
    return skip + y


def add_embedding(y, positions, embed_size):
    """Concatenate embedding based of embed_size based on positions to y"""
    embed = SinusoidalEmbedding(embed_size)(positions)
    return kl.Concatenate()([y, embed])


class FishingModelAttn(GenericFishingModel):
    """Create an attention (transformer) based fishing model

    See GenericFishingModel for how to instantiate.

    Attributes:
        dropout: amount of dropout applied to final Dense layers
        xy_noise: amount of spatial noise to apply
        speed_noise: amount of noise to apply to speed feature
        pos_noise: amount of noise to apply to position feature
            (where we are along the track)
        angle_noise: amount of noise to apply to heading feature
        n_blocks: number of transformer blocks
        n_filters: number of filters per transformer
        key_dim: transformer key size
        internal_droupout: amount of dropout applied in attention blocks
    """

    dropout: float = 0.5
    xy_noise: float = 0.01
    speed_noise: float = 0.01
    pos_noise: float = 0.01
    angle_noise: float = 0.1

    n_blocks: int = 8
    n_filters: int = 128
    key_dim: int = 32
    internal_dropout: float = 0.1

    def apply_noise(self, y):
        """Add position to y and apply noise to features"""
        tn = y.shape[1]

        pos = (
            tf.reshape(
                tf.range(start=0, limit=tn, delta=1, dtype="float32"),
                tf.shape(y[:1, :, :1]),
            )
            + 0 * y[:, :, :1]
        )

        dpos_noise = tf.random.normal(tf.shape(y[:, :, :1]), stddev=self.pos_noise)
        pos_noise = tf.math.cumsum(dpos_noise, axis=1)
        pos_noise -= pos_noise[:, tn // 2 : tn // 2 + 1, :]

        theta = tf.random.uniform(tf.shape(y[:, :1, :1]), 0, 2 * 3.14159)

        speed_noise = tf.random.normal(tf.shape(y[:, :, :1]), stddev=self.speed_noise)

        dx_noise = tf.random.normal(tf.shape(y[:, :, :1]), stddev=self.xy_noise)
        x_noise = tf.math.cumsum(dx_noise, axis=1)
        x_noise -= x_noise[:, tn // 2 : tn // 2 + 1, :]

        dy_noise = tf.random.normal(tf.shape(y[:, :, :1]), stddev=self.xy_noise)
        y_noise = tf.math.cumsum(dy_noise, axis=1)
        y_noise -= y_noise[:, tn // 2 : tn // 2 + 1, :]

        h_noise = tf.random.normal(tf.shape(y[:, :, :1]), stddev=self.angle_noise)
        h = tf.math.atan2(y[:, :, 3:4], y[:, :, 2:3]) + theta

        y0_noise = kl.Concatenate()(
            [
                pos + pos_noise,
                y[:, :, 0:1] * tf.cos(theta) - y[:, :, 1:2] * tf.sin(theta) + x_noise,
                y[:, :, 0:1] * tf.sin(theta) + y[:, :, 1:2] * tf.cos(theta) + y_noise,
                tf.cos(h + h_noise),
                tf.sin(h + h_noise),
                y[:, :, 4:5] + speed_noise,
            ]
        )
        y0 = kl.Concatenate()([pos, y[:, :, 0:2], y[:, :, 2:4], y[:, :, 4:5]])

        return K.in_test_phase(y0, y0_noise)

    def build_keras_model(self):
        n_inputs = len(self.metadata["inputs"])
        sample_length = self.metadata["sample_length"]
        n_filters = self.n_filters
        key_dim = self.key_dim

        input_layer = kl.Input(shape=(sample_length, n_inputs))

        y0 = self.apply_noise(input_layer)
        dt, dx, dy, cc, sc, y = (
            y0[:, :, 0],
            y0[:, :, 1],
            y0[:, :, 2],
            y0[:, :, 3],
            y0[:, :, 4],
            y0[:, :, 5:],
        )

        embedder = SinusoidalEmbedding(embed_size=n_filters)
        emb_t = embedder(dt)
        emb_x = embedder(dx)
        emb_y = embedder(dy)
        emb_cc = embedder(cc)
        emb_sc = embedder(sc)

        y = kl.Concatenate()([y0, emb_t, emb_x, emb_y, emb_cc, emb_sc])
        y = kl.Conv1D(n_filters, 1)(y)

        for i in range(self.n_blocks):
            y = add_transformer_block(
                y,
                n_heads=n_filters // key_dim,
                key_dim=key_dim,
                n_filters=n_filters,
                dropout=self.internal_dropout,
            )

        y = gelu(kl.BatchNormalization()(y))
        y = kl.Flatten()(y)
        y = kl.Dropout(self.dropout)(y)
        y = kl.Dense(1024, activation=gelu)(y)
        y = kl.Dropout(self.dropout)(y)
        output = kl.Dense(3, activation="softmax")(y)

        model = KerasModel(inputs=input_layer, outputs=output)
        loss = CategoricalCrossentropy(label_smoothing=0.0)
        model.compile(
            loss=loss, optimizer=Adam(learning_rate=0.001), metrics=["accuracy"]
        )
        return model

    def set_norm(self, features):
        """Sets the norm: this model sets the norm to not apply normalization"""
        _, _, n = np.shape(features)
        self.metadata["feature_means"] = np.zeros([1, 1, n]).tolist()
        self.metadata["features_stds"] = np.ones([1, 1, n]).tolist()

    def fit(
        self,
        features: TrainingData,
        weights: Optional[np.ndarray] = None,
        validation_data: NumpyData = None,
        steps: int = 100,
        batch_size: int = 16,
    ):
        """Train the model

        Args:
            features: either a data generator or a tuple of (features, labels)
            weights (optional): weights per sample
            validation_data (optional): tuple of (features, labels) used to
                compute validation metrics during training
            steps (optional): number of epochs to train for after warmup
            batch_size (optional)
        """
        if isinstance(features, tf.keras.utils.Sequence):
            _, _, n = features.means.shape
            self.metadata["feature_means"] = np.zeros([1, 1, n]).tolist()
            self.metadata["features_stds"] = np.ones([1, 1, n]).tolist()
            labels = None
        else:
            # Otherwise we expect a tuple of arrays.
            features, labels = features
            self.set_norm(features)
            features = self.apply_norm(features)
            labels = tf.one_hot(np.asarray(labels), 3)
        if validation_data:
            validation_data = (
                self.apply_norm(validation_data[0]),
                tf.one_hot(np.asarray(validation_data[1]), 3),
            )
        scheduler = CosineAnnealingScheduler(
            warmup=10, n0=steps, min_lr=0, max_lr=0.0002, length_scale=1
        )
        n_epochs = scheduler.epochs_for_cycles(1)
        callback = callbacks.LearningRateScheduler(scheduler)
        return self.keras_model.fit(
            features,
            labels,
            sample_weight=weights,
            epochs=n_epochs,
            callbacks=[callback],
            validation_data=validation_data,
            verbose=2,
        )
