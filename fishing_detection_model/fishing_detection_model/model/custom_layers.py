import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as kl
from tensorflow.keras.activations import gelu
from tensorflow.keras.utils import register_keras_serializable

import numpy as np
from tensorflow_addons.layers import StochasticDepth


@register_keras_serializable()
class DropBlock1D(kl.Layer):
    """DropBlock Regularization layer

    See: https://arxiv.org/pdf/1810.12890.pdf

    Originally from https://github.com/CyberZHG/keras-drop-block,
    but modified to support our use cases.

    Args:
        block_size: size for each mask block.
        keep_prob: probability of keeping the original features in a block.
        **kwargs (optional): passed onto kl.Layer
    """

    def __init__(self, block_size: int, keep_prob: float, **kwargs):
        super().__init__(**kwargs)
        self.block_size = block_size
        self.keep_prob = keep_prob

        self.supports_masking = True

    def get_config(self):
        config = super().get_config().copy()
        keep_prob = (
            self.keep_prob
            if isinstance(self.keep_prob, (int, float))
            else self.keep_prob.numpy()
        )
        config.update(
            {
                "block_size": self.block_size,
                "keep_prob": float(keep_prob),
            }
        )
        return config

    def _get_gamma(self, shape):
        """Get the number of activation units to drop"""
        width = K.cast(shape[1], K.floatx())
        block_size = K.constant(self.block_size, dtype=K.floatx())
        return ((1.0 - self.keep_prob) / block_size) * (
            width / (width - block_size + 1.0)
        )

    def _compute_drop_mask(self, shape):
        mask = K.random_bernoulli(shape, p=self._get_gamma(shape))
        half_block_size = self.block_size // 2
        mask = mask[:, half_block_size:-half_block_size]
        mask = K.temporal_padding(
            mask,
            ((half_block_size, half_block_size)),
        )
        mask = kl.MaxPool1D(
            pool_size=(self.block_size),
            padding="same",
            strides=1,
            data_format="channels_last",
        )(mask)
        return 1.0 - mask

    def call(self, inputs, training=None):
        def dropped_inputs():
            outputs = inputs
            shape = K.shape(outputs)
            mask = self._compute_drop_mask(shape)
            outputs = outputs * mask / (K.mean(mask, axis=(1,), keepdims=True) + 1e-9)
            return outputs

        return K.in_train_phase(dropped_inputs, inputs, training=training)


@register_keras_serializable()
class SinusoidalEmbedding(kl.Layer):
    """Create a sinusoidal embedding suitable for a transformer.

    See https://arxiv.org/abs/1706.03762?context=cs
    """

    def __init__(self, embed_size, **kwargs):
        super().__init__(**kwargs)
        assert embed_size % 2 == 0, "embedding size should be divisible by 2"
        self.embed_size = embed_size

    def build(self, input_shape):
        omegas = []
        thetas = []
        for i in range(self.embed_size):
            k = i // 2
            omegas.append(1 / 10000 ** (2 * k / self.embed_size))
            thetas.append((i % 2) * np.pi / 2)
        self.omegas = tf.constant(
            omegas, shape=(1, 1, self.embed_size), dtype="float32"
        )
        self.thetas = tf.constant(
            thetas, shape=(1, 1, self.embed_size), dtype="float32"
        )

    def call(self, inputs):
        return tf.cos(self.omegas * tf.expand_dims(inputs, 2) + self.thetas)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"embed_size": self.embed_size})
        return cfg


@register_keras_serializable()
class ConvNeXt(kl.Layer):
    """1D version of ConvNeXt residual layer

    ConvNeXt is described in https://arxiv.org/pdf/2201.03545.pdf

    Args:
        kernel_size (optional)
        scale (optional): increase/reduce inner filters by this factor.
        dilation : (optional): dilate internal DepthwiseConv1D by this factor
        survival_prob : (optional): survival prob for stochastic depth
        **kwargs (optional): passed onto kl.Layer
    """

    def __init__(
        self,
        kernel_size: int = 7,
        scale: int = 4,
        dilation: int = 1,
        survival: float = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.scale = scale
        self.dilation = dilation
        self.survival = survival

    def build(self, input_shape):
        n = input_shape[-1]

        self.conv_lyr0 = kl.DepthwiseConv1D(
            kernel_size=2 * self.dilation + 1,
            padding="same",
        )
        self.conv_lyr1 = kl.DepthwiseConv1D(
            kernel_size=self.kernel_size, padding="same", dilation_rate=self.dilation
        )
        self.norm_lyr = kl.BatchNormalization(center=False, scale=False)
        self.conv_lyr2 = kl.Conv1D(self.scale * n, 1, activation=gelu)
        self.conv_lyr3 = kl.Conv1D(n, 1)
        self.add_lyr = StochasticDepth(self.survival)

    def call(self, x):
        skip = x
        x = self.conv_lyr0(x)
        x = self.conv_lyr1(x)
        x = self.norm_lyr(x)
        x = self.conv_lyr2(x)
        x = self.conv_lyr3(x)
        return self.add_lyr([skip, x])

    def get_config(self):
        cfg = super().get_config()
        cfg["kernel_size"] = self.kernel_size
        cfg["scale"] = self.scale
        cfg["dilation"] = self.dilation
        cfg["survival"] = self.survival
        return cfg


custom_layers = {
    x.__name__: x
    for x in [
        ConvNeXt,
        DropBlock1D,
        SinusoidalEmbedding,
    ]
}
