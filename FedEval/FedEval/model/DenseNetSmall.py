import tensorflow as tf

from ..config.configuration import ConfigurationManager


def make_output(num_classes, output_raw):
    if output_raw:
        return tf.keras.layers.Dense(num_classes, activation=None)
    if num_classes > 1:
        return tf.keras.layers.Dense(num_classes, activation="softmax")
    return tf.keras.layers.Dense(num_classes, activation="sigmoid")

class DenseLayer(tf.keras.layers.Layer):
    def __init__(self, growth_rate, activation):
        super().__init__()
        self.bn = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.Activation(activation)
        self.conv = tf.keras.layers.Conv2D(
            growth_rate, 3, padding="same", use_bias=False
        )

    def call(self, inputs, training=None):
        x = self.bn(inputs, training=training)
        x = self.act(x)
        x = self.conv(x)
        return tf.concat([inputs, x], axis=-1)


class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, layers, growth_rate, activation):
        super().__init__()
        self.layers_ = [
            DenseLayer(growth_rate, activation) for _ in range(layers)
        ]

    def call(self, inputs, training=None):
        x = inputs
        for layer in self.layers_:
            x = layer(x, training=training)
        return x


class DenseNetSmall(tf.keras.Model):
    """Small DenseNet-style architecture."""

    def __init__(self, target_shape, **kwargs):
        super().__init__()
        cfg = ConfigurationManager().model_config
        activation = kwargs.get("activation", cfg.activation)
        num_classes = target_shape[-1]

        self.stem = tf.keras.layers.Conv2D(
            32, 3, padding="same", use_bias=False
        )
        self.block1 = DenseBlock(3, 16, activation)
        self.transition1 = tf.keras.layers.Conv2D(
            64, 1, padding="same"
        )
        self.pool1 = tf.keras.layers.AveragePooling2D(2)

        self.block2 = DenseBlock(3, 16, activation)
        self.transition2 = tf.keras.layers.Conv2D(
            128, 1, padding="same"
        )
        self.pool2 = tf.keras.layers.AveragePooling2D(2)

        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.output_layer = make_output(
            num_classes, kwargs.get("output_raw", False)
        )

    def call(self, inputs, training=None, mask=None):
        x = self.stem(inputs)
        x = self.block1(x, training=training)
        x = self.transition1(x)
        x = self.pool1(x)

        x = self.block2(x, training=training)
        x = self.transition2(x)
        x = self.pool2(x)

        x = self.global_pool(x)
        return self.output_layer(x)
