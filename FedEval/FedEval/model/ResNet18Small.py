import tensorflow as tf

from ..config.configuration import ConfigurationManager


def make_output(num_classes, output_raw):
    if output_raw:
        return tf.keras.layers.Dense(num_classes, activation=None)
    if num_classes > 1:
        return tf.keras.layers.Dense(num_classes, activation="softmax")
    return tf.keras.layers.Dense(num_classes, activation="sigmoid")

class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, filters, activation, stride=1):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters, 3, strides=stride, padding="same", use_bias=False
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(
            filters, 3, padding="same", use_bias=False
        )
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.Activation(activation)
        self.projection = None
        self.filters = filters
        self.stride = stride

    def build(self, input_shape):
        if input_shape[-1] != self.filters or self.stride != 1:
            self.projection = tf.keras.Sequential([
                tf.keras.layers.Conv2D(
                    self.filters, 1, strides=self.stride,
                    padding="same", use_bias=False
                ),
                tf.keras.layers.BatchNormalization(),
            ])

    def call(self, inputs, training=None):
        shortcut = inputs

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        if self.projection is not None:
            shortcut = self.projection(shortcut, training=training)

        return self.act(x + shortcut)


class ResNet18Small(tf.keras.Model):
    """ResNet-18-style network adapted for small images."""

    def __init__(self, target_shape, **kwargs):
        super().__init__()
        cfg = ConfigurationManager().model_config
        activation = kwargs.get("activation", cfg.activation)
        num_classes = target_shape[-1]

        self.stem = tf.keras.layers.Conv2D(
            64, 3, padding="same", use_bias=False
        )
        self.stem_bn = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.Activation(activation)

        self.b1 = [BasicBlock(64, activation), BasicBlock(64, activation)]
        self.b2 = [
            BasicBlock(128, activation, 2),
            BasicBlock(128, activation)
        ]
        self.b3 = [
            BasicBlock(256, activation, 2),
            BasicBlock(256, activation)
        ]
        self.b4 = [
            BasicBlock(512, activation, 2),
            BasicBlock(512, activation)
        ]

        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.output_layer = make_output(
            num_classes, kwargs.get("output_raw", False)
        )

    def _run(self, x, blocks, training):
        for block in blocks:
            x = block(x, training=training)
        return x

    def call(self, inputs, training=None, mask=None):
        x = self.stem(inputs)
        x = self.stem_bn(x, training=training)
        x = self.act(x)

        x = self._run(x, self.b1, training)
        x = self._run(x, self.b2, training)
        x = self._run(x, self.b3, training)
        x = self._run(x, self.b4, training)

        return self.output_layer(self.pool(x))
