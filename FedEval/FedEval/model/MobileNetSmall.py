import tensorflow as tf

from ..config.configuration import ConfigurationManager


def make_output(num_classes, output_raw):
    if output_raw:
        return tf.keras.layers.Dense(num_classes, activation=None)
    if num_classes > 1:
        return tf.keras.layers.Dense(num_classes, activation="softmax")
    return tf.keras.layers.Dense(num_classes, activation="sigmoid")

class DepthwiseBlock(tf.keras.layers.Layer):
    def __init__(self, filters, activation, stride=1):
        super().__init__()
        self.dw = tf.keras.layers.DepthwiseConv2D(
            3, strides=stride, padding="same", use_bias=False
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.act1 = tf.keras.layers.Activation(activation)
        self.pw = tf.keras.layers.Conv2D(
            filters, 1, padding="same", use_bias=False
        )
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.act2 = tf.keras.layers.Activation(activation)

    def call(self, inputs, training=None):
        x = self.dw(inputs)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.pw(x)
        x = self.bn2(x, training=training)
        return self.act2(x)


class MobileNetSmall(tf.keras.Model):
    """MobileNet-style network using depthwise separable convolutions."""

    def __init__(self, target_shape, **kwargs):
        super().__init__()
        cfg = ConfigurationManager().model_config
        activation = kwargs.get("activation", cfg.activation)
        num_classes = target_shape[-1]

        self.stem = tf.keras.layers.Conv2D(
            32, 3, strides=2, padding="same", use_bias=False
        )
        self.stem_bn = tf.keras.layers.BatchNormalization()
        self.stem_act = tf.keras.layers.Activation(activation)

        self.block1 = DepthwiseBlock(32, activation)
        self.block2 = DepthwiseBlock(64, activation, 2)
        self.block3 = DepthwiseBlock(64, activation)
        self.block4 = DepthwiseBlock(128, activation, 2)
        self.block5 = DepthwiseBlock(128, activation)

        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.output_layer = make_output(
            num_classes, kwargs.get("output_raw", False)
        )

    def call(self, inputs, training=None, mask=None):
        x = self.stem(inputs)
        x = self.stem_bn(x, training=training)
        x = self.stem_act(x)

        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.block4(x, training=training)
        x = self.block5(x, training=training)

        return self.output_layer(self.pool(x))
