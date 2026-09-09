import tensorflow as tf

from ..config.configuration import ConfigurationManager


def make_output(num_classes, output_raw):
    if output_raw:
        return tf.keras.layers.Dense(num_classes, activation=None)
    if num_classes > 1:
        return tf.keras.layers.Dense(num_classes, activation="softmax")
    return tf.keras.layers.Dense(num_classes, activation="sigmoid")

class VGGSmall(tf.keras.Model):
    """VGG-style CNN with repeated 3x3 convolutions."""

    def __init__(self, target_shape, **kwargs):
        super().__init__()
        cfg = ConfigurationManager().model_config
        activation = kwargs.get("activation", cfg.activation)
        dropout = float(kwargs.get("dropout", cfg.dropout))
        num_classes = target_shape[-1]

        self.features = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, padding="same", activation=activation),
            tf.keras.layers.Conv2D(32, 3, padding="same", activation=activation),
            tf.keras.layers.MaxPooling2D(2),

            tf.keras.layers.Conv2D(64, 3, padding="same", activation=activation),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation=activation),
            tf.keras.layers.MaxPooling2D(2),

            tf.keras.layers.Conv2D(128, 3, padding="same", activation=activation),
            tf.keras.layers.Conv2D(128, 3, padding="same", activation=activation),
            tf.keras.layers.MaxPooling2D(2),
        ])

        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation=activation),
            tf.keras.layers.Dropout(dropout),
            make_output(num_classes, kwargs.get("output_raw", False)),
        ])

    def call(self, inputs, training=None, mask=None):
        x = self.features(inputs, training=training)
        return self.classifier(x, training=training)
