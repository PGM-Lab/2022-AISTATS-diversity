# Regression models

import tensorflow as tf


def mlp_1LDenseDeterministic(input_shape):
    """Builds a multi-layer perceptron with 1 layer.

    Args:
      input_shape: tf.Tensor.
      num_classes: Number of output classes.

    Returns:
      tf.keras.Model.
    """
    inputs = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs=inputs, outputs=x)
  
def mlp_1LDenseDeterministic_Dropout(input_shape):
    """Builds a multi-layer perceptron with 1 layer.

    Args:
      input_shape: tf.Tensor.
      num_classes: Number of output classes.

    Returns:
      tf.keras.Model.
    """
    inputs = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


def mlp_2LDenseDeterministic(input_shape, depth):
    """Builds a multi-layer perceptron with 2 layers.

    Args:
      input_shape: tf.Tensor.
      depth: Width of the intermediate layer.
    Returns:
      tf.keras.Model.
    """
    inputs = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Flatten()(inputs)

    x = tf.keras.layers.Dense(depth)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.Dense(1)(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
