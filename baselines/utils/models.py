# coding=utf-8
# Copyright 2020 The Uncertainty Baselines Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wide ResNet with variational Bayesian layers."""
import functools
import warnings
import numpy as np
import tensorflow as tf
from baselines.utils import trainableNormalDuplicate

try:
  import edward2 as ed  # pylint: disable=g-import-not-at-top
except ImportError as e:
  warnings.warn(f'Skipped due to ImportError: {e}')

BatchNormalization = functools.partial(  # pylint: disable=invalid-name
    tf.keras.layers.BatchNormalization,
    epsilon=1e-5,  # using epsilon and momentum defaults from Torch
    momentum=0.9)

Conv2DFlipout = functools.partial(  # pylint: disable=invalid-name
    ed.layers.Conv2DFlipout,
    kernel_size=3,
    padding='same',
    use_bias=False)


def mlp_1LDenseEnsemble(input_shape,
                            depth,
                            width_multiplier,
                            num_classes,
                            prior_stddev,
                            dataset_size,
                            stddev_init):
  """Builds Wide ResNet.

  Following Zagoruyko and Komodakis (2016), it accepts a width multiplier on the
  number of filters. Using three groups of residual blocks, the network maps
  spatial features of size 32x32 -> 16x16 -> 8x8.

  Args:
    input_shape: tf.Tensor.
    depth: Total number of convolutional layers. "n" in WRN-n-k. It differs from
      He et al. (2015)'s notation which uses the maximum depth of the network
      counting non-conv layers like dense.
    width_multiplier: Integer to multiply the number of typical filters by. "k"
      in WRN-n-k.
    num_classes: Number of output classes.
    prior_stddev: Fixed standard deviation for weight prior.
    dataset_size: Dataset size to properly scale the KL.
    stddev_init: float to initialize variational posterior stddev parameters.

  Returns:
    tf.keras.Model.
  """
  if (depth - 4) % 6 != 0:
    raise ValueError('depth should be 6n+4 (e.g., 16, 22, 28, 40).')
  inputs = tf.keras.layers.Input(shape=input_shape)

  x = tf.keras.layers.Flatten()(inputs)
  x2 = tf.keras.layers.Flatten()(inputs)

  x = ed.layers.DenseFlipout(
      num_classes,
      use_bias=True,
      kernel_initializer=ed.initializers.TrainableHeNormal(
      stddev_initializer=tf.keras.initializers.TruncatedNormal(
          mean=np.log(np.expm1(stddev_init)), stddev=0.1)),
      kernel_regularizer=ed.regularizers.NormalKLDivergenceWithTiedMean(
          stddev=prior_stddev, scale_factor=1./dataset_size))(x)

  x2 = ed.layers.DenseFlipout(
      num_classes,
      use_bias=True,
      kernel_initializer=ed.initializers.TrainableHeNormal(
      stddev_initializer=tf.keras.initializers.TruncatedNormal(
          mean=np.log(np.expm1(stddev_init)), stddev=0.1)),
      kernel_regularizer=ed.regularizers.NormalKLDivergenceWithTiedMean(
          stddev=prior_stddev, scale_factor=1./dataset_size))(x2)

  x_stack = tf.stack([x, x2])

  return tf.keras.Model(inputs=inputs, outputs=x_stack)




def mlp_1LDense(input_shape,
                            depth,
                            width_multiplier,
                            num_classes,
                            prior_stddev,
                            dataset_size,
                            stddev_init):
  """Builds Wide ResNet.

  Following Zagoruyko and Komodakis (2016), it accepts a width multiplier on the
  number of filters. Using three groups of residual blocks, the network maps
  spatial features of size 32x32 -> 16x16 -> 8x8.

  Args:
    input_shape: tf.Tensor.
    depth: Total number of convolutional layers. "n" in WRN-n-k. It differs from
      He et al. (2015)'s notation which uses the maximum depth of the network
      counting non-conv layers like dense.
    width_multiplier: Integer to multiply the number of typical filters by. "k"
      in WRN-n-k.
    num_classes: Number of output classes.
    prior_stddev: Fixed standard deviation for weight prior.
    dataset_size: Dataset size to properly scale the KL.
    stddev_init: float to initialize variational posterior stddev parameters.

  Returns:
    tf.keras.Model.
  """
  if (depth - 4) % 6 != 0:
    raise ValueError('depth should be 6n+4 (e.g., 16, 22, 28, 40).')
  inputs = tf.keras.layers.Input(shape=input_shape)

  x = tf.keras.layers.Flatten()(inputs)
  x2 = tf.keras.layers.Flatten()(inputs)

  initializer = ed.initializers.TrainableHeNormal(
      stddev_initializer=tf.keras.initializers.TruncatedNormal(
          mean=np.log(np.expm1(stddev_init)), stddev=0.1))

  x = ed.layers.DenseFlipout(
      num_classes,
      use_bias=True,
      kernel_initializer=initializer,
      kernel_regularizer=ed.regularizers.NormalKLDivergenceWithTiedMean(
          stddev=prior_stddev, scale_factor=1./dataset_size))(x)

  x2 = ed.layers.DenseFlipout(
      num_classes,
      use_bias=True,
      kernel_initializer=trainableNormalDuplicate.TrainableNormalDuplicate(initializer),
      kernel_regularizer=None)(x2)

  x_stack = tf.stack([x, x2])

  return tf.keras.Model(inputs=inputs, outputs=x_stack)


def mlp_1LDenseSingle(input_shape,
                            depth,
                            width_multiplier,
                            num_classes,
                            prior_stddev,
                            dataset_size,
                            stddev_init):
  """Builds Wide ResNet.

  Following Zagoruyko and Komodakis (2016), it accepts a width multiplier on the
  number of filters. Using three groups of residual blocks, the network maps
  spatial features of size 32x32 -> 16x16 -> 8x8.

  Args:
    input_shape: tf.Tensor.
    depth: Total number of convolutional layers. "n" in WRN-n-k. It differs from
      He et al. (2015)'s notation which uses the maximum depth of the network
      counting non-conv layers like dense.
    width_multiplier: Integer to multiply the number of typical filters by. "k"
      in WRN-n-k.
    num_classes: Number of output classes.
    prior_stddev: Fixed standard deviation for weight prior.
    dataset_size: Dataset size to properly scale the KL.
    stddev_init: float to initialize variational posterior stddev parameters.

  Returns:
    tf.keras.Model.
  """
  if (depth - 4) % 6 != 0:
    raise ValueError('depth should be 6n+4 (e.g., 16, 22, 28, 40).')
  inputs = tf.keras.layers.Input(shape=input_shape)

  x = tf.keras.layers.Flatten()(inputs)


  x = ed.layers.DenseFlipout(
      num_classes,
      kernel_initializer=ed.initializers.TrainableHeNormal(
      stddev_initializer=tf.keras.initializers.TruncatedNormal(
          mean=np.log(np.expm1(stddev_init)), stddev=0.1)),
      kernel_regularizer=ed.regularizers.NormalKLDivergenceWithTiedMean(
          stddev=prior_stddev, scale_factor=1./dataset_size))(x)

  return tf.keras.Model(inputs=inputs, outputs=x)


def mlp_1LDenseDeterministic(input_shape,
                            num_classes):
  """Builds Wide ResNet.

  Following Zagoruyko and Komodakis (2016), it accepts a width multiplier on the
  number of filters. Using three groups of residual blocks, the network maps
  spatial features of size 32x32 -> 16x16 -> 8x8.

  Args:
    input_shape: tf.Tensor.
    depth: Total number of convolutional layers. "n" in WRN-n-k. It differs from
      He et al. (2015)'s notation which uses the maximum depth of the network
      counting non-conv layers like dense.
    width_multiplier: Integer to multiply the number of typical filters by. "k"
      in WRN-n-k.
    num_classes: Number of output classes.
    prior_stddev: Fixed standard deviation for weight prior.
    dataset_size: Dataset size to properly scale the KL.
    stddev_init: float to initialize variational posterior stddev parameters.

  Returns:
    tf.keras.Model.
  """
  inputs = tf.keras.layers.Input(shape=input_shape)
  x = tf.keras.layers.Flatten()(inputs)
  x = tf.keras.layers.Dense(num_classes)(x)
  return tf.keras.Model(inputs=inputs, outputs=x)

def mlp_1LDenseDeterministic_Dropout(input_shape,
                            num_classes):
  """Builds Wide ResNet.

  Following Zagoruyko and Komodakis (2016), it accepts a width multiplier on the
  number of filters. Using three groups of residual blocks, the network maps
  spatial features of size 32x32 -> 16x16 -> 8x8.

  Args:
    input_shape: tf.Tensor.
    depth: Total number of convolutional layers. "n" in WRN-n-k. It differs from
      He et al. (2015)'s notation which uses the maximum depth of the network
      counting non-conv layers like dense.
    width_multiplier: Integer to multiply the number of typical filters by. "k"
      in WRN-n-k.
    num_classes: Number of output classes.
    prior_stddev: Fixed standard deviation for weight prior.
    dataset_size: Dataset size to properly scale the KL.
    stddev_init: float to initialize variational posterior stddev parameters.

  Returns:
    tf.keras.Model.
  """
  inputs = tf.keras.layers.Input(shape=input_shape)
  x = tf.keras.layers.Flatten()(inputs)
  x = tf.keras.layers.Dropout(0.1)(x)
  x = tf.keras.layers.Dense(num_classes)(x)
  return tf.keras.Model(inputs=inputs, outputs=x)


def mlp_1LDenseBatchEnsemble(input_shape,
                              num_classes,
                              ensemble_size,
                              random_sign_init,
                              l2):
    """Builds LeNet5."""
    inputs = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Flatten()(inputs)
    x = ed.layers.DenseBatchEnsemble(
        num_classes,
        alpha_initializer=make_sign_initializer(random_sign_init),
        gamma_initializer=make_sign_initializer(random_sign_init),
        activation=None,
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(l2),
        bias_regularizer=tf.keras.regularizers.l2(l2),
        ensemble_size=ensemble_size)(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

def mlp_2LDense(input_shape,
                            depth,
                            width_multiplier,
                            num_classes,
                            prior_stddev,
                            dataset_size,
                            stddev_init):
  """Builds Wide ResNet.

  Following Zagoruyko and Komodakis (2016), it accepts a width multiplier on the
  number of filters. Using three groups of residual blocks, the network maps
  spatial features of size 32x32 -> 16x16 -> 8x8.

  Args:
    input_shape: tf.Tensor.
    depth: Total number of convolutional layers. "n" in WRN-n-k. It differs from
      He et al. (2015)'s notation which uses the maximum depth of the network
      counting non-conv layers like dense.
    width_multiplier: Integer to multiply the number of typical filters by. "k"
      in WRN-n-k.
    num_classes: Number of output classes.
    prior_stddev: Fixed standard deviation for weight prior.
    dataset_size: Dataset size to properly scale the KL.
    stddev_init: float to initialize variational posterior stddev parameters.

  Returns:
    tf.keras.Model.
  """
  if (depth - 4) % 6 != 0:
    raise ValueError('depth should be 6n+4 (e.g., 16, 22, 28, 40).')
  inputs = tf.keras.layers.Input(shape=input_shape)

  x = tf.keras.layers.Flatten()(inputs)
  x2 = tf.keras.layers.Flatten()(inputs)

  initializer = ed.initializers.TrainableHeNormal(
      stddev_initializer=tf.keras.initializers.TruncatedNormal(
          mean=np.log(np.expm1(stddev_init)), stddev=0.1))


  x = ed.layers.DenseFlipout(
      50,
      kernel_initializer=initializer,
      kernel_regularizer=ed.regularizers.NormalKLDivergenceWithTiedMean(
          stddev=prior_stddev, scale_factor=1./dataset_size))(x)
  x2 = ed.layers.DenseFlipout(
      50,
      kernel_initializer=trainableNormalDuplicate.TrainableNormalDuplicate(initializer),
      kernel_regularizer=None)(x2)

  x = tf.keras.layers.Activation('relu')(x)
  x2 = tf.keras.layers.Activation('relu')(x2)

  initializer = ed.initializers.TrainableHeNormal(
          stddev_initializer=tf.keras.initializers.TruncatedNormal(
              mean=np.log(np.expm1(stddev_init)), stddev=0.1))

  x = ed.layers.DenseFlipout(
      num_classes,
      kernel_initializer=initializer,
      kernel_regularizer=ed.regularizers.NormalKLDivergenceWithTiedMean(
          stddev=prior_stddev, scale_factor=1./dataset_size))(x)

  x2 = ed.layers.DenseFlipout(
      num_classes,
      kernel_initializer=trainableNormalDuplicate.TrainableNormalDuplicate(initializer),
      kernel_regularizer=None)(x2)

  x_stack = tf.stack([x, x2])

  return tf.keras.Model(inputs=inputs, outputs=x_stack)


def mlp_2LDenseSingle(input_shape,
                            depth,
                            width_multiplier,
                            num_classes,
                            prior_stddev,
                            dataset_size,
                            stddev_init):
  """Builds Wide ResNet.

  Following Zagoruyko and Komodakis (2016), it accepts a width multiplier on the
  number of filters. Using three groups of residual blocks, the network maps
  spatial features of size 32x32 -> 16x16 -> 8x8.

  Args:
    input_shape: tf.Tensor.
    depth: Total number of convolutional layers. "n" in WRN-n-k. It differs from
      He et al. (2015)'s notation which uses the maximum depth of the network
      counting non-conv layers like dense.
    width_multiplier: Integer to multiply the number of typical filters by. "k"
      in WRN-n-k.
    num_classes: Number of output classes.
    prior_stddev: Fixed standard deviation for weight prior.
    dataset_size: Dataset size to properly scale the KL.
    stddev_init: float to initialize variational posterior stddev parameters.

  Returns:
    tf.keras.Model.
  """
  if (depth - 4) % 6 != 0:
    raise ValueError('depth should be 6n+4 (e.g., 16, 22, 28, 40).')
  inputs = tf.keras.layers.Input(shape=input_shape)

  x = tf.keras.layers.Flatten()(inputs)

  x = ed.layers.DenseFlipout(
      50,
      kernel_initializer=ed.initializers.TrainableHeNormal(
      stddev_initializer=tf.keras.initializers.TruncatedNormal(
          mean=np.log(np.expm1(stddev_init)), stddev=0.1)),
      kernel_regularizer=ed.regularizers.NormalKLDivergenceWithTiedMean(
          stddev=prior_stddev, scale_factor=1./dataset_size))(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = ed.layers.DenseFlipout(
      num_classes,
      kernel_initializer=ed.initializers.TrainableHeNormal(
          stddev_initializer=tf.keras.initializers.TruncatedNormal(
              mean=np.log(np.expm1(stddev_init)), stddev=0.1)),
      kernel_regularizer=ed.regularizers.NormalKLDivergenceWithTiedMean(
          stddev=prior_stddev, scale_factor=1./dataset_size))(x)



  return tf.keras.Model(inputs=inputs, outputs=x)



def mlp_2LDenseDeterministic(input_shape,
                            depth,
                            num_classes):
  """Builds Wide ResNet.

  Following Zagoruyko and Komodakis (2016), it accepts a width multiplier on the
  number of filters. Using three groups of residual blocks, the network maps
  spatial features of size 32x32 -> 16x16 -> 8x8.

  Args:
    input_shape: tf.Tensor.
    depth: Total number of convolutional layers. "n" in WRN-n-k. It differs from
      He et al. (2015)'s notation which uses the maximum depth of the network
      counting non-conv layers like dense.
    width_multiplier: Integer to multiply the number of typical filters by. "k"
      in WRN-n-k.
    num_classes: Number of output classes.
    prior_stddev: Fixed standard deviation for weight prior.
    dataset_size: Dataset size to properly scale the KL.
    stddev_init: float to initialize variational posterior stddev parameters.

  Returns:
    tf.keras.Model.
  """
  inputs = tf.keras.layers.Input(shape=input_shape)

  x = tf.keras.layers.Flatten()(inputs)

  x = tf.keras.layers.Dense(depth)(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.Dense(num_classes)(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

def mlp_2LDenseEnsemble(input_shape,
                            depth,
                            width_multiplier,
                            num_classes,
                            prior_stddev,
                            dataset_size,
                            stddev_init):
  """Builds Wide ResNet.

  Following Zagoruyko and Komodakis (2016), it accepts a width multiplier on the
  number of filters. Using three groups of residual blocks, the network maps
  spatial features of size 32x32 -> 16x16 -> 8x8.

  Args:
    input_shape: tf.Tensor.
    depth: Total number of convolutional layers. "n" in WRN-n-k. It differs from
      He et al. (2015)'s notation which uses the maximum depth of the network
      counting non-conv layers like dense.
    width_multiplier: Integer to multiply the number of typical filters by. "k"
      in WRN-n-k.
    num_classes: Number of output classes.
    prior_stddev: Fixed standard deviation for weight prior.
    dataset_size: Dataset size to properly scale the KL.
    stddev_init: float to initialize variational posterior stddev parameters.

  Returns:
    tf.keras.Model.
  """
  if (depth - 4) % 6 != 0:
    raise ValueError('depth should be 6n+4 (e.g., 16, 22, 28, 40).')
  inputs = tf.keras.layers.Input(shape=input_shape)

  x = tf.keras.layers.Flatten()(inputs)
  x2 = tf.keras.layers.Flatten()(inputs)


  x = ed.layers.DenseFlipout(
      50,
      kernel_initializer=ed.initializers.TrainableHeNormal(
      stddev_initializer=tf.keras.initializers.TruncatedNormal(
          mean=np.log(np.expm1(stddev_init)), stddev=0.1)),
      kernel_regularizer=ed.regularizers.NormalKLDivergenceWithTiedMean(
          stddev=prior_stddev, scale_factor=1./dataset_size))(x)

  x2 = ed.layers.DenseFlipout(
      50,
      kernel_initializer=ed.initializers.TrainableHeNormal(
      stddev_initializer=tf.keras.initializers.TruncatedNormal(
          mean=np.log(np.expm1(stddev_init)), stddev=0.1)),
      kernel_regularizer=ed.regularizers.NormalKLDivergenceWithTiedMean(
          stddev=prior_stddev, scale_factor=1./dataset_size))(x2)

  x = tf.keras.layers.Activation('relu')(x)
  x2 = tf.keras.layers.Activation('relu')(x2)


  x = ed.layers.DenseFlipout(
      num_classes,
      kernel_initializer=ed.initializers.TrainableHeNormal(
      stddev_initializer=tf.keras.initializers.TruncatedNormal(
          mean=np.log(np.expm1(stddev_init)), stddev=0.1)),
      kernel_regularizer=ed.regularizers.NormalKLDivergenceWithTiedMean(
          stddev=prior_stddev, scale_factor=1./dataset_size))(x)

  x2 = ed.layers.DenseFlipout(
      num_classes,
      kernel_initializer=ed.initializers.TrainableHeNormal(
      stddev_initializer=tf.keras.initializers.TruncatedNormal(
          mean=np.log(np.expm1(stddev_init)), stddev=0.1)),
      kernel_regularizer=ed.regularizers.NormalKLDivergenceWithTiedMean(
          stddev=prior_stddev, scale_factor=1./dataset_size))(x2)

  x_stack = tf.stack([x, x2])

  return tf.keras.Model(inputs=inputs, outputs=x_stack)


def mlp_2LDenseBatchEnsemble(input_shape,
                              depth,
                              num_classes,
                              ensemble_size,
                              random_sign_init,
                              l2):
    """Builds LeNet5."""
    inputs = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Flatten()(inputs)
    x = ed.layers.DenseBatchEnsemble(
        depth,
        alpha_initializer=make_sign_initializer(random_sign_init),
        gamma_initializer=make_sign_initializer(random_sign_init),
        activation=tf.nn.relu,
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(l2),
        bias_regularizer=tf.keras.regularizers.l2(l2),
        ensemble_size=ensemble_size)(x)

    x = ed.layers.DenseBatchEnsemble(
        num_classes,
        alpha_initializer=make_sign_initializer(random_sign_init),
        gamma_initializer=make_sign_initializer(random_sign_init),
        activation=None,
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(l2),
        bias_regularizer=tf.keras.regularizers.l2(l2),
        ensemble_size=ensemble_size)(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

def mlp_2LConv(input_shape,
                            depth,
                            width_multiplier,
                            num_classes,
                            prior_stddev,
                            dataset_size,
                            stddev_init):
  """Builds Wide ResNet.

  Following Zagoruyko and Komodakis (2016), it accepts a width multiplier on the
  number of filters. Using three groups of residual blocks, the network maps
  spatial features of size 32x32 -> 16x16 -> 8x8.

  Args:
    input_shape: tf.Tensor.
    depth: Total number of convolutional layers. "n" in WRN-n-k. It differs from
      He et al. (2015)'s notation which uses the maximum depth of the network
      counting non-conv layers like dense.
    width_multiplier: Integer to multiply the number of typical filters by. "k"
      in WRN-n-k.
    num_classes: Number of output classes.
    prior_stddev: Fixed standard deviation for weight prior.
    dataset_size: Dataset size to properly scale the KL.
    stddev_init: float to initialize variational posterior stddev parameters.

  Returns:
    tf.keras.Model.
  """
  if (depth - 4) % 6 != 0:
    raise ValueError('depth should be 6n+4 (e.g., 16, 22, 28, 40).')
  inputs = tf.keras.layers.Input(shape=input_shape)

  initializer = ed.initializers.TrainableHeNormal(
      stddev_initializer=tf.keras.initializers.TruncatedNormal(
          mean=np.log(np.expm1(stddev_init)), stddev=0.1))

  x = Conv2DFlipout(
      16,
      strides=1,
      kernel_initializer=initializer,
      kernel_regularizer=ed.regularizers.NormalKLDivergenceWithTiedMean(
          stddev=prior_stddev, scale_factor=1./dataset_size))(inputs)

  x2 = Conv2DFlipout(
      16,
      strides=1,
      kernel_initializer=trainableNormalDuplicate.TrainableNormalDuplicate(initializer),
      kernel_regularizer=None)(inputs)

  x = BatchNormalization()(x)
  x2 = BatchNormalization()(x2)
  x = tf.keras.layers.Activation('relu')(x)
  x2 = tf.keras.layers.Activation('relu')(x2)
  x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
  x2 = tf.keras.layers.AveragePooling2D(pool_size=8)(x2)

  x = tf.keras.layers.Flatten()(x)
  x2 = tf.keras.layers.Flatten()(x2)

  initializer = ed.initializers.TrainableHeNormal(
      stddev_initializer=tf.keras.initializers.TruncatedNormal(
          mean=np.log(np.expm1(stddev_init)), stddev=0.1))


  x = ed.layers.DenseReparameterization(
      num_classes,
      kernel_initializer=initializer,
      kernel_regularizer=ed.regularizers.NormalKLDivergenceWithTiedMean(
          stddev=prior_stddev, scale_factor=1./dataset_size))(x)
  x2 = ed.layers.DenseReparameterization(
      num_classes,
      kernel_initializer=trainableNormalDuplicate.TrainableNormalDuplicate(initializer),
      kernel_regularizer=None)(x2)

  x_stack = tf.stack([x, x2])

  return tf.keras.Model(inputs=inputs, outputs=x_stack)


def mlp_leNet5(input_shape,
                            depth,
                            width_multiplier,
                            num_classes,
                            prior_stddev,
                            dataset_size,
                            stddev_init):
    if (depth - 4) % 6 != 0:
        raise ValueError('depth should be 6n+4 (e.g., 16, 22, 28, 40).')
    inputs = tf.keras.layers.Input(shape=input_shape)

    initializer = ed.initializers.TrainableHeNormal(
        stddev_initializer=tf.keras.initializers.TruncatedNormal(
            mean=np.log(np.expm1(stddev_init)), stddev=0.1))
### Conv - 6
    x = ed.layers.Conv2DFlipout(
        6,
        kernel_size=5,
        strides=1,
        padding='SAME',
        activation=tf.nn.relu,
        kernel_initializer=initializer,
        kernel_regularizer=ed.regularizers.NormalKLDivergenceWithTiedMean(
            stddev=prior_stddev, scale_factor=1. / dataset_size))(inputs)

    x2 = ed.layers.Conv2DFlipout(
        6,
        kernel_size=5,
        padding='SAME',
        activation=tf.nn.relu,
        kernel_initializer=trainableNormalDuplicate.TrainableNormalDuplicate(initializer),
        kernel_regularizer=None)(inputs)

    x = tf.keras.layers.MaxPooling2D(pool_size=[2, 2],
                                         strides=[2, 2],
                                         padding='SAME')(x)

    x2 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2],
                                         strides=[2, 2],
                                         padding='SAME')(x2)

    ### Conv - 16
    initializer = ed.initializers.TrainableHeNormal(
        stddev_initializer=tf.keras.initializers.TruncatedNormal(
            mean=np.log(np.expm1(stddev_init)), stddev=0.1))

    x = ed.layers.Conv2DFlipout(
        16,
        kernel_size=5,
        padding='SAME',
        activation=tf.nn.relu,
        kernel_initializer=initializer,
        kernel_regularizer=ed.regularizers.NormalKLDivergenceWithTiedMean(
            stddev=prior_stddev, scale_factor=1. / dataset_size))(x)

    x2 = ed.layers.Conv2DFlipout(
        16,
        kernel_size=5,
        padding='SAME',
        activation=tf.nn.relu,
        kernel_initializer=trainableNormalDuplicate.TrainableNormalDuplicate(initializer),
        kernel_regularizer=None)(x2)

    x = tf.keras.layers.MaxPooling2D(pool_size=[2, 2],
                                     strides=[2, 2],
                                     padding='SAME')(x)

    x2 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2],
                                      strides=[2, 2],
                                      padding='SAME')(x2)

    ### Conv - 120

    initializer = ed.initializers.TrainableHeNormal(
        stddev_initializer=tf.keras.initializers.TruncatedNormal(
            mean=np.log(np.expm1(stddev_init)), stddev=0.1))

    x = ed.layers.Conv2DFlipout(
        120,
        kernel_size=5,
        padding='SAME',
        activation=tf.nn.relu,
        kernel_initializer=initializer,
        kernel_regularizer=ed.regularizers.NormalKLDivergenceWithTiedMean(
            stddev=prior_stddev, scale_factor=1. / dataset_size))(x)

    x2 = ed.layers.Conv2DFlipout(
        120,
        kernel_size=5,
        padding='SAME',
        activation=tf.nn.relu,
        kernel_initializer=trainableNormalDuplicate.TrainableNormalDuplicate(initializer),
        kernel_regularizer=None)(x2)

    # Dense 84

    x = tf.keras.layers.Flatten()(x)
    x2 = tf.keras.layers.Flatten()(x2)

    initializer = ed.initializers.TrainableHeNormal(
        stddev_initializer=tf.keras.initializers.TruncatedNormal(
            mean=np.log(np.expm1(stddev_init)), stddev=0.1))

    x = ed.layers.DenseReparameterization(
      84,
      activation=tf.nn.relu,
      kernel_initializer=initializer,
      kernel_regularizer=ed.regularizers.NormalKLDivergenceWithTiedMean(
          stddev=prior_stddev, scale_factor=1./dataset_size))(x)
    x2 = ed.layers.DenseReparameterization(
      84,
      activation=tf.nn.relu,
      kernel_initializer=trainableNormalDuplicate.TrainableNormalDuplicate(initializer),
      kernel_regularizer=None)(x2)

    # Dense - classes
    initializer = ed.initializers.TrainableHeNormal(
        stddev_initializer=tf.keras.initializers.TruncatedNormal(
            mean=np.log(np.expm1(stddev_init)), stddev=0.1))

    x = ed.layers.DenseReparameterization(
      num_classes,
      kernel_initializer=initializer,
      kernel_regularizer=ed.regularizers.NormalKLDivergenceWithTiedMean(
          stddev=prior_stddev, scale_factor=1./dataset_size))(x)
    x2 = ed.layers.DenseReparameterization(
      num_classes,
      kernel_initializer=trainableNormalDuplicate.TrainableNormalDuplicate(initializer),
      kernel_regularizer=None)(x2)

    x_stack = tf.stack([x, x2])

    return tf.keras.Model(inputs=inputs, outputs=x_stack)



def mlp_leNet5_Single(input_shape,
                            depth,
                            width_multiplier,
                            num_classes,
                            prior_stddev,
                            dataset_size,
                            stddev_init):
    if (depth - 4) % 6 != 0:
        raise ValueError('depth should be 6n+4 (e.g., 16, 22, 28, 40).')
    inputs = tf.keras.layers.Input(shape=input_shape)

    initializer = ed.initializers.TrainableHeNormal(
        stddev_initializer=tf.keras.initializers.TruncatedNormal(
            mean=np.log(np.expm1(stddev_init)), stddev=0.1))
### Conv - 6
    x = ed.layers.Conv2DFlipout(
        6,
        kernel_size=5,
        strides=1,
        padding='SAME',
        activation=tf.nn.relu,
        kernel_initializer=initializer,
        kernel_regularizer=ed.regularizers.NormalKLDivergenceWithTiedMean(
            stddev=prior_stddev, scale_factor=1. / dataset_size))(inputs)

    x = tf.keras.layers.MaxPooling2D(pool_size=[2, 2],
                                         strides=[2, 2],
                                         padding='SAME')(x)

    ### Conv - 16
    initializer = ed.initializers.TrainableHeNormal(
        stddev_initializer=tf.keras.initializers.TruncatedNormal(
            mean=np.log(np.expm1(stddev_init)), stddev=0.1))

    x = ed.layers.Conv2DFlipout(
        16,
        kernel_size=5,
        padding='SAME',
        activation=tf.nn.relu,
        kernel_initializer=initializer,
        kernel_regularizer=ed.regularizers.NormalKLDivergenceWithTiedMean(
            stddev=prior_stddev, scale_factor=1. / dataset_size))(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=[2, 2],
                                     strides=[2, 2],
                                     padding='SAME')(x)

    ### Conv - 120

    initializer = ed.initializers.TrainableHeNormal(
        stddev_initializer=tf.keras.initializers.TruncatedNormal(
            mean=np.log(np.expm1(stddev_init)), stddev=0.1))

    x = ed.layers.Conv2DFlipout(
        120,
        kernel_size=5,
        padding='SAME',
        activation=tf.nn.relu,
        kernel_initializer=initializer,
        kernel_regularizer=ed.regularizers.NormalKLDivergenceWithTiedMean(
            stddev=prior_stddev, scale_factor=1. / dataset_size))(x)

    # Dense 84

    x = tf.keras.layers.Flatten()(x)

    initializer = ed.initializers.TrainableHeNormal(
        stddev_initializer=tf.keras.initializers.TruncatedNormal(
            mean=np.log(np.expm1(stddev_init)), stddev=0.1))

    x = ed.layers.DenseReparameterization(
      84,
      activation=tf.nn.relu,
      kernel_initializer=initializer,
      kernel_regularizer=ed.regularizers.NormalKLDivergenceWithTiedMean(
          stddev=prior_stddev, scale_factor=1./dataset_size))(x)

    # Dense - classes
    initializer = ed.initializers.TrainableHeNormal(
        stddev_initializer=tf.keras.initializers.TruncatedNormal(
            mean=np.log(np.expm1(stddev_init)), stddev=0.1))

    x = ed.layers.DenseReparameterization(
      num_classes,
      kernel_initializer=initializer,
      kernel_regularizer=ed.regularizers.NormalKLDivergenceWithTiedMean(
          stddev=prior_stddev, scale_factor=1./dataset_size))(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def mlp_leNet5_Deterministic(input_shape,
                            num_classes):
    """Builds LeNet5."""
    inputs = tf.keras.layers.Input(shape=input_shape)
    conv1 = tf.keras.layers.Conv2D(6,
                                   kernel_size=5,
                                   padding='SAME',
                                   activation='relu')(inputs)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2],
                                         strides=[2, 2],
                                         padding='SAME')(conv1)
    conv2 = tf.keras.layers.Conv2D(16,
                                   kernel_size=5,
                                   padding='SAME',
                                   activation='relu')(pool1)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2],
                                         strides=[2, 2],
                                         padding='SAME')(conv2)
    conv3 = tf.keras.layers.Conv2D(120,
                                   kernel_size=5,
                                   padding='SAME',
                                   activation=tf.nn.relu)(pool2)
    flatten = tf.keras.layers.Flatten()(conv3)
    dense1 = tf.keras.layers.Dense(84, activation=tf.nn.relu)(flatten)
    logits = tf.keras.layers.Dense(num_classes)(dense1)
    return tf.keras.Model(inputs=inputs, outputs=logits)


def make_sign_initializer(random_sign_init):
  if random_sign_init > 0:
    return ed.initializers.RandomSign(random_sign_init)
  else:
    return tf.keras.initializers.RandomNormal(mean=1.0,
                                              stddev=-random_sign_init)

def mlp_leNet5_BatchEnsemble(input_shape,
                              num_classes,
                              ensemble_size,
                              random_sign_init,
                              l2):
    """Builds LeNet5."""
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = ed.layers.Conv2DBatchEnsemble(6,
                                   kernel_size=5,
                                   padding='SAME',
                                   activation='relu',
                                    alpha_initializer=make_sign_initializer(random_sign_init),
                                    gamma_initializer=make_sign_initializer(random_sign_init),
                                    kernel_regularizer=tf.keras.regularizers.l2(l2),
                                    ensemble_size=ensemble_size)(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=[2, 2],
                                         strides=[2, 2],
                                         padding='SAME')(x)
    x = ed.layers.Conv2DBatchEnsemble(16,
                                   kernel_size=5,
                                   padding='SAME',
                                   activation='relu',
                                    alpha_initializer = make_sign_initializer(random_sign_init),
                                    gamma_initializer = make_sign_initializer(random_sign_init),
                                    kernel_regularizer = tf.keras.regularizers.l2(l2),
                                    ensemble_size = ensemble_size)(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=[2, 2],
                                         strides=[2, 2],
                                         padding='SAME')(x)
    x = ed.layers.Conv2DBatchEnsemble(120,
                                   kernel_size=5,
                                   padding='SAME',
                                   activation='relu',
                                    alpha_initializer = make_sign_initializer(random_sign_init),
                                    gamma_initializer = make_sign_initializer(random_sign_init),
                                    kernel_regularizer = tf.keras.regularizers.l2(l2),
                                    ensemble_size = ensemble_size)(x)
    x = tf.keras.layers.Flatten()(x)
    x = ed.layers.DenseBatchEnsemble(
        84,
        alpha_initializer=make_sign_initializer(random_sign_init),
        gamma_initializer=make_sign_initializer(random_sign_init),
        activation=tf.nn.relu,
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(l2),
        bias_regularizer=tf.keras.regularizers.l2(l2),
        ensemble_size=ensemble_size)(x)

    x = ed.layers.DenseBatchEnsemble(
        num_classes,
        alpha_initializer=make_sign_initializer(random_sign_init),
        gamma_initializer=make_sign_initializer(random_sign_init),
        activation=None,
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(l2),
        bias_regularizer=tf.keras.regularizers.l2(l2),
        ensemble_size=ensemble_size)(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


