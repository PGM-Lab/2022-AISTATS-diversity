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

# Lint as: python3
"""TF Keras definition for Resnet-20 for CIFAR."""

from typing import Any, Dict

import tensorflow as tf


def _resnet_layer(
    inputs: tf.Tensor,
    num_filters: int = 16,
    kernel_size: int = 3,
    strides: int = 1,
    use_activation: bool = True,
    use_norm: bool = True) -> tf.Tensor:
  """2D Convolution-Batch Normalization-Activation stack builder.

  Args:
    inputs: input tensor from input image or previous layer.
    num_filters: Conv2D number of filters.
    kernel_size: Conv2D square kernel dimensions.
    strides: Conv2D square stride dimensions.
    use_activation: whether or not to use a non-linearity.
    use_norm: whether to include normalization.
    l2_weight: the L2 regularization coefficient to use for the convolution
      kernel regularizer.

  Returns:
      Tensor output of this layer.
  """
  conv_layer = tf.keras.layers.Conv2D(
      num_filters,
      kernel_size=kernel_size,
      strides=strides,
      padding='same',
      kernel_initializer='he_normal')

  x = conv_layer(inputs)
  x = tf.keras.layers.BatchNormalization()(x) if use_norm else x
  x = tf.keras.layers.ReLU()(x) if use_activation is not None else x
  return x


def resnet20_deterministic(input_shape, num_classes):
  """Resnet-20 v1."""
  # TODO(znado): support NCHW data format.
  input_layer = tf.keras.layers.Input(
      shape=input_shape)
  depth = 20
  num_filters = 16
  num_res_blocks = int((depth - 2) / 6)

  x = _resnet_layer(
      inputs=input_layer,
      num_filters=num_filters)
  for stack in range(3):
    for res_block in range(num_res_blocks):
      strides = 1
      if stack > 0 and res_block == 0:
        strides = 2
      y = _resnet_layer(
          inputs=x,
          num_filters=num_filters,
          strides=strides)
      y = _resnet_layer(
          inputs=y,
          num_filters=num_filters,
          use_activation=False)
      if stack > 0 and res_block == 0:
        x = _resnet_layer(
            inputs=x,
            num_filters=num_filters,
            kernel_size=1,
            strides=strides,
            use_activation=False,
            use_norm=False)
      x = tf.keras.layers.add([x, y])
      x = tf.keras.layers.ReLU()(x)
    num_filters *= 2

  x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
  x = tf.keras.layers.Flatten()(x)
  logits = tf.keras.layers.Dense(num_classes, kernel_initializer='he_normal')(x)
  return tf.keras.models.Model(
      inputs=input_layer, outputs=logits, name='ResNet20')
