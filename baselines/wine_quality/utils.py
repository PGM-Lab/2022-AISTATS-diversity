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

"""Utilities for Wine-quality"""
import logging
import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from experiments.utils import fix_path


def load_input_fn(
    name,
    batch_size,
    use_bfloat16,
    data_dir=None,
    drop_remainder=True,
    lower_prop=0.0,
    upper_prop=1.0,
):
    """Loads WineQuality dataset for training or testing.

    Args:
      batch_size: The global batch size to use.
      use_bfloat16: data type, bfloat16 precision or float32.
      drop_remainder: bool.
      lower_prop: float, with upper_prop, determines the percentage of used
        dataset as [lower_prop, upper_prop).
      upper_prop: float

    Returns:
      Input function which returns a locally-sharded dataset batch.
    """
    if use_bfloat16:
        dtype = tf.bfloat16
    else:
        dtype = tf.float32

    ds_info = tfds.builder(name, data_dir=data_dir).info
    dataset_size = ds_info.splits["train"].num_examples

    def preprocess(features_dict, label):
        """Dictionary preprocessing function."""
        features = []
        for k in features_dict.keys():
            features.append(tf.cast(tf.reshape(features_dict[k], [-1, 1]), dtype))
        features = tf.concat(features, axis=0)
        label = tf.cast(label, dtype)
        return tf.squeeze(features), label

    def input_fn(ctx=None):
        """Returns a locally sharded (i.e., per-core) dataset batch."""
        if upper_prop == 1.0:
            new_split = "train[{}%:]".format(int(100 * lower_prop))
        elif lower_prop == 0.0:
            new_split = "train[:{}%]".format(int(100 * upper_prop))
        else:
            new_split = "train[{}%:{}%]".format(
                int(100 * lower_prop), int(100 * upper_prop)
            )
        dataset = tfds.load(name, split=new_split, as_supervised=True, try_gcs=True)
        dataset = dataset.shuffle(buffer_size=dataset_size).repeat()

        dataset = dataset.map(
            preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        if ctx and ctx.num_input_pipelines > 1:
            dataset = dataset.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)

        return dataset

    return input_fn


class LearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule.

    It starts with a linear warmup to the initial learning rate over
    `warmup_epochs`. This is found to be helpful for large batch size training
    (Goyal et al., 2018). The learning rate's value then uses the initial
    learning rate, and decays by a multiplier at the start of each epoch in
    `decay_epochs`. The stepwise decaying schedule follows He et al. (2015).
    """

    def __init__(
        self,
        steps_per_epoch,
        initial_learning_rate,
        decay_ratio,
        decay_epochs,
        warmup_epochs,
    ):
        super(LearningRateSchedule, self).__init__()
        self.steps_per_epoch = steps_per_epoch
        self.initial_learning_rate = initial_learning_rate
        self.decay_ratio = decay_ratio
        self.decay_epochs = decay_epochs
        self.warmup_epochs = warmup_epochs

    def __call__(self, step):
        lr_epoch = tf.cast(step, tf.float32) / self.steps_per_epoch
        learning_rate = self.initial_learning_rate
        if self.warmup_epochs >= 1:
            learning_rate *= lr_epoch / self.warmup_epochs
        decay_epochs = [self.warmup_epochs] + self.decay_epochs
        for index, start_epoch in enumerate(decay_epochs):
            learning_rate = tf.where(
                lr_epoch >= start_epoch,
                self.initial_learning_rate * self.decay_ratio ** index,
                learning_rate,
            )
        return learning_rate

    def get_config(self):
        return {
            "steps_per_epoch": self.steps_per_epoch,
            "initial_learning_rate": self.initial_learning_rate,
        }
