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

"""Deep Ensemble for regression on wine-quality dataset."""

import sys

sys.path.append(".")

from baselines.utils.loss_functions import compute_l2_loss

import functools
import os, datetime
import time
from absl import app
from absl import flags
from absl import logging
import itertools

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import uncertainty_baselines
from baselines.cifar.utils import (
    restore_checkpoint,
    save_checkpoint,
)
from baselines.utils import models_regression
from baselines.utils.loss_functions_regression import (
    compute_loss,
)
from baselines.utils.metrics_regression import (
    createMetrics,
    update_metrics,
)
from experiments.utils_regression import res_to_csv, fix_path
from baselines.wine_quality import utils  # local file import
from baselines.utils import varianceBound
from baselines.utils import resnet20_deterministic
import uncertainty_metrics as um

import sys
import hashlib

tf.config.run_functions_eagerly(True)

flags.DEFINE_integer("ensemble_size", 4, "Size of ensemble.")
flags.DEFINE_integer(
    "per_core_batch_size",
    32,
    "Batch size per TPU core/GPU. The number of new "
    "datapoints gathered per batch is this number divided by "
    "ensemble_size (we tile the batch by that # of times).",
)
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_float("fast_weight_lr_multiplier", 0.5, "fast weights lr multiplier.")
flags.DEFINE_float(
    "train_proportion", default=0.7, help="only use a proportion of training set."
)
flags.DEFINE_float(
    "base_learning_rate",
    0.1,
    "Base learning rate when total training batch size is 128.",
)
flags.DEFINE_integer(
    "lr_warmup_epochs",
    1,
    "Number of epochs for a linear warmup to the initial "
    "learning rate. Use 0 to do no warmup.",
)
flags.DEFINE_float("lr_decay_ratio", 0.2, "Amount to decay learning rate.")
flags.DEFINE_list(
    "lr_decay_epochs", ["60", "120", "160"], "Epochs to decay learning rate by."
)

flags.DEFINE_float("l2", 2e-4, "L2 regularization coefficient.")
flags.DEFINE_enum(
    "dataset",
    "wine_quality",
    enum_values=["wine_quality"],
    help="Dataset.",
)
flags.DEFINE_integer('checkpoint_interval', 25,
                     'Number of epochs between saving checkpoints. Use -1 to '
                     'never save checkpoints.')
flags.DEFINE_integer("num_bins", 15, "Number of bins for ECE.")
flags.DEFINE_string(
    "output_dir",
    "/tmp/cifar",
    "The directory where the model weights and "
    "training/evaluation summaries are stored.",
)
flags.DEFINE_integer("train_epochs", 250, "Number of training epochs.")

# Accelerator flags.
flags.DEFINE_bool("use_gpu", False, "Whether to run on GPU or otherwise TPU.")
flags.DEFINE_bool("use_bfloat16", False, "Whether to use mixed precision.")
flags.DEFINE_integer("num_cores", 8, "Number of TPU cores or number of GPUs.")
flags.DEFINE_string("tpu", None, "Name of the TPU. Only used if use_gpu is False.")

flags.DEFINE_string(
    "data_dir",
    None,
    "GS data folder (eg., gs://tfds-data/datasets ) None for local execution",
)

flags.DEFINE_string(
    "loss_fn", "PAC2B", "Loss Function: PACB, PAC2B, PAC2B-val, PAC2B-Log, Ensemble-CE"
)

flags.DEFINE_enum(
    "model",
    "leNet5",
    enum_values=[
        "leNet5",
        "ResNet20",
        "ResNet50",
        "wideResNet50",
        "mlp_1LDense",
        "mlp_1LDense_Dropout",
        "mlp_2LDense",
    ],
    help="Model.",
)

flags.DEFINE_string(
    "results_file", "results.csv", "CSV file where the results are stored"
)
flags.DEFINE_bool(
    "divide_l2_loss", False, "Whether to divide the l2 loss by the number of ensembles."
)
flags.DEFINE_integer(
    "mlp_hidden_dim",
    100,
    "Number of hidden layers for the multilayer perceptron model.",
)
flags.DEFINE_bool(
    "random_init", False, "Wether to random initializa each model of the ensemble"
)

FLAGS = flags.FLAGS


def main(argv):

    print("Command arguments:")
    print(sys.argv)

    del argv  # unused arg
    tf.io.gfile.makedirs(FLAGS.output_dir)
    logging.info("Saving checkpoints at %s", fix_path(FLAGS.output_dir, True))
    tf.random.set_seed(FLAGS.seed)
    tf.keras.backend.set_floatx("float32")

    # Uncomment this part to execute locally.
    # FLAGS.dataset = "wine_quality"
    # FLAGS.loss_fn = "PAC2B"
    # FLAGS.data_dir = None
    # FLAGS.use_gpu = True
    # FLAGS.checkpoint_interval = 1
    # FLAGS.ensemble_size = 4
    # FLAGS.model = "mlp_1LDense"
    # FLAGS.train_epochs = 100
    # FLAGS.base_learning_rate = 0.001
    # FLAGS.num_cores = 4
    # FLAGS.output_dir = "./output/"
    # FLAGS.train_proportion = 0.8
    # FLAGS.random_init = True

    ## or add this command line args:
    ##  --train_proportion 1.0   --dataset cifar10  --model mlp_1LDense  --loss_fn PACB  --ensemble_size 2  --train_epochs 2  --base_learning_rate 0.001  --output_dir ./output/  --checkpoint_interval 1  --corruptions_interval -1  --use_gpu True

    FLAG_str = (
        "P2B_Ensemble"
        + "--"
        + "model="
        + FLAGS.model
        + "--"
        + "dataset="
        + FLAGS.dataset
        + "--"
        + "loss_fn="
        + FLAGS.loss_fn
        + "--"
        + "ensemble_size="
        + str(FLAGS.ensemble_size)
        + "--"
        + "train_epochs="
        + str(FLAGS.train_epochs)
        + "--"
        + "base_learning_rate="
        + str(FLAGS.base_learning_rate)
        + "--"
        + "train_proportion="
        + str(FLAGS.train_proportion)
        + "--"
        + "seed="
        + str(FLAGS.seed)
        + "--"
        + "mlp_hidden_dim"
        + str(FLAGS.mlp_hidden_dim)
    )

    FLAG_hash = hashlib.sha1(FLAG_str.encode("utf8")).hexdigest()

    # Data download
    # builder = tfds.builder(FLAGS.dataset)
    # builder.download_and_prepare()

    ds_info = tfds.builder(FLAGS.dataset).info
    dataset = tfds.load(FLAGS.dataset, split="train", as_supervised=True, try_gcs=True)

    n_features = len(ds_info.features["features"].keys())

    num_examples = ds_info.splits["train"].num_examples

    val_proportion = (1 - FLAGS.train_proportion) / 2
    test_proportion = (1 - FLAGS.train_proportion) / 2

    per_core_batch_size = FLAGS.per_core_batch_size
    batch_size = per_core_batch_size * FLAGS.num_cores
    # Train_proportion is a float so need to convert steps_per_epoch to int.
    steps_per_epoch = int((num_examples * FLAGS.train_proportion) // batch_size)

    if FLAGS.use_gpu:
        logging.info("Use GPU")
        strategy = tf.distribute.MirroredStrategy()
    else:
        logging.info("Use TPU at %s", FLAGS.tpu if FLAGS.tpu is not None else "local")
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=FLAGS.tpu)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.experimental.TPUStrategy(resolver)

    train_input_fn = utils.load_input_fn(
        name=FLAGS.dataset,
        batch_size=per_core_batch_size,
        use_bfloat16=FLAGS.use_bfloat16,
        data_dir=FLAGS.data_dir,
        lower_prop=0.0,
        upper_prop=FLAGS.train_proportion,
    )

    train_dataset = strategy.experimental_distribute_datasets_from_function(
        train_input_fn
    )

    # trains_steps_per_epoch must be equal to evals_steps_per_epoch
    train_steps_per_epoch = int(
        (ds_info.splits["train"].num_examples * FLAGS.train_proportion)
        // (per_core_batch_size * FLAGS.num_cores)
    )

    per_core_batch_size_train_eval = (
        per_core_batch_size * (val_proportion) / FLAGS.train_proportion
    )
    evals_steps_per_epoch = int(
        (ds_info.splits["train"].num_examples * (val_proportion))
        // (per_core_batch_size_train_eval * FLAGS.num_cores)
    )
    if evals_steps_per_epoch != train_steps_per_epoch:
        raise ValueError("Error.")

    per_core_batch_size_train_eval = int(
        per_core_batch_size * (val_proportion) / FLAGS.train_proportion
    )

    trainval_input_fn = utils.load_input_fn(
        name=FLAGS.dataset,
        batch_size=per_core_batch_size_train_eval,
        use_bfloat16=FLAGS.use_bfloat16,
        data_dir=FLAGS.data_dir,
        lower_prop=FLAGS.train_proportion,
        upper_prop=FLAGS.train_proportion + val_proportion,
    )

    train_validation_dataset = strategy.experimental_distribute_datasets_from_function(
        trainval_input_fn
    )

    validation_input_fn = utils.load_input_fn(
        name=FLAGS.dataset,
        batch_size=per_core_batch_size,
        use_bfloat16=FLAGS.use_bfloat16,
        data_dir=FLAGS.data_dir,
        lower_prop=FLAGS.train_proportion,
        upper_prop=FLAGS.train_proportion + val_proportion,
    )

    validation_dataset = strategy.experimental_distribute_datasets_from_function(
        validation_input_fn
    )

    test_input_fn = utils.load_input_fn(
        name=FLAGS.dataset,
        batch_size=per_core_batch_size,
        use_bfloat16=FLAGS.use_bfloat16,
        data_dir=FLAGS.data_dir,
        lower_prop=FLAGS.train_proportion + val_proportion,
        upper_prop=1.0,
    )

    test_dataset = strategy.experimental_distribute_datasets_from_function(
        test_input_fn
    )

    if FLAGS.use_bfloat16:
        policy = tf.keras.mixed_precision.experimental.Policy("mixed_bfloat16")
        tf.keras.mixed_precision.experimental.set_policy(policy)

    logdir = os.path.join(
        "logs", FLAG_str, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    summary_writer = tf.summary.create_file_writer(
        os.path.join(FLAGS.output_dir, logdir)
    )

    with strategy.scope():
        logging.info("Building model_{}".format(FLAGS.model))

        model = []
        for _ in range(FLAGS.ensemble_size):
            if FLAGS.model == "mlp_1LDense":
                model.append(models_regression.mlp_1LDenseDeterministic(n_features))
            elif FLAGS.model == "mlp_1LDense_Dropout":
                model.append(models_regression.mlp_1LDenseDeterministic_Dropout(n_features))
            elif FLAGS.model == "mlp_2LDense":
                model.append(
                    models_regression.mlp_2LDenseDeterministic(
                        input_shape=n_features, depth=FLAGS.mlp_hidden_dim
                    )
                )

        if not FLAGS.random_init:
            for i in range(1, FLAGS.ensemble_size):
                model[i].set_weights(model[0].get_weights())

        logging.info("Model input shape: %s", model[0].input_shape)
        logging.info("Model output shape: %s", model[0].output_shape)
        logging.info("Model number of weights: %s", model[0].count_params())
        # Linearly scale learning rate and the decay epochs by vanilla settings.
        base_lr = FLAGS.base_learning_rate * batch_size / 128
        lr_decay_epochs = [
            (int(start_epoch_str) * FLAGS.train_epochs) // 200
            for start_epoch_str in FLAGS.lr_decay_epochs
        ]
        lr_schedule = utils.LearningRateSchedule(
            steps_per_epoch,
            base_lr,
            decay_ratio=FLAGS.lr_decay_ratio,
            decay_epochs=lr_decay_epochs,
            warmup_epochs=FLAGS.lr_warmup_epochs,
        )
        if FLAGS.lr_decay_ratio == -1:
            optimizer = tf.keras.optimizers.SGD(
                FLAGS.base_learning_rate, momentum=0.9, nesterov=True
            )
        else:
            optimizer = tf.keras.optimizers.SGD(
                lr_schedule, momentum=0.9, nesterov=True
            )

        metrics_train = createMetrics(FLAGS.ensemble_size)
        metrics_train["loss"] = tf.keras.metrics.Mean()
        metrics_test = createMetrics(FLAGS.ensemble_size)
        metrics_test["ms_per_example"] = tf.keras.metrics.Mean()
        metrics_validation = createMetrics(FLAGS.ensemble_size)

        # checkpoints_path = f"{FLAGS.output_dir}/checkpoints/param_optim/"
        # checkpoint, initial_epoch = restore_checkpoint(
        #     checkpoints_path, FLAGS, FLAG_hash, model, optimizer, steps_per_epoch
        # )

    @tf.function
    def train_step(train_iterator, valid_iterator=None):
        """Training StepFn."""

        def step_fn(inputs_train, inputs_valid=None):
            """Per-Replica StepFn."""

            # extract the features and labels from the train and validation data
            features, y_true = inputs_train
            features_valid, y_true_valid = inputs_valid or (None, None)
            with tf.GradientTape() as tape:
                # Shape (n_models, n_samples)
                predictions = tf.stack(
                    [
                        model[i](features, training=True)
                        for i in range(FLAGS.ensemble_size)
                    ]
                )
                #predictions = tf.clip_by_value(predictions, -1, 15)

                if FLAGS.use_bfloat16:
                    predictions = tf.cast(predictions, tf.float32)

                # shape [n_models]
                mse = tf.keras.losses.MSE(
                    y_true=tf.reshape(
                        tf.tile(y_true, [FLAGS.ensemble_size]),
                        [FLAGS.ensemble_size, y_true.shape[0], 1],
                    ),
                    y_pred=predictions,
                )
                # Average over batch
                mse = tf.reduce_mean(mse, axis=-1)

                l2_loss = 0
                for i in range(FLAGS.ensemble_size):
                    l2_loss += compute_l2_loss(
                        model[i], FLAGS.l2
                    )  # sum(model[i].losses)#/FLAGS.num_eval_samples

                if FLAGS.divide_l2_loss:
                    l2_loss = l2_loss / FLAGS.ensemble_size

                # arguments for the loss function
                args = dict(
                    mse=mse,
                    l2_loss=l2_loss,
                    FLAGS=FLAGS,
                    predictions=predictions,
                    y_true=y_true,
                )

                if FLAGS.loss_fn == "PAC2B-val":
                    # shape (n_models, bath_size, 1)
                    predictions_valid = tf.stack(
                        [
                            model[i](features_valid, training=True)
                            for i in range(FLAGS.ensemble_size)
                        ]
                    )
                    mse_valid = tf.keras.losses.MSE(
                        y_true=tf.reshape(
                            tf.tile(y_true_valid, [FLAGS.ensemble_size]),
                            [FLAGS.ensemble_size, y_true_valid.shape[0], 1],
                        ),
                        y_pred=predictions_valid,
                    )

                    # Average over batch
                    mse = tf.reduce_mean(mse, axis=-1)

                    args["predictions_valid"] = predictions_valid
                    args["mse_valid"] = mse_valid

                loss = compute_loss(FLAGS.loss_fn, **args)

                # Scale the loss given the TPUStrategy will reduce sum all gradients.
                scaled_loss = loss / strategy.num_replicas_in_sync

            trainable_vars = []
            for i in range(FLAGS.ensemble_size):
                trainable_vars.extend(model[i].trainable_variables)

            grads = tape.gradient(scaled_loss, trainable_vars)

            if FLAGS.loss_fn == "PAC2B" and FLAGS.random_init and FLAGS.model == "mlp_1LDense":
                grads, _ = tf.clip_by_global_norm(grads, 1.0)

            optimizer.apply_gradients(zip(grads, trainable_vars))

            update_metrics(metrics_train, y_true, predictions, predictions.shape[0])

            metrics_train["loss"].update_state(loss)

        # get the arguments of each step: validation data is optional
        args = [
            next(train_iterator),
        ]

        if valid_iterator is not None:
            args.append(next(valid_iterator))

        # run an iteration
        strategy.run(step_fn, args)

    @tf.function
    def test_step(iterator, metrics):
        """Evaluation StepFn."""

        def step_fn(inputs, metrics):
            """Per-Replica StepFn."""
            features, y_true = inputs

            # Shape (n_models, n_samples)
            predictions = tf.squeeze(
                tf.stack(
                    [
                        model[i](features, training=False)
                        for i in range(FLAGS.ensemble_size)
                    ]
                ),
                axis=2,
            )

            if FLAGS.use_bfloat16:
                predictions = tf.cast(predictions, tf.float32)

            update_metrics(metrics, y_true, predictions, predictions.shape[0])

        strategy.run(step_fn, args=(next(iterator), metrics))

    train_iterator = iter(train_dataset)
    valid_iterator = iter(train_validation_dataset)

    start_time = time.time()
    for epoch in range(FLAGS.train_epochs):
        logging.info("Starting to run epoch: %s", epoch)
        for step in range(steps_per_epoch):
            train_step(train_iterator, valid_iterator)

            current_step = epoch * steps_per_epoch + (step + 1)
            max_steps = steps_per_epoch * FLAGS.train_epochs
            time_elapsed = time.time() - start_time
            steps_per_sec = float(current_step) / time_elapsed
            eta_seconds = (max_steps - current_step) / steps_per_sec
            message = (
                "{:.1%} completion: epoch {:d}/{:d}. {:.1f} steps/s. "
                "ETA: {:.0f} min. Time elapsed: {:.0f} min".format(
                    current_step / max_steps,
                    epoch + 1,
                    FLAGS.train_epochs,
                    steps_per_sec,
                    eta_seconds / 60,
                    time_elapsed / 60,
                )
            )
            if step % 20 == 0:
                logging.info(message)

        datasets_to_evaluate = {"test": test_dataset, "validation": validation_dataset}

        for dataset_name, test_dataset in datasets_to_evaluate.items():
            test_iterator = iter(test_dataset)
            logging.info("Testing on dataset %s", dataset_name)

            if dataset_name == "validation":
                steps_per_eval = int(num_examples * val_proportion) // batch_size
                metrics = metrics_validation
            if dataset_name == "test":
                steps_per_eval = int(num_examples * test_proportion) // batch_size
                metrics = metrics_test

            for step in range(steps_per_eval):
                if step % 20 == 0:
                    logging.info(
                        "Starting to run eval step %s of epoch: %s", step, epoch + 1
                    )
                test_start_time = time.time()
                test_step(test_iterator, metrics)
                ms_per_example = (time.time() - test_start_time) * 1e6 / batch_size
                # metrics_test['ms_per_example'].update_state(ms_per_example)

            logging.info("Done with testing on %s", dataset_name)

        logging.info(
            "Ensemble MSE: %.4f, Gibbs MSE: %.4f, Variance: %.4f,",
            metrics_train["ensemble_mse"].result().numpy(),
            metrics_train["gibbs_mse"].result().numpy(),
            metrics_train["variance"].result().numpy(),
        )

        logging.info(
            "Test Ensemble MSE: %.4f, Gibbs MSE: %.4f",
            metrics_test["ensemble_mse"].result().numpy(),
            metrics_test["gibbs_mse"].result().numpy(),
        )

        total_results = {
            "train/" + name: metric.result().numpy()
            for name, metric in metrics_train.items()
        }
        total_results.update(
            {
                "test/" + name: metric.result().numpy()
                for name, metric in metrics_test.items()
            }
        )
        total_results.update(
            {
                "validation/" + name: metric.result().numpy()
                for name, metric in metrics_validation.items()
            }
        )

        if epoch == FLAGS.train_epochs - 1:
            flags_dict = FLAGS.flag_values_dict()
            flags_dict["method"] = "ensemble"
            res_to_csv(flags_dict, total_results, FLAG_hash)

        with summary_writer.as_default():
            for name, result in total_results.items():
                logging.info(str(name) + ": %.4f", result)
                tf.summary.scalar(name, result, step=epoch + 1)

        for metric in metrics_train.values():
            metric.reset_states()

        for metric in metrics_test.values():
            metric.reset_states()

        for metric in metrics_validation.values():
            metric.reset_states()

        # if FLAGS.checkpoint_interval > 0 and (
        #     (epoch + 1) % FLAGS.checkpoint_interval == 0
        #     or epoch == FLAGS.train_epochs - 1
        # ):
        #     save_checkpoint(
        #         checkpoint, f"{FLAGS.output_dir}/checkpoints/param_optim/", FLAG_hash
        #     )


if __name__ == "__main__":
    app.run(main)
