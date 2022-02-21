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

"""Deep Ensemble on CIFAR-10 and CIFAR-100."""

import functools
import os,datetime
import time
from absl import app
from absl import flags
from absl import logging
import itertools

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import uncertainty_baselines
from baselines.cifar.utils import restore_checkpoint, save_checkpoint
from baselines.utils import models
from baselines.utils.loss_functions import PAC2B, PAC2B_val, PACB, Ensemble_CE, compute_loss, compute_l2_loss
from baselines.utils.metrics import gibbs_error, ensemble_error, logLike_Variance, update_pairwise_diversity, \
    aggregateMetrics, createMetrics, update_metrics
from experiments.utils import res_to_csv, fix_path
from uncertainty_baselines.models import wide_resnet
from baselines.cifar import utils # local file import
from baselines.utils import varianceBound
from baselines.utils import resnet20_deterministic
import uncertainty_metrics as um

import sys
import hashlib




flags.DEFINE_integer('ensemble_size', 4, 'Size of ensemble.')
flags.DEFINE_integer('per_core_batch_size', 64,
                     'Batch size per TPU core/GPU. The number of new '
                     'datapoints gathered per batch is this number divided by '
                     'ensemble_size (we tile the batch by that # of times).')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_float('fast_weight_lr_multiplier', 0.5,
                   'fast weights lr multiplier.')
flags.DEFINE_float('train_proportion', default=1.0,
                   help='only use a proportion of training set.')
flags.DEFINE_float('base_learning_rate', 0.1,
                   'Base learning rate when total training batch size is 128.')
flags.DEFINE_integer('lr_warmup_epochs', 1,
                     'Number of epochs for a linear warmup to the initial '
                     'learning rate. Use 0 to do no warmup.')
flags.DEFINE_float('lr_decay_ratio', 0.2, 'Amount to decay learning rate.')
flags.DEFINE_list('lr_decay_epochs', ['60', '120', '160'],
                  'Epochs to decay learning rate by.')
flags.DEFINE_float('l2', 2e-4, 'L2 regularization coefficient.')
flags.DEFINE_enum('dataset', 'cifar10',
                  enum_values=['cifar10', 'cifar100', 'mnist', 'fashion_mnist'],
                  help='Dataset.')
# TODO(ghassen): consider adding CIFAR-100-C to TFDS.
flags.DEFINE_string('cifar100_c_path', None,
                    'Path to the TFRecords files for CIFAR-100-C. Only valid '
                    '(and required) if dataset is cifar100 and corruptions.')
flags.DEFINE_integer('corruptions_interval', -1,
                     'Number of epochs between evaluating on the corrupted '
                     'test data. Use -1 to never evaluate.')
flags.DEFINE_integer('checkpoint_interval', 25,
                     'Number of epochs between saving checkpoints. Use -1 to '
                     'never save checkpoints.')
flags.DEFINE_integer('num_bins', 15, 'Number of bins for ECE.')
flags.DEFINE_string('output_dir', '/tmp/cifar',
                    'The directory where the model weights and '
                    'training/evaluation summaries are stored.')
flags.DEFINE_integer('train_epochs', 250, 'Number of training epochs.')

# Accelerator flags.
flags.DEFINE_bool('use_gpu', False, 'Whether to run on GPU or otherwise TPU.')
flags.DEFINE_bool('use_bfloat16', False, 'Whether to use mixed precision.')
flags.DEFINE_integer('num_cores', 8, 'Number of TPU cores or number of GPUs.')
flags.DEFINE_string('tpu', None,
                    'Name of the TPU. Only used if use_gpu is False.')

flags.DEFINE_string('data_dir', None,
                    'GS data folder (eg., gs://tfds-data/datasets ) None for local execution')

flags.DEFINE_string('loss_fn', 'PAC2B',
                    'Loss Function: PACB, PAC2B, PAC2B-val, PAC2B-val-unsupervised, PAC2B-Log, Ensemble-CE')

flags.DEFINE_enum('model', 'leNet5',
                  enum_values=['leNet5', 'ResNet20', 'ResNet50', 'wideResNet50','mlp_1LDense', "mlp_1LDense_Dropout", 'mlp_2LDense'],
                  help='Model.')

flags.DEFINE_string('results_file', 'results.csv',
                    'CSV file where the results are stored')

flags.DEFINE_bool('divide_l2_loss', False, 'Whether to divide the l2 loss by the number of ensembles.')
flags.DEFINE_float('beta', 1, 'Weight of the validation variance when using unsupervised loss function.')

flags.DEFINE_bool('random_init', True, 'Whether to random initialize each model of the ensemble')

FLAGS = flags.FLAGS

# IMPORTANTE: al añadir un nuevo flag, modificar:
# FLAG_str
# res_to_csv

# tf.config.run_functions_eagerly(True)

def main(argv):

  print("Command arguments:")
  print(sys.argv)

  del argv  # unused arg
  tf.io.gfile.makedirs(FLAGS.output_dir)
  logging.info('Saving checkpoints at %s', fix_path(FLAGS.output_dir, True))
  tf.random.set_seed(FLAGS.seed)
  tf.keras.backend.set_floatx('float32')

  # Uncomment this part to execute locally.
  # FLAGS.dataset = 'cifar10'
  # FLAGS.loss_fn = 'PAC2B-val-unsupervised2'
  # FLAGS.data_dir = None
  # FLAGS.use_gpu=True
  # FLAGS.checkpoint_interval=-1
  # FLAGS.corruptions_interval=-1
  # FLAGS.ensemble_size=2
  # FLAGS.model = 'mlp_1LDense'
  # FLAGS.train_epochs=2
  # FLAGS.base_learning_rate = 0.001
  # FLAGS.num_cores = 4
  # FLAGS.output_dir = './output/'
  # FLAGS.train_proportion=0.8
  # FLAGS.beta = 0.55


  ## or add this command line args:
  ##  --train_proportion 1.0   --dataset cifar10  --model mlp_1LDense  --loss_fn PACB  --ensemble_size 2  --train_epochs 2  --base_learning_rate 0.001  --output_dir ./output/  --checkpoint_interval 1  --corruptions_interval -1  --use_gpu True

  strtime = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

  FLAG_str = strtime +"_P2B_Ensemble" + "--" + \
             "model=" + FLAGS.model + "--" + \
             "dataset=" + FLAGS.dataset + "--" + \
             "loss_fn=" + FLAGS.loss_fn + "--" +\
             "ensemble_size=" + str(FLAGS.ensemble_size) + "--" + \
             "train_epochs=" + str(FLAGS.train_epochs) + "--" + \
             "base_learning_rate=" + str(FLAGS.base_learning_rate) + "--" + \
             "train_proportion=" + str(FLAGS.train_proportion) + "--" + \
             "corruptions_interval=" + str(FLAGS.corruptions_interval) + "--" + \
             "divide_l2_loss=" + str(FLAGS.divide_l2_loss) + "--" + \
             "beta=" + str(FLAGS.beta) + "--" + \
             "random_init=" + str(FLAGS.random_init) + "--" + \
             "seed=" + str(FLAGS.seed)


  FLAG_hash = hashlib.sha1(FLAG_str.encode("utf8")).hexdigest()
  print("FLAGS:\n=======================")
  print(FLAG_str)
  print(FLAG_hash)
  print("=======================\n")




  ds_info = tfds.builder(FLAGS.dataset).info
  per_core_batch_size = FLAGS.per_core_batch_size
  batch_size = per_core_batch_size * FLAGS.num_cores
  # Train_proportion is a float so need to convert steps_per_epoch to int.
  steps_per_epoch = int((ds_info.splits['train'].num_examples *
                         FLAGS.train_proportion) // batch_size)
  steps_per_eval = ds_info.splits['test'].num_examples // batch_size
  num_classes = ds_info.features['label'].num_classes

  if FLAGS.use_gpu:
    logging.info('Use GPU')
    strategy = tf.distribute.MirroredStrategy()
  else:
    logging.info('Use TPU at %s',
                 FLAGS.tpu if FLAGS.tpu is not None else 'local')
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=FLAGS.tpu)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)


  train_input_fn = utils.load_input_fn(
      split=tfds.Split.TRAIN,
      name=FLAGS.dataset,
      batch_size=per_core_batch_size,
      use_bfloat16=FLAGS.use_bfloat16,
      data_dir=FLAGS.data_dir,
      proportion=FLAGS.train_proportion)

  train_dataset = strategy.experimental_distribute_datasets_from_function(
      train_input_fn)


  clean_test_input_fn = utils.load_input_fn(
      split=tfds.Split.TEST,
      name=FLAGS.dataset,
      batch_size=per_core_batch_size,
      use_bfloat16=FLAGS.use_bfloat16,
      data_dir=FLAGS.data_dir)

  test_datasets = {
      'clean': strategy.experimental_distribute_datasets_from_function(
          clean_test_input_fn),
  }

  if FLAGS.train_proportion<1.0:
    val_input_fn = utils.load_input_fn(
        split=tfds.Split.TRAIN,
        name=FLAGS.dataset,
        batch_size=per_core_batch_size,
        use_bfloat16=FLAGS.use_bfloat16,
        data_dir=FLAGS.data_dir,
        proportion=-(1.-FLAGS.train_proportion))

    test_datasets['validation']=strategy.experimental_distribute_datasets_from_function(
        val_input_fn)

    # trains_steps_per_epoch must be equal to evals_steps_per_epoch
    train_steps_per_epoch = int((ds_info.splits['train'].num_examples *
                           FLAGS.train_proportion) // (per_core_batch_size * FLAGS.num_cores))

    per_core_batch_size_train_eval = per_core_batch_size * (1 - FLAGS.train_proportion) / FLAGS.train_proportion
    evals_steps_per_epoch = int((ds_info.splits['train'].num_examples *
                                 (1 - FLAGS.train_proportion)) // (per_core_batch_size_train_eval * FLAGS.num_cores))
    if evals_steps_per_epoch != train_steps_per_epoch:
        raise ValueError('Error.')

    per_core_batch_size_train_eval = int(per_core_batch_size*(1-FLAGS.train_proportion)/FLAGS.train_proportion)

    trainval_input_fn = utils.load_input_fn(
        split=tfds.Split.TRAIN,
        name=FLAGS.dataset,
        batch_size=per_core_batch_size_train_eval,
        use_bfloat16=FLAGS.use_bfloat16,
        data_dir=FLAGS.data_dir,
        proportion=-(1.-FLAGS.train_proportion))

    train_validation_dataset=strategy.experimental_distribute_datasets_from_function(
        trainval_input_fn)



  if FLAGS.corruptions_interval > 0:
    if FLAGS.dataset == 'cifar10':
      load_c_input_fn = utils.load_cifar10_c_input_fn
    else:
      load_c_input_fn = functools.partial(utils.load_cifar100_c_input_fn,
                                          path=FLAGS.cifar100_c_path)
    corruption_types, max_intensity = utils.load_corrupted_test_info(
        FLAGS.dataset)
    for corruption in corruption_types:
      for intensity in range(1, max_intensity + 1):
        input_fn = load_c_input_fn(
            corruption_name=corruption,
            corruption_intensity=intensity,
            batch_size=per_core_batch_size,
            use_bfloat16=FLAGS.use_bfloat16,
            data_dir=FLAGS.data_dir)
        test_datasets['{0}_{1}'.format(corruption, intensity)] = (
            strategy.experimental_distribute_datasets_from_function(input_fn))


  if FLAGS.use_bfloat16:
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    tf.keras.mixed_precision.experimental.set_policy(policy)

  logdir = os.path.join("logs", FLAG_str, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.output_dir, logdir))

  with strategy.scope():
    logging.info('Building ResNet model_{}'.format(FLAGS.model))
    if FLAGS.model == 'leNet5':
        model = []
        for _ in range(FLAGS.ensemble_size):
            model.append(models.mlp_leNet5_Deterministic(
            input_shape=ds_info.features['image'].shape,
            num_classes=num_classes))

    elif FLAGS.model == 'mlp_1LDense':
        model = []
        for _ in range(FLAGS.ensemble_size):
            model.append(models.mlp_1LDenseDeterministic(
            input_shape=ds_info.features['image'].shape,
            num_classes=num_classes))
    elif FLAGS.model == 'mlp_1LDense_Dropout':
        model = []
        for _ in range(FLAGS.ensemble_size):
            model.append(models.mlp_1LDenseDeterministic_Dropout(
            input_shape=ds_info.features['image'].shape,
            num_classes=num_classes))
    elif FLAGS.model == 'mlp_2LDense':
        model = []
        for _ in range(FLAGS.ensemble_size):
            model.append(models.mlp_2LDenseDetxerministic(
            input_shape=ds_info.features['image'].shape,
            num_classes=num_classes))
    elif FLAGS.model == 'ResNet20':
        model = []
        for _ in range(FLAGS.ensemble_size):
            model.append(resnet20_deterministic.resnet20_deterministic(
            input_shape=ds_info.features['image'].shape,
            num_classes=num_classes))
    elif FLAGS.model == 'ResNet50':
        model = []
        for _ in range(FLAGS.ensemble_size):
            model.append(uncertainty_baselines.models.resnet50_deterministic(
            input_shape=ds_info.features['image'].shape,
            num_classes=num_classes))
    elif FLAGS.model == 'wideResNet50':
        model = []
        for _ in range(FLAGS.ensemble_size):
            model.append(uncertainty_baselines.models.wide_resnet(
            input_shape=ds_info.features['image'].shape,
            depth=28,
            width_multiplier=10,
            num_classes=num_classes,
            l2=FLAGS.l2,
            version=2))
    else:
        model = None

    if not FLAGS.random_init:
        for i in range(1,FLAGS.ensemble_size):
            model[i].set_weights(model[0].get_weights())


    logging.info('Model input shape: %s', model[0].input_shape)
    logging.info('Model output shape: %s', model[0].output_shape)
    logging.info('Model number of weights: %s', model[0].count_params())
    # Linearly scale learning rate and the decay epochs by vanilla settings.
    base_lr = FLAGS.base_learning_rate * batch_size / 128
    lr_decay_epochs = [(int(start_epoch_str) * FLAGS.train_epochs) // 200
                       for start_epoch_str in FLAGS.lr_decay_epochs]
    lr_schedule = utils.LearningRateSchedule(
        steps_per_epoch,
        base_lr,
        decay_ratio=FLAGS.lr_decay_ratio,
        decay_epochs=lr_decay_epochs,
        warmup_epochs=FLAGS.lr_warmup_epochs)
    optimizer = tf.keras.optimizers.SGD(lr_schedule,
                                        momentum=0.9,
                                        nesterov=True)
    metrics_train = createMetrics(FLAGS.ensemble_size)
    metrics_train['loss']= tf.keras.metrics.Mean()
    metrics_test = createMetrics(FLAGS.ensemble_size)
    metrics_test['ms_per_example'] = tf.keras.metrics.Mean()
    metrics_validation = createMetrics(FLAGS.ensemble_size)

    corrupt_metrics = {}
    if FLAGS.corruptions_interval > 0:
      corrupt_metrics = {}
      for intensity in range(1, max_intensity + 1):
        for corruption in corruption_types:
          dataset_name = '{0}_{1}'.format(corruption, intensity)
          corrupt_metrics[dataset_name]= createMetrics(FLAGS.ensemble_size)


    checkpoints_path = f'{FLAGS.output_dir}/checkpoints/param_optim/'
    checkpoint, initial_epoch = restore_checkpoint(checkpoints_path, FLAGS, FLAG_hash, model, optimizer, steps_per_epoch)



  @tf.function
  def train_step(train_iterator, valid_iterator=None):
    """Training StepFn."""
    def step_fn(inputs_train, inputs_valid=None):
      """Per-Replica StepFn."""

      # extract the features and labels from the train and validation data
      images, labels = inputs_train
      images_valid, labels_valid = inputs_valid or (None,None)

      with tf.GradientTape() as tape:
        logits = tf.stack([model[i](images, training=True) for i in range(FLAGS.ensemble_size)])

        if FLAGS.use_bfloat16:
          logits = tf.cast(logits, tf.float32)

        log_likelihood = -tf.keras.losses.sparse_categorical_crossentropy(tf.reshape(tf.tile(labels, [FLAGS.ensemble_size]), [FLAGS.ensemble_size, labels.shape[0]]),
                                                                          logits, from_logits=True)
        l2_loss = 0
        for i in range(FLAGS.ensemble_size):
            l2_loss += compute_l2_loss(model[i],FLAGS.l2)

        if (FLAGS.divide_l2_loss):
            l2_loss = l2_loss/FLAGS.ensemble_size

        # arguments for the loss function
        args = dict(log_likelihood = log_likelihood,
                    l2_loss = l2_loss,
                    FLAGS = FLAGS,
                    logits = logits,
                    labels = labels)

        if FLAGS.loss_fn == 'PAC2B-val':
          # terms from validation
          logits_valid = tf.stack([model[i](images_valid, training=True) for i in range(FLAGS.ensemble_size)])
          log_likelihood_valid = -tf.keras.losses.sparse_categorical_crossentropy(
            tf.reshape(tf.tile(labels_valid, [FLAGS.ensemble_size]), [FLAGS.ensemble_size, labels_valid.shape[0]]),
            logits_valid, from_logits=True)

          args['logits_valid'] = logits_valid
          args['log_likelihood_valid'] = log_likelihood_valid
          
        elif FLAGS.loss_fn == 'PAC2B-val-unsupervised':
          # terms from validation [n_models, batch_size, n_classes]
          logits_valid = tf.stack([model[i](images_valid, training=True) 
                                   for i in range(FLAGS.ensemble_size)])
          
          # Compute predicted labels [n_models, batch_size, n_classes]  ->  [batch_size, n_classes]
          ensemble_probs = tf.reduce_mean(tf.nn.softmax(logits_valid), axis=0)
          
          # [batch_size, n_classes} -> [batch_size]
          unsupervised_labels_valid = tf.stop_gradient(
            tf.math.argmax(ensemble_probs, axis = 1)
            )
          
          # Compute log likelihood using unsupervised labels as true labels
          # [n_models, batch_size]
          log_likelihood_valid = -tf.keras.losses.sparse_categorical_crossentropy(
            tf.reshape(tf.tile(unsupervised_labels_valid, [FLAGS.ensemble_size]), [FLAGS.ensemble_size, unsupervised_labels_valid.shape[0]]),
            logits_valid, from_logits=True)

          args['logits_valid'] = logits_valid
          args['log_likelihood_valid'] = log_likelihood_valid
          
        elif FLAGS.loss_fn == 'PAC2B-val-unsupervised2':
          # terms from validation [n_models, batch_size, n_classes]
          logits_valid = tf.stack([model[i](images_valid, training=True) 
                                   for i in range(FLAGS.ensemble_size)])

          # log_likelihood_valid = [n_models, batch_size, classes]
          log_likelihood_valid = []
          for i in range(logits_valid.shape[-1]):
            # [n_models, batch_size]
            y_true = tf.ones(logits_valid.shape[0:2]) * i
            ce = -tf.keras.losses.sparse_categorical_crossentropy(
              y_true = y_true,
              y_pred = logits_valid,
              from_logits = True
            )
            log_likelihood_valid.append(ce)
            
          # Shape [batch, classes]
          args['probs_valid'] = tf.reduce_mean(tf.nn.softmax(logits_valid), axis = 0)
          args['log_likelihood_valid'] = log_likelihood_valid
          
        loss = compute_loss(FLAGS.loss_fn, **args)

        # Scale the loss given the TPUStrategy will reduce sum all gradients.
        scaled_loss = loss / strategy.num_replicas_in_sync

      trainable_vars = []
      for i in range(FLAGS.ensemble_size):
        trainable_vars.extend(model[i].trainable_variables)

      grads = tape.gradient(scaled_loss, trainable_vars)
      optimizer.apply_gradients(zip(grads, trainable_vars))

      update_metrics(metrics_train, labels, logits, logits.shape[0])

      metrics_train['loss'].update_state(loss)

    # get the arguments of each step: validation data is optional
    args = [next(train_iterator),]
    if valid_iterator is not None: args.append(next(valid_iterator))

    #run an iteration
    strategy.run(step_fn, args)

  @tf.function
  def test_step(iterator, dataset_name):
    """Evaluation StepFn."""
    def step_fn(inputs):
      """Per-Replica StepFn."""
      images, labels = inputs
      # TODO(trandustin): Use more eval samples only on corrupted predictions;
      # it's expensive but a one-time compute if scheduled post-training.
      logits = tf.stack([model[i](images, training=False)
                         for i in range(FLAGS.ensemble_size)], axis=0)
      if FLAGS.use_bfloat16:
        logits = tf.cast(logits, tf.float32)

      if dataset_name == 'clean':
        metrics = metrics_test
      elif dataset_name == 'validation':
        metrics = metrics_validation
      else:
        metrics = corrupt_metrics[dataset_name]

      update_metrics(metrics, labels, logits, logits.shape[0])

    strategy.run(step_fn, args=(next(iterator),))

  train_iterator = iter(train_dataset)
  valid_iterator = iter(train_validation_dataset) if 'validation' in test_datasets else None

  start_time = time.time()
  for epoch in range(initial_epoch, FLAGS.train_epochs):
    logging.info('Starting to run epoch: %s', epoch)
    for step in range(steps_per_epoch):
      train_step(train_iterator, valid_iterator)

      current_step = epoch * steps_per_epoch + (step + 1)
      max_steps = steps_per_epoch * FLAGS.train_epochs
      time_elapsed = time.time() - start_time
      steps_per_sec = float(current_step) / time_elapsed
      eta_seconds = (max_steps - current_step) / steps_per_sec
      message = ('{:.1%} completion: epoch {:d}/{:d}. {:.1f} steps/s. '
                 'ETA: {:.0f} min. Time elapsed: {:.0f} min'.format(
                     current_step / max_steps,
                     epoch + 1,
                     FLAGS.train_epochs,
                     steps_per_sec,
                     eta_seconds / 60,
                     time_elapsed / 60))
      if step % 20 == 0:
        logging.info(message)

    datasets_to_evaluate = {'clean': test_datasets['clean']}
    if FLAGS.train_proportion < 1.0:
        datasets_to_evaluate['validation']= test_datasets['validation']

    if (FLAGS.corruptions_interval > 0 and
        (epoch + 1) % FLAGS.corruptions_interval == 0):
      datasets_to_evaluate = test_datasets
    for dataset_name, test_dataset in datasets_to_evaluate.items():
      test_iterator = iter(test_dataset)
      logging.info('Testing on dataset %s', dataset_name)
      if dataset_name=='validation':
          steps_per_eval = int((ds_info.splits['train'].num_examples *
                                (1. - FLAGS.train_proportion)) // batch_size)
      else:
          steps_per_eval = ds_info.splits['test'].num_examples // batch_size

      for step in range(steps_per_eval):
        if step % 20 == 0:
          logging.info('Starting to run eval step %s of epoch: %s', step,
                       epoch)
        test_start_time = time.time()
        test_step(test_iterator, dataset_name)
        ms_per_example = (time.time() - test_start_time) * 1e6 / batch_size
        #metrics_test['ms_per_example'].update_state(ms_per_example)

      logging.info('Done with testing on %s', dataset_name)

    corrupt_results = {}
    if (FLAGS.corruptions_interval > 0 and
        (epoch + 1) % FLAGS.corruptions_interval == 0):
      corrupt_results = aggregateMetrics(corrupt_metrics)

    logging.info('Train Loss: %.4f, Accuracy: %.2f%%, Variance: %.4f,',
                 metrics_train['ensemble_ce'].result(),
                 metrics_train['ensemble_accuracy'].result() * 100,
                 metrics_train['variance'].result())

    logging.info('Test NLL: %.4f, Accuracy: %.2f%%',
                 metrics_test['ensemble_ce'].result(),
                 metrics_test['ensemble_accuracy'].result() * 100)


    total_results = {'train/'+name: metric.result() for name, metric in metrics_train.items()}
    total_results.update({'test/'+name: metric.result() for name, metric in metrics_test.items()})
    total_results.update({'validation/'+name: metric.result() for name, metric in metrics_validation.items()})
    total_results.update(corrupt_results)

    if epoch == FLAGS.train_epochs-1 :
      flags_dict = FLAGS.flag_values_dict()
      flags_dict["method"] = "ensemble"
      res_to_csv(flags_dict, total_results, FLAG_hash)

    with summary_writer.as_default():
      for name, result in total_results.items():
        #logging.info(str(name)+': %.4f',result)
        tf.summary.scalar(name, result, step=epoch + 1)

    for metric in metrics_train.values():
      metric.reset_states()

    for metric in metrics_test.values():
      metric.reset_states()

    for metric in metrics_validation.values():
      metric.reset_states()

    if (FLAGS.checkpoint_interval > 0 and ((epoch + 1) % FLAGS.checkpoint_interval == 0 or epoch==FLAGS.train_epochs-1)):
        save_checkpoint(checkpoint, f'{FLAGS.output_dir}/checkpoints/param_optim/', FLAG_hash)


if __name__ == '__main__':
  app.run(main)
