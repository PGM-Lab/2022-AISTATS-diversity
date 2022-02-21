#!/bin/bash
#


#env $(!pwd)

env PYTHONPATH=$(pwd) python baselines/cifar/p2b_ensemble.py --train_proportion 0.8 --dataset cifar10 --model mlp_1LDense --loss_fn PACB --ensemble_size 2 --train_epochs 2 --base_learning_rate 0.001 --checkpoint_interval -1 --corruptions_interval -1 --use_gpu True