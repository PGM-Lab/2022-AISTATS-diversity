# Diversity and Generalization in Neural Network Ensembles

This bundle contains all the code needed for replicaitng the experiments in the paper accepted at AISTATS 2022
and entitled "Diversity and Generalization in Neural Network Ensembles".


## Setup

For running the experiments **Python 3.7** or higher is required. Moreover, for running them locally, a requirements 
file is provided which can be installed as follows.

```
pip install -r requirements/local.txt
```

Alternatively, the code can be executed under a TPU environment. This can be done under Google Colab Prob, 
which requires to install the following dependencies:

```
!pip install --upgrade git+https://github.com/google/edward2.git@f36188e58bd6e0454f551fed6b29beded1b777ad --quiet
!pip install --upgrade git+https://github.com/google-research/robustness_metrics.git@5a380c4ec11fa85255b5db643c64efa42d749110 --quiet
!pip install fsspec --quiet
!pip install gcsfs --quiet
!pip uninstall tf-nightly -y
!pip uninstall tf-estimator-nightly -y
!pip install --upgrade tensorflow==2.4.1
!pip install --upgrade tensorflow-datasets==4.0.1
!pip install --upgrade tensorflow-estimator==2.4.0
!pip install --upgrade tensorflow-gcs-config==2.4.0
!pip install --upgrade tensorflow-hub==0.12.0
!pip install --upgrade tensorflow-metadata==0.30.0
!pip install --upgrade tensorflow-probability==0.12.1

```





## Running locally

The python scripts for learning the different ensembles proposed in the paper are:

- `baselines/cifar/p2b_ensemble.py`
- `baselines/wine_quality/p2b_ensemble_regression.py`


The input parameters of these python scripts are listed below.

- `--base_learning_rate `: Base learning rate when total training batch size is 128. Default is 0.1.
- `--beta `: Weight of the validation variance when using unsupervised loss function. Default is 1.
- `--checkpoint_interval`: Number of epochs between saving checkpoints. Use -1 to never save checkpoints. Default is 25.
- `--corruptions_interval `: Number of epochs between evaluating on the corrupted test data. Use -1 to never evaluate (default).
- `--data_dir `: GS data folder (eg., gs://tfds-data/datasets ). None for local execution.
- `--dataset `: Name of the dataset. Possible values: 'cifar10', 'cifar100', 'mnist', 'fashion_mnist'
- `--divide_l2_loss `: Whether to divide the l2 loss by the number of ensembles. Default is false
- `--ensemble_size `: Size of the ensemble. Default is 4.
- `--loss_fn `: Loss Function: PACB, PAC2B, PAC2B-val, PAC2B-val-unsupervised, PAC2B-Log, Ensemble-CE
- `--model `: NN considered. Possible values leNet5', 'ResNet20', 'ResNet50', 'wideResNet50','mlp_1LDense', "mlp_1LDense_Dropout", 'mlp_2LDense'.
- `--output_dir `: The directory where the model weights and training/evaluation summaries are stored. Default is '/tmp/cifar'.
- `--random_init`: Whether to random initialize each model of the ensemble. Default is True.
- `--seed`: random seed.
- `--train_epochs `: Number of training epochs. Default is 250.
- `--train_proportion `: Proportion that is used for training. Default is 1.
- `--use_gpu`: Whether to run on GPU or otherwise TPU.

As an example, the following command will (locally) learn the P2b-Ensemble-RI for leNet5 network using PACB as loss funciton:

```
env PYTHONPATH=$(pwd) python baselines/cifar/p2b_ensemble.py --random_init=True --dataset cifar10 --model leNet5 --loss_fn PACB --train_proportion 0.8  --ensemble_size 2 --train_epochs 2 --base_learning_rate 0.001 --checkpoint_interval -1 --corruptions_interval -1 --use_gpu True

```


## Running with TPUs (colab)

The same code can be executed with TPUs under Google Colab Pro. This requires to use the flag 
`--use_gpu False` and setting the flags `--data-dir`  and `--data-dir` to valid GS paths.

```
env PYTHONPATH=$(pwd) python baselines/cifar/p2b_ensemble.py --random_init=True --dataset cifar10 --model leNet5 --loss_fn PACB --train_proportion 0.8  --ensemble_size 2 --train_epochs 2 --base_learning_rate 0.001 --checkpoint_interval -1 --corruptions_interval -1 --use_gpu False --data_dir=gs://bucket-name/data/ --output_dir=gs:///bucket-name/output/
```
