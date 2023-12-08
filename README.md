# EdVAE
This code base is given as an EdVAE implementation using CIFAR10 dataset. All of the datasets and the methods will be shared later:

1. "config.yaml" file includes the configuration of an experiment.
2. "data" folder consists of the data loaders for the datasets. Currently, we only share the data loader for the CIFAR10 dataset.
3. "models" folder consists of the files related to the model such as the architecture and the quantizer etc. Currently, we only share the architecture and the quantizer for the EdVAE.
4. "train.py" file is the main file to run for the training.
5. "utils.py" file includes the helper functions.

In order to run an experiment, you can follow these steps:

1. First, you can create an environment using "package-list.txt" file to be able to run the codes.
2. In order to replicate a CIFAR10 experiment, you can use "python train.py --config_path config.yaml" command.
3. After that, you will see that two folders named as "DVAE" and "datasets" are created. "DVAE" folder will consist of other folders named after the used quantizer type. "datasets" folder will consist of the downloaded datasets. For this experiment, it is CIFAR10.
4. In our experiment, we use DirichletQuantizer of EdVAE. DirichletQuantizer will consist of other folders named after the used dataset.
5. As we use CIFAR10, we will have "CIFAR10Data" folder. In that folder, we will have a folder for each experiment that is named after the experiment date.
6. In each experiment folder, we will have the checkpoints, saved images during the training and validation, the codes that we ran, and the tensorboard logs.
7. You can view the results of each experiments using tensorboard.
