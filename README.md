# EdVAE
Official implementation of "EdVAE: Mitigating Codebook Collapse with Evidential Discrete Variational Autoencoders".

## Installation
```
conda create --name edvae --file package-list.txt
conda activate edvae
```
## Data Preparation
Create a folder named `datasets`. Download [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and [LSUN Church](https://github.com/fyu/lsun) datasets explicitly, and put them to `datasets`.
## Repository Configuration

`configs`: Consists of EdVAE configuration files for each dataset.

`data`: Consists of the data loaders for the datasets.

`models`: Consists of the files related to the model such as the architecture and the quantizer etc.

`train.py`: Training file.

`utils.py`: Helper functions.

## Training

In order to train a model, run the following command:
```
python train.py --config_path configs/{dataset_name}.py
```
Example:
```
python train.py --config_path configs/cifar10.py
```
Each experiment's result be saved in a folder `{model_name}/{quantizer_name}/{lightning_data_module_name}/{experiment_date}`, e.g. `DVAE/DirichletQuantizer/CIFAR10Data/2024-09-03_17-18-38`. 
Each folder will consist of `ckpt`, `codes`, `imgs`, and `logs` folders. Checkpoints will be saved in `ckpt`, codes that are used in this experiment will be saved in `codes`, reconstruction results will be saved in `imgs`, and tensorboard logs will be saved in `logs`.

## Citation
If you use this code for your research, please cite:
```
@article{baykal2024edvae,
  title = {EdVAE: Mitigating codebook collapse with evidential discrete variational autoencoders},
  journal = {Pattern Recognition},
  volume = {156},
  pages = {110792},
  year = {2024},
  issn = {0031-3203},
  doi = {https://doi.org/10.1016/j.patcog.2024.110792},
  url = {https://www.sciencedirect.com/science/article/pii/S0031320324005430},
  author = {Gulcin Baykal and Melih Kandemir and Gozde Unal}
}
```
If you have any questions, please contact `baykalg@itu.edu.tr`.
