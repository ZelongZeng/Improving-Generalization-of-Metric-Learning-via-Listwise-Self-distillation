# Official code repository for the paper "Improving Generalization of Metric Learning via Listwise Self-distillation" in Pytorch

---

This repository contains all code and implementations used in:

```
Improving Generalization of Metric Learning via Listwise Self-distillation
```

**Link**: https://arxiv.org/abs/2206.08880

---
## Some Notes:

If you use this code in your research, please cite
```
@article{zeng2022improving,
  title={Improving Generalization of Metric Learning via Listwise Self-distillation},
  author={Zeng, Zelong and Yang, Fan and Wang, Zheng and Satoh, Shin'ichi},
  journal={arXiv preprint arXiv:2206.08880},
  year={2022}
}
```

This code is adapted from a nice repository:

* https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch

---

## How to use this Repo

### Requirements:

* PyTorch 1.2.0+ & Faiss-Gpu
* Python 3.6+
* pretrainedmodels, torchvision 0.3.0+

An exemplary setup of a virtual environment containing everything needed:
```
(1) wget  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
(2) bash Miniconda3-latest-Linux-x86_64.sh (say yes to append path to bashrc)
(3) source .bashrc
(4) conda create -n DL python=3.6
(5) conda activate DL
(6) conda install matplotlib scipy scikit-learn scikit-image tqdm pandas pillow
(7) conda install pytorch torchvision faiss-gpu cudatoolkit=10.0 -c pytorch
(8) pip install wandb pretrainedmodels
(9) Run the scripts!
```

### Datasets:
Data for
* CUB200-2011 (http://www.vision.caltech.edu/visipedia/CUB-200.html)
* CARS196 (https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
* Stanford Online Products (http://cvgl.stanford.edu/projects/lifted_struct/)

can be downloaded either from the respective project sites or directly via Dropbox:

* CUB200-2011 (1.08 GB): https://www.dropbox.com/s/tjhf7fbxw5f9u0q/cub200.tar?dl=0
* CARS196 (1.86 GB): https://www.dropbox.com/s/zi2o92hzqekbmef/cars196.tar?dl=0
* SOP (2.84 GB): https://www.dropbox.com/s/fu8dgxulf10hns9/online_products.tar?dl=0

**The latter ensures that the folder structure is already consistent with this pipeline and the dataloaders**.   

Otherwise, please make sure that the datasets have the following internal structure:

* For CUB200-2011/CARS196:
```
cub200/cars196
└───images
|    └───001.Black_footed_Albatross
|           │   Black_Footed_Albatross_0001_796111
|           │   ...
|    ...
```

* For Stanford Online Products:
```
online_products
└───images
|    └───bicycle_final
|           │   111085122871_0.jpg
|    ...
|
└───Info_Files
|    │   bicycle.txt
|    │   ...
```

Assuming your folder is placed in e.g. `<$datapath/cub200>`, pass `$datapath` as input to `--source`.

### Training:
Training is done by using `main.py` and setting the respective flags, all of which are listed and explained in `parameters.py`. A vast set of exemplary runs is provided in `Revisit_Runs.sh`.

**[I.]** **A basic sample for baseline would like this**:

```
python main_original.py --loss triplet --batch_mining distance --log_online \
              --project DML_Project --group CUB_Triplet_distance --seed 0 \
              --gpu 0 --bs 112 --data_sampler class_random --samples_per_class 2 \
              --arch resnet50_frozen_normalize --source $datapath --dataset cub200 \
              --n_epochs 150 --lr 0.00001 --embed_dim 128 --evaluate_on_gpu
```

The purpose of each flag explained:

* `--loss <loss_name>`: Name of the training objective used. See folder `criteria` for implementations of these methods.
* `--batch_mining <batchminer_name>`: Name of the batch-miner to use (for tuple-based ranking methods). See folder `batch_mining` for implementations of these methods.
* `--log_online`: Log metrics online via either W&B (Default) or CometML. Regardless, plots, weights and parameters are all stored offline as well.
*  `--project`, `--group`: Project name as well as name of the run. Different seeds will be logged into the same `--group` online. The group as well as the used seed also define the local savename.
* `--seed`, `--gpu`, `--source`: Basic Parameters setting the training seed, the used GPU and the path to the parent folder containing the respective Datasets.
* `--arch`: The utilized backbone, e.g. ResNet50. You can append `_frozen` and `_normalize` to the name to ensure that BatchNorm layers are frozen and embeddings are normalized, respectively.
* `--data_sampler`, `--samples_per_class`: How to construct a batch. The default method, `class_random`, selects classes at random and places `<samples_per_class>` samples into the batch until the batch is filled.
* `--lr`, `--n_epochs`, `--bs` ,`--embed_dim`: Learning rate, number of training epochs, the batchsize and the embedding dimensionality.  
* `--evaluate_on_gpu`: If set, all metrics are computed using the gpu - requires Faiss-GPU and may need additional GPU memory.
* `--dataset cub200`: Dataset to use. Currently supported: cub200, cars196, online_products.

#### Some Notes:
* During training, metrics listed in `--evaluation_metrics` will be logged for both training and validation/test set. If you do not care about detailed training metric logging, simply set the flag `--no_train_metrics`. A checkpoint is saved for improvements in metrics listed in `--storage_metrics` on training, validation or test sets. Detailed information regarding the available metrics can be found at the bottom of this `README`.
* If one wishes to use a training/validation split, simply set `--use_tv_split` and `--tv_split_perc <train/val split percentage>`.


**[II.]** **How to plus our LSD?**:

```
python main.py --loss triplet_lsd --batch_mining distance --log_online \
              --project DML_Project --group CUB_Triplet_distance_lsd --seed 0 \
              --gpu 0 --bs 112 --data_sampler class_random --samples_per_class 2 \
              --arch resnet50_frozen_normalize --source $datapath --dataset cub200 \
              --n_epochs 150 --lr 0.00001 --embed_dim 128 --evaluate_on_gpu\
              --kd_alpha 50.0 --kd_tau 1.0
```

* `--loss <loss_name_of_lsd>`: Name of the training objective (applied lsd) used. See folder `criteria` for implementations of these methods.
* `--kd_alpha` and `--kd_tau`: The hyper-parameters for knowledge distillation.

**[II.]** **How to use noisy data?**:

```
... (basic command) --noisy_label --noisy_ratio 0.3

```

* `--noisy_label`:If set, using noisy data.
* `--noisy_ratio`: The noisy ratio.

### Evaluating Results with W&B
Here some information on using W&B (highly encouraged!)

* Create an account here (free): https://wandb.ai
* After the account is set, make sure to include your API key in `parameters.py` under `--wandb_key`.
* To make sure that W&B data can be stored, ensure to run `wandb on` in the folder pointed to by `--save_path`.
* When data is logged online to W&B, one can use `Result_Evaluations.py` to download all data, create named metric and correlation plots and output a summary in the form of a latex-ready table with mean and standard deviations of all metrics. **This ensures that there are no errors between computed and reported results.**
