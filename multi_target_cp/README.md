# Minimax Multi-Target Conformal Prediction with Applications in Imaging Inverse Problems

## Description
This is the code for the paper [Minimax Multi-Target Conformal Prediction with Applications in Imaging Inverse Problems](https://openreview.net/forum?id=53FEYwDQK0).

## Installation
Please follow the instructions to setup the environment to run the repo.
1. Create a new environment with the following commands
```
conda create -n multitarget python=3.9 numpy=1.23 pip
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c conda-forge cupy cudatoolkit=11.8 cudnn cutensor nccl
conda install -c anaconda h5py=3.6.0
```
2. From the project root directory, install the requirements with the following command
```
pip install -r requirements.txt
```

# Overview
The implementation of all of the conformal methods can be found in **conformal_multi/conformal_utils_cp.py**.
The different conformal methods reported in the paper can be utilized by setting the configuration files to have the following parameters:
```
# For the IA method:
conformal_method : naive
normalization_type : none
alpha_correction_type : independent

# For the QN method:
conformal_method : joint_unified
normalization_type: length
alpha_correction_type: none

# For our proposed minimax method:
conformal_method: unified
normalization_type: quantile_score
alpha_correction_type: none

# For our proposed minimax method + QN:
conformal_method: joint_unified
normalization_type: qn_mini
alpha_correction_type: none
```
These are automatically adjusted in the proceeding scripts to run the experiments for all of the conformal methods reported in the paper.


# Toy Problem Experiments
The toy problem experiments can be easily run by using the following commands.

For the toy problem with a fixed number of training and tuning samples, run the following commands:
```
# Navigate to the conformal folder
cd conformal_multi

# Run the toy problem with fixed number of training and tuning samples
python toy_problem_many_trials_diff_alphas.py
```

For varying numbers of training samples, run the following commands:
```
python toy_problem_many_trials_diff_training_nums.py
```

For varying the number of tuning samples, run the following commands:
```
python toy_problem_many_trials_diff_tuning_many_tunes.py
```

The results will be saved in the `results/toy_problem/plots` folder. Note: to change whether the noise is correlated or independent,
change the independent variable to `True` or `False` in the script.


# fastMRI Experiments

## Using Precomputed Metric Values
To run our conformal methods without needing to download the fastMRI datasets or train the models, 
we include the precomputed task/metric targets and estimates under **results/mri_knee/MulticoilCNF/task_outputs**.
The quality metric estimates and targets are computed for a single E2EVarNet recovery compared against $c$ posterior samples from a CNF model.
The task estimates and targets are computed for $c$ posterior samples from the same CNF model.
To use these precomputed values, skip straight to the **Conformal Evaluation** section.
Otherwise, follow the instructions below to train a model and generate posterior samples.


## Usage Prerequisites
1. Download the fastMRI knee and brain datasets from [here](https://fastmri.org/)
2. Set the directory of the multicoil fastMRI knee and brain datasets to where they are stored on your device
    - Change [variables.py](variables.py) to set the paths to the dataset and your prefered logging folder
3. Change the configurations for training in the config files located in **train/configs/**. The current values are set to the ones used in the paper.


## Overview
- All model code used can be found in the **models** folder
- The training scripts for the reconstruction models can be found in the **train** folder


## Training Recovery Models
First, set the directory to the **train** folder
```
cd train
```

To train a model, modify the configuration file in **train/configs/** and run the following commands for the model you want to train.
```
# Training a CNF model
python train_cnf.py --model_type MulticoilCNF 

# Training an E2E Model
python train_varnet.py
```

All models will be saved in the logging folder specified in [variables.py](variables.py)

## Training Classifier Model
First, set the directory to the **classifier** folder
```
cd classifier
```
To train a model, modify the configuration file in **classifier/configs/** and run the following commands for the model you want to train.
```
# Training a SIMCLR model
python pretrain_classifier_simclr.py

# Training a classifier model (note, specify a  pretrained SIMCLR model if desired in the config file)
python train_multilabel_classifier.py
```



## Conformal Evaluation

To perform the Monte Carlo evaluation, change the config files in **conformal_multi/configs** to specify the location of the trained models, 
the type of conformal bound, the error rate, and all other parameters. 
The configs for the quality metric experiments are in **conformal_multi/configs/eval_config_mri_knee_cp.yaml**.
The configs for the task classification experiments are in **conformal_multi/configs/eval_config_mri_knee_classification_cp.yaml**.

Then, run the following commands
```
# Navigate to the conformal folder
cd conformal_multi

# Run the evaluation
python eval_many_trials.py --objective metric #Objective can be either 'metric' or 'classification'
```

Results are saved in 'results/plots/'.

To run the multi-round measurement protocol, set the 'accels' parameter in the config files to include $R=16,8,4,2$. Then,
run the following commands
```
# Navigate to the conformal folder
cd conformal_metrics

# Run the evaluation
python eval_multiround.py --objective metric #Objective can be either 'metric' or 'classification'
```



## Notes
- The first time using a dataset will invoke the preprocessing step required for compressing the coils. 
Subsequent runs will be much faster since this step can be skipped.

