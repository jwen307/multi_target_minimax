#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module to evaluate the multi-label classifier

train_cnf.py
    - Script to train a conditional normalizing flow
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers,seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import os
from pathlib import Path
import traceback
import numpy as np

import sys
sys.path.append("../")

from datasets.fastmri_annotated_multi import FastMRIMultiLabel
from datasets.fastmri_multicoil import FastMRIDataModule
from util import helper
import variables


#Get the input arguments
#args = helper.flags()

#Get the checkpoint arguments if needed
load_ckpt_dir = '/storage/logs/MultiLabelClassifier/version_29'
load_last_ckpt = False


if __name__ == "__main__":
    
    #Use new configurations if not loading a pretrained model
    if load_ckpt_dir == 'None':
        model_type = 'MultiLabelClassifier'

        #Get the configurations
        configuration = helper.select_config(model_type)
        config = configuration.config
        ckpt=None
    
    #Load the previous configurations
    else:
        ckpt_name = 'last.ckpt' if load_last_ckpt else 'best_val_auroc.ckpt'
        ckpt = os.path.join(load_ckpt_dir,
                            'checkpoints',
                            ckpt_name)

        #Get the configuration file
        config_file = os.path.join(load_ckpt_dir, 'configs.pkl')
        config = helper.read_pickle(config_file)
        config['train_args']['epochs'] = 150
    

    try:
        # Get the directory of the dataset
        base_dir = variables.fastmri_paths[config['data_args']['mri_type']]

        # Get the model type
        model_type = 'MultiLabelClassifier'

        # Get the data
        if config['data_args']['challenge'] == 'multicoil':
            data = FastMRIDataModule(base_dir,
                                    batch_size=config['train_args']['batch_size'],
                                    num_data_loader_workers=4,
                                    **config['data_args'],
                                    )
        else:
            data = FastMRIMultiLabel(base_dir,
                                     batch_size=config['train_args']['batch_size'],
                                     num_data_loader_workers=8,
                                     #evaluating=True,
                                     **config['data_args'],
                                     )
        data.prepare_data()
        data.setup()



        #Load the model
        model = helper.load_model(model_type, config, ckpt)

        # Compile the model (Doesn't work if there's complex numbers like in fft2c)
        #model = torch.compile(model)

        # Evaluate the model
        model.test(data, load_ckpt_dir)


    except:

        traceback.print_exc()
       



    #%% Side thing to add labels to config files
    # Add the labels to the configuration
    # config['data_args']['labels'] = data.train.label_names
    # helper.write_pickle(config, config_file)

