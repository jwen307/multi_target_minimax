import math
import numpy as np
import torch
import os
import fastmri
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from omegaconf import OmegaConf
import sys

sys.path.append("../")

from datasets.fastmri_multicoil import FastMRIDataModule
from util import helper
import variables
from util import network_utils
import pickle
import torchmetrics.functional as tmf
from sklearn.metrics import balanced_accuracy_score
import compute_tasks

def load_task_output(cf, experiment, accel, p, training=False, give_true_labels=False):
    # Store all the selected tasks in one dictionary
    task_outputs = {}

    # Load the task outputs for each task
    for task_num, task in cf.tasks.items():
        task_output_dir = os.path.join(experiment.experiment_dir, 'task_outputs', task.name)
        task_accel_dir = os.path.join(task_output_dir, f'accel_{accel}')



        # Load
        if training:
            # Check if the task outputs have already been computed
            if not os.path.exists(os.path.join(task_accel_dir, 'task_outputs_training_5000.pkl')):
                compute_tasks.compute_tasks(cf, train=True)

            with open(os.path.join(task_accel_dir, 'task_outputs_training_5000.pkl'), 'rb') as f:
                loaded_task_outputs = pickle.load(f)
        else:
            # Check if the task outputs have already been computed
            if not os.path.exists(os.path.join(task_accel_dir, 'task_outputs.pkl')):
                compute_tasks.compute_tasks(cf, train=False)


            with open(os.path.join(task_accel_dir, 'task_outputs.pkl'), 'rb') as f:
                loaded_task_outputs = pickle.load(f)

        # Store
        if task.name == 'classify':
            # Add each classification label separately
            for label_num, (label, values) in enumerate(loaded_task_outputs.items()):
                task_outputs[label] = values[p]

                # Make sure the shape of the ground truth is correct
                if len(task_outputs[label]['gt'].shape) > 1:
                    task_outputs[label]['gt'] = task_outputs[label]['gt'].flatten()

                if label_num+1 == cf.tasks.classify.num_classes:
                    break

            n = task_outputs[label]['gt'].shape[0]

            # Also get the true labels
            if give_true_labels:
                with open(os.path.join(task_output_dir, 'true_labels.pkl'), 'rb') as f:
                    true_labels = pickle.load(f)
                    true_labels = true_labels[:, :cf.tasks.classify.num_classes]
        else:
            task_outputs[task.name] = loaded_task_outputs[p]

            # Get the number of samples
            n = task_outputs[task.name]['gt'].shape[0]

            # Make sure the shape of the ground truth is correct
            if len(task_outputs[task.name]['gt'].shape) > 1:
                task_outputs[task.name]['gt'] = task_outputs[task.name]['gt'].flatten()

    if give_true_labels:
        return task_outputs, true_labels

    return task_outputs


# Get the correct experiment
def get_experiment(cf):
    if cf.data == 'mri_knee':
        return MRIKneeExperiment(cf)
    elif cf.data == 'ffhq':
        return FFHQExperiment(cf)
    else:
        raise ValueError('Data type not recognized')

# Make a class for each dataset following the same structure as this
class Experiment():
    def __init__(self, cf):
        self.cf = cf
        self.experiment_dir = None


    # Load the model and dataset
    def load_model_and_dataset(self, accel=None):
        pass

    # Get the reconstruction samples
    def get_posterior_samples(self, data_index):
        pass


# MRI Experiments
class MRIKneeExperiment(Experiment):
    def __init__(self, cf):
        super().__init__(cf)

        # Get the model type
        self.recon_type = self.cf.model.load_ckpt_dir.split('/')[-3]

        # Define the experiment directory
        self.experiment_dir = os.path.join(cf.log_folder, cf.data, self.recon_type)

    # Load the posterior model and dataset
    def load_model_and_dataset(self, accel=None):

        # Load the configuration for the CNF
        cnf_ckpt_name = 'best.ckpt'
        cnf_ckpt = os.path.join(self.cf.model.load_ckpt_dir,
                                'checkpoints',
                                cnf_ckpt_name)

        # Get the configuration file for the CNF
        cnf_config_file = os.path.join(self.cf.model.load_ckpt_dir, 'configs.pkl')
        cnf_config = helper.read_pickle(cnf_config_file)

        # Get the directory of the dataset
        base_dir = variables.fastmri_paths[cnf_config['data_args']['mri_type']]

        # Load the model
        self.recon_model = helper.load_model(self.recon_type, cnf_config, cnf_ckpt)
        self.recon_model.eval()
        self.recon_model.cuda()

        # Load the dataset
        self.data = FastMRIDataModule(base_dir,
                                 batch_size=cnf_config['train_args']['batch_size'],
                                 num_data_loader_workers=4,
                                 evaluating=True,
                                 specific_accel=accel,
                                 **cnf_config['data_args'],
                                 )
        self.data.prepare_data()
        self.data.setup()

        return self.recon_model, self.data


    # Get the reconstruction samples
    def get_posterior_samples(self, data_index, train=False):

        # Separate the data point
        if train:
            c, x, masks, norm_val, _, _, _ = self.data.train[data_index]
        else:
            c, x, masks, norm_val, _, _, _ = self.data.val[data_index]

        c = c.unsqueeze(0).to(self.recon_model.device)
        x = x.to(self.recon_model.device)
        if self.recon_type == 'VarNet':
            masks = masks.unsqueeze(0).unsqueeze(1).to(self.recon_model.device)
        else:
            masks = masks.to(self.recon_model.device)
        norm_val = norm_val.unsqueeze(0).to(self.recon_model.device)

        # Get the reconstructions
        with torch.no_grad():
            samples = self.recon_model.reconstruct(c,
                                                   num_samples=32, #TODO: Change the second value for larger c
                                                   temp=1.0,
                                                   check=True,
                                                   maps=None,
                                                   mask=masks,
                                                   norm_val=norm_val,
                                                   split_num=8,
                                                   multicoil=False,
                                                   rss=True)

        if self.recon_type == 'VarNet':
            recons = samples.to(self.recon_model.device).unsqueeze(0)
        else:
            recons = samples[0].to(self.recon_model.device)

        # Get the ground truth
        gt = fastmri.rss_complex(network_utils.format_multicoil(network_utils.unnormalize(x.unsqueeze(0), norm_val),
                                                                chans=False), dim=1).to(self.recon_model.device)


        return recons, gt


    # Load the recovery model
    def load_recovery_model(self):

        # Load recovery model if specified
        if 'recovery_load_ckpt_dir' not in self.cf.model.keys():
            self.recovery_model = None
            return None


        # Load the configuration for the recovery model
        recovery_ckpt_name = 'best.ckpt'
        recovery_ckpt = os.path.join(self.cf.model.recovery_load_ckpt_dir,
                                     'checkpoints',
                                     recovery_ckpt_name)

        # Get the configuration file for the recovery model
        recovery_config_file = os.path.join(self.cf.model.recovery_load_ckpt_dir, 'configs.pkl')
        recovery_config = helper.read_pickle(recovery_config_file)

        # Load the model
        self.recovery_type = self.cf.model.recovery_load_ckpt_dir.split('/')[-3]
        self.recovery_model = helper.load_model(self.recovery_type, recovery_config, recovery_ckpt)
        self.recovery_model.eval()
        self.recovery_model.cuda()

        return self.recovery_model

    # Get the recovery samples
    def get_recovery_sample(self, data_index, train=False):

        # Get recovery samples if specified
        if self.recovery_model is None:
            return None

        # Separate the data point
        if train:
            c, x, masks, norm_val, _, _, _ = self.data.train[data_index]
        else:
            c, x, masks, norm_val, _, _, _ = self.data.val[data_index]

        c = c.unsqueeze(0).to(self.recovery_model.device)
        x = x.to(self.recovery_model.device)
        if self.recovery_type == 'VarNet':
            masks = masks.unsqueeze(0).unsqueeze(1).to(self.recovery_model.device)
        else:
            masks = masks.to(self.recovery_model.device)
        norm_val = norm_val.unsqueeze(0).to(self.recovery_model.device)

        # Get the reconstructions
        with torch.no_grad():
            samples = self.recovery_model.reconstruct(c,
                                                     num_samples=32,
                                                     temp=1.0,
                                                     check=True,
                                                     maps=None,
                                                     mask=masks,
                                                     norm_val=norm_val,
                                                     split_num=8,
                                                     multicoil=False,
                                                     rss=True)

        if self.recovery_type == 'VarNet':
            recovery = samples.to(self.recon_model.device).unsqueeze(0)
        else:
            recovery = samples[0].to(self.recon_model.device)

        return recovery





def get_classification_results(predictions, true_labels, reduction='average'):
    '''
    Get the classification results
    :param predictions: np.array (num_samples, num_classes) with sigmoid values (between 0 and 1)
    :param true_labels: np.array (num_samples, num_classes) with the true labels
    :return:
    '''

    if not torch.is_tensor(predictions):
        predictions = torch.tensor(predictions)
    if not torch.is_tensor(true_labels):
        true_labels = torch.tensor(true_labels).int()

    num_classes = int(true_labels.shape[-1])

    classif_task = 'multilabel' if num_classes > 1 else 'binary'

    # Get the metrics
    auroc = tmf.auroc(predictions, true_labels, task=classif_task, num_labels=num_classes, average='none')
    precision = tmf.precision(predictions, true_labels, task=classif_task, num_labels=num_classes, average='none')
    recall = tmf.recall(predictions, true_labels, task=classif_task, num_labels=num_classes, average='none')
    accuracy = tmf.accuracy(predictions, true_labels, task=classif_task, num_labels=num_classes, average='none')
    f1 = tmf.f1_score(predictions, true_labels, task=classif_task, num_labels=num_classes, average='none')

    # Get the balanced accuracy
    # Convert to numpy arrays
    predictions = predictions.cpu().numpy()
    true_labels = true_labels.cpu().numpy()
    # Get the predicted labels (as 0 or 1)
    pred_labels = np.zeros_like(predictions)
    pred_labels[predictions >= 0.5] = 1
    pred_labels[predictions < 0.5] = 0

    balanced_accuracy = [balanced_accuracy_score(true_labels[:,i], pred_labels[:,i]) for i in range(num_classes)]



    # Apply the reduction
    if reduction == 'min':
        auroc = torch.min(auroc)
        precision = torch.min(precision)
        recall = torch.min(recall)
        accuracy = torch.min(accuracy)
        f1 = torch.min(f1)
        balanced_accuracy = np.min(balanced_accuracy)
    elif reduction == 'max':
        auroc = torch.max(auroc)
        precision = torch.max(precision)
        recall = torch.max(recall)
        accuracy = torch.max(accuracy)
        f1 = torch.max(f1)
        balanced_accuracy = np.max(balanced_accuracy)
    else:
        auroc = torch.mean(auroc)
        precision = torch.mean(precision)
        recall = torch.mean(recall)
        accuracy = torch.mean(accuracy)
        f1 = torch.mean(f1)
        balanced_accuracy = np.mean(balanced_accuracy)

    #print(auroc)

    #return auroc.item(), precision.item(), recall.item(), accuracy.item()
    return {'auroc': auroc.item(), 'precision': precision.item(), 'recall': recall.item(), 'accuracy': accuracy.item(), 'f1': f1.item(),
            'balanced_accuracy': balanced_accuracy}