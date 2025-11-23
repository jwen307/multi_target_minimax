import numpy as np
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
import sys

sys.path.append("../")

import pickle
import utils
import tasks

import os


def compute_tasks(cf, train=False, num_train_samples=5000):
    # Ensure the same reconstructions are created between different runs
    np.random.seed(0)
    torch.manual_seed(0)


    # Get the experiment setup
    experiment = utils.get_experiment(cf)

    # Go through all the accelerations if set
    for accel in cf.accels:

        # Check if the tasks have already been computed, if not, add them to the list
        compute_tasks_names = []
        compute_tasks_dirs = []
        for task_num, task in cf.tasks.items():
            task_output_dir = os.path.join(experiment.experiment_dir, 'task_outputs', task.name )
            task_accel_dir = os.path.join(task_output_dir, f'accel_{accel}')


            if not os.path.exists(task_accel_dir):
                os.makedirs(task_accel_dir, exist_ok=True)
                compute_tasks_names.append(task.name)
                compute_tasks_dirs.append(task_accel_dir)

        # If all tasks have already been computed, skip this acceleration
        if len(compute_tasks_names) == 0:
            continue

        # Load the model and dataset
        recon_model, data = experiment.load_model_and_dataset(accel)

        # Load the recovery model if needed
        recovery_model = experiment.load_recovery_model()


        # Prepare all the tasks that need to be computed (quality or classify)
        task_types = tasks.get_tasks(compute_tasks_names, cf)

        # Store all the computed tasks
        task_outputs = {}
        for name in compute_tasks_names:
            # For classification, add all the labels as separate tasks
            if name == 'classify':
                labels = task_types[-1].classifier.label_names

                for label in labels:
                    task_outputs[label] = {p: {'posteriors': [], 'gt': []} for p in cf.num_ps}

            # For other tasks, just add the task name
            else:
                task_outputs[name] = {p: {'posteriors': [], 'gt': []} for p in cf.num_ps}


        # Select the samples to compute the tasks on
        if train:
            num_train = len(data.train)
            indices = np.random.permutation(num_train)[:num_train_samples]
        else:
            num_val = len(data.val)
            indices = np.linspace(0, num_val, num_val, endpoint=False).astype(int)


        for i in tqdm(indices):

            # Get the reconstruction samples
            posteriors, gt = experiment.get_posterior_samples(i, train=train)

            # Get the recovery
            recovery = experiment.get_recovery_sample(i, train=train)

            # Cycle through the task types and compute the tasks for the posteriors and ground truth
            for task in task_types:

                # Cycle through the different amounts of averaging
                for p in cf.num_ps:

                    if recovery is not None:
                        reference = recovery[:p].mean(dim=0).unsqueeze(0)
                    else:
                        reference = None

                    task_output = task.compute_task_output(reference, posteriors, gt)

                    for name, output in task_output.items():
                        task_outputs[name][p]['posteriors'].append(output['posteriors'])
                        task_outputs[name][p]['gt'].append(output['gt'])


        # Save the task outputs
        for i, item in enumerate(task_outputs.items()):
            name, value = item
            for p in cf.num_ps:
                task_outputs[name][p]['posteriors'] = np.stack(task_outputs[name][p]['posteriors'], axis=0)
                task_outputs[name][p]['gt'] = np.stack(task_outputs[name][p]['gt'], axis=0)

        for i, name in enumerate(compute_tasks_names):
            # Save the task outputs for this acceleration
            task_output_dir = compute_tasks_dirs[i]
            filename = f'task_outputs_training_{num_train_samples}.pkl' if train else 'task_outputs.pkl'
            with open(os.path.join(task_output_dir, filename), 'wb') as f:

                # For the classification type, save all labels into same file, remove metrics from the dictionary
                if name == 'classify':
                    classify_task_outputs = {}
                    for label, output in task_outputs.items():
                        if 'metric' not in label:
                            classify_task_outputs[label] = output

                    pickle.dump(classify_task_outputs, f)

                else:
                    pickle.dump(task_outputs[name], f)



