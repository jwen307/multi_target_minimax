'''
This script evaluates the conformal multi-round method but treats each acceleration rate as a separate task.
There is only one calibration phase for all acceleration rates, but the test phase is done separately for each acceleration rate.
This one does not compute the classification results, so can be used for any task.
'''

import argparse
import numpy as np
import torch
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from omegaconf import OmegaConf
import sys

sys.path.append("../")

import pickle
import utils
import conformal_utils_cp as conformal


parser = argparse.ArgumentParser()

parser.add_argument(
    '--objective',
    type=str,
    default='metric',
    help='Type of objective. Options: classification, metric'
)

args = parser.parse_args()

if args.objective == 'metric':
    data_type = 'mri_knee'
elif args.objective == 'classification':
    data_type = 'mri_knee_classification'
else:
    raise ValueError('Objective must be either classification or metric')

# Get the configurations
cf = OmegaConf.load(f'configs/eval_config_{data_type}_cp.yaml')

# Ensure the same reconstructions are used between different runs
np.random.seed(0)
torch.manual_seed(0)

alphas = cf.alpha

methods = ['independent',  'length', 'unified']
configs = {'independent': {'conformal_method': 'naive',
                           'normalization_type': 'none',
                           'alpha_correction_type': 'independent',},
           'length': {'conformal_method': 'joint_unified',
                      'normalization_type': 'length',
                      'alpha_correction_type': 'none',},
            'unified': {'conformal_method': 'unified',
                        'normalization_type': 'quantile_score',
                        'alpha_correction_type': 'none'}}


# Remember the original calibration split
og_calib_split = cf.eval.calib_split


if __name__ == "__main__":

    for method in methods:

        cf.conformal_method = configs[method]['conformal_method']
        cf.normalization_type = configs[method]['normalization_type']
        cf.alpha_correction_type = configs[method]['alpha_correction_type']

        # Get the experiment setup
        experiment = utils.get_experiment(cf)

        # Get a new experiment save directory based on new number
        experiments_dir = os.path.join(experiment.experiment_dir, 'experiments')
        if not os.path.exists(experiments_dir):
            os.makedirs(experiments_dir)


        for alpha in alphas:
            cf.alpha = alpha

            # Ensure the same reconstructions and training indices are used between different runs with different alphas
            np.random.seed(0)
            torch.manual_seed(0)

            # Make the directory for the experiment
            conformal_method = cf.conformal_method
            normalization_type = cf.normalization_type
            experiment_save_dir = os.path.join(experiments_dir, f'{data_type}_cp', 'multiround_multi_accel_calib',
                                               f'{conformal_method}_{normalization_type}',
                                               f'alpha_{cf.alpha}', f'tau_{cf.tau}')
            if not os.path.exists(experiment_save_dir):
                os.makedirs(experiment_save_dir)

            # Save the configuration
            with open(os.path.join(experiment_save_dir, 'config.yaml'), 'w') as f:
                OmegaConf.save(cf, f)


            # Prep the tuning data (indices) and separate from calibration/test dadta
            # Get the tasks outputs
            task_outputs = utils.load_task_output(cf, experiment, cf.accels[0], cf.num_ps[0])
            # Get the number of samples
            n = task_outputs[list(task_outputs.keys())[0]]['gt'].shape[0]
            rand_indices = np.random.permutation(n)
            if cf.normalization_type == 'quantile_score':
                n_train = int(n * cf.eval.train_split)
                train_indices = rand_indices[:n_train]
                cal_test_indices = rand_indices[n_train:]
            else:
                n_train = 0
                # Change the calib_split so the number of test samples is the same whether or not training samples are used
                # Only change this once
                if cf.eval.calib_split == og_calib_split:
                    cf.eval.calib_split = 1 - (1 - cf.eval.train_split) * (1 - cf.eval.calib_split)

            # Get a list of all the tasks
            task_list = []
            for task in cf.tasks:
                if task == 'classify':
                    label_names = list(task_outputs.keys())
                    task_list = task_list + label_names
                    # Add
                else:
                    task_list.append(task)

            # Make a separate task for each task at each acceleration
            accel_task_list = []
            for accel in cf.accels:
                for task in task_list:
                    accel_task_list.append(f'{task}_{accel}')



            # Perform the conformal multiround
            print('Testing the conformal multiround...')

            # Store the results for each trial to average later
            all_coverages = []
            all_avg_accels = []
            all_num_accepted = []



            # Combine the task outputs for each acceleration
            task_outputs_all = {}
            for accel in cf.accels:
                # Get the tasks outputs
                task_outputs = utils.load_task_output(cf, experiment, accel, cf.num_ps[0])

                # Combine the task outputs for each acceleration
                for task in task_list:
                    # task_outputs_all[f'{task}_{accel}']['posteriors'] = task_outputs[task]['posteriors']
                    # task_outputs_all[f'{task}_{accel}']['gt'] = task_outputs[task]['gt']
                    task_outputs_all[f'{task}_{accel}'] = {'posteriors': task_outputs[task]['posteriors'],
                                                            'gt': task_outputs[task]['gt']}



            # Perform the multi-round experiment for many trials
            for t in tqdm(range(cf.eval.num_trials)):

                # Get the number of samples
                n_cal_test = n - n_train
                n_cal = int(n_cal_test * cf.eval.calib_split) # Number of calibration samples
                # Get the indices for the calibration and test sets (Keep same for all accelerations)
                indices = np.random.permutation(n_cal_test)
                cal_indices = indices[:n_cal]
                test_indices = indices[n_cal:]
                total_test = n_cal_test - n_cal


                # Keep track of the number of samples accepted for each acceleration
                num_accepted = [0 for _ in range(len(cf.accels) + 1)]  # Number of samples accepted for each acceleration
                accepted_coverages = [0 for _ in range(len(cf.accels))]
                slice_accels = []
                num_in_interval = 0

                # Store the predictions at acceptance
                min_pred_at_acceptance = []
                max_pred_at_acceptance = []
                median_pred_at_acceptance = []
                gt_pred_at_acceptance = [] # Store the gt prediction (accel=1) in the ordering of accepatnce
                true_labels_reorder = [] # Reorder the true labels to match the order of the predictions


                # %% Get the data
                # Get the training and calibration data

                # Get the training task outputs (same indices for every iteration)
                if cf.normalization_type == 'quantile_score':
                    task_outputs_training = {task: {'posteriors': task_outputs_all[task]['posteriors'][train_indices],
                                                    'gt': task_outputs_all[task]['gt'][train_indices]} for task in
                                             accel_task_list}
                    # Get the calibration and test data
                    cal_test_task_outputs = {task: {'posteriors': task_outputs_all[task]['posteriors'][cal_test_indices],
                                           'gt': task_outputs_all[task]['gt'][cal_test_indices]} for task in accel_task_list}


                else:
                    task_outputs_training = None

                    # Get the calibration and test data
                    cal_test_task_outputs = {task: {'posteriors': task_outputs_all[task]['posteriors'][indices],
                                                    'gt': task_outputs_all[task]['gt'][indices]} for task in
                                             accel_task_list}


                # Get the calibration and test data
                cal_data = {task: {'posteriors': cal_test_task_outputs[task]['posteriors'][cal_indices],
                                   'gt': cal_test_task_outputs[task]['gt'][cal_indices]} for task in accel_task_list}

                # %% Create and calibrate the conformal object
                # Create an conformal object
                cinf = conformal.select_conformal(cf, cal_data, task_outputs_training, joint_accels=cf.accels)

                # Calibrate
                cinf.calibrate(c=cf.num_cs[0])


                #%% Perform multi-round going through all the accelerations
                # Go through all the accelerations if set
                for a, accel in enumerate(cf.accels):
                    #print(f'Accel: {accel}')

                    # Get the directory to save the results
                    save_dir = experiment_save_dir
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)


                    # Get the remaining test data
                    test_data = {task: {'posteriors': cal_test_task_outputs[task]['posteriors'][test_indices],
                                        'gt': cal_test_task_outputs[task]['gt'][test_indices]} for task in accel_task_list}



                    # Test
                    # Only use the acceleration task for the test
                    test_task_list = [f'{task}_{accel}' for task in task_list]
                    outputs = cinf.test_certain_tasks(test_data, c=cf.num_cs[0], test_task_list = test_task_list)

                    #%% Evaluate if bound size is below threshold
                    accepted_all_task = [] # Store if the test sample is accepted for all tasks

                    for task_num, task in enumerate(task_list):

                        # Joint unified only has one conformal interval object and is always two sided
                        if cf.conformal_method == 'joint_unified':
                            bound_type = 'two_sided'
                        else:
                            bound_type = cinf.conformal_intervals[task_num].bound_type

                        # For metric tasks, want to use the upper or lower bound as the interval measurement
                        if 'metric' in task:
                            if 'psnr' in task or 'ssim' in task:
                                interval_measurement = outputs[f'{task}_{accel}']['intervals'][:, 0]
                                bound_type = 'bound_lower'

                            elif 'lpips' in task or 'dists' in task:
                                interval_measurement = outputs[f'{task}_{accel}']['intervals'][:, 1]
                                bound_type = 'bound_upper'

                        else:
                            # Note: interval_measurement is either interval size or upper or lower bound
                            interval_measurement = outputs[f'{task}_{accel}']['interval_measurement']

                        # Check if the interval measurement is below the threshold
                        # Note: bound type will be the same for the task regardless of the acceleration
                        if bound_type == 'bound_upper' or bound_type == 'two_sided':
                            accepted = interval_measurement <= cf.tau
                        elif bound_type == 'bound_lower':
                            accepted = interval_measurement >= cf.tau
                        else:
                            raise ValueError('Invalid bound type')

                        # Store if the test sample is accepted for this task
                        accepted_all_task.append(accepted)

                    # Get the samples that would have all tasks accepted
                    accepted_all_task = np.all(np.stack(accepted_all_task, axis=0),axis=0)
                    # Get the number of accepted samples at this acceleration
                    num_accepted[a] = np.sum(accepted_all_task)

                    if any(accepted_all_task):

                        # Get whether the intervals are valid at the accepted locations
                        joint_valid = outputs['joint_valid']
                        joint_valid_at_accepted = joint_valid[accepted_all_task]
                        joint_coverage = np.mean(joint_valid_at_accepted)

                        # Keep track of the number of samples that were jointly valid at acceptance
                        num_in_interval += np.sum(joint_valid_at_accepted)

                        # Add the slice accelerations to the list for looking at histogram later
                        slice_accels += [accel for _ in range(num_accepted[a])]


                    # Only keep the test indices that were not accepted
                    test_indices = test_indices[~accepted_all_task]

                    # Break if all test samples have been accepted
                    if test_indices.shape[0] == 0:
                        break


                # Include the fully sampled images if threshold never went below tau
                if test_indices.shape[0] > 0:
                    num_accepted[-1] = len(test_indices)
                    num_in_interval += len(test_indices)
                    slice_accels += [1 for _ in range(len(test_indices))]

                    # Get the remaining test data for acceleration 1
                    test_data = {task: {'posteriors': cal_test_task_outputs[task]['posteriors'][test_indices],
                                        'gt': cal_test_task_outputs[task]['gt'][test_indices]} for task in accel_task_list}






                # Make sure the number of samples accepted is the same as the total number of samples
                assert np.sum(num_accepted) == total_test

                # Compute the empirical joint coverage when each slice was accepted
                coverage = num_in_interval / total_test
                all_coverages.append(coverage)

                # Get the average acceleration
                avg_accel = 1/np.mean(1/np.array(slice_accels))
                all_avg_accels.append(avg_accel)

                # Get the number of samples accepted at each acceleration rate
                all_num_accepted.append(num_accepted)


            #%% Average over all the trials and save the results
            print(f'Average Coverage: {np.mean(all_coverages)} +/- {np.std(all_coverages)/np.sqrt(cf.eval.num_trials)}')
            print(f'Average Acceleration: {np.mean(all_avg_accels)} +/- {np.std(all_avg_accels)/np.sqrt(cf.eval.num_trials)}')

            # Get the average number of samples accepted at each acceleration rate
            all_percent_accepted = np.array(all_num_accepted) / total_test
            mean_percent_accepted = np.mean(all_percent_accepted, axis=0)
            std_error_percent_accepted = np.std(all_percent_accepted, axis=0)/np.sqrt(cf.eval.num_trials)
            std_dev_percent_accepted = np.std(all_percent_accepted, axis=0)
            print('Average Number of Samples Accepted: ', mean_percent_accepted)
            print('Standard Error of Number of Samples Accepted: ', std_error_percent_accepted)
            print('Standard Deviation of Number of Samples Accepted: ', std_dev_percent_accepted)



            #%% Save the results
            with open(os.path.join(experiment_save_dir, 'avg_coverage.txt'), 'w') as f:
                f.write(f'Average Coverage: {np.mean(all_coverages)} +/- {np.std(all_coverages)/np.sqrt(cf.eval.num_trials)}')
            with open(os.path.join(experiment_save_dir, 'avg_accel.txt'), 'w') as f:
                f.write(f'Average Acceleration: {np.mean(all_avg_accels)} +/- {np.std(all_avg_accels)/np.sqrt(cf.eval.num_trials)}')
            with open(os.path.join(experiment_save_dir, 'avg_num_accepted.pkl'), 'wb') as f:
                pickle.dump({'mean': mean_percent_accepted, 'std_error': std_error_percent_accepted, 'std_dev': std_dev_percent_accepted}, f)





    #%% Plot the results
    experiment_load_dir = os.path.join(experiments_dir, f'{data_type}_cp','multiround_multi_accel_calib')

    folder = 'metrics' if data_type == 'mri_knee_cp' else 'classif'
    save_dir = os.path.join(f'../results/plots/', folder, 'multiround_multi_accel_calib/')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    round_types = [ 'multiround_multi_accel_calib', 'multiround_multi_accel_calib']
    round_type_names = ['IA', 'Minimax']
    configs = ['naive_none', 'unified_quantile_score']
    colors = [ 'blue', 'green']
    markers = ['x', 'o', '>']
    #alphas = cf.alpha #[0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
    tau = cf.tau

    coverages = {}
    avg_accels = {}

    for i, round_type in enumerate(round_types):

        coverages[round_type_names[i]] = []
        avg_accels[round_type_names[i]] = []

        for alpha in alphas:
            # Get the directory to load the results
            load_dir = os.path.join(experiment_load_dir, configs[i], f'alpha_{alpha}', f'tau_{tau}')

            # Load the coverage results
            with open(os.path.join(load_dir, 'avg_coverage.txt'), 'r') as f:
                lines = f.readlines()
            for line in lines:
                if '+/-' in line:
                    metric, value = line.split('Coverage:')
                    value = value.split('+/-')[0].strip()
                    coverage = float(value)
                    coverages[round_type_names[i]].append(coverage)

            # Load the average acceleration results
            with open(os.path.join(load_dir, 'avg_accel.txt'), 'r') as f:
                lines = f.readlines()
            for line in lines:
                if '+/-' in line:
                    metric, value = line.split('Acceleration:')
                    value = value.split('+/-')[0].strip()
                    avg_accel = float(value)
                    avg_accels[round_type_names[i]].append(avg_accel)

    # Plot the coverages against 1-alphas
    plt.figure()
    for i, round_type in enumerate(round_types):
        plt.plot(1 - np.array(alphas), coverages[round_type_names[i]], color=colors[i], marker=markers[i],
                 label=round_type_names[i],
                 markersize=10, linewidth=2)

    # Plot line for desired coverage
    plt.plot(1 - np.array(alphas), 1 - np.array(alphas), linestyle='--', color='k', label='Desired Coverage',
             markersize=10, linewidth=2)

    # plt.xlabel('Desired Coverage')
    # plt.ylabel('Avg Accepted Coverage')
    # plt.ylim(0.7, 1.0)
    plt.ylim(top=1.0)
    # plt.legend()
    plt.grid(alpha=0.5)
    # plt.show()

    if not os.path.exists(os.path.join(save_dir, 'multiround')):
        os.makedirs(os.path.join(save_dir, 'multiround'))
    #
    plt.savefig(os.path.join(save_dir, 'multiround', 'acceptedcoverage_dists.pdf'), dpi=1200)
    plt.close()

    # Plot the average acceleration
    plt.figure()
    for i, round_type in enumerate(round_types):
        plt.plot(1 - np.array(alphas), avg_accels[round_type_names[i]], color=colors[i], marker=markers[i],
                 label=round_type_names[i],
                 markersize=10, linewidth=2)
    # plt.xlabel('Desired Coverage')
    # plt.ylabel('Avg Acceleration')
    plt.ylim(2, 5)
    plt.grid(alpha=0.5)
    # plt.legend()
    # plt.show()
    plt.savefig(os.path.join(save_dir, 'multiround', 'avg_accel_dists.pdf'), dpi=1200)
    plt.close()











