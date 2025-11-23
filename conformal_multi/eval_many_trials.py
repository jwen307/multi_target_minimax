'''
eval_many_trials.py
- Script to evaluate the multi-target conformal prediction methods
'''
import math
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import sys
import argparse

sys.path.append("../")

import pickle
import utils
import conformal_utils_cp as conformal
from compute_tasks import compute_tasks

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

# Remember all the alphas
alphas = cf.alpha

# For the minimax method, you can perform multiple runs since the tuning data can vary
n_runs_desired = 1


# Defines the multi-target conformal prediction methods
methods = ['independent', 'length', 'unified'] # Notes: independent is IA, length is QN, and unified is minimax from the paper
configs = {'independent': {'conformal_method': 'naive',
                           'normalization_type': 'none',
                           'alpha_correction_type': 'independent',},
           'length': {'conformal_method': 'joint_unified',
                      'normalization_type': 'length',
                      'alpha_correction_type': 'none',},
            'unified': {'conformal_method': 'unified',
                        'normalization_type': 'quantile_score',
                        'alpha_correction_type': 'none'}}


if __name__ == "__main__":

    # Get the experiment setup
    experiment = utils.get_experiment(cf)

    # Get a new experiment save directory
    experiments_dir = os.path.join(experiment.experiment_dir, 'experiments')
    if not os.path.exists(experiments_dir):
        os.makedirs(experiments_dir)

    # Remember the original calibration split
    og_calib_split = cf.eval.calib_split


    # Get a list of all the tasks
    # Get the tasks outputs (tasks are computed if they don't already exist)
    task_outputs = utils.load_task_output(cf, experiment, accel=cf.accels[0], p= cf.num_ps[0])
    task_list = []
    for task in cf.tasks:
        if task == 'classify':
            label_names = list(task_outputs.keys())
            task_list = task_list + label_names
            # Add
        else:
            task_list.append(task)

    for method in methods:

        cf.conformal_method = configs[method]['conformal_method']
        cf.normalization_type = configs[method]['normalization_type']
        cf.alpha_correction_type = configs[method]['alpha_correction_type']


        # Make the directory for the experiment
        conformal_method = cf.conformal_method
        normalization_type = cf.normalization_type
        experiment_save_dir = os.path.join(experiments_dir, f'{data_type}_cp',
                                           f'{conformal_method}_{normalization_type}')
        if not os.path.exists(experiment_save_dir):
            os.makedirs(experiment_save_dir)

        # Figure out the number of runs to average over if it is the quantile score
        n_runs = n_runs_desired if cf.normalization_type == 'quantile_score' else 1

        # Save the configuration
        with open(os.path.join(experiment_save_dir, 'config.yaml'), 'w') as f:
            OmegaConf.save(cf, f)

        #Go through multiple alphas
        for alpha in alphas:
            cf.alpha = alpha

            # Ensure the same reconstructions and training indices are used between different runs with different alphas
            np.random.seed(0)
            torch.manual_seed(0)

            # Go through all the accelerations if set
            for accel in cf.accels:

                # Go through different amounts of averaging
                for p in cf.num_ps:


                    for c in cf.num_cs:
                        print(f'alpha: {cf.alpha}, Accel: {accel}, p: {p}, c: {c}')

                        # Get the directory to save the results
                        save_dir = os.path.join(experiment_save_dir, f'alpha_{cf.alpha}', f'accel_{accel}', f'p_{p}', f'c_{c}')
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)

                        all_runs_results = {'coverage': {f'{task}': [] for task in task_list},
                                            'interval_measurement': {f'{task}': [] for task in task_list},
                                            'max_abs_diff': {f'{task}': [] for task in task_list},
                                            'joint_coverage': []}

                        for _ in range(n_runs):

                            # Get the tasks outputs
                            task_outputs = utils.load_task_output(cf, experiment, accel, p)

                            # Get the training task outputs
                            #task_outputs_training = utils.load_task_output(cf, experiment, accel, p, training=True)
                            # Get the training task outputs if needed
                            #if cf.normalization_type != 'none':
                            if cf.normalization_type == 'quantile_score':
                                # Split the data into training and calibration/test
                                task_outputs_training, task_outputs = conformal.split_training_task_outputs(cf, task_outputs)
                                cf.eval_calib_split = og_calib_split * 1.0
                            else:
                                task_outputs_training = None
                                # Change the calib_split so the number of test samples is the same whether or not training samples are used
                                # Only change this once
                                if cf.eval.calib_split == og_calib_split:
                                    cf.eval.calib_split = 1 - (1 - cf.eval.train_split) * (1 - cf.eval.calib_split)


                            # Create a conformal object
                            cinf = conformal.select_conformal(cf, task_outputs, task_outputs_training)

                            # Perform many trials of the conformal experiment
                            outputs = cinf.many_trials(cf, task_outputs, c)

                            #%% Save and plot the results
                            all_task_coverages = []

                            # Get the distribution of risk for each task
                            for k, task in enumerate(cinf.task_list):

                                # Decide if you should write or append files
                                over_write = 'w' if k == 0 else 'a'

                                coverage = outputs[task]['coverage']
                                coverage_mean = np.mean(coverage, axis=0)
                                coverage_std_err = np.std(coverage, axis=0) / math.sqrt(len(coverage))
                                all_runs_results['coverage'][f'{task}'].append(coverage_mean)

                                # Get the mean interval measurement
                                interval_measurement = outputs[task]['interval_measurement']
                                interval_measurement_mean = np.mean(np.mean(interval_measurement, axis=1))
                                all_runs_results['interval_measurement'][f'{task}'].append(interval_measurement_mean.item())

                                # Get the max absolute difference
                                max_abs_diff = outputs[task]['max_abs_diff']
                                max_abs_diff_mean = np.mean(np.mean(max_abs_diff, axis=1)) # Over all test samples and then over all trials
                                all_runs_results['max_abs_diff'][f'{task}'].append(max_abs_diff_mean.item())


                                # Get the lambda hat values
                                lambda_hats = outputs[task]['lambda_hat']
                                lambda_hats_mean = np.mean(lambda_hats)
                                lambda_hats_std_err = np.std(lambda_hats) / math.sqrt(len(lambda_hats))

                                # Store the mean coverage for each task
                                all_task_coverages.append(coverage_mean)

                                # Remove any '/' in the task name for saving
                                task = task.replace('/', '_')

                                # Save the results in txt file
                                with open(os.path.join(save_dir, f'coverage.txt'), over_write) as f:
                                    f.write(f'{task} risk mean: {coverage_mean:.4f} +/- {coverage_std_err:.4f}\n')
                                #print(f'{task} risk mean: {risk_mean:.4f}, std: {risk_std:.4f}')

                                # Plot the results
                                plt.figure()
                                plt.hist(coverage, bins=25, density=True)
                                # Mark where the alpha is at
                                plt.axvline(1-cf.alpha, color='r', linestyle='--', label='Desired')
                                plt.axvline(coverage_mean, color='b', linestyle='--', label='Mean')
                                plt.xlabel('Coverage')
                                plt.ylabel('Density')
                                plt.title(f'{task} risk')
                                plt.legend()
                                #plt.show()
                                plt.savefig(os.path.join(save_dir, f'{task}_coverage_density.png'))
                                plt.close()


                                #print(f'{task} interval measurement mean: {interval_measurement_mean.item():.4f}')
                                with open(os.path.join(save_dir, f'interval_measurement.txt'), over_write) as f:
                                    f.write(f'{task} interval measurement mean: {interval_measurement_mean.item():.4f}\n')

                                # Plot the distribution of interval measurements
                                plt.figure()
                                plt.hist(np.mean(interval_measurement,axis=1), bins=25, density=True)
                                plt.axvline(interval_measurement_mean, color='b', linestyle='--', label='Mean')
                                plt.xlabel('Interval Measurement')
                                plt.ylabel('Density')
                                plt.title(f'{task} interval measurement')
                                plt.legend()
                                #plt.show()
                                plt.savefig(os.path.join(save_dir, f'{task}_interval_measurement_density.png'))
                                plt.close()

                                with open(os.path.join(save_dir, f'max_abs_diff.txt'), over_write) as f:
                                    f.write(f'{task} max abs diff mean: {max_abs_diff_mean.item():.4f}\n')


                                # Save the lambda hat values too
                                with open(os.path.join(save_dir, f'lambda_hat.txt'), over_write) as f:
                                    f.write(f'{task} lambda hat mean: {lambda_hats_mean:.4f} +/- {lambda_hats_std_err:.4f}\n')


                            # Get the joint coverage
                            joint_coverage = outputs['joint_coverage']
                            joint_coverage_mean = np.mean(joint_coverage)
                            joint_coverage_std_err = np.std(joint_coverage) / math.sqrt(len(joint_coverage))
                            all_runs_results['joint_coverage'].append(joint_coverage_mean)
                            print(f'Joint coverage: {joint_coverage_mean:.4f} +/- {joint_coverage_std_err:.4f}')
                            with open(os.path.join(save_dir, 'joint_coverage.txt'), 'w') as f:
                                f.write(f'Joint coverage: {joint_coverage_mean:.4f} +/- {joint_coverage_std_err:.4f}')

                            # Save the outputs
                            with open(os.path.join(save_dir, 'results.pkl'), 'wb') as f:
                                pickle.dump(outputs, f)


                            # Get the balance coverage metrics
                            all_task_coverages = np.array(all_task_coverages)
                            # Max difference between coverages and mean coverage
                            max_diff = np.max(np.abs(all_task_coverages - np.mean(all_task_coverages)))
                            # Range of coverages
                            coverage_range = np.max(all_task_coverages) - np.min(all_task_coverages)
                            # Standard deviation of coverages
                            coverage_std = np.std(all_task_coverages)
                            # Mean coverage
                            mean_coverage = np.mean(all_task_coverages)

                            print(f'Max Coverage: {np.max(all_task_coverages):.4f}')

                            # Save the max coverage
                            with open(os.path.join(save_dir, 'max_coverage.txt'), 'w') as f:
                                f.write(f'Max coverage: {np.max(all_task_coverages):.4f}\n')

                            # Save the balance coverage metrics
                            with open(os.path.join(save_dir, 'balance_coverage_metrics.txt'), 'w') as f:
                                f.write(f'Max difference: {max_diff:.4f}\n')
                                f.write(f'Coverage range: {coverage_range:.4f}\n')
                                f.write(f'Coverage std: {coverage_std:.4f}\n')
                                f.write(f'Mean coverage: {mean_coverage:.4f}\n')



                        # Average the results over the number of runs if needed
                        if n_runs > 1:
                            # Average the coverage and get the standard deviation
                            with open(os.path.join(save_dir, f'coverage.txt'),'w') as f:
                                for task in all_runs_results['coverage']:
                                    coverage = np.mean(np.array(all_runs_results['coverage'][task]), axis=0)
                                    coverage_std = np.std(np.array(all_runs_results['coverage'][task]), axis=0)
                                    f.write(f'{task} risk mean: {coverage:.4f} +/- {coverage_std:.4f}\n')

                            # Average the mean interval length and get the standard deviation
                            with open(os.path.join(save_dir, f'interval_measurement_many_tunes.txt'),'w') as f:
                                for task in all_runs_results['interval_measurement']:
                                    interval_measurement = np.mean(np.array(all_runs_results['interval_measurement'][task]), axis=0)
                                    interval_measurement_std = np.std(np.array(all_runs_results['interval_measurement'][task]), axis=0)
                                    f.write(f'{task} interval measurement mean: {interval_measurement:.4f} +/- {interval_measurement_std:.4f}\n')


                            # Average the joint coverage and get the standard deviation
                            joint_coverage = np.mean(np.array(all_runs_results['joint_coverage']), axis=0)
                            joint_coverage_std = np.std(np.array(all_runs_results['joint_coverage']), axis=0)
                            with open(os.path.join(save_dir, 'joint_coverage.txt'), 'w') as f:
                                f.write(f'Joint coverage: {joint_coverage:.4f} +/- {joint_coverage_std:.4f}\n')


                            # Save the max and min single task coverage (take max and min first, then average)
                            all_coverages = np.stack(list(all_runs_results['coverage'].values()), axis=0)
                            max_coverages = np.max(all_coverages, axis=0)
                            min_coverages = np.min(all_coverages, axis=0)
                            with open(os.path.join(save_dir, 'max_min_single_task_coverage.txt'), 'w') as f:
                                max_mean = np.mean(max_coverages, axis=0)
                                min_mean = np.mean(min_coverages, axis=0)
                                max_std = np.std(max_coverages, axis=0)
                                min_std = np.std(min_coverages, axis=0)
                                f.write(f'Max coverage mean: {max_mean:.4f} +/- {max_std:.4f}\n')
                                f.write(f'Min coverage mean: {min_mean:.4f} +/- {min_std:.4f}\n')
                            with open(os.path.join(save_dir, 'max_min_single_task_coverages.txt'), 'w') as f:
                                for i in range(len(max_coverages)):
                                    f.write(f'Run {i}: max:{max_coverages[i]:.4f} min:{min_coverages[i]:.4f}\n')


                            # Compute the max and min interval measurements, then average
                            all_interval_measurements = np.stack(list(all_runs_results['interval_measurement'].values()), axis=0)
                            max_interval_measurements = np.max(all_interval_measurements, axis=0)
                            min_interval_measurements = np.min(all_interval_measurements, axis=0)
                            with open(os.path.join(save_dir, 'max_min_single_task_interval_measurement.txt'), 'w') as f:
                                max_mean = np.mean(max_interval_measurements, axis=0)
                                min_mean = np.mean(min_interval_measurements, axis=0)
                                max_std = np.std(max_interval_measurements, axis=0)
                                min_std = np.std(min_interval_measurements, axis=0)
                                f.write(f'Max interval measurement mean: {max_mean:.4f} +/- {max_std:.4f}\n')
                                f.write(f'Min interval measurement mean: {min_mean:.4f} +/- {min_std:.4f}\n')

                            # Compute the max and min max absolute differences, then average
                            all_max_abs_diffs = np.stack(list(all_runs_results['max_abs_diff'].values()), axis=0)
                            max_max_abs_diffs = np.max(all_max_abs_diffs, axis=0)
                            min_max_abs_diffs = np.min(all_max_abs_diffs, axis=0)
                            with open(os.path.join(save_dir, 'max_min_single_task_max_abs_diff.txt'), 'w') as f:
                                max_mean = np.mean(max_max_abs_diffs, axis=0)
                                min_mean = np.mean(min_max_abs_diffs, axis=0)
                                max_std = np.std(max_max_abs_diffs, axis=0)
                                min_std = np.std(min_max_abs_diffs, axis=0)
                                f.write(f'Max max abs diff mean: {max_mean:.4f} +/- {max_std:.4f}\n')
                                f.write(f'Min max abs diff mean: {min_mean:.4f} +/- {min_std:.4f}\n')



    #%% Generate the joint coverage, single target coverage, and interval measurement plots
    experiment_load_dir = os.path.join(experiments_dir, f'{data_type}_cp')
    plt.rcParams.update({'font.size': 20})

    save_dir = f'../results/plots/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    configs = ['naive_none', 'joint_unified_length', 'unified_quantile_score']
    config_names = ['independent', 'quantile length', 'ours']
    colors = ['blue', 'orange', 'green']
    markers = ['x', '^', 'o', '<', '>']
    accel = 8
    p = 1 if data_type == 'mri_knee' else 4
    c = 32
    alphas = [0.3, 0.25, 0.2, 0.15, 0.1, 0.05]


    #%% Generate the single target coverages

    results = {}

    # Get the results for each config
    for i, config in enumerate(configs):
        results[config] = {}
        results[config]['min_coverages'] = []
        results[config]['max_coverages'] = []
        results[config]['min_coverage_std'] = []
        results[config]['max_coverage_std'] = []

        for alpha in alphas:
            # Get the directory to load the results
            load_dir = os.path.join(experiment_load_dir, config, f'alpha_{alpha}', f'accel_{accel}', f'p_{p}', f'c_{c}')

            # Load the results
            with open(os.path.join(load_dir, 'coverage.txt'), 'r') as f:
                lines = f.readlines()

            # Get the coverages
            coverages = []
            stddevs = []
            for line in lines:
                if '+/-' in line:
                    metric, value = line.split('risk mean:')
                    coverage = value.split('+/-')[0].strip()
                    coverages.append(float(coverage))
                    stddev = float(value.split('+/-')[1].strip())
                    stddevs.append(stddev)

            results[config]['min_coverages'].append(min(coverages))
            results[config]['max_coverages'].append(max(coverages))
            results[config]['min_coverage_std'].append(stddevs[np.argmin(coverages)])
            results[config]['max_coverage_std'].append(stddevs[np.argmax(coverages)])

    # Plot the results
    desired_coverages = 1 - np.array(alphas)
    plt.figure()
    for i, config in enumerate(configs):

        plt.plot(desired_coverages, results[config]['min_coverages'], color=colors[i], marker=markers[i], markersize=10,
                 linewidth=2)
        plt.plot(desired_coverages, results[config]['max_coverages'], color=colors[i], marker=markers[i], markersize=10,
                 linewidth=2)
        plt.fill_between(desired_coverages, results[config]['min_coverages'], results[config]['max_coverages'],
                         alpha=0.5,
                         color=colors[i],
                         )  # label=config_names[i])
    # plt.xlabel('Desired Coverage')
    # plt.ylabel('Single Task Coverage')
    # plt.legend(loc='lower right')
    # plt.ylim(0.7, 1.0)
    plt.grid(alpha=0.5)

    # plt.show()

    folder = 'metrics' if data_type == 'mri_knee' else 'classif'
    if not os.path.exists(os.path.join(save_dir, folder)):
        os.makedirs(os.path.join(save_dir, folder))
    plt.savefig(os.path.join(save_dir, folder, 'single_coverage.pdf'), dpi=1200)
    plt.close()


    #%% Generate the joint coverages
    results = {}

    # Get the results for each config
    for i, config in enumerate(configs):
        results[config] = {}
        results[config]['joint_coverages'] = []
        results[config]['joint_coverage_std'] = []

        for alpha in alphas:
            # Get the directory to load the results
            load_dir = os.path.join(experiment_load_dir, config, f'alpha_{alpha}', f'accel_{accel}', f'p_{p}', f'c_{c}')

            # Load the results
            with open(os.path.join(load_dir, 'joint_coverage.txt'), 'r') as f:
                lines = f.readlines()

            for line in lines:
                if '+/-' in line:
                    metric, value = line.split('coverage:')
                    coverage = value.split('+/-')[0].strip()
                    results[config]['joint_coverages'].append(float(coverage))
                    stddev = float(value.split('+/-')[1].strip())
                    results[config]['joint_coverage_std'].append(stddev)

    # Plot the results
    desired_coverages = 1 - np.array(alphas)
    plt.figure()
    for i, config in enumerate(configs):
        if config == 'unified_quantile_score':
            plt.errorbar(desired_coverages, results[config]['joint_coverages'],
                         yerr=results[config]['joint_coverage_std'],
                         color=colors[i], marker=markers[i], label=config_names[i], capsize=7,
                         markersize=10, linewidth=2)
        else:
            plt.plot(desired_coverages, results[config]['joint_coverages'], color=colors[i], marker=markers[i],
                     markersize=10, linewidth=2)
    # Plot line for desired coverage
    plt.plot(desired_coverages, desired_coverages, linestyle='--', color='k', label='Desired Coverage', linewidth=4)

    # plt.xlabel('Desired Coverage')
    # plt.ylabel('Joint Coverage')
    # plt.legend(loc='lower right')
    # plt.ylim(0.7, 1.0)
    plt.grid(alpha=0.5)

    # plt.show()
    plt.savefig(os.path.join(save_dir, folder, 'joint_coverage.pdf'), dpi=1200)
    plt.close()



    #%% Plot the interval lengths

    results = {}

    # Get the results for each config
    for i, config in enumerate(configs):
        results[config] = {}
        results[config]['min_interval_length'] = []
        results[config]['max_interval_length'] = []
        results[config]['mean_interval_length'] = []
        results[config]['min_interval_std'] = []
        results[config]['max_interval_std'] = []

        for alpha in alphas:
            # Get the directory to load the results
            load_dir = os.path.join(experiment_load_dir, config, f'alpha_{alpha}', f'accel_{accel}', f'p_{p}', f'c_{c}')

            # Load the results
            with open(os.path.join(load_dir, 'interval_measurement.txt'), 'r') as f:
                lines = f.readlines()

            # Get the measurements
            interval_measurements = []
            stddevs = []
            for line in lines:
                if '+/-' in line:
                    metric, value = line.split('mean:')
                    interval_measurement = value.split('+/-')[0].strip()
                    interval_measurements.append(float(interval_measurement))
                    stddev = float(value.split('+/-')[1].strip())
                    stddevs.append(stddev)

                else:
                    metric, value = line.split('mean:')
                    interval_measurement = value.split('\n')[0].strip()
                    interval_measurements.append(float(interval_measurement))
                    stddevs.append(0)

            results[config]['mean_interval_length'].append(np.mean(interval_measurements))
            results[config]['min_interval_length'].append(min(interval_measurements))
            results[config]['max_interval_length'].append(max(interval_measurements))
            results[config]['min_interval_std'].append(stddevs[np.argmin(interval_measurements)])
            results[config]['max_interval_std'].append(stddevs[np.argmax(interval_measurements)])

    # Plot the results
    desired_coverages = 1 - np.array(alphas)

    plt.figure()
    for i, config in enumerate(configs):

        plt.plot(desired_coverages, results[config]['min_interval_length'], color=colors[i], marker=markers[i],
                 markersize=10, linewidth=2)
        plt.plot(desired_coverages, results[config]['max_interval_length'], color=colors[i], marker=markers[i],
                 markersize=10, linewidth=2)

        # plt.plot(desired_coverages, results[config]['mean_interval_length'], color=colors[i], marker=markers[i], linestyle='--', label=config_names[i])
        plt.fill_between(desired_coverages, results[config]['min_interval_length'],
                         results[config]['max_interval_length'], alpha=0.5,
                         color=colors[i],
                         )  # label=config_names[i])
    # plt.xlabel('Desired Coverage')
    # plt.ylabel('Single Task Coverage')
    # plt.legend(loc='lower right')
    # plt.ylim(0.7, 1.0)
    plt.grid(alpha=0.5)

    # plt.show()
    # if not os.path.exists(os.path.join(save_dir, 'classif')):
    #     os.makedirs(os.path.join(save_dir, 'classif'))
    plt.savefig(os.path.join(save_dir, folder, 'single_target_interval_measurements.pdf'), dpi=1200)
    plt.close()



