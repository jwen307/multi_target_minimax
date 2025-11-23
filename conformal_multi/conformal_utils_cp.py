'''
conformal_utils_cp.py
- Contains the classes for all of the multi-task conformal prediction methods
'''

import time

import torch
import numpy as np
from scipy.stats import binom
from tqdm import tqdm
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer
from omegaconf import OmegaConf
from collections import OrderedDict



def to_tensor(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)

    if torch.cuda.is_available():
        x = x.cuda()
    return x


#%% Conformal Classes

def select_conformal(cf, calib_task_outputs=None, train_task_outputs=None, joint_accels=[]):
    if cf.conformal_method == 'naive':
        return BaseConformal(calib_task_outputs, cf, joint_accels=joint_accels)

    elif cf.conformal_method == 'unified':
        return UnifiedConformal(calib_task_outputs, cf, train_task_outputs, joint_accels=joint_accels)

    elif cf.conformal_method == 'joint_unified':
        return JointUnifiedConformal(calib_task_outputs, cf, train_task_outputs, joint_accels=joint_accels)
    
    elif cf.conformal_method == 'copula_cpts':
        return CopulaCPTSConformal(calib_task_outputs, cf, train_task_outputs, joint_accels=joint_accels)
    else:
        raise ValueError(f'Conformal type {cf.conformal_type} not recognized')

def select_interval(task_config, alpha=None, interval_type=None, joint_predictor=False, interval_limits=None):

    if joint_predictor:
        if interval_type == 'cqr':
            return JointCQR(task_config, alpha, interval_limits)

    else:
        if interval_type == 'cqr':
            return CQR(task_config, alpha, interval_limits)
        else:
            raise ValueError(f'Interval type {task_config.interval_type} not recognized')


#%% Base Conformal Class (serves as a naive approach where each task is treated separately)
class BaseConformal:
    def __init__(self, calib_task_outputs, config, train_task_outputs=None, joint_accels=[]):
        # Store the experiment configuration
        self.cf = config

        # Store the calibration task outputs
        self.calib_task_outputs = calib_task_outputs

        # Store the error rate
        self.alpha = self.cf.alpha

        # Get a list of all the tasks
        self.task_list = []
        for task in self.cf.tasks:
            if task == 'classify':
                label_names = list(calib_task_outputs.keys())

                # Make sure the label don't already have the acceleration
                label_names = [label.split('_')[0] for label in label_names]

                # Make sure the label names are unique while preserving the order
                label_names = list(OrderedDict.fromkeys(label_names))
                # # Make sure the label names are unique
                #label_names = list(set(label_names))

                self.task_list = self.task_list + label_names

            else:
                self.task_list.append(task)


        # If joint_accels is provided, then add all the tasks at different accelerations as differents tasks
        if len(joint_accels) > 0:
            accel_task_list = []
            for accel in joint_accels:
                for task in self.task_list:
                    if str(accel) not in task:
                        accel_task_list.append(f'{task}_{accel}')
                    else:
                        accel_task_list.append(task)
            self.task_list = accel_task_list

        # Compute the number of tasks
        self.num_tasks = len(self.task_list)


        # Correct the alpha if set (for worse case negative correlation)
        if self.cf.alpha_correction_type == 'bonferroni':
            # For negative correlation
            self.alpha = self.alpha/self.num_tasks
        elif self.cf.alpha_correction_type == 'independent':
            #For independent assumption
            self.alpha = 1 - (1-self.alpha)**(1/self.num_tasks)

        # Interval type
        self.interval_type = self.cf.interval_type

        # Get an ConformalInterval object for each task
        self.conformal_intervals = []

        # If joint_accels is provided, then add all the tasks at different accelerations as differents tasks
        # Note: we use conformal intervals based on indexing the task list, so ordering differentiates the tasks at
        # different accelerations
        if len(joint_accels) > 0:
            for accel in joint_accels:
                for task in self.cf.tasks:
                    # Create an interval object for each task
                    if task == 'classify':
                        for label in label_names:
                            task_config = OmegaConf.create(
                                {'name': label, 'bound_type': self.cf.tasks['classify']['bound_type']})
                            self.conformal_intervals.append(
                                select_interval(task_config, self.alpha, self.interval_type, interval_limits=[0, 1]))

                    else:
                        self.conformal_intervals.append(
                            select_interval(self.cf.tasks[task], self.alpha, self.interval_type))

        else:
            for task in self.cf.tasks:
                # Create an interval object for each task
                if task == 'classify':
                    for label in label_names:
                        task_config = OmegaConf.create({'name': label, 'bound_type': self.cf.tasks['classify']['bound_type']})
                        self.conformal_intervals.append(select_interval(task_config, self.alpha, self.interval_type, interval_limits=[0,1]))

                else:
                    self.conformal_intervals.append(select_interval(self.cf.tasks[task], self.alpha, self.interval_type))

    def calibrate(self, calib_task_outputs=None, c=32):
        # Use the initialized calibration task outputs if none are provided
        if calib_task_outputs is None:
            calib_task_outputs = self.calib_task_outputs

        # Get the lambda_hat for each task separately
        self.lambda_hat = []

        for i in range(self.num_tasks):
            task = self.task_list[i]

            # Get the posteriors and ground truth
            posteriors = calib_task_outputs[task]['posteriors'][:, :c]
            gt = calib_task_outputs[task]['gt']

            # Get the lambda that minimizes the risk
            self.lambda_hat.append(self.get_lambda_hat(posteriors, gt, self.conformal_intervals[i]))

        return self.lambda_hat


    def get_lambda_hat(self, posteriors, gt,  conformal_interval):

        # Get the number of calibration samples
        n = gt.shape[0]

        # Compute the scores
        scores = conformal_interval.compute_score(posteriors, gt)

        # Get the quantile that satisfies the error rate
        lambda_hat = np.quantile(scores,  np.ceil((1-self.alpha)*(n+1))/n)

        return lambda_hat

    def test(self, test_task_outputs, c):
        outputs = {}

        all_valids = []

        for i in range(self.num_tasks):
            task = self.task_list[i]

            # Get the posteriors and ground truth
            posteriors = test_task_outputs[task]['posteriors'][:, :c]
            gt = test_task_outputs[task]['gt']

            # Get the coverage and intervals
            intervals = self.conformal_intervals[i].compute_interval(posteriors, self.lambda_hat[i])
            coverage, valid = self.conformal_intervals[i].compute_coverage(intervals, gt)

            # Get the max absolute difference between the intervals and the ground truth
            max_abs_diff = self.conformal_intervals[i].compute_max_abs_diff(intervals, gt)


            # Collect all the valid indicators for each sample for each task
            all_valids.append(valid)

            # Get interval measurement (either interval width or worst case bound)
            interval_measurement = self.conformal_intervals[i].get_interval_measurement(intervals)

            outputs[task] = {'coverage': coverage, 'interval_measurement': interval_measurement, 'intervals': intervals, 'lambda_hat': self.lambda_hat[i],
                             'max_abs_diff':max_abs_diff}

        # Check the joint coverage
        all_valids = np.stack(all_valids, axis=0)
        joint_valid = np.all(all_valids, axis=0)
        joint_coverage = np.mean(joint_valid)

        outputs['joint_coverage'] = joint_coverage
        outputs['joint_valid'] = joint_valid

        return outputs


    # Function to only test certain tasks (mostly used for multi-round with tasks defined for each acceleration)
    def test_certain_tasks(self, test_task_outputs, c, test_task_list):
        outputs = {}

        all_valids = []

        #for i in range(len(test_task_list)):
        for task in test_task_list:

            # Find the index that corresponds to the task in self.task_list
            i = self.task_list.index(task)

            # Get the posteriors and ground truth
            posteriors = test_task_outputs[task]['posteriors'][:, :c]
            gt = test_task_outputs[task]['gt']

            # Get the coverage and intervals
            intervals = self.conformal_intervals[i].compute_interval(posteriors, self.lambda_hat[i])
            coverage, valid = self.conformal_intervals[i].compute_coverage(intervals, gt)

            # Collect all the valid indicators for each sample for each task
            all_valids.append(valid)

            # Get interval measurement (either interval width or worst case bound)
            interval_measurement = self.conformal_intervals[i].get_interval_measurement(intervals)

            outputs[task] = {'coverage': coverage, 'interval_measurement': interval_measurement, 'intervals': intervals,
                             'lambda_hat': self.lambda_hat[i]}

        # Check the joint coverage
        all_valids = np.stack(all_valids, axis=0)
        joint_valid = np.all(all_valids, axis=0)
        joint_coverage = np.mean(joint_valid)

        outputs['joint_coverage'] = joint_coverage
        outputs['joint_valid'] = joint_valid

        return outputs




    def many_trials(self, cf, task_outputs, c):
        # Get the number of samples
        n = task_outputs[self.task_list[0]]['gt'].shape[0]

        # Split for the calibration data
        n_cal = int(n * cf.eval.calib_split)

        # Store all the outputs
        all_outputs = {}

        # Go through all the trials
        for i in tqdm(range(cf.eval.num_trials)):
            # Get the indices for the calibration and test sets
            indices = np.random.permutation(n)
            cal_indices = indices[:n_cal]
            test_indices = indices[n_cal:]

            calib_task_outputs = {task: {'posteriors': task_outputs[task]['posteriors'][cal_indices],
                                         'gt': task_outputs[task]['gt'][cal_indices]} for task in self.task_list}

            test_task_outputs = {task: {'posteriors': task_outputs[task]['posteriors'][test_indices],
                                            'gt': task_outputs[task]['gt'][test_indices]} for task in self.task_list}

            # Calibrate (Note: self.lambda_hat is updated each time a calibration happens, so don't need to pass into test)
            lambda_hat = self.calibrate(calib_task_outputs, c=c)

            # Test
            outputs = self.test(test_task_outputs, c=c)

            # Collect all of the outputs
            #for task_name, task in cf.tasks.items():
            for task_name in self.task_list:
                if task_name not in all_outputs:
                    all_outputs[task_name] = {}

                for output_name in outputs[task_name]:
                    if output_name not in all_outputs[task_name]:
                        all_outputs[task_name][output_name] = []

                    all_outputs[task_name][output_name].append(outputs[task_name][output_name])

                if 'joint_coverage' not in all_outputs:
                    all_outputs['joint_coverage'] = []
                all_outputs['joint_coverage'].append(outputs['joint_coverage'])

                if 'shape_coverage' in outputs:
                    if 'shape_coverage' not in all_outputs:
                        all_outputs['shape_coverage'] = []
                    all_outputs['shape_coverage'].append(outputs['shape_coverage'])

        return all_outputs



#%% Unified Conformal Class (computes one lambda for all tasks)
class UnifiedConformal(BaseConformal):

    def __init__(self, calib_task_outputs, config, train_task_outputs=None, joint_accels=[]):
        super().__init__(calib_task_outputs, config, train_task_outputs, joint_accels=joint_accels)

        self.normalization_type = config.normalization_type


        # Find the normalization factors
        if self.normalization_type != 'none' and self.normalization_type != 'length':

            #print('Computing normalization factors from training data...')

            for i in range(self.num_tasks):
                task = self.task_list[i]
                training_gt = train_task_outputs[task]['gt']
                training_posteriors = train_task_outputs[task]['posteriors']

                self.conformal_intervals[i].compute_normalization_factors(training_posteriors, training_gt, normalization_type=self.normalization_type)

            #print('Normalization factors computed, initialization completed')


    # Take the max score for all tasks before computing the lambda
    def calibrate(self, calib_task_outputs=None, c=32):
        # Use the initialized calibration task outputs if none are provided
        if calib_task_outputs is None:
            calib_task_outputs = self.calib_task_outputs

        # Accumulate the scores for all tasks
        all_scores = []

        for i in range(self.num_tasks):
            task = self.task_list[i]

            # Get the posteriors and ground truth
            posteriors = calib_task_outputs[task]['posteriors'][:, :c]
            gt = calib_task_outputs[task]['gt']

            # Compute the scores
            scores = self.conformal_intervals[i].compute_score(posteriors, gt)

            # Append the scores
            all_scores.append(scores)

        # Take the maximum score for all tasks
        all_scores = np.stack(all_scores, axis=0)
        max_scores = np.max(all_scores, axis=0)
        #print(max_scores.shape)

        # Get the number of calibration samples
        n = len(max_scores)

        # Get the quantile that satisfies the error rate
        lambda_hat = np.quantile(max_scores, np.ceil((1 - self.alpha) * (n + 1)) / n)

        # Repeat the lambda for all tasks
        self.lambda_hat = lambda_hat * np.ones(self.num_tasks)

        return self.lambda_hat



class CopulaCPTSConformal(UnifiedConformal):
    """
    Copula-based joint conformal method (CPTS).

    This method performs a two-step calibration:
      1) Split the provided calibration set into two parts: one to form per-task
         nonconformity distributions, and one to infer a copula over the
         dependence structure of the per-task exceedance levels.
      2) Use a learned per-task exceedance threshold (via `utils.search_alpha`)
         to select per-task radii that achieve the target joint coverage.

    It then produces per-task intervals by adding/subtracting the learned
    radii around each task's point prediction (mean of posterior samples).
    """

    def __init__(self, calib_task_outputs, config, train_task_outputs=None, joint_accels=[]):
        super().__init__(calib_task_outputs, config, train_task_outputs, joint_accels=joint_accels)



    # --- conformal API ---
    def calibrate(self, calib_task_outputs=None, c=32):
        # Use the initialized calibration task outputs if none are provided
        if calib_task_outputs is None:
            calib_task_outputs = self.calib_task_outputs

        # Accumulate the scores for all tasks
        all_scores = []

        for i in range(self.num_tasks):
            task = self.task_list[i]

            # Get the posteriors and ground truth
            posteriors = calib_task_outputs[task]['posteriors'][:, :c]
            gt = calib_task_outputs[task]['gt']

            # Compute the scores
            scores = self.conformal_intervals[i].compute_score(posteriors, gt)

            # Append the scores
            all_scores.append(scores)

        # Find the optimized u* using the author's code
        all_scores = np.stack(all_scores, axis=0).T
        self.lambda_hat = search_alpha(all_scores, epsilon=self.alpha, epochs=800)  # (num_tasks,) # Note: lambda_hat is in the transformed domain (move out later)

        return self.lambda_hat




#%% Generic Conformal Interval Class
class ConformalInterval:
    def __init__(self, task_config, alpha, interval_limits = None):
        # Store the experiment configuration
        self.task_cf = task_config

        # Store the error rate
        self.alpha = alpha

        # Interval type
        self.bound_type = self.task_cf.bound_type

        # Normalization factors
        self.mu = 0
        self.sigma = 1

        self.normalization_type = 'standardize'

        self.interval_limits = interval_limits if interval_limits is not None else [-np.inf, np.inf]

    def compute_interval(self, posterior_task_outputs, lam):
        raise NotImplementedError

    def compute_score(self, posterior_task_outputs, gt):
        raise NotImplementedError

    # Normalize using training data statistics
    def compute_normalization_factors(self, posteriors, gt, normalization_type='standardize'):

        self.normalization_type = normalization_type

        if normalization_type == 'standardize':
            self.mu = np.mean(gt)
            self.sigma = np.std(gt)


        elif normalization_type == 'quantile_score':
            # Compute the scores first without normalization
            self.normalization_type = 'standardize' # Set the normalization type to standardize with 0 mean and 1 std
            scores = self.compute_score(posteriors, gt)

            # Compute the score correction (quantile transformer)
            self.quantile_transformer = QuantileTransformer(output_distribution='uniform', n_quantiles=len(gt))
            self.quantile_transformer.fit(scores.reshape(-1, 1))
            self.normalization_type = 'quantile_score' # Set the normalization type back to quantile score


    def normalize(self,x, posterior=False):

        in_shape = x.shape

        if self.normalization_type == 'standardize':
            x = (x-self.mu)/self.sigma

        elif self.normalization_type == 'quantile_score':
            return x

        return x

    def denormalize(self,x):

        in_shape = x.shape

        # If the input is np.inf or -np.inf, then return the same
        if np.any(np.isinf(x)):
            return x

        if self.normalization_type == 'standardize':
            x = x*self.sigma + self.mu

        elif self.normalization_type == 'quantile_score':
            return x

        return x

    # Get the interval measurement (either interval width or worst case bound)
    def get_interval_measurement(self, intervals):
        if self.bound_type == 'bound_upper':
            return intervals[:, 1]
        elif self.bound_type == 'bound_lower':
            return intervals[:, 0]
        elif self.bound_type == 'two_sided':
            return intervals[:, 1] - intervals[:, 0]
        else:
            raise ValueError(f'Interval type {self.bound_type} not recognized')

    # Compute the coverage
    def compute_coverage(self, intervals, gt):
        if self.bound_type == 'bound_upper':
            valid = (gt <= intervals[:, 1])
        elif self.bound_type == 'bound_lower':
            valid = (gt >= intervals[:, 0])
        elif self.bound_type == 'two_sided':
            valid = ((gt >= intervals[:, 0]) & (gt <= intervals[:, 1]))
        else:
            raise ValueError(f'Interval type {self.bound_type} not recognized')

        coverage = np.mean(valid)

        return coverage, valid


    def compute_max_abs_diff(self, intervals, gt):
        if self.bound_type == 'bound_upper':
            max_abs_diff = np.abs(gt - intervals[:, 1])
        elif self.bound_type == 'bound_lower':
            max_abs_diff = np.abs(gt - intervals[:, 0])
        elif self.bound_type == 'two_sided':
            lower_abs_diff = np.abs(gt - intervals[:, 0])
            upper_abs_diff = np.abs(gt - intervals[:, 1])
            max_abs_diff = np.max(np.stack([lower_abs_diff, upper_abs_diff], axis=-1), axis=-1)
        else:
            raise ValueError(f'Interval type {self.bound_type} not recognized')

        return max_abs_diff





#%% Conformal interval for adaptive conformal prediction with additive lambda using quantile (CQR)
class CQR(ConformalInterval):

    def compute_interval(self, posterior_task_outputs, lam):
        '''
        :param posterior_task_outputs: (n, c) array of posterior outputs where n is number of samples and c is number of posteriors
        :param lam: lambda value to use for interval
        :return:
        '''

        # Normalize the posterior task outputs
        posterior_task_outputs = self.normalize(posterior_task_outputs)

        if self.normalization_type == 'quantile_score':
            lam = self.quantile_transformer.inverse_transform(lam.reshape(-1, 1)).reshape(lam.shape)

        # Compute the interval
        if self.bound_type == 'bound_upper':
            upper = np.quantile(posterior_task_outputs, 1 - self.alpha, axis=1) + lam

            if self.normalization_type == 'score_shift' or self.normalization_type == 'score_shift_scale':
                upper = upper + self.mu

            lower = -np.infty * np.ones_like(upper)
        elif self.bound_type == 'bound_lower':
            lower = np.quantile(posterior_task_outputs, self.alpha, axis=1) - lam

            if self.normalization_type == 'score_shift' or self.normalization_type == 'score_shift_scale':
                lower = lower - self.mu

            upper = np.infty * np.ones_like(lower)
        elif self.bound_type == 'two_sided':
            upper = np.quantile(posterior_task_outputs, 1 - self.alpha/2, axis=1) + lam
            lower = np.quantile(posterior_task_outputs, self.alpha/2, axis=1) - lam

        else:
            raise ValueError(f'Bound type {self.bound_type} not recognized')



        # Denormalize the interval
        lower = self.denormalize(lower)
        upper = self.denormalize(upper)

        upper = np.clip(upper, self.interval_limits[0], self.interval_limits[1])
        lower = np.clip(lower, self.interval_limits[0], self.interval_limits[1])

        interval = np.stack([lower, upper], axis=-1)

        return interval

    def compute_score(self, posterior_task_outputs, gt):

        # Normalize the posterior task outputs and gt
        posterior_task_outputs = self.normalize(posterior_task_outputs)
        gt = self.normalize(gt)

        # Compute the scores
        if self.bound_type == 'bound_upper':
            scores = gt - np.quantile(posterior_task_outputs, 1 - self.alpha, axis=1)
        elif self.bound_type == 'bound_lower':
            scores = np.quantile(posterior_task_outputs, self.alpha, axis=1) - gt
        elif self.bound_type == 'two_sided':
            upper = np.quantile(posterior_task_outputs, 1 - self.alpha/2, axis=1)
            lower = np.quantile(posterior_task_outputs, self.alpha/2, axis=1)

            # Clip the bounds to the interval limits
            upper = np.clip(upper, self.interval_limits[0], self.interval_limits[1])
            lower = np.clip(lower, self.interval_limits[0], self.interval_limits[1])
            scores = np.maximum(lower - gt, gt - upper)
        else:
            raise ValueError(f'Bound type {self.bound_type} not recognized')

        if self.normalization_type == 'score_shift':
            scores = scores - self.mu
        elif self.normalization_type == 'score_shift_scale':
            scores = scores - self.mu
            scores = scores / self.sigma

        elif self.normalization_type == 'quantile_score':
            scores = self.quantile_transformer.transform(scores.reshape(-1, 1)).reshape(scores.shape)

        return scores









#%% Unified Class that only uses a single Conformal Interval object to handle all tasks
class JointUnifiedConformal(UnifiedConformal):

    def __init__(self, calib_task_outputs, config, train_task_outputs=None, joint_accels=[]):
        super().__init__(calib_task_outputs, config, train_task_outputs, joint_accels=joint_accels)

        # Modify the configuration to have a separate task config for each task
        self.new_config = config.copy()

        # Remove the task list from the config
        self.new_config['tasks'] = {}

        # Remove the classify task if it exists
        if 'classify' in config['tasks']:
            interval_limits = [0,1]

        else:
            interval_limits = None

        for task in self.task_list:
            task_config = OmegaConf.create(
                                {'name': task, 'bound_type': 'two_sided'})
            self.new_config['tasks'][task] = task_config



        # Use a single interval class for all tasks
        self.conformal_intervals = select_interval(self.new_config, self.alpha, self.interval_type, joint_predictor=True, interval_limits=interval_limits)


        # Compute the noramlization factors if needed
        if self.normalization_type == "qn_mini":
            # Combine the posteriors for all tasks in a tensor (n, c, num_tasks)
            # Combine the ground truth for all tasks in a tensor (n, num_tasks)
            posteriors_all_tasks = np.stack([train_task_outputs[task]['posteriors'] for task in self.task_list],
                                            axis=-1)
            gt_all_tasks = np.stack([train_task_outputs[task]['gt'] for task in self.task_list], axis=-1)

            self.conformal_intervals.compute_normalization_factors(posteriors_all_tasks, gt_all_tasks)


    # Calibrate
    def calibrate(self, calib_task_outputs=None, c=32):
        # Use the initialized calibration task outputs if none are provided
        if calib_task_outputs is None:
            calib_task_outputs = self.calib_task_outputs

        # Combine the posteriors for all tasks in a tensor (n, c, num_tasks)
        # Combine the ground truth for all tasks in a tensor (n, num_tasks)
        posteriors_all_tasks = np.stack([calib_task_outputs[task]['posteriors'][:, :c] for task in self.task_list],
                                        axis=-1)
        gt_all_tasks = np.stack([calib_task_outputs[task]['gt'] for task in self.task_list], axis=-1)

        self.lambda_hat = self.get_lambda_hat(posteriors_all_tasks, gt_all_tasks, self.conformal_intervals)

        return self.lambda_hat

    def test(self, test_task_outputs, c):
        outputs = {}

        posterior_all_tasks = np.stack([test_task_outputs[task]['posteriors'][:, :c] for task in self.task_list], axis=-1)
        gt_all_tasks = np.stack([test_task_outputs[task]['gt'] for task in self.task_list], axis=-1)

        # Get the coverage and intervals
        intervals = self.conformal_intervals.compute_interval(posterior_all_tasks, self.lambda_hat)
        coverages, joint_coverages, _ = self.conformal_intervals.compute_coverage(intervals, gt_all_tasks)

        # Get the interval measurement (either interval width or worst case bound)
        interval_measurement = self.conformal_intervals.get_interval_measurement(intervals)

        # Get the max abs diff
        max_abs_diffs = self.conformal_intervals.compute_max_abs_diff(intervals, gt_all_tasks)

        for i, task in enumerate(self.task_list):
            outputs[task] = {'coverage': coverages[i], 'interval_measurement': interval_measurement[:, i], 'intervals': intervals[:,i,:], 'lambda_hat': self.lambda_hat,
                             'max_abs_diff': max_abs_diffs[:, i]}

        outputs['joint_coverage'] = joint_coverages

        return outputs

    # Function to only test certain tasks (mostly used for multi-round with tasks defined for each acceleration)
    def test_certain_tasks(self, test_task_outputs, c, test_task_list):
        outputs = {}

        all_valids = []

        # Compute for all tasks
        posterior_all_tasks = np.stack([test_task_outputs[task]['posteriors'][:, :c] for task in self.task_list],
                                       axis=-1)
        gt_all_tasks = np.stack([test_task_outputs[task]['gt'] for task in self.task_list], axis=-1)

        # Get the coverage and intervals
        intervals = self.conformal_intervals.compute_interval(posterior_all_tasks, self.lambda_hat)
        coverages, joint_coverages, valids = self.conformal_intervals.compute_coverage(intervals, gt_all_tasks)

        # Get the interval measurement (either interval width or worst case bound)
        interval_measurement = self.conformal_intervals.get_interval_measurement(intervals)

        # Only save the information for the tasks that are in the test_task_list
        # for i in range(len(test_task_list)):
        for task in test_task_list:
            # Find the index that corresponds to the task in self.task_list
            i = self.task_list.index(task)

            # Collect all the valid indicators for each sample for each task
            all_valids.append(valids[i])

            outputs[task] = {'coverage': coverages[i], 'interval_measurement': interval_measurement[:, i],
                             'intervals': intervals[:,i,:],
                             'lambda_hat': self.lambda_hat}

        # Check the joint coverage
        all_valids = np.stack(all_valids, axis=0)
        joint_valid = np.all(all_valids, axis=0)


        outputs['joint_coverage'] = joint_coverages
        outputs['joint_valid'] = joint_valid

        return outputs



class JointConformalInterval:

        def __init__(self, task_config, alpha, interval_limits = None):
            # Store the entire config
            self.cf = task_config

            # Store the error rate
            self.alpha = alpha

            self.task_list = list(self.cf.tasks.keys())
            self.num_tasks = len(self.task_list)

            self.normalization_type = self.cf.normalization_type

            self.interval_limits = interval_limits if interval_limits is not None else [-np.inf, np.inf]

        # Normalize using training data statistics
        def compute_normalization_factors(self, posterior_task_outputs, gt):
            # Note: posteriors is (n, c, num_tasks) array

            if self.normalization_type == 'qn_mini':
                # Store all the quantile transfomers for each task
                self.quantile_transformers = []

                # Set to be just the QN normalization first
                self.normalization_type = 'length'

                # Compute the individual scores for each task
                # Score should be (n, num_tasks)
                scores = self.compute_individual_score(posterior_task_outputs, gt)

                # Compute the quantile transform for each task
                for i in range(self.num_tasks):
                    task_scores = scores[:, i]
                    qt = QuantileTransformer(output_distribution='uniform', n_quantiles=len(gt))
                    qt.fit(task_scores.reshape(-1, 1))
                    self.quantile_transformers.append(qt)

                # Set the normalization type back to qn_mini
                self.normalization_type = 'qn_mini'


        def normalize(self, x, posteriors=None):

            lengths = []
            ratios = []

            for i in range(self.num_tasks):
                task = self.task_list[i]

                bound_type = self.cf.tasks[task].bound_type

                if self.normalization_type == 'length' or self.normalization_type == 'qn_mini':
                    # if bound_type == 'bound_upper':
                    #     length = np.quantile(posteriors[:, :, i], 1 - self.alpha, axis=1)
                    # elif bound_type == 'bound_lower':
                    #     length = np.quantile(posteriors[:, :, i], self.alpha, axis=1)
                    # elif bound_type == 'two_sided':
                    upper = np.quantile(posteriors[:, :, i], 1 - self.alpha / 2, axis=1)
                    lower = np.quantile(posteriors[:, :, i], self.alpha / 2, axis=1)
                    length = upper - lower
                    # else:
                    #    raise ValueError(f'Bound type {bound_type} not recognized')

                    lengths.append(length)

                    ratio = lengths[0]/length
                    ratios.append(ratio)

            ratios = np.stack(ratios, axis=1)
            if len(x.shape) == 3:
                x = x * ratios[:,None,:]
            else:
                x = x * ratios


            return x

        def denormalize(self, x, posteriors=None):

            lengths = []
            ratios = []

            for i in range(self.num_tasks):
                task = self.task_list[i]

                bound_type = self.cf.tasks[task].bound_type

                if self.normalization_type == 'length' or self.normalization_type == 'qn_mini':
                    # if bound_type == 'bound_upper':
                    #     length = np.quantile(posteriors[:, :, i], 1 - self.alpha, axis=1)
                    # elif bound_type == 'bound_lower':
                    #     length = np.quantile(posteriors[:, :, i], self.alpha, axis=1)
                    # elif bound_type == 'two_sided':
                    upper = np.quantile(posteriors[:, :, i], 1 - self.alpha / 2, axis=1)
                    lower = np.quantile(posteriors[:, :, i], self.alpha / 2, axis=1)
                    length = upper - lower
                    # else:
                    #    raise ValueError(f'Bound type {bound_type} not recognized')

                    lengths.append(length)

                    ratio = lengths[0] / length
                    ratios.append(ratio)

            ratios = np.stack(ratios, axis=1)
            if len(x.shape) == 3:
                x = x / ratios[:, None, :]
            else:
                x = x / ratios

            return x


        def get_interval_measurement(self, intervals):
            measures = []

            for i in range(self.num_tasks):
                task = self.task_list[i]
                bound_type = self.cf.tasks[task].bound_type

                if bound_type == 'bound_upper':
                    measure = intervals[:,i,1]
                elif bound_type == 'bound_lower':
                    measure = intervals[:,i,0]
                elif bound_type == 'two_sided':
                    measure = intervals[:,i,1] - intervals[:,i,0]
                else:
                    raise ValueError(f'Bound type {bound_type} not recognized')
                measures.append(measure)

            return np.stack(measures, axis=-1)



        def compute_coverage(self, intervals, gt):
            '''
            :param intervals: (n, num_tasks, 2) array of intervals
            :param gt: (n, num_tasks) array of ground truth values
            :return: Coverage for each task
            '''

            coverages = []
            valids = []

            for i in range(self.num_tasks):
                # if self.bound_type == 'bound_upper':
                #     valid = (gt[:,i] <= intervals[:,i,1])
                # elif self.bound_type == 'bound_lower':
                #     valid = (gt[:,i] >= intervals[:,i,0])
                # elif self.bound_type == 'two_sided':
                valid = ((gt[:,i] >= intervals[:,i,0]) & (gt[:,i] <= intervals[:,i,1]))
                #else:
                #    raise ValueError(f'Interval type {self.bound_type} not recognized')

                coverage = np.mean(valid)
                coverages.append(coverage)

                valids.append(valid)

            valids = np.stack(valids, axis=0)
            joint_valid = np.all(valids, axis=0)
            joint_coverage = np.mean(joint_valid)

            return coverages, joint_coverage, valids


        def compute_max_abs_diff(self, intervals, gt):
            max_abs_diffs = []

            for i in range(self.num_tasks):
                task = self.task_list[i]
                bound_type = self.cf.tasks[task].bound_type

                if bound_type == 'bound_upper':
                    max_abs_diff = np.abs(gt[:,i] - intervals[:,i,1])

                elif bound_type == 'bound_lower':
                    max_abs_diff = np.abs(gt[:,i] - intervals[:,i,0])

                elif bound_type == 'two_sided':
                    max_abs_diff = np.maximum(np.abs(gt[:,i] - intervals[:,i,0]), np.abs(gt[:,i] - intervals[:,i,1]))

                max_abs_diffs.append(max_abs_diff)

            max_abs_diffs = np.stack(max_abs_diffs, axis=-1)

            return max_abs_diffs

class JointCQR(JointConformalInterval):

    def compute_interval(self, posterior_task_outputs, lam):
        '''
        :param posterior_task_outputs: (n, c, num_tasks) array of posterior outputs where n is number of samples, c is number of posteriors, and num_tasks is number of tasks
        :param lam: lambda value to use for interval
        :return:
        '''

        # Normalize the posterior task outputs
        posterior_task_outputs_norm = self.normalize(posterior_task_outputs, posterior_task_outputs)

        lowers = []
        uppers = []

        lam_og = lam * 1.0  # Keep original lambda

        for i in range(self.num_tasks):
            task = self.task_list[i]

            bound_type = self.cf.tasks[task].bound_type

            # Apply the inverse quantile transformation if needed to lambda to get in the space of QN scores
            if self.normalization_type == 'qn_mini':
                qt = self.quantile_transformers[i]
                lam = qt.inverse_transform(lam_og.reshape(-1, 1)).reshape(lam_og.shape)

            if bound_type == 'bound_upper':
                upper = np.quantile(posterior_task_outputs_norm[:,:,i], 1-self.alpha, axis=1) + lam
                lower = -np.infty * np.ones_like(upper)
            elif bound_type == 'bound_lower':
                lower = np.quantile(posterior_task_outputs_norm[:,:,i], self.alpha, axis=1) - lam
                upper = np.infty * np.ones_like(lower)
            elif bound_type == 'two_sided':
                upper = np.quantile(posterior_task_outputs_norm[:,:,i], 1-self.alpha/2, axis=1) + lam
                lower = np.quantile(posterior_task_outputs_norm[:,:,i], self.alpha/2, axis=1) - lam
            else:
                raise ValueError(f'Bound type {bound_type} not recognized')

            lowers.append(lower)
            uppers.append(upper)

        lowers = np.stack(lowers, axis=1)
        uppers = np.stack(uppers, axis=1)

        # Denormalize the interval
        lowers = self.denormalize(lowers, posterior_task_outputs)
        uppers = self.denormalize(uppers, posterior_task_outputs)

        # Clip the intervals to the interval limits when needed
        uppers = np.clip(uppers, self.interval_limits[0], self.interval_limits[1])
        lowers = np.clip(lowers, self.interval_limits[0], self.interval_limits[1])

        intervals = np.stack([lowers, uppers], axis=-1) # Dims: (n, num_tasks, 2)

        return intervals

    def compute_score(self, posterior_task_outputs, gt):

        # Normalize the posterior task outputs and gt
        posterior_task_outputs_norm = self.normalize(posterior_task_outputs, posterior_task_outputs)
        gt_norm = self.normalize(gt, posterior_task_outputs)

        scores = []

        for i in range(self.num_tasks):
            task = self.task_list[i]

            bound_type = self.cf.tasks[task].bound_type

            if bound_type == 'bound_upper':
                score = gt_norm[:,i] - np.quantile(posterior_task_outputs_norm[:,:,i], 1 - self.alpha, axis=1)
            elif bound_type == 'bound_lower':
                score = np.quantile(posterior_task_outputs_norm[:,:,i], self.alpha, axis=1) - gt_norm[:,i]
            elif bound_type == 'two_sided':
                upper = np.quantile(posterior_task_outputs_norm[:,:,i], 1 - self.alpha / 2, axis=1)
                lower = np.quantile(posterior_task_outputs_norm[:,:,i], self.alpha / 2, axis=1)


                # Compute the score
                score = np.maximum(lower - gt_norm[:,i], gt_norm[:,i] - upper)
            else:
                raise ValueError(f'Bound type {bound_type} not recognized')
            

            # Apply the quantile transformation if needed
            if self.normalization_type == 'qn_mini':
                qt = self.quantile_transformers[i]
                score = qt.transform(score.reshape(-1, 1)).reshape(score.shape)


            scores.append(score)

        scores = np.stack(scores, axis=1)

        # Take the max of the scores
        scores = np.max(scores, axis=1)

        return scores

    def compute_individual_score(self, posterior_task_outputs, gt):

        # Normalize the posterior task outputs and gt
        posterior_task_outputs_norm = self.normalize(posterior_task_outputs, posterior_task_outputs)
        gt_norm = self.normalize(gt, posterior_task_outputs)

        scores = []

        for i in range(self.num_tasks):
            task = self.task_list[i]

            bound_type = self.cf.tasks[task].bound_type

            if bound_type == 'bound_upper':
                score = gt_norm[:,i] - np.quantile(posterior_task_outputs_norm[:,:,i], 1 - self.alpha, axis=1)
            elif bound_type == 'bound_lower':
                score = np.quantile(posterior_task_outputs_norm[:,:,i], self.alpha, axis=1) - gt_norm[:,i]
            elif bound_type == 'two_sided':
                upper = np.quantile(posterior_task_outputs_norm[:,:,i], 1 - self.alpha / 2, axis=1)
                lower = np.quantile(posterior_task_outputs_norm[:,:,i], self.alpha / 2, axis=1)


                # Compute the score
                score = np.maximum(lower - gt_norm[:,i], gt_norm[:,i] - upper)
            else:
                raise ValueError(f'Bound type {bound_type} not recognized')
            

            # Apply the quantile transformation if needed
            if self.normalization_type == 'qn_mini':
                qt = self.quantile_transformers[i]
                score = qt.transform(score.reshape(-1, 1)).reshape(score.shape)

            scores.append(score)

        scores = np.stack(scores, axis=1)


        return scores



#%% Split the training task outputs
def split_training_task_outputs(cf, task_outputs):

    # Get a list of the tasks
    task_list = []
    for task in cf.tasks:
        if task == 'classify':
            label_names = list(task_outputs.keys())
            task_list = task_list + label_names
        else:
            task_list.append(task)

    # Get the total number of samples
    n = task_outputs[list(task_outputs.keys())[0]]['gt'].shape[0]

    # Get the number of training samples
    n_train = int(n * cf.eval.train_split)

    # Get the training task outputs
    # task_outputs_training = utils.load_task_output(cf, experiment, accel, p, training=True)

    # Use a random num_training samples from task_outputs as training
    rand_indices = np.random.permutation(n)
    train_indices = rand_indices[:n_train]
    cal_test_indices = rand_indices[n_train:]

    # Get the training data
    task_outputs_training = {task: {'posteriors': task_outputs[task]['posteriors'][train_indices],
                                    'gt': task_outputs[task]['gt'][train_indices]} for task in task_list}
    # Get the calibration and test data
    task_outputs = {task: {'posteriors': task_outputs[task]['posteriors'][cal_test_indices],
                            'gt': task_outputs[task]['gt'][cal_test_indices]} for task in task_list}

    return task_outputs_training, task_outputs








# Note: This was from the original CPTS codebase
import torch.nn as nn
import torch
class CP(nn.Module):
    def __init__(self, dimension, epsilon):
        super(CP, self).__init__()
        self.alphas = nn.Parameter(torch.ones(dimension))
        self.epsilon = epsilon
        self.relu = torch.nn.ReLU()

    def forward(self, pseudo_data):
        coverage = torch.mean(
            torch.relu(
                torch.prod(torch.sigmoid((self.alphas - pseudo_data) * 1000), dim=1)
            )
        )
        return torch.abs(coverage - 1 + self.epsilon)
    
def search_alpha(alpha_input, epsilon, epochs=500):
    # pseudo_data = torch.tensor(pseudo_obs(alpha_input))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pseudo_data = torch.tensor(alpha_input, device=device)
    dim = alpha_input.shape[-1]
    cp = CP(dim, epsilon).to(device)
    optimizer = torch.optim.Adam(cp.parameters(), weight_decay=1e-4)

    for i in range(epochs):
        optimizer.zero_grad()
        loss = cp(pseudo_data)

        loss.backward()
        optimizer.step()

    return cp.alphas.detach().cpu().numpy()