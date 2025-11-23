import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import QuantileRegressor
import matplotlib.pyplot as plt
import conformal_utils_cp as conformal
from omegaconf import OmegaConf
import math
import os
import pickle
from tqdm import tqdm


#%% Function to convert the quantile predictions
def convert_quantile_predictions(q_low, q_high, alpha):
    # alpha = 0.1
    # q_low = 0.2
    # q_high = 0.56
    n = 100

    low_level = alpha/2
    high_level = 1-alpha/2
    level_range = 1-alpha
    per_level_change = (q_high - q_low)/ (level_range * 100)

    lower_bound = q_low - (low_level * 100) * per_level_change
    upper_bound = q_high + ((1- high_level) * 100) * per_level_change

    #quantiles = np.quantile([lower_bound, upper_bound], [low_level, high_level])
    #print("Quantiles:", quantiles)

    return np.stack([lower_bound, upper_bound], axis=-1)


# posteriors = np.stack([lower_bound[:,0], upper_bound[:,0]], axis=1)
#
# low_quant = np.quantile(posteriors, alpha/2, axis=1)


def generate_date(n, independent=False):
    # Generate synthetic data
    X = np.random.uniform(-5, 5, size=n)
    mu1 = 10 * X + 10
    mu2 = -2 * X
    mu3 = 0.1 * X**2

    # Correlation structure for errors
    if independent:
        cov_matrix = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
    else:
        cov_matrix = np.array([
            [1.0, 0.8, 0.7],
            [0.8, 1.0, 0.4],
            [0.7, 0.4, 1.0]
        ])

    # Errrors added on
    errors = np.zeros((n, 3))
    errors[:, 0] = np.random.normal(10, 1, size=n)
    errors[:, 1] = np.random.gamma(shape=1.0, scale=1.0, size=n)  # Centered gamma
    errors[:, 2] = np.random.exponential(scale=1.0, size=n)  # Centered exponential

    # Add the correlations
    L = np.linalg.cholesky(cov_matrix)
    Z = np.random.normal(0, 1, size=(n, 3))
    errors = Z @ L.T

    Y = np.stack([mu1 + errors[:, 0],
                  mu2 + errors[:, 1],
                  mu3 + errors[:, 2]], axis=1)

    return X, Y

#%% Define all the parameters for the experiments

if __name__ == '__main__':
    np.random.seed(20)
    n_trains = [50, 200, 500, 1000, 2000, 5000, 10000]
    n_tune = 5000
    n_cal = 5000
    n_test = 1000

    independent = False

    # Number of montecarlo trials
    n_trials = 10000

    alpha = 0.1 # [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]


    # Number of labels
    p = 3
    tasks = {}
    task_list = []
    for j in range(0,p):
        tasks[f'dim_{j}'] = {'name': f'dim_{j}',
                             'bound_type': 'two_sided'}
        task_list.append(f'dim_{j}')


    # Methods
    methods = ['independent', 'length', 'unified']
    configs = {'independent': {'conformal_method': 'naive',
                               'normalization_type': 'none',
                               'alpha_correction_type': 'independent',},
               'length': {'conformal_method': 'joint_unified',
                          'normalization_type': 'length',
                          'alpha_correction_type': 'none',},
                'unified': {'conformal_method': 'unified',
                            'normalization_type': 'quantile_score',
                            'alpha_correction_type': 'none'}}



    #%% --------------------------
    for n_train in n_trains:

        # Each training run should use the same random seed to build the training data off of previous runs
        np.random.seed(32)

        # Generate the tuning data if needed (Will be the same for each n_train)
        # if cf.normalization_type == 'quantile_score':
        X_tune, Y_tune = generate_date(n_tune, independent=independent)



        # %% --------------------------
        # 1. Train the Quantile Regressors
        # --------------------------
        # Define the quantile levels
        q_low = alpha / 2
        q_high = 1 - alpha / 2

        X_train, Y_train = generate_date(n_train, independent=independent)

        # Store the quantile regressors for each dimension
        qr_lows = []
        qr_highs = []

        # Train the quantile regressors for each dimension
        for j in range(p):
            qr_low = QuantileRegressor(quantile=q_low, alpha=0, solver="highs").fit(X_train.reshape(-1, 1), Y_train[:, j])
            qr_high = QuantileRegressor(quantile=q_high, alpha=0, solver="highs").fit(X_train.reshape(-1, 1), Y_train[:, j])
            qr_lows.append(qr_low)
            qr_highs.append(qr_high)



        # Make predictions for the quantile regressors on the tuning set
        q_preds_low = []
        q_preds_high = []

        for j in range(p):
            # Predict on the tuning set
            q_low_pred = qr_lows[j].predict(X_tune.reshape(-1, 1))
            q_high_pred = qr_highs[j].predict(X_tune.reshape(-1, 1))

            q_preds_low.append(q_low_pred)
            q_preds_high.append(q_high_pred)

        q_preds_low = np.array(q_preds_low).T  # shape (n_cal, p)
        q_preds_high = np.array(q_preds_high).T

        # Convert the data to conformal format
        tune_preds = convert_quantile_predictions(q_preds_low, q_preds_high, alpha)
        tune_data = {task: {'posteriors': tune_preds[:, i, :],
                            'gt': Y_tune[:, i]} for i, task in enumerate(tasks)}



        for method in methods:

            # Each method should use the similar calibration and test data
            #np.random.seed(32)

            print('Method:', method, 'n_train:', n_train)

            indep = 'independent' if independent else 'correlated'

            save_dir = os.path.join('../results/toy_problem',
                                    f'{configs[method]["conformal_method"]}_{configs[method]["normalization_type"]}',
                                    f'{indep}',
                                    'diff_ntrains',
                                    f'n_train_{n_train}',
                                    f'alpha_{alpha}',
                                    )

            # Create the directory if it doesn't exist
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)


            # Configuratiosn of conformal predictor
            cf = OmegaConf.create({
                'tasks': tasks,
                'conformal_method': configs[method]['conformal_method'], # Conformal method (naive, unified, shaped, whiten, joint_unified)
                'interval_type': 'cqr', # 'nonadaptive', 'cqr', 'lwr'
                'local_cov_weight': 0.0, # Weight for local covariance
                'normalization_type': configs[method]['normalization_type'], # 'standardize', 'quantile', 'none', 'score_scaling', 'score_shift','score_shift_scale', 'length', 'quantile_score'
                'alpha_correction_type': configs[method]['alpha_correction_type'], # 'none', 'bonferroni', 'independent'
                'alpha': alpha, #0.05 # Error rate
                'eval': {
                    'num_trials': 10000, # Number of trials for evaluation
                    'calib_split': 0.7, # Calibration split
                    'train_split': 0.3, # Train split
                }
            })


            # Store all the outputs
            all_outputs = {}


            for t in tqdm(range(n_trials)):
            #%% --------------------------
            # 2. Calibrate on new data
            # --------------------------

                # Use less data for calibration if you are tuning
                if cf.normalization_type == 'quantile_score':
                    X_cal, Y_cal = generate_date(n_cal, independent=independent)
                else:
                    X_cal, Y_cal = generate_date(n_cal + n_tune, independent=independent)


                q_preds_low = []
                q_preds_high = []

                # Make predictions
                for j in range(p):

                    # Predict on the calibration set
                    q_low_pred = qr_lows[j].predict(X_cal.reshape(-1, 1))
                    q_high_pred = qr_highs[j].predict(X_cal.reshape(-1, 1))

                    q_preds_low.append(q_low_pred)
                    q_preds_high.append(q_high_pred)

                q_preds_low = np.array(q_preds_low).T  # shape (n_cal, p)
                q_preds_high = np.array(q_preds_high).T

                # Convert the data to conformal format
                cal_preds = convert_quantile_predictions(q_preds_low, q_preds_high, alpha)
                cal_data = {task: {'posteriors': cal_preds[:, i, :],
                                   'gt': Y_cal[:, i]} for i, task in enumerate(tasks)}

                # Create the conformal object
                cinf = conformal.select_conformal(cf, cal_data, train_task_outputs=tune_data)

                # Calibrate
                cinf.calibrate(c=2) # Note: c set to 2 since we change to have only two predictions so that give use the QR predictions




            #%% --------------------------
            # 3. Test on new data
            # --------------------------
                X_test, Y_test = generate_date(n_test, independent=independent)

                q_pred_low_test, q_pred_high_test = [], []

                # Make predictions
                for j in range(p):

                    # Predict on the test set
                    q_low_pred = qr_lows[j].predict(X_test.reshape(-1, 1))
                    q_high_pred = qr_highs[j].predict(X_test.reshape(-1, 1))

                    q_pred_low_test.append(q_low_pred)
                    q_pred_high_test.append(q_high_pred)

                q_pred_low_test = np.array(q_pred_low_test).T
                q_pred_high_test = np.array(q_pred_high_test).T

                # Convert the data to conformal format
                test_preds = convert_quantile_predictions(q_pred_low_test, q_pred_high_test, alpha)
                test_data = {task: {'posteriors': test_preds[:, i, :],
                                    'gt': Y_test[:, i]} for i, task in enumerate(tasks)}

                # Get the test predictions
                outputs = cinf.test(test_data, c=2)

                # Collect all of the outputs
                for task_name in task_list:
                    if task_name not in all_outputs:
                        all_outputs[task_name] = {}

                    for output_name in outputs[task_name]:
                        if output_name not in all_outputs[task_name]:
                            all_outputs[task_name][output_name] = []

                        all_outputs[task_name][output_name].append(outputs[task_name][output_name])

                    if 'joint_coverage' not in all_outputs:
                        all_outputs['joint_coverage'] = []
                    all_outputs['joint_coverage'].append(outputs['joint_coverage'])



            #%% Get the average results
            all_task_coverages = []
            outputs = all_outputs

            # Save the results in txt file
            with open(os.path.join(save_dir, f'coverage.txt'), 'w') as f:
                for i, task in enumerate(task_list):
                    coverage = outputs[task]['coverage']
                    coverage_mean = np.mean(coverage, axis=0)
                    coverage_std_err = np.std(coverage, axis=0) / math.sqrt(len(coverage))
                    f.write(f'{task} risk mean: {coverage_mean:.4f} +/- {coverage_std_err:.4f}\n')
            # print(f'{task} risk mean: {risk_mean:.4f}, std: {risk_std:.4f}')

            # print(f'{task} interval measurement mean: {interval_measurement_mean.item():.4f}')
            with open(os.path.join(save_dir, f'interval_measurement.txt'), 'w') as f:
                # Get the mean interval measurement
                for task in task_list:
                    interval_measurement = outputs[task]['interval_measurement']
                    interval_measurement_mean = np.mean(np.mean(interval_measurement, axis=1))
                    f.write(f'{task} interval measurement mean: {interval_measurement_mean.item():.4f}\n')

            # Get the joint coverage
            joint_coverage = outputs['joint_coverage']
            joint_coverage_mean = np.mean(joint_coverage)
            joint_coverage_std_err = np.std(joint_coverage) / math.sqrt(len(joint_coverage))
            print(f'Joint coverage: {joint_coverage_mean:.4f} +/- {joint_coverage_std_err:.4f}')
            with open(os.path.join(save_dir, 'joint_coverage.txt'), 'w') as f:
                f.write(f'Joint coverage: {joint_coverage_mean:.4f} +/- {joint_coverage_std_err:.4f}')

            # Save the outputs
            with open(os.path.join(save_dir, 'results.pkl'), 'wb') as f:
                pickle.dump(outputs, f)



    #%% Plot individual coverages across number of training samples
    experiment_dir = '../results/toy_problem'

    save_dir = os.path.join(experiment_dir, 'plots')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.rcParams.update({'font.size': 20})

    indep = 'correlated' # 'independent' or 'correlated'
    configs = ['naive_none', 'joint_unified_length', 'unified_quantile_score']
    config_names = ['independent', 'quantile length', 'ours']
    colors = ['blue', 'orange', 'green']
    alpha = 0.1 #[0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    n_trains = [50, 200, 500, 1000, 2000, 5000,10000]
    markers = ['x', '^', 'o', '<', '>']

    results = {}

    # Get the results for each config
    for i, config in enumerate(configs):
        results[config] = {}
        results[config]['min_coverages'] = []
        results[config]['max_coverages'] = []

        for n_train in n_trains:
            # Get the directory to load the results
            load_dir = os.path.join(experiment_dir, config, f'{indep}', 'diff_ntrains', f'n_train_{n_train}', f'alpha_{alpha}')

            # Load the results
            with open(os.path.join(load_dir, 'coverage.txt'), 'r') as f:
                lines = f.readlines()

            # Get the coverages
            coverages = []
            for line in lines:
                if '+/-' in line:
                    metric, value = line.split('risk mean:')
                    value = value.split('+/-')[0].strip()
                    coverages.append(float(value))

            results[config]['min_coverages'].append(min(coverages))
            results[config]['max_coverages'].append(max(coverages))

    # Plot the results
    n_trains = np.array(n_trains)
    plt.figure()
    for i, config in enumerate(configs):
        plt.plot(n_trains, results[config]['min_coverages'], color=colors[i], marker=markers[i], markersize=10, linewidth=2)
        plt.plot(n_trains, results[config]['max_coverages'], color=colors[i], marker=markers[i], markersize=10, linewidth=2)
        plt.fill_between(n_trains, results[config]['min_coverages'], results[config]['max_coverages'], alpha=0.5,
                         color = colors[i],
                         label=config_names[i])

    # Plot line for desired coverage
    #plt.plot(desired_coverages, desired_coverages, linestyle='--', color='k', label='Desired Coverage')

    # plt.xlabel('Number of Training Samples')
    # plt.ylabel('Single Task Coverage')
    plt.ylim([0.93,0.98])
    # plt.legend(loc='lower right')

    # plt.grid(alpha=0.5)
    # plt.show()
    plt.savefig(os.path.join(save_dir, f'single_target_coverage_diff_train.pdf'), dpi=1200)
    plt.close()











