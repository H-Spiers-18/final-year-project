import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

import constants as constants

# create output folders when this module is imported if they don't exist already
output_dirs = constants.RQ_ANALYSIS_PATHS + \
              list(constants.SUBJECT_SYSTEM_PATHS.values()) + \
              [y for l in constants.SCATTER_PLOT_PATHS.values() for y in l]  # flatten into 1d list
for path in output_dirs:
    if not os.path.exists(path):
        os.makedirs(path)


def get_wilcoxon_p_value(s1, s2):
    """
    Performs the Wilcoxon test to determine whether the difference between the test and predicted values
    is statistically significant
    Parameters
    ----------
    s1: numpy.ndarray - array of prediction values supporting the null hypothesis
    s2: numpy.ndarray - array of prediction values supporting the alternative hypothesis
    Returns
    -------
    p: float - p-value used to quantify the difference between y_test and y_pred. if the value is <= 0.05 then
               we can consider the difference to be statistically significant
    """
    _, p = wilcoxon(s1, s2)
    return p


def get_cliffs_delta(s1, s2):
    """
    Performs the Cliff's delta test to determine the effect size of my measurements. Based on the following link's
    implementation, just written to efficiently use numpy arrays - https://github.com/txt/ase16/blob/master/doc/stats.md
    Parameters
    ----------
    s1: numpy.ndarray - array of prediction values supporting the null hypothesis
    s2: numpy.ndarray - array of prediction values supporting the alternative hypothesis
    Returns
    -------
    z: float - z-value used to determine effect size.
               z < 0.147 - negligible effect size
               0.147 <= z < 0.33 - small effect size,
               0.33 <= z < 0.474 - medium effect size,
               0.474 <= z - large effect size
               (see figure 11 in report, based on D. Chen et. al [31])
    """
    # ensure that the 2 samples are numpy arrays
    if type(s1) != np.ndarray:
        s1 = np.array(s1)
    if type(s2) != np.ndarray:
        s2 = np.array(s2)

    # for each sample in sample 1, count the number of samples greater than and less than the sample
    less_than_count = greater_than_count = 0
    for x in s1:
        greater_than_count += np.sum(s2 > x)
        less_than_count += np.sum(s2 < x)

    # calculate cliff's delta
    z = abs(less_than_count - greater_than_count) / (len(s1) * len(s2))
    return z


def save_results(results, subject_system, wilcox_p, cliffs_delta, rq):
    """
    Save results to csv file
    Parameters
    ----------
    results: pd.DataFrame - DataFrame containing recorded results
    subject_system: str - subject system that the results belong to
    wilcox_p: list - results' wilcoxon p value(s) (with and optionally without cross validation)
    cliffs_delta: float - results' cliff's delta value(s) (with and optionally without cross validation)
    rq: int - which research question the csv belongs to

    Returns
    -------
    None
    """
    # grab what the output filename and output path
    output_dir, csv_filename = constants.SUBJECT_SYSTEM_PATHS[subject_system], constants.RQ_CSV_NAMES[rq-1]
    csv_path = os.path.join(output_dir, csv_filename)

    results.to_csv(csv_path, index=False)
    add_effect_size_to_csv(results, rq)
    print('Output written to', csv_path, '\n')


def add_effect_size_to_csv(results, rq):
    """
    Add Wilcoxon's p value and cliff's delta as a footer to the given csv file
    Parameters
    ----------
    wilcox_p: list - results' wilcoxon p value(s) (with and optionally without cross validation)
    cliffs_delta: float - results' cliff's delta value(s) (with and optionally without cross validation)
    csv_path: str - path to the results csv file
    rq: int - which research question the csv belongs to

    Returns
    -------
    None
    """
    csv_path = constants.STAT_TEST_CSV_NAMES[rq-1]
    wilcox_p = cliffs_delta = 0
    out = ''
    dataset_size_table = [['', '20%', '40%', '60%', '80%'],
                          ['20%', '', '', '', ''],
                          ['40%', '', '', '', ''],
                          ['60%', '', '', '', ''],
                          ['80%', '', '', '', '']]
    if rq == 1:
        wilcox_p = get_wilcoxon_p_value(results['mape_accuracy_tgt_cv'], results['mse_accuracy_trans_cv'])
        cliffs_delta = get_cliffs_delta(results['mape_accuracy_tgt_cv'], results['mse_accuracy_trans_cv'])
        out = 'Wilcoxon p value: ' + str(wilcox_p) + '\n' + 'Cliff\'s delta: ' + str(cliffs_delta)

    elif rq == 2:
        out =  'Wilcoxon p value: ' + '\n' + dataset_size_table + '\n\n' + \
               'Cliff\'s delta: ' + '\n' + dataset_size_table

    elif rq == 3:
        out = 'Wilcoxon p value: ' + '\n' + dataset_size_table + '\n\n' + \
              'Cliff\'s delta: ' + '\n' + dataset_size_table

    with open(csv_path, 'w') as results_file:
        results_file.write(out)

    results_file.close()


def read_results_csv(csv_path):
    """
    Read results (without p values and effect size) from csv into pandas dataframe
    Parameters
    ----------
    csv_path: str - path to the results csv

    Returns
    -------
    results: pd.DataFrame - dataframe containing experiment results
    """
    results = pd.read_csv(csv_path)[:constants.EXPERIMENT_REPS]
    return results.astype(np.float64)


def get_mean_and_minmax(results):
    """
    Gets the mean, minimum, and maximum values for each results column
    Parameters
    ----------
    results: pd.DataFrame - dataframe containing experiment results

    Returns
    -------
    tuple(mean, min, max)  - contains mean, min and max values from each results column
    """
    cols = results.columns
    out = pd.DataFrame()
    results = np.abs(results)
    for col in cols:
        mean = np.mean(results[col])
        _min = np.min(results[col])
        _max = np.max(results[col])
        out[col] = [mean, _min, _max]

    return out


def write_mean_min_max(rq):
    """
    Perform analysis of all results csv files and output analysis to csv files
    Parameters
    ----------
    rq: int - number representing the research question results currently being analysed

    Returns
    -------
    None
    """
    # get input csv name and output path
    rq_csv, rq_analysis_folder = constants.RQ_CSV_NAMES[rq-1], constants.RQ_ANALYSIS_PATHS[rq-1]
    mean_min_max_dfs = []

    # get the mean, min and max values from each results csv file and append them to a list
    for subject_system in constants.SUBJECT_SYSTEMS:
        results_csv = os.path.join(constants.SUBJECT_SYSTEM_PATHS[subject_system], rq_csv)
        results = read_results_csv(results_csv)
        mean_min_max_dfs.append(get_mean_and_minmax(results))

    # combine all subject systems' mean, min, max into a single dataframe
    out = pd.concat(mean_min_max_dfs, ignore_index=True)
    # label each row with what it represents to enhance readability
    out['measurement'] = constants.MEAN_MIN_MAX_ROW_DESCRIPTIONS
    out.to_csv(os.path.join(rq_analysis_folder, 'mean_min_max.csv'))


def make_box_plots(rq, show_outliers=False):
    """
    Creates a box plot for the data from each research question and writes to file
    Parameters
    ----------
    rq: int - number representing the research question results currently being analysed
    show_outliers: bool - determines whether or not to include outliers in the box plot

    Returns
    -------
    None
    """
    rq_csv, rq_analysis_folder, box_plot_fields, box_plot_filename_descriptions, y_label = \
        constants.RQ_CSV_NAMES[rq-1], \
        constants.RQ_ANALYSIS_PATHS[rq-1], \
        constants.BOX_PLOT_FIELDS[rq-1], \
        constants.BOX_PLOT_DESCRIPTIONS[rq-1], \
        constants.Y_AXIS_LABELS[rq-1]

    # loop through each subject system and create box plots for it
    for subject_system in constants.SUBJECT_SYSTEMS:
        subject_system_path = constants.SUBJECT_SYSTEM_PATHS[subject_system]
        results_csv = os.path.join(subject_system_path, rq_csv)
        results = read_results_csv(results_csv)

        for fields, box_plot_description in zip(box_plot_fields, box_plot_filename_descriptions):
            # plot results in box plot
            fig, ax = plt.subplots()
            ax.boxplot(results[fields].values, showfliers=show_outliers)
            ax.set_title(box_plot_description[:-1] + ' - ' + subject_system)
            ax.set_ylabel(y_label, fontsize=16)
            ax.set_xticklabels(fields, fontsize=12, rotation=30, horizontalalignment='right')
            plt.savefig(os.path.join(rq_analysis_folder, box_plot_description + subject_system + '_box_plot.png'),
                        bbox_inches='tight')


def make_transfer_model_scatter_plot(model, x_train, y_train, rq, mape_error,
                                     experiment_rep, subject_system, dataset_size=''):
    """
    Creates and writes a scatter plot containing the source and target data points and linear regression model line
    Parameters
    ----------
    model: sklearn.LinearRegression - Trained transfer model
    x_train: numpy.ndarray - Source compile-time configuration performance measurements
    y_train: numpy.ndarray - Target compile-time configuration performance measurements
    rq: int - Number representing the research question results currently being analysed
    mape_error: float - Cross-validation MAPE error (not the error between the model and plotted data)
    experiment_rep: int - Current experiment repetition

    Returns
    -------
    None
    """
    output_dir = constants.SCATTER_PLOT_PATHS[subject_system.lower()][rq-1]
    fig, ax = plt.subplots()
    ax.scatter(x_train, y_train)
    ax.plot(x_train, model.predict(x_train))
    ax.set_title('MAPE error: ' + str(mape_error))
    ax.set_ylabel('Predicted target performance value', fontsize=12)
    ax.set_xlabel('Measured source performance value', fontsize=12)
    plt.savefig(os.path.join(output_dir, 'rep_' + str(experiment_rep) + '_' + str(dataset_size) + '.png'),
                bbox_inches='tight')
    plt.clf()
