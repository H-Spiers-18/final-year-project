import os
import itertools

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import constants


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

    less_than_count = greater_than_count = 0
    for x in s1:
        greater_than_count += np.sum(s2 > x)
        less_than_count += np.sum(s2 < x)

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
    csv_filename = 'rq'+str(rq)+'.csv'
    output_dir = os.path.join('../results/', subject_system.lower())
    csv_path = os.path.join(output_dir, csv_filename)

    print('Writing output to', csv_path, '\n')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results.to_csv(csv_path, index=False)
    add_effect_size_to_csv(wilcox_p, cliffs_delta, csv_path, rq)
    print('Output written to', csv_path, '\n')


def add_effect_size_to_csv(wilcox_p, cliffs_delta, csv_path, rq):
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
    with open(csv_path, 'a') as results_file:
        if rq == 1 or rq == 2:
            results_file.write(
                '\n' + 'Wilcoxon p value: ' + str(wilcox_p[0]) + '\n' +
                'Cliff\'s delta: ' + str(cliffs_delta[0]))
            results_file.close()

        elif rq == 3:
            results_file.write(
                '\n' + 'Wilcoxon p value CV: ' + str(wilcox_p[0]) + '\n' +
                'Cliff\'s delta CV: ' + str(cliffs_delta[0]))
            results_file.write(
                '\n' + 'Wilcoxon p value no CV: ' + str(wilcox_p[1]) + '\n' +
                'Cliff\'s delta no CV: ' + str(cliffs_delta[1]))
            results_file.close()


def read_results_csv(csv_path):
    """
    Read results (without p values) from csv to pandas dataframe
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


def analyse_results():
    """
    Perform analysis of all results csv files and output analysis to csv files
    Returns
    -------
    None
    """
    for rq_csv, rq_analysis_folder in zip(constants.RQ_CSV_NAMES, constants.RQ_ANALYSIS_FOLDERS):
        mean_min_max_dfs = []
        # get the mean, min and max values from each research question results csv file
        for subject_system_path in constants.SUBJECT_SYSTEM_PATHS:
            results_csv = os.path.join(subject_system_path, rq_csv)
            results = read_results_csv(results_csv)
            mean_min_max_dfs.append(get_mean_and_minmax(results))

        # combine all subject systems' results into a single dataframe
        out = pd.concat(mean_min_max_dfs, ignore_index=True)
        # label each row to enhance readability
        out['measurement'] = ['nodejs mean', 'nodejs min', 'nodejs max',
                              'x264 mean', 'x264 min', 'x264 max',
                              'xz mean', 'xz min', 'xz max']

        # create output folder if it doesn't exist and write analysis to csv
        if not os.path.exists(rq_analysis_folder):
            os.makedirs(rq_analysis_folder)
        out.to_csv(os.path.join(rq_analysis_folder, 'mean_min_max.csv'))
