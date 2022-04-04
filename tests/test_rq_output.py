import os

import src.constants as constants
from src.analysis import read_results_csv


def test_results_structure():
    """Tests that the results output structure is as expected"""
    for subject_system in constants.SUBJECT_SYSTEMS:
        assert os.listdir(constants.SUBJECT_SYSTEM_PATHS[subject_system]) == ['rq1.csv', 'rq2.csv', 'rq3.csv']


def test_rq_csvs_contents():
    """Checks that the results csv contents are as they should be. Includes checking column names and column lengths"""
    expected_columns = constants.RESULTS_DATAFRAME_COLUMN_NAMES
    # read all results CSVs
    results_csvs = [read_results_csv(os.path.join('../results/', subject_system, rq_csv))
                    for subject_system, rq_csv in zip(constants.SUBJECT_SYSTEMS, constants.RQ_CSV_NAMES)]
    for x in range(len(results_csvs)):
        results_csv = results_csvs[x]
        assert all(results_csv.columns == expected_columns[x])
        assert len(results_csv[expected_columns[x][-1]]) == constants.EXPERIMENT_REPS
