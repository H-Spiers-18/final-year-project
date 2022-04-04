import os

import src.constants as constants
from src.analysis import read_results_csv


def test_results_structure():
    """Tests that the results output structure is as expected"""
    for subject_system in constants.SUBJECT_SYSTEMS:
        assert os.listdir(constants.SUBJECT_SYSTEM_PATHS[subject_system]) == ['rq1.csv', 'rq2.csv', 'rq3.csv']


def test_rq_csvs_contents():
    """Checks that the results csv contents are as they should be. Includes checking column names and column lengths"""
    expected_columns = [
        ['mse_accuracy_tgt_no_cv', 'mape_accuracy_tgt_no_cv', 'mse_accuracy_tgt_cv', 'mape_accuracy_tgt_cv',
         'mse_accuracy_trans_no_cv', 'mape_accuracy_trans_no_cv', 'mse_accuracy_trans_cv', 'mape_accuracy_trans_cv'],
        ['mape_accuracy_pred_20pct', 'mape_accuracy_pred_40pct', 'mape_accuracy_pred_60pct',
         'mape_accuracy_pred_80pct', 'mape_accuracy_trans_20pct', 'mape_accuracy_trans_40pct',
         'mape_accuracy_trans_60pct', 'mape_accuracy_trans_80pct', 'mse_accuracy_pred_20pct',
         'mse_accuracy_pred_40pct', 'mse_accuracy_pred_60pct', 'mse_accuracy_pred_80pct',
         'mse_accuracy_trans_20pct', 'mse_accuracy_trans_40pct', 'mse_accuracy_trans_60pct',
         'mse_accuracy_trans_80pct'],
        ['training_time_pred_20pct_no_cv', 'training_time_pred_40pct_no_cv',
         'training_time_pred_60pct_no_cv',
         'training_time_pred_80pct_no_cv', 'training_time_pred_20pct_cv', 'training_time_pred_40pct_cv',
         'training_time_pred_60pct_cv', 'training_time_pred_80pct_cv',
         'training_time_trans_20pct_no_cv',
         'training_time_trans_40pct_no_cv', 'training_time_trans_60pct_no_cv',
         'training_time_trans_80pct_no_cv',
         'training_time_trans_20pct_cv', 'training_time_trans_40pct_cv', 'training_time_trans_60pct_cv',
         'training_time_trans_80pct_cv']
        ]
    # read all results CSVs
    results_csvs = [read_results_csv(os.path.join('../results/', subject_system, rq_csv))
                    for subject_system, rq_csv in zip(constants.SUBJECT_SYSTEMS, constants.RQ_CSV_NAMES)]
    for x in range(len(results_csvs)):
        results_csv = results_csvs[x]
        assert all(results_csv.columns == expected_columns[x])
        assert len(results_csv[expected_columns[x][-1]]) == constants.EXPERIMENT_REPS