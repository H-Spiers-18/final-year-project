import os

import pandas as pd

import src.constants as constants
from src.analysis import *


def test_output_structure():
    """Check that the analysis output file structure is as expected"""
    assert os.path.exists('../results/_analysis')
    assert sorted(os.listdir('../results/_analysis/rq1_analysis')) == ['mape_error_optimised_nodejs_box_plot.png',
                                                                       'mape_error_optimised_x264_box_plot.png',
                                                                       'mape_error_optimised_xz_box_plot.png',
                                                                       'mape_error_unoptimised_nodejs_box_plot.png',
                                                                       'mape_error_unoptimised_x264_box_plot.png',
                                                                       'mape_error_unoptimised_xz_box_plot.png',
                                                                       'mean_min_max.csv',
                                                                       'transfer_model_scatter_plots']
    assert sorted(os.listdir('../results/_analysis/rq2_analysis')) == ['mape_error_predictor_model_nodejs_box_plot.png',
                                                                       'mape_error_predictor_model_x264_box_plot.png',
                                                                       'mape_error_predictor_model_xz_box_plot.png',
                                                                       'mape_error_transfer_model_nodejs_box_plot.png',
                                                                       'mape_error_transfer_model_x264_box_plot.png',
                                                                       'mape_error_transfer_model_xz_box_plot.png',
                                                                       'mean_min_max.csv',
                                                                       'transfer_model_scatter_plots']
    assert sorted(os.listdir('../results/_analysis/rq3_analysis')) == ['mean_min_max.csv',
                                                                       'training_time_predictor_model_nodejs_box_plot.png',
                                                                       'training_time_predictor_model_x264_box_plot.png',
                                                                       'training_time_predictor_model_xz_box_plot.png',
                                                                       'training_time_transfer_model_nodejs_box_plot.png',
                                                                       'training_time_transfer_model_x264_box_plot.png',
                                                                       'training_time_transfer_model_xz_box_plot.png']


def test_scatter_plot_output():
    """Check that the scatter plot output file structure is as expected. Includes checking that plots for all subject
    systems are present and that there is a plot for each experiment repetition"""
    scatter_plot_paths = constants.RQ_ANALYSIS_PATHS[:2]  # only check RQ1 and 2 since 3 has no scatter plots

    for rq_analysis_path in scatter_plot_paths:
        _path = os.path.join(rq_analysis_path, 'transfer_model_scatter_plots')
        # check that scatter plots have been created for all subject systems
        assert sorted(os.listdir(_path)) == constants.SUBJECT_SYSTEMS


def test_mean_min_max_output():
    """Check that the mean min max file output is as expected. Includes checking that the mean, min and max values are
    calculated for each subject system"""
    for rq_analysis_path in constants.RQ_ANALYSIS_PATHS:
        mean_min_max_csv = pd.read_csv(os.path.join(rq_analysis_path, 'mean_min_max.csv'))
        assert len(mean_min_max_csv['measurement']) == 9
        assert all(mean_min_max_csv['measurement'] == ['nodejs mean', 'nodejs min', 'nodejs max',
                                                       'x264 mean', 'x264 min', 'x264 max',
                                                       'xz mean', 'xz min', 'xz max'])
