import os

import src.constants as constants
from src.analysis import *


def test_output_structure():
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
    scatter_plot_paths = constants.RQ_ANALYSIS_PATHS[:2]  # only check RQ1 and 2 since 3 has no scatter plots
    for rq_analysis_path in scatter_plot_paths:
        _path = os.path.join(rq_analysis_path, 'transfer_model_scatter_plots')
        assert sorted(os.listdir(_path)) == constants.SUBJECT_SYSTEMS

        for subject_system in constants.SUBJECT_SYSTEMS:
            assert sorted(os.listdir(os.path.join(_path, subject_system))) == \
                   sorted(['rep_'+str(x+1)+'.png' for x in range(constants.EXPERIMENT_REPS)])
