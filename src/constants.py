# Contains constants for my codebase. used to keep values consistent across the whole codebase

# Cross validation number of folds
CV_FOLDS = 5

# Number of times to run cross validation to take avg accuracy
CV_RUNS = 5

# Experiment repetitions
EXPERIMENT_REPS = 25

# Subject systems
SUBJECT_SYSTEMS = ['nodejs', 'x264', 'xz']

# Subject system paths
SUBJECT_SYSTEM_PATHS = ['../results/nodejs', '../results/x264', '../results/xz']

# RQ analysis folders
RQ_ANALYSIS_PATHS = ['../results/_analysis/rq1_analysis',
                       '../results/_analysis/rq2_analysis',
                       '../results/_analysis/rq3_analysis']

# Folders to contain transfer model scatter plots
SCATTER_PLOT_PATHS = ['../results/_analysis/rq1_analysis/transfer_model_scatter_plots',
                      '../results/_analysis/rq2_analysis/transfer_model_scatter_plots']

# RQ csv names
RQ_CSV_NAMES = ['rq1.csv', 'rq2.csv', 'rq3.csv']

# Mean min max output csv row descriptions
MEAN_MIN_MAX_ROW_DESCRIPTIONS = ['nodejs mean', 'nodejs min', 'nodejs max',
                                 'x264 mean', 'x264 min', 'x264 max',
                                 'xz mean', 'xz min', 'xz max']

# Results fields to use in each RQ's box plot
BOX_PLOT_FIELDS = [[['mape_accuracy_trans_cv', 'mape_accuracy_tgt_cv'], ['mape_accuracy_trans_no_cv', 'mape_accuracy_tgt_no_cv']],
                   [['mape_accuracy_trans_20pct', 'mape_accuracy_trans_40pct', 'mape_accuracy_trans_60pct', 'mape_accuracy_trans_80pct'],
                    ['mape_accuracy_pred_20pct', 'mape_accuracy_pred_40pct', 'mape_accuracy_pred_60pct', 'mape_accuracy_pred_80pct']],
                   [['training_time_trans_20pct_cv', 'training_time_trans_40pct_cv', 'training_time_trans_60pct_cv', 'training_time_trans_80pct_cv'],
                    ['training_time_pred_20pct_cv', 'training_time_pred_40pct_cv', 'training_time_pred_60pct_cv', 'training_time_pred_80pct_cv']]]

# Box plot descriptions to include in filenames
BOX_PLOT_DESCRIPTIONS = [['mape_error_optimised_', 'mape_error_unoptimised_'],
                         ['mape_error_transfer_model_', 'mape_error_predictor_model_'],
                         ['training_time_transfer_model_', 'training_time_predictor_model_']]

# y axis labels for matplotlib plots
Y_AXIS_LABELS = ['MAPE error (%)', 'MAPE error (%)', 'Training time (ms)']

# Dataset folder paths
NODEJS_PATH = '../dataset/nodejs'
POPPLER_PATH = '../dataset/poppler'
X264_PATH = '../dataset/x264'
XZ_PATH = '../dataset/xz'

# Dataset csv paths
NODEJS_CSV_PATH = 'buffer1.csv'
POPPLER_CSV_PATH = 'apache_camel.csv'
X264_CSV_PATH = 'original_videos_Animation_480P_Animation_480P-087e.csv'
XZ_CSV_PATH = 'dickens.csv'

# Index for each dataset where the configuration options end and non-functional properties begin
NODEJS_NF_BOUNDARY = -1
POPPLER_NF_BOUNDARY = -2
X264_NF_BOUNDARY = -5
XZ_NF_BOUNDARY = -2

# Sklearn scoring parameters to convert from my names for scoring techniques to sklearn's

MAPE_SCORING = 'neg_mean_absolute_percentage_error'
MSE_SCORING = 'neg_mean_squared_error'

# Exception messages

INVALID_ACCURACY_MEASURE_MSG = 'Invalid accuracy measure parameter. Available measures are \'MSE\' and \'MAPE\''

# Hyperparameter grids

MAX_DEPTH = list(range(3, 30)) + [None]
MIN_SAMPLES_SPLIT = list(range(2, 10))

REGRESSION_TREE_PARAM_GRID = {'max_depth': MAX_DEPTH,
                              'min_samples_split': MIN_SAMPLES_SPLIT}

FIT_INTERCEPT = [True, False]
COPY_X = [True, False]
POSITIVE = [True, False]

LINEAR_REGRESSION_PARAM_GRID = {'fit_intercept': FIT_INTERCEPT,
                                'copy_X': COPY_X,
                                'positive': POSITIVE}
