# Contains constants for my codebase. used to keep values consistent across the whole codebase

# Index for each dataset where the configuration options end and non-functional properties begin
NODEJS_NF_BOUNDARY = -1
POPPLER_NF_BOUNDARY = -2
X264_NF_BOUNDARY = -5
XZ_NF_BOUNDARY = -2

# Exception messages

INVALID_ACCURACY_MEASURE_MSG = 'Invalid accuracy measure parameter. Available measures are \'MSE\' and \'MAPE\''

# hyperparameter grids

MAX_DEPTH = list(range(3, 50)) + [None]
MIN_SAMPLES_SPLIT = list(range(2, 10))
MIN_WEIGHT_FRACTION_LEAF = [x*0.1 for x in range(0, 5)]
MAX_FEATURES = ['auto', 'sqrt', 'log2']
MAX_LEAF_NODES = list(range(3, 50)) + [None]

REGRESSION_TREE_PARAM_GRID = {'max_depth': MAX_DEPTH}
                              #'min_samples_split': MIN_SAMPLES_SPLIT,
                              #'min_weight_fraction_leaf': MIN_WEIGHT_FRACTION_LEAF,
                              #'max_features': MAX_FEATURES,
                              #'max_leaf_nodes': MAX_LEAF_NODES}

FIT_INTERCEPT = [True, False]
COPY_X = [True, False]
POSITIVE = [True, False]

LINEAR_REGRESSION_PARAM_GRID = {'fit_intercept': FIT_INTERCEPT,
                                'copy_X': COPY_X,
                                'positive': POSITIVE}