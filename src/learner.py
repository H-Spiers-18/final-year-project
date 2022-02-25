from abc import ABC, abstractmethod
from time import time
from enum import Enum

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

import constants


class InvalidAccuracyMeasureException(Exception):
    """Custom exception for when an accuracy measure parameter other than 'mse' or 'mape' is passed to get_accuracy()"""
    pass


class CrossValScoringMethods(Enum):
    """Used to access the non-functional property boundary indexes for each subject system's dataset
    (i.e. which column number does the configuration options stop and the non-functional property measurements begin)"""
    MAPE = constants.MAPE_SCORING
    MSE = constants.MSE_SCORING


class Learner(ABC):
    """Abstract class for ML models. Contains the model itself and a number of results-gathering methods"""

    def __init__(self):
        self.model = None
        self.training_time = 0
        super().__init__()

    @abstractmethod
    def fit(self, x_train, y_train, premade_model=None):
        """
        Trains a machine learning model given one or more input samples and expected outcomes
        Implemented in PredictorLearner and TransferLearner
        Parameters
        ----------
        x_train: numpy.ndarray - array of training sample feature vectors
        y_train: numpy.ndarray - array of measured training sample performance values
        premade_model: sklearn model - Optional parameter for an optimised model defined using cross validation

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def predict(self, x_test):
        """
        Predicts the performance value of one or more run-time configurations. Implemented in PredictorLearner and
        TransferLearner
        Parameters
        ----------
        x_test: numpy.ndarray - array of test sample feature vectors

        Returns
        -------
        None
        """
        pass

    @staticmethod
    def get_optimal_params(X_validate, y_validate):
        """
        Perform a grid search of all possible hyperparameter configurations with 5-fold cross validation and MAPE
        to find the best performing hyperparameter set. Implemented in PredictorLearner and TransferLearner
        Parameters
        ----------
        X_validate: numpy.ndarray - array of validation set feature vectors
        y_validate: numpy.ndarray - array of validation set measured performance values

        Returns
        -------
        None
        """
        pass

    def get_error(self, xs, ys, measure='mape'):
        """
        Calculate the accuracy of the machine learning model using either MSE or MAPE
        Parameters
        ----------
        xs: numpy.ndarray - 2D array (shape (N,1)) of training feature vectors
        ys: numpy.ndarray - 1D array of performance values for target compile-time configuration
        measure: str - defines which error function to use (possible values are 'mape' and 'mse')

        Returns
        -------
        numpy.ndarray - contains the measured error of each input prediction value
        """
        measure = measure.upper() # convert measure to upper to ensure it matches CrossValScoringMethods key
        return cross_val_score(self.model, xs, ys, cv=5, scoring=CrossValScoringMethods[measure].value)

    def get_training_time(self):
        """
        Gets time taken to train the model
        Returns
        -------
        float - time taken in seconds to train model
        """
        return self.training_time*1000


class PredictorLearner(Learner):
    """Class for training and using a model for predicting the performance values for run-time configurations using a
    regression tree learner. Implements abstract class Learner"""

    def __init__(self):
        super().__init__()

    def fit(self, x_train, y_train, premade_model=None):
        """
        Implements abstract method. Fits data using a regression tree. For more info, see Learner.fit
        Returns
        -------
        None
        """
        start_time = time()
        # check if hyperparameter-optimised model is provided by premade_model param
        if premade_model is not None:
            self.model = premade_model
        else:
            self.model = DecisionTreeRegressor()
        # train the model
        self.model.fit(x_train, y_train)
        self.training_time = time()-start_time

    def predict(self, x_test):
        """
        Implements abstract method. Predicts the performance value of one or more run-time configurations.
        For more info, see learner.predict

        Returns
        -------
        y_pred: numpy.ndarray - predicted performance values for each of the input samples
        """
        y_pred = self.model.predict(x_test)
        return y_pred

    @staticmethod
    def get_optimal_params(X_validate, y_validate):
        """
        Implements abstract method. Perform a grid search of all possible hyperparameter configurations with
        5-fold cross validation and MAPE to find the best performing hyperparameter set.
        Parameters
        ----------
        X_validate: numpy.ndarray - array of validation set feature vectors
        y_validate: numpy.ndarray - array of validation set measured performance values

        Returns
        -------
        2-tuple containing:
        DecisionTreeRegressor - untrained regression tree containing the hyperparameters which exhibited the lowest MAPE
        float - MAPE score of best performing configuration
        """
        param_grid = constants.REGRESSION_TREE_PARAM_GRID
        temp_model = DecisionTreeRegressor()
        # test out the performance every possible permutation of hyperparameters using 5-fold cross validation
        cv = GridSearchCV(temp_model, param_grid, cv=5, scoring='neg_mean_absolute_percentage_error')
        cv.fit(X_validate, y_validate)
        # return the estimator with the lowest error, along with the MAPE that that estimator achieved
        return cv.best_estimator_, max(cv.cv_results_['mean_test_score']) * -100


class TransferLearner(Learner):
    """Class for training and using a model for transferring run-time configurations between compile-time configurations
     using a linear regression learner. Implements abstract class Learner"""

    def fit(self, x_train, y_train, premade_model=None):
        """
        Implements abstract method fit(). Fits data using linear regression. For more info, see Learner.fit
        Returns
        -------
        None
        """
        start_time = time()
        # check if hyperparameter-optimised model is provided by premade_model param
        if premade_model is not None:
            self.model = premade_model
        else:
            self.model = LinearRegression()
        # train the model
        self.model.fit(x_train, y_train)
        self.training_time = time() - start_time

    def predict(self, x_test):
        """
        Implements abstract method predict(). Predicts the performance value of one or more run-time configurations.
        For more info, see learner.predict

        Returns
        -------
        y_pred: numpy.ndarray - predicted performance values for each of the input samples
        """
        y_pred = self.model.predict(x_test)
        return y_pred

    @staticmethod
    def get_optimal_params(X_validate, y_validate):
        """
        Implements abstract method. Perform a grid search of all possible hyperparameter configurations with
        5-fold cross validation and MAPE to find the best performing hyperparameter set.
        Parameters
        ----------
        X_validate: numpy.ndarray - array of validation set feature vectors
        y_validate: numpy.ndarray - array of validation set measured performance values

        Returns
        -------
        2-tuple containing:
        DecisionTreeRegressor - untrained regression tree containing the hyperparameters which exhibited the lowest MAPE
        float - MAPE score of best performing configuration
        """
        param_grid = constants.LINEAR_REGRESSION_PARAM_GRID
        temp_model = LinearRegression()
        # test out the performance every possible permutation of hyperparameters using 5-fold cross validation
        cv = GridSearchCV(temp_model, param_grid, cv=5, scoring='neg_mean_absolute_percentage_error')
        cv.fit(X_validate, y_validate)
        # return the estimator with the lowest error, along with the MAPE that that estimator achieved
        return cv.best_estimator_, max(cv.cv_results_['mean_test_score']) * -100
