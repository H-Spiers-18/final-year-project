from abc import ABC, abstractmethod

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse

import constants
from time import time


class InvalidAccuracyMeasureException(Exception):
    """Custom exception for when an accuracy measure parameter other than 'mse' or 'mape' is passed to get_accuracy()"""
    pass


class Learner(ABC):
    """Abstract class for ML models. Contains the model itself and a number of results-gathering methods"""

    def __init__(self):
        self.model = None
        self.training_time = 0
        super().__init__()

    @abstractmethod
    def fit(self, x_train, y_train):
        """
        Trains a machine learning model given one or more input samples and expected outcomes
        Implemented in PredictorLearner and TransferLearner
        Parameters
        ----------
        x_train: numpy.ndarray - array of training sample feature vectors
        y_train: numpy.ndarray - array of measured training sample performance values

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def predict(self, x_test):
        """
        Predicts the performance value of one or more run-time configurations
        Parameters
        ----------
        x_test: numpy.ndarray - array of test sample feature vectors

        Returns
        -------
        None
        """
        pass

    @staticmethod
    def get_error(y_test, y_pred, measure='mape'):
        """
        Calculate the accuracy of the machine learning model using either MSE or MAPE
        Parameters
        ----------
        y_test: numpy.ndarray - array of test sample performance values
        y_pred: numpy.ndarray - array of predicted performance values
        measure: str - defines which error function to use (possible values are 'mape' and 'mse')

        Returns
        -------
        numpy.ndarray - contains the measured error of each input prediction value
        """
        measure = measure.upper()
        if measure == 'MAPE':
            return mape(y_test, y_pred)*100
        elif measure == 'MSE':
            return mse(y_test, y_pred)
        else:
            raise InvalidAccuracyMeasureException(constants.INVALID_ACCURACY_MEASURE_MSG)

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

    def fit(self, x_train, y_train):
        """
        Implements abstract method fit(). Fits data using a regression tree. For more info, see Learner.fit
        Returns
        -------
        None
        """
        start_time = time()
        self.model = DecisionTreeRegressor(random_state=20)
        self.model.fit(x_train, y_train)
        self.training_time = time()-start_time

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


class TransferLearner(Learner):
    """Class for training and using a model for transferring run-time configurations between compile-time configurations
     using a linear regression learner. Implements abstract class Learner"""

    def fit(self, x_train, y_train):
        """
        Implements abstract method fit(). Fits data using linear regression. For more info, see Learner.fit
        Returns
        -------
        None
        """
        start_time = time()
        self.model = LinearRegression()
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
