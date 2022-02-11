from abc import ABC, abstractmethod

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse

import constants
from time import time


class InvalidAccuracyMeasureException(Exception):
    pass


class Learner(ABC):

    def __init__(self):
        self.model = None
        self.training_time = 0
        super().__init__()

    @abstractmethod
    def fit(self, x_train, y_train):
        pass

    @abstractmethod
    def predict(self, x_test):
        pass

    @staticmethod
    def get_accuracy(y_test, y_pred, measure='mape'):
        measure = measure.upper()
        if measure == 'MAPE':
            return mape(y_test, y_pred)
        elif measure == 'MSE':
            return mse(y_test, y_pred)
        else:
            raise InvalidAccuracyMeasureException(constants.INVALID_ACCURACY_MEASURE_MSG)

    def get_training_time(self):
        return self.training_time*1000


class PredictorLearner(Learner):

    def __init__(self):
        super().__init__()

    def fit(self, x_train, y_train):
        start_time = time()
        self.model = DecisionTreeRegressor(random_state=20)
        self.model.fit(x_train, y_train)
        self.training_time = time()-start_time

    def predict(self, x_test):
        y_test = self.model.predict(x_test)
        return y_test


class TransferLearner(Learner):

    def fit(self, x_train, y_train):
        pass

    def predict(self, x_test):
        pass
