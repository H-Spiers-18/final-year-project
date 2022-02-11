from abc import ABC, abstractmethod

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse

import constants


class InvalidAccuracyMeasureException(Exception):
    pass


class Learner(ABC):

    def __init__(self):
        self.model = None
        super().__init__()

    @abstractmethod
    def fit(self, x_train, y_train):
        pass

    @abstractmethod
    def predict(self, x_test):
        pass

    @staticmethod
    def get_accuracy(self, y_test, y_pred, measure='mape'):
        measure = measure.upper()
        if measure == 'MAPE':
            return mape(y_test, y_pred)
        elif measure == 'MSE':
            return mse(y_test, y_pred)
        else:
            raise InvalidAccuracyMeasureException(constants.INVALID_ACCURACY_MEASURE_MSG)


class PredictorLearner(Learner):

    def __init__(self):
        super().__init__()

    def fit(self, x_train, y_train):
        self.model = DecisionTreeRegressor(random_state=20)
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        y_test = self.model.predict(x_test)
        return y_test


class TransferLearner(Learner):

    def fit(self, x_train, y_train):
        pass

    def predict(self, x_test):
        pass
