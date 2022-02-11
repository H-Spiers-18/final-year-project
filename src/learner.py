from abc import ABC, abstractmethod


class Learner(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass


class PredictorLearner(Learner):

    def fit(self):
        pass

    def predict(self):
        pass


class TransferLearner(Learner):

    def fit(self):
        pass

    def predict(self):
        pass
