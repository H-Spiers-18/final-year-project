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
