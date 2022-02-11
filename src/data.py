from enum import Enum

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

import constants


class NFPropertyBoundaryIndexes(Enum):
    """Used to access the non-functional property boundary indexes for each subject system's dataset
    (i.e. which column number does the configuration options stop and the non-functional property measurements begin)"""
    NODEJS = constants.NODEJS_NF_BOUNDARY
    POPPLER = constants.POPPLER_NF_BOUNDARY
    X264 = constants.X264_NF_BOUNDARY
    XZ = constants.XZ_NF_BOUNDARY


class Dataset:
    """Used to contain dataset used for testing and training models"""

    dataset: pd.DataFrame

    def __init__(self, csv_path):
        self.dataset = self.__set_dataset(csv_path)

    @staticmethod
    def __set_dataset(csv_path):
        """
        Reads dataset from csv file into a pandas dataframe
        Parameters
        ----------
        csv_path: str - path to the csv dataset (can be relative or absolute)

        Returns
        -------
        pandas.DataFrame - dataframe containing the dataset. read in the same column format as it is in the csv file
        """
        return pd.read_csv(csv_path)

    def get_dataset(self):
        """
        Get dataset
        Returns
        -------
        pandas.DataFrame - dataset's current state
        """
        return self.dataset

    def prepare_dataset(self, subject_system):
        """
        Prepares the dataset so that it is ready for the predictor learner. this involves:
            - split the dataset into xs (configuration options) and ys (measured performance value)
            - using LabelEncoder to encode all string feature values into integers based on classes
            - crop the data so that only the configuration options remain in the feature set
        Parameters
        ----------
        subject_system: str - says which subject system the dataset belongs to. value can be nodejs, poppler, x264 or xz

        Returns
        -------
        xs: numpy.ndarray - array of prepared feature vectors for all input samples
        ys: numpy.ndarray - array of performance values for all input samples
        """
        le = LabelEncoder()
        cols = self.dataset.columns.values
        nf_boundary = NFPropertyBoundaryIndexes[subject_system.upper()].value

        for col in cols[1:nf_boundary]:
            self.dataset[col] = le.fit_transform(self.dataset[col].to_numpy())

        xs = self.dataset[cols[1:nf_boundary]].to_numpy()
        ys = self.dataset[cols[-1]].to_numpy()

        return xs, ys

    @staticmethod
    def get_split_dataset(xs, ys, test_size=0.2):
        """
        Splits dataset into training and test set
        Parameters
        ----------
        xs: numpy.ndarray - array of prepared feature vectors for all input samples
        ys: numpy.ndarray - array of performance values for all input samples
        test_size: float - portion of dataset to assign as the test set. range between 0-1

        Returns
        -------
        X_train: numpy.ndarray - array of training sample feature vectors
        X_test: numpy.ndarray - array of test sample feature vectors
        y_train: numpy.ndarray - array of measured training sample performance values
        y_test: numpy.ndarray - array of test sample performance values
        """
        return train_test_split(xs, ys, test_size=test_size, shuffle=True)

