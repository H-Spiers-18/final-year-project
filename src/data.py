from enum import Enum

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

import constants


class NFPropertyBoundaryIndexes(Enum):
    """Used to access the non-functional property boundary indexes for each subject system's dataset
    (i.e. which column number does the configuration options stop and the non-functional property measurements begin)"""
    NODEJS = constants.NODEJS_NF_BOUNDARY
    POPPLER = constants.POPPLER_NF_BOUNDARY
    X264 = constants.X264_NF_BOUNDARY
    XZ = constants.XZ_NF_BOUNDARY


def get_transfer_dataset(xs_source, ys_source, xs_target, ys_target, random_state=42):
    """
    Splits 2 datasets into a single train/test split for transfer learning between compile-time configurations
    Parameters
    ----------
    xs_source: numpy.ndarray - 2D array (shape (N,1)) of measured performance values for source compile-time
                               configuration
    ys_source: numpy.ndarray - 1D array of performance values for source compile-time configuration
    xs_target: numpy.ndarray - 2D array (shape (N,1)) of measured performance values for target compile-time
                               configuration
    ys_target: numpy.ndarray - 1D array of performance values for target compile-time configuration

    Returns
    -------
    A training set (X_train, y_train) and a validation set (X_validate, y_validate)
    X_train: numpy.ndarray - 2D array (shape (N,1)) of measured performance values for source compile-time configuration
    X_validate: numpy.ndarray - 2D array (shape (N,1)) of measured performance values for source compile-time
                                configuration
    y_train: numpy.ndarray - 1D array of performance values for target compile-time configuration
    y_validate: numpy.ndarray - 1D array of performance values for target compile-time configuration
    """
    # split both datasets. we ignore the first 2 values since we only want the performance values for our transfer model
    _, _, X_train, X_validate = Dataset.get_split_dataset(xs_source, ys_source, random_state=random_state)
    _, _, y_train, y_validate = Dataset.get_split_dataset(xs_target, ys_target, random_state=random_state)
    # convert the source configuration performance values from shape (N) to (N,1) since that's the shape sklearn wants
    # from feature vectors regardless of their dimensionality
    X_train = np.array(list(map(lambda x: np.array([x]), X_train)))
    X_validate = np.array(list(map(lambda x: np.array([x]), X_validate)))

    return X_train, X_validate, y_train, y_validate


class Dataset:
    """Used to contain dataset used for training models"""

    dataset: pd.DataFrame
    features: np.ndarray
    values: np.ndarray

    def __init__(self, csv_path, subject_system):
        self.dataset = self.__set_dataset(csv_path)
        self.__prepare_dataset(subject_system)
        self.features, self.values = self.__get_features_and_values(subject_system)

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

    def __prepare_dataset(self, subject_system):
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
        # grab the column index at which the configuration options stop and the non-functional property values begin
        nf_boundary = NFPropertyBoundaryIndexes[subject_system.upper()].value

        # change nominal feature values to numeric
        for col in cols[1:nf_boundary]:
            self.dataset[col] = le.fit_transform(self.dataset[col].to_numpy())

    def __get_features_and_values(self, subject_system):
        """
        Gets and returns the feature vectors and measured performance values for all samples in our dataset
        Parameters
        ----------
        subject_system: str - says which subject system the dataset belongs to. value can be nodejs, poppler, x264 or xz

        Returns
        -------
        xs: numpy.ndarray - 2d array containing feature vectors for all samples
        ys: numpy.ndarray - 1d array containing all performance values
        """
        cols = self.dataset.columns.values
        # grab the column index at which the configuration options stop and the non-functional property values begin
        nf_boundary = NFPropertyBoundaryIndexes[subject_system.upper()].value

        # split our dataset into feature vectors and measured performance values
        # skip the first item since that's just the index of the run-time configuration
        features = self.dataset[cols[1:nf_boundary]].to_numpy()
        # grab the performance value
        values = self.dataset[cols[-1]].to_numpy()

        return features, values

    def get_split_dataset(self, validation_size=0.2, random_state=42):
        """
        Splits dataset into training and test set
        Parameters
        ----------
        validation_size: float - portion of dataset to assign as the test set. range between 0-1
        random_state: int - Seed for random data sampling

        Returns
        -------
        X_train: numpy.ndarray - array of training set feature vectors
        X_validate: numpy.ndarray - array of validation set feature vectors
        y_train: numpy.ndarray - array of measured training set performance values
        y_validate: numpy.ndarray - array of measured validation set performance values
        """
        return train_test_split(self.features, self.values, test_size=validation_size, random_state=random_state)
