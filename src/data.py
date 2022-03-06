import os
import random
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


def get_random_datasets():
    """
    Selects 2 random datasets for each subject system for each experiment repetition (without replacement)
    Returns
    -------
    datasets: dict of Dataset - A tuple for each subject system, each with a source and target dataset (8 in total)
    """
    nodejs_datasets = []
    poppler_datasets = []
    x264_datasets = []
    xz_datasets = []
    for rep in range(constants.EXPERIMENT_REPS):
        # randomly select 2 datasets per subject system per experiment repetition
        nodejs_tgt_src = \
            tuple(random.sample([Dataset(os.path.join(constants.NODEJS_PATH, ctime_dir, constants.NODEJS_CSV_PATH),
                                         'nodejs')
                                for ctime_dir in os.listdir(constants.NODEJS_PATH)
                                if os.path.isdir(os.path.join(constants.NODEJS_PATH, ctime_dir))], 2))
        poppler_tgt_src = \
            tuple(random.sample([Dataset(os.path.join(constants.POPPLER_PATH, ctime_dir, constants.POPPLER_CSV_PATH),
                                         'poppler')
                                for ctime_dir in os.listdir(constants.POPPLER_PATH)
                                if os.path.isdir(os.path.join(constants.POPPLER_PATH, ctime_dir))], 2))
        x264_tgt_src = \
            tuple(random.sample([Dataset(os.path.join(constants.X264_PATH, ctime_dir, constants.X264_CSV_PATH), 'x264')
                                for ctime_dir in os.listdir(constants.X264_PATH)
                                if os.path.isdir(os.path.join(constants.X264_PATH, ctime_dir))], 2))
        xz_tgt_src = \
            tuple(random.sample([Dataset(os.path.join(constants.XZ_PATH, ctime_dir, constants.XZ_CSV_PATH), 'xz')
                                for ctime_dir in os.listdir(constants.XZ_PATH)
                                if os.path.isdir(os.path.join(constants.XZ_PATH, ctime_dir))], 2))

        nodejs_datasets.append(nodejs_tgt_src)
        poppler_datasets.append(poppler_tgt_src)
        x264_datasets.append(x264_tgt_src)
        xz_datasets.append(xz_tgt_src)

    datasets = {
        'NODEJS': nodejs_datasets,
        'POPPLER': poppler_datasets,
        'X264': x264_datasets,
        'XZ': xz_datasets
    }

    return datasets


def get_transfer_dataset(d_src, d_target, train_size=0.8, validation_size=0.2, random_state=42):
    """
    Splits 2 datasets into a single train/test split for transfer learning between compile-time configurations
    Parameters
    ----------
    d_src: Dataset - Dataset object for source compile-time configuration
    d_target: Dataset - Dataset object for target compile-time configuration
    train_size: float - Size of the training set
    validation_size: float - Size of the validation set
    random_state: int - Seed for random sampler

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
    _, _, X_train, X_validate = d_src.get_split_dataset(train_size=train_size,
                                                        validation_size=validation_size,
                                                        random_state=random_state)
    _, _, y_train, y_validate = d_target.get_split_dataset(train_size=train_size,
                                                           validation_size=validation_size,
                                                           random_state=random_state)
    # add an extra dimension to the source performance measurements since that's the input shape that sklearn wants
    X_train = np.array(list(map(lambda x: np.array([x]), X_train)))
    X_validate = np.array(list(map(lambda x: np.array([x]), X_validate)))

    return X_train, X_validate, y_train, y_validate


class Dataset:
    """Used to contain dataset used for training models"""

    dataset: pd.DataFrame
    features: np.ndarray
    values: np.ndarray
    csv_path: str

    def __init__(self, csv_path, subject_system):
        self.csv_path = csv_path
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

    def get_csv_path(self):
        return self.csv_path

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

    def get_split_dataset(self, train_size=0.8, validation_size=0.2, random_state=42):
        """
        Splits dataset into training and test set
        Parameters
        ----------
        train_size: float - portion of dataset to assign as the training set. range between 0-1
        validation_size: float - portion of dataset to assign as the validation set. range between 0-1
        random_state: int - Seed for random data sampling

        Returns
        -------
        X_train: numpy.ndarray - array of training set feature vectors
        X_validate: numpy.ndarray - array of validation set feature vectors
        y_train: numpy.ndarray - array of measured training set performance values
        y_validate: numpy.ndarray - array of measured validation set performance values
        """
        return train_test_split(self.features,
                                self.values,
                                train_size=train_size,
                                test_size=validation_size,
                                random_state=random_state)
