from enum import Enum

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

import constants


class NFPropertyBoundaryIndexes(Enum):
    NODEJS = constants.NODEJS_NF_BOUNDARY
    POPPLER = constants.POPPLER_NF_BOUNDARY
    X264 = constants.X264_NF_BOUNDARY
    XZ = constants.XZ_NF_BOUNDARY


class Dataset:

    dataset: pd.DataFrame

    def __init__(self, csv_path):
        self.dataset = self.__set_dataset(csv_path)

    @staticmethod
    def __set_dataset(csv_path):
        return pd.read_csv(csv_path)

    def get_dataset(self):
        return self.dataset

    def prepare_dataset(self, subject_system):
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
        return train_test_split(xs, ys, test_size=test_size, shuffle=True)

