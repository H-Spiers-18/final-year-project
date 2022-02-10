from sklearn.model_selection import train_test_split
import pandas as pd


class Dataset:

    dataset: pd.DataFrame

    def __init__(self, csv_path):
        self.dataset = self.__set_dataset(csv_path)

    @staticmethod
    def __set_dataset(csv_path):
        return pd.read_csv(csv_path)

    def get_dataset(self):
        return self.dataset

    def split_dataset(self, prepped_dataset):
        return train_test_split(prepped_dataset)

