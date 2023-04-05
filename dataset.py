import os
from abc import ABC
from typing import Optional

import numpy as np
import pandas as pd
import scipy.io as io


def load_data_mat(filename, max_samples, seed=42):
    raw = io.loadmat(filename)
    X = raw['X']  # Array of [32, 32, 3, n_samples]
    y = raw['y']  # Array of [n_samples, 1]
    X = np.moveaxis(X, [3], [0])
    y = y.flatten()
    # Fix up class 0 to be 0
    y[y == 10] = 0

    np.random.seed(seed)
    samples = np.random.choice(np.arange(X.shape[0]),
                               max_samples,
                               replace=False)

    return X[samples].astype(np.float32), y[samples]


def load_svhn(folder, max_train, max_test):
    train_X, train_y = load_data_mat(os.path.join(folder, "train_32x32.mat"), max_train)
    test_X, test_y = load_data_mat(os.path.join(folder, "test_32x32.mat"), max_test)
    return train_X, train_y, test_X, test_y


def random_split_train_val(X, y, num_val, seed=42):
    np.random.seed(seed)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    train_indices = indices[:-num_val]
    train_X = X[train_indices]
    train_y = y[train_indices]

    val_indices = indices[-num_val:]
    val_X = X[val_indices]
    val_y = y[val_indices]

    return train_X, train_y, val_X, val_y


def prepare_for_neural_network(train_X, test_X):
    train_flat = train_X.reshape(train_X.shape[0], -1).astype(np.float64) / 255.0
    test_flat = test_X.reshape(test_X.shape[0], -1).astype(np.float64) / 255.0

    # Subtract mean
    mean_image = np.mean(train_flat, axis=0)
    train_flat -= mean_image
    test_flat -= mean_image

    return train_flat, test_flat


class Dataset(ABC):
    X: Optional[np.ndarray]
    y: Optional[np.ndarray]

    def __init__(self):
        self.batch_size = None
        self.batches_indices = None
        self.val_y = None
        self.val_X = None
        self.train_y = None
        self.train_X = None
        self.iter = -1

    def random_split_train_val(self):
        self.train_X, self.train_y, self.val_X, self.val_y = random_split_train_val(
            self.train_X, self.train_y, num_val=int(self.train_y.shape[0] * 0.25)
        )

    def set_batches_indices(self):
        shuffled_indices = np.arange(self.train_X.shape[0])
        np.random.shuffle(shuffled_indices)
        sections = np.arange(self.batch_size, self.train_X.shape[0], self.batch_size)
        self.batches_indices = np.array_split(shuffled_indices, sections)

    def get_dim_features(self):
        return self.train_X.shape[1], len(np.unique(self.train_y))

    def mode(self, mode='train'):
        if mode == 'train':
            self.X, self.y = self.train_X, self.train_y
        elif mode == 'val':
            self.X, self.y = self.val_X, self.val_y

    def __iter__(self):
        return self

    def __next__(self):
        self.iter += 1
        if self.iter >= len(self.batches_indices):
            self.iter = -1
            raise StopIteration

        batch_ind = self.batches_indices[self.iter]
        return self.X[batch_ind], self.y[batch_ind]


class TitanicDataset(Dataset):

    def __init__(
            self,
            csv_data_path='data/titanic.csv', csv_sep=',',
            batch_size=32, drop_column=None,
    ):
        super().__init__()
        self.val_y = None
        self.val_X = None
        self.test_y = None
        self.test_X = None
        self.batches_indices = None

        if drop_column is None:
            drop_column = ['PassengerId', 'Name']

        self.batch_size = batch_size

        df = pd.read_csv(csv_data_path, sep=csv_sep)
        df = df.drop(drop_column, axis=1)

        cat_columns = df.select_dtypes(exclude=[int, float]).columns
        df[cat_columns] = df[cat_columns].apply(lambda x: pd.factorize(x)[0])
        self.columns = df.columns
        df = df.fillna(-1).to_numpy()

        self.train_X, self.train_y = df[:, 1:], df[:, 0].astype(int)
        self.random_split_train_val()
        self.set_batches_indices()


class SVHNDataset(Dataset):
    """Utility class to hold training and validation data"""

    def __init__(self, data_path="./data", batch_size=32):
        super().__init__()
        self.batches_indices = None
        self.val_y = None
        self.val_X = None
        self.X, self.y = None, None
        self.iter = -1

        self.train_X, self.train_y, self.test_X, self.test_y = load_svhn(data_path, max_train=10000, max_test=1000)
        self.train_X, self.test_X = prepare_for_neural_network(self.train_X, self.test_X)

        self.random_split_train_val()

        self.batch_size = batch_size
        self.set_batches_indices()

    def get_data(self):
        """Return train/test split"""
        return self.train_X, self.train_y, self.test_X, self.test_y

    def set_data(self, X, y):
        self.train_X, self.train_y = X, y
        self.test_X, self.test_y = X, y

        self.random_split_train_val()
        self.set_batches_indices()
