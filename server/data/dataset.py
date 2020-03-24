#!/usr/bin/env python
# coding: utf-8

# In[15]:


import glob
import os
import numpy as np
from . import dirs
from . import data_file_utils


# In[20]:


class DataLoader(object):
    def __init__(self):

        self.x_train = glob.glob(os.path.join(dirs.train_dir, "*.jpg"))
        self.x_test = glob.glob(os.path.join(dirs.test_dir, "*.jpg"))
        self.y_train = list(map(DataLoader.to_label, self.x_train))
        self.y_test = list(map(DataLoader.to_label, self.x_test))

        self.train_dir = dirs.train_dir
        self.test_dir = dirs.test_dir
        self.feature_dir = dirs.feature_dir

    @staticmethod
    def to_label(full_path):
        r"""Convert a full file path into its integer label"""
        return data_file_utils.get_age_by_file_name(
            os.path.split(full_path)[-1])

    def load_train(self, feature=False, flatten=True):
        r"""File names of training data and their labels"""
        if not feature:
            return self.x_train, self.y_train

        x_train = np.load(os.path.join(self.feature_dir,
                                       "features-train-worker0.npy"))
        y_train = np.load(os.path.join(
            self.feature_dir, "labels-train-worker0.npy"))

        if flatten:
            x_train = x_train.reshape((x_train.shape[0], -1))
            y_train = y_train.reshape((y_train.shape[0], -1))
        return x_train, y_train

    def load_test(self, feature=False, flatten=True):
        r"""File names of test data and their labels"""
        if not feature:
            return self.x_test, self.y_test

        x_test = np.load(os.path.join(
            self.feature_dir, "features-test-worker0.npy"))
        y_test = np.load(os.path.join(
            self.feature_dir, "labels-test-worker0.npy"))

        if flatten:
            x_test = x_test.reshape((x_test.shape[0], -1))
            y_test = y_test.reshape((y_test.shape[0], -1))
        return x_test, y_test
