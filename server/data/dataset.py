#!/usr/bin/env python
# coding: utf-8

# In[15]:


import glob
import os
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

    @staticmethod
    def to_label(full_path):
        r"""Convert a full file path into its integer label"""
        return data_file_utils.get_age_by_file_name(
            os.path.split(full_path)[-1])

    def load_train(self):
        r"""File names of training data and their labels"""
        return self.x_train, self.y_train

    def load_test(self):
        r"""File names of test data and their labels"""
        return self.x_test, self.y_test
