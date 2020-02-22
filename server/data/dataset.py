#!/usr/bin/env python
# coding: utf-8

# In[15]:


import glob
import os
from sklearn.model_selection import train_test_split
from . import dirs
from . import data_file_utils


# In[20]:


class DataLoader():
    def __init__(self):
        def to_label(x): return data_file_utils.get_age_by_file_name(
            os.path.split(x)[-1])

        self.x_train = glob.glob(os.path.join(dirs.train_dir, "*.jpg"))
        self.x_test = glob.glob(os.path.join(dirs.test_dir, "*.jpg"))
        self.y_train = list(map(to_label, self.x_train))
        self.y_test = list(map(to_label, self.x_test))

    def load_train(self):
        r"""File names of training data and their labels"""
        return self.x_train, self.y_train

    def load_test(self):
        r"""File names of test data and their labels"""
        return self.x_test, self.y_test

    def load_train_val_split(self, shuffle=False):
        r"""File names of training/validation data and
            their labels"""
        return train_test_split(
            self.x_train, self.y_train, test_size=0.2, shuffle=shuffle)
