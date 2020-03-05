#!/usr/bin/env python
# coding: utf-8

# In[4]:


import glob
import os
import re
import shutil
import matplotlib.image as img
import numpy as np
from server.data import data_file_utils


# ## Utils

# In[3]:


def resize_rgb_img(image, new_shape=(100, 100), file_name=None):
    r"""Resize the given RGB image.

    @Args:
        image:      the image read by "matplotlib.image.imread"
        new_shape:  tuple of dimension of new rows and new columns
        file_name:  optional file name deciding saving the file on disk
                    or return the resized image
    @Returns:
                    The resized image if no file name passed in
    """
    assert(len(image.shape) == 3)  # row + col + channel
    assert(image.shape[2] == 3)

    new_rows, new_cols = new_shape

    old_rows = image.shape[0]
    old_cols = image.shape[1]

    ratio_r = new_rows / old_rows
    ratio_c = new_cols / old_cols

    pos_row = np.floor(np.arange(old_rows * ratio_r) / ratio_r).astype('int64')
    pos_col = np.floor(np.arange(old_cols * ratio_c) / ratio_c).astype('int64')

    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    red = r[pos_row, :]
    red = red[:, pos_col]
    green = g[pos_row, :]
    green = green[:, pos_col]
    blue = b[pos_row, :]
    blue = blue[:, pos_col]

    output_img = np.zeros([new_rows, new_cols, 3])
    output_img[:, :, 0] = red
    output_img[:, :, 1] = green
    output_img[:, :, 2] = blue

    if file_name:
        img.imsave(file_name, output_img.astype(np.uint8))

    else:
        return output_img


# In[57]:


def init_dir(dir_name):
    try:
        shutil.rmtree(dir_name)
    except:
        pass

    os.mkdir(dir_name)


# ## UTKFace

# In[7]:


def preprocess_utkface(target_dir, padding_digits=5):
    for i, file_name in enumerate(glob.glob("utkface/*.jpg")):
        image = img.imread(file_name)

        file_name = re.sub(r'^\D+/', '', file_name)
        age = data_file_utils.get_age_by_file_name(file_name)

        resize_rgb_img(image, (250, 250), '%s/%d_utk_%s.jpg' % (
            target_dir,
            age,
            str(i + 1).zfill(padding_digits)
        ))


# ## Main

# In[ ]:


def main():
    target_dir = "dim_unified"
    init_dir(target_dir)

    # Preprocessing
    preprocess_utkface(target_dir)


# In[ ]:


if __name__ == "__main__":
    main()
