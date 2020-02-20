#!/usr/bin/env python
# coding: utf-8

# In[72]:


import glob
import os
import re
import shutil
import matplotlib.image as img
import numpy as np


# ## Utils

# In[ ]:


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


def get_age_by_file_name(file_name):
    r"""Extract the age label from the given file name

    @Args:
        file_name:  file name prefixed by the age label
    @Returns:
                    The age in int
    """
    return int(re.split(r"_", file_name)[0])


def save_result_as_file(res_list, file_name):
    r"""Save the result into a new file

    @Args:
        res_list:   a list of string or int
        file_name:  file name prefixed by the age label
    """
    file_content = "\n".join(list(map(str, res_list)))
    with open(file_name, "w") as fd:
        fd.write(file_content)


# In[56]:


def calc_padding_digits(directories):
    max_digits = 0
    for directory in directories:
        size = len([name for name in os.listdir(directory)
                    if os.path.isfile(directory + name)])
        digits = len(str(size))
        max_digits = max(digits, max_digits)
    return max_digits


# In[57]:


def init_dir(dir_name):
    try:
        shutil.rmtree(dir_name)
    except:
        pass

    os.mkdir(dir_name)


# ## UTKFace

# In[60]:


def preprocess_utkface(target_dir, padding_digits):
    age_labels = []

    for i, file_name in enumerate(glob.glob("utkface/*.jpg")):
        image = img.imread(file_name)

        file_name = re.sub(r'^\D+\/', '', file_name)
        age = get_age_by_file_name(file_name)
        age_labels.append(age)

        resize_rgb_img(image, (250, 250), '%s/utk_%s.jpg' % (
            target_dir,
            str(i + 1).zfill(padding_digits)
        ))

    return age_labels


# ## CACD

# In[64]:


def preprocess_cacd(target_dir, padding_digits):
    age_labels = []

    for i, file_name in enumerate(glob.glob("CACD/*.jpg")):
        file_name = re.sub(r'^\D+\/', '', file_name)
        age = get_age_by_file_name(file_name)
        age_labels.append(age)

        shutil.copyfile("CACD/" + file_name, '%s/cacd_%s.jpg' % (
            target_dir,
            str(i + 1).zfill(padding_digits)
        ))

    return age_labels


# ## Main

# In[ ]:


def main():
    # Calculate padding for file names
    padding_digits = calc_padding_digits(["utkface/", "CACD/"])
    target_dir = "dim_unified"

    init_dir(target_dir)

    # Preprocessing
    age_labels_utk = preprocess_utkface(target_dir, padding_digits)
    age_labels_cacd = preprocess_cacd(target_dir, padding_digits)

    # Create label file
    labels = [*age_labels_cacd, *age_labels_utk]
    save_result_as_file(labels, "labels.dat")


# In[ ]:


if __name__ == "__main__":
    main()
