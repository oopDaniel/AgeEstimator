#!/usr/bin/env python
# coding: utf-8

# In[72]:


import glob
import re
import matplotlib.image as img
import os
import shutil
import utils


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


# In[58]:


# Calculate padding for file names
padding_digits = calc_padding_digits(["utkface/", "CACD/"])
target_dir = "dim_unified"

init_dir(target_dir)


# ## UTKFace

# In[60]:


age_labels_utk = []
for i, file_name in enumerate(glob.glob("utkface/*.jpg")):
    image = img.imread(file_name)

    file_name = re.sub(r'^\D+\/', '', file_name)
    age = utils.get_age_by_file_name(file_name)
    age_labels_utk.append(age)

    utils.resize_rgb_img(image, (250, 250), '%s/utk_%s.jpg' % (
        target_dir,
        str(i + 1).zfill(padding_digits)
    ))


# ## CACD

# In[64]:


age_labels_cacd = []
for i, file_name in enumerate(glob.glob("CACD/*.jpg")):
    file_name = re.sub(r'^\D+\/', '', file_name)
    age = utils.get_age_by_file_name(file_name)
    age_labels_cacd.append(age)

    shutil.copyfile("CACD/" + file_name, '%s/cacd_%s.jpg' % (
        target_dir,
        str(i + 1).zfill(padding_digits)
    ))


# ## Create label file

# In[66]:


labels = [*age_labels_cacd, *age_labels_utk]


# In[75]:


utils.save_result_as_file(labels, "labels.dat")


# In[ ]:
