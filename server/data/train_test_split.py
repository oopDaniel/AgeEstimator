#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random
import shutil
import dirs


# In[2]:


def main():
    dirs.init_dir()

    # Shuffle
    random.shuffle(dirs.dataset_dir)

    # 80/20 split
    n_test = len(dirs.dataset_dir) * 2 // 10

    for src in dirs.dataset_dir[:n_test]:
        dst = os.path.join(dirs.test_dir, os.path.split(src)[-1])
        shutil.copyfile(src, dst)

    for src in dirs.dataset_dir[n_test:]:
        dst = os.path.join(dirs.train_dir, os.path.split(src)[-1])
        shutil.copyfile(src, dst)


# In[3]:


if __name__ == "__main__":
    main()
