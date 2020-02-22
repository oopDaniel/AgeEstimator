import glob
import os
import shutil

curr_dir = os.path.dirname(os.path.realpath(__file__))
dataset_dir = glob.glob(os.path.join(curr_dir, "images", "*.jpg"))
train_dir = os.path.join(curr_dir, "dataset", "train")
test_dir = os.path.join(curr_dir, "dataset", "test")


def init_dir():
    try:
        shutil.rmtree(train_dir)
        shutil.rmtree(test_dir)
    except:
        pass

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
