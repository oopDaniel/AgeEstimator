r"""Extended data loader that supports validation set.
Import this module instead of calling it in CLI.
"""
import glob
import os
from server.data.dataset import DataLoader as DL


class DataLoader(DL):
    r"""The data loader supports "fixed" validation set, which
    may be useful in deep learning in particular.
    """

    def __init__(self):
        super(DataLoader, self).__init__()
        self.valid_dir = os.path.join(
            os.path.split(self.train_dir)[0], "valid")
        self.x_valid = glob.glob(os.path.join(self.valid_dir, "*.jpg"))
        self.y_valid = list(map(DL.to_label, self.x_valid))

    def load_valid(self):
        r"""File names of validation data and their labels"""
        return self.x_valid, self.y_valid
© 2020 GitHub, Inc.