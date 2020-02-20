import re
import matplotlib.image as img
import numpy as np


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
