import numpy as np
import re


def save_result_as_file(res_list, file_name):
    r"""Save the result into a new file.

    @Args:
        res_list:   a list of string or int
        file_name:  file name prefixed by the age label
    """
    file_content = "\n".join(list(map(str, res_list)))
    with open(file_name, "w") as fd:
        fd.write(file_content)


def load_int_data_into_list(path):
    r"""Load a file by the given path as a list of strings
        separated by lines, and convert elements into int.

    @Args:
        path:   string path to the file
    @Return:
        List of int
    """
    with open(path, "r") as fd:
        lst = [int(line.rstrip('\n')) for line in fd]
    return lst


def load_as_np_list(path):
    r"""Load a file by the given path as a Numpy list of int.

    @Args:
        path:   string path to the file
    @Return:
        Numpy list of int
    """
    return np.array(load_int_data_into_list(path))


def get_age_by_file_name(file_name):
    r"""Extract the age label from the given file name

    @Args:
        file_name:  file name prefixed by the age label
    @Returns:
                    The age in int
    """
    return int(re.split(r"_", file_name)[0])
