r"""This script provides the model definition and the path
to pre-trained weight(s).

Import this module instead of calling it in CLI.
"""
import os
import itertools
import pandas as pd

from tensorflow.keras.optimizers import Nadam, Adam, SGD
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, BatchNormalization, Dropout


# Tested. No need to import them unless necessary
# from server.models.cnn.model_combination_resnet50 import *
# from server.models.cnn.model_combination_vgg16 import *
from server.models.cnn.model_combination_vggface import *

# --------------- Global Vars -------------------

OLD_WEIGHTS_PATH = "old_nn_classification_weights.hdf5"
BEST_WEIGHTS_PATH = "best_nn_classification_weights.hdf5"

# We define our IMAGE_SIZE here because 250x250 is too big for VGGFace
IMAGE_SIZE = (224, 224)
IMAGE_SHAPE = IMAGE_SIZE + (3,)
INPUT_SHAPE = 2048

# Make compatible path for Python and Jupyter Notebook
try:
    CURR_DIR = os.path.dirname(os.path.realpath(__file__))
except NameError:
    CURR_DIR = os.getcwd()

LABEL_MAPPING = pd.Series.from_csv(
    os.path.join(CURR_DIR, "age_class_mapping.csv"),
    header=0).to_dict()
N_CLASSES = len(set(LABEL_MAPPING.values()))

# ---------- Pretrained models ------------


def get_model(summary=True):
    r"""Get the model definition

    @Args:
        summary:    Whether to show the model summary
    @Returns:
                    The model's definition
    """
    # _, m = get_vgg_face3()
    x = x_in = Input(INPUT_SHAPE, name="input")
    # x = Dense(512, name="d1", activation="relu")(x)
    x = Dense(1024, name="d0", activation="relu",
              kernel_initializer="he_uniform")(x)
    x = BatchNormalization(name="bn0")(x)
    x = Dense(512, name="d1", activation="relu",
              kernel_initializer="he_uniform")(x)
    x = BatchNormalization(name="bn1")(x)
    # x = Dense(256, name="d2", activation="relu")(x)
    x = Dense(256, name="d2", activation="relu",
              kernel_initializer="he_uniform")(x)
    x = BatchNormalization(name="bn2")(x)
    x = Dropout(0.5, name="dr1")(x)
    x = Dense(N_CLASSES, name="d3", activation="softmax")(x)

    m = Model(inputs=x_in, outputs=x)

    if summary:
        m.summary()

    return m


def get_models():
    r"""Test all combination of models

    @Returns:
                List of tuple of (model name, optimizer, models' definition)
    """
    nadam1 = Nadam(lr=0.01, beta_1=0.9, beta_2=0.999)
    nadam2 = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999)
    adam1 = Adam(lr=0.01)
    adam2 = Adam(lr=0.002)
    sgd1 = SGD(lr=0.01, momentum=0.9)
    sgd2 = SGD(lr=0.002, momentum=0.9, nesterov=True)

    exclude_model_names = [
        "nadam2_vgg16-256-2Dense_nodrop_he_uniform",
        "nadam1_vgg16-1024-1Dense",
        "nadam2_vgg16-1024-1Dense",
        "nadam1_vgg16-512-2Dense_he_uniform",
        "nadam2_vgg16-512-2Dense_he_uniform",
        "nadam1_res50-1024-1Dense",
        "nadam1_res50-512-2Dense_he_uniform",
        "nadam1_res50-1024-1Dense",
        "nadam1_res50-512-2Dense_he_normal",
        "nadam1_res50-256-2Dense_nodrop_he_uniform",
    ]

    opts = [
        ("nadam1", nadam1),
        # ("nadam2", nadam2),
        # ("adam1", adam1),
        # ("adam2", adam2),
        # ("sgd1", sgd1),
        # ("sgd2", sgd2),
    ]

    vggs = [
        # get_vgg_face(),
        # get_vgg_face2(),

        # Some relatively best models
        # get_vgg_9(),
        # get_vgg_7(),
        # get_vgg_14(),
        # get_vgg_12(),
        # get_vgg_4(),

        # Let's party :/

        # get_vgg_1(),
        # get_vgg_2(),
        # get_vgg_3(),
        # get_vgg_4(),
        # get_vgg_5(),
        # get_vgg_6(),
        # get_vgg_7(),
        # get_vgg_8(),
        # get_vgg_9(),
        # get_vgg_10(),
        # get_vgg_11(),
        # get_vgg_12(),
        # get_vgg_13(),
        # get_vgg_14(),
        # get_vgg_15(),
        # get_vgg2_1(),
        # get_vgg2_2(),
        # get_vgg2_3(),
        # get_vgg2_4(),
        # get_vgg2_5(),
        # get_vgg2_6(),
        # get_vgg2_7(),
        # get_vgg2_8(),
        # get_vgg2_9(),
        # get_vgg2_10(),
        # get_vgg2_11(),
        # get_vgg2_12(),
        # get_vgg2_13(),
        # get_vgg2_14(),
        # get_vgg2_15(),
        # get_res50_1(),
        # get_res50_2(),
        # get_res50_3(),
        # get_res50_4(),
        # get_res50_5(),
        # get_res50_6(),
        # get_res50_7(),
        # get_res50_8(),
        # get_res50_9(),
        # get_res50_10(),
        # get_res50_11(),
        # get_res50_12(),
        # get_res50_13(),
        # get_res50_14(),
        # get_res50_15(),
        # get_res50_2_1(),
        # get_res50_2_2(),
        # get_res50_2_3(),
        # get_res50_2_4(),
        # get_res50_2_5(),
        # get_res50_2_6(),
        # get_res50_2_7(),
        # get_res50_2_8(),
        # get_res50_2_9(),
        # get_res50_2_10(),
        # get_res50_2_11(),
        # get_res50_2_12(),
        # get_res50_2_13(),
        # get_res50_2_14(),
        # get_res50_2_15(),
    ]

    exclude_model_dict = dict.fromkeys(exclude_model_names, 1)
    model_combinations = list(map(lambda x: (x[0][0] + "_" + x[1][0], x[0][1], x[1][1]),
                                  itertools.product(opts, vggs)))

    def not_excluded(x): return x[0] not in exclude_model_dict

    return list(filter(not_excluded, model_combinations))
