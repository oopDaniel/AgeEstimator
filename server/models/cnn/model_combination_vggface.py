r"""This script provides the model definition and the path
to pre-trained weight(s).

Import this module instead of calling it in CLI.
"""
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, BatchNormalization, Dropout
from keras_vggface.vggface import VGGFace
import pandas as pd

# --------------- Global Vars -------------------

OLD_WEIGHTS_PATH = "old_vggface_classification_weights.hdf5"
BEST_WEIGHTS_PATH = "best_vggface_classification_weights.hdf5"

IMAGE_SIZE = (224, 224)
INPUT_SHAPE = IMAGE_SIZE + (3,)

# Make compatible path for Python and Jupyter Notebook
try:
    CURR_DIR = os.path.dirname(os.path.realpath(__file__))
except NameError:
    CURR_DIR = os.getcwd()

LABEL_MAPPING = pd.Series.from_csv(
    os.path.join(CURR_DIR, "age_class_mapping.csv"),
    header=0).to_dict()
N_CLASSES = len(set(LABEL_MAPPING.values()))

# ---------- Shared pretrained models ------------


vgg_face = VGGFace(model="resnet50", include_top=False,
                   input_shape=INPUT_SHAPE)

# ---------------- VGGFace start ----------------


def get_vgg_face():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = vgg_face(x)
    x = Flatten(name="fl")(x)
    x = Dense(512, name="d1", kernel_initializer="he_uniform",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(256, name="d1.5", kernel_initializer="he_uniform",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5, name="dr1")(x)
    x = Dense(N_CLASSES, name="d2", activation="softmax",
              kernel_initializer="he_uniform")(x)

    m = Model(inputs=x_in, outputs=x)
    return "vggface-512-2Dense-he_uniform", m


def get_vgg_face2():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = vgg_face(x)
    x = Flatten(name="fl")(x)
    x = Dense(256, name="d1.5", kernel_initializer="he_uniform",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(N_CLASSES, name="d2", activation="softmax",
              kernel_initializer="he_uniform")(x)

    m = Model(inputs=x_in, outputs=x)
    return "vggface-256-2Dense-he_uniform", m


def get_vgg_face3():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = vgg_face(x)
    x = Flatten(name="fl")(x)
    x = Dense(256, name="d1.5", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(N_CLASSES, name="d2", activation="softmax")(x)

    m = Model(inputs=x_in, outputs=x)
    return "vggface-256-2Dense", m
