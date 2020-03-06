r"""This script provides the model definition and the path
to pre-trained weight(s).

Import this module instead of calling it in CLI.
"""
import os
from tensorflow.keras.layers import Input, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
import pandas as pd

# --------------- Global Vars -------------------

OLD_WEIGHTS_PATH = "old_vgg16_classification_weights.hdf5"
BEST_WEIGHTS_PATH = "best_vgg16_classification_weights.hdf5"

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

vgg = VGG16(weights="imagenet", include_top=False,
            input_shape=INPUT_SHAPE)
vgg_avg = VGG16(weights="imagenet", include_top=False,
                input_shape=INPUT_SHAPE, pooling="avg")

vgg.trainable = False
vgg_avg.trainable = False

# ---------------- VGG16 start ----------------


def get_vgg_1():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = vgg(x)
    x = Flatten(name="fl")(x)
    x = Dense(1024, name="d0",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(512, name="d1",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(256, name="d1.5",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5, name="dr1")(x)
    x = Dense(N_CLASSES, name="d2", activation="softmax")(x)

    m = Model(inputs=x_in, outputs=x)
    return "vgg16-1024-3Dense", m


def get_vgg_2():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = vgg(x)
    x = Flatten(name="fl")(x)
    x = Dense(512, name="d1",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(256, name="d1.5",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5, name="dr1")(x)
    x = Dense(N_CLASSES, name="d2", activation="softmax")(x)

    m = Model(inputs=x_in, outputs=x)
    return "vgg16-512-2Dense", m


def get_vgg_3():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = vgg(x)
    x = Flatten(name="fl")(x)
    x = Dense(512, name="d1",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(256, name="d1.5",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(N_CLASSES, name="d2", activation="softmax")(x)

    m = Model(inputs=x_in, outputs=x)
    return "vgg16-512-2Dense_nodrop", m


def get_vgg_4():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = vgg(x)
    x = Flatten(name="fl")(x)
    x = Dense(1024, name="d1",
              activation="relu")(x)
    x = Dense(N_CLASSES, name="d2", activation="softmax")(x)

    m = Model(inputs=x_in, outputs=x)
    return "vgg16-1024-1Dense", m


def get_vgg_5():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = vgg(x)
    x = Flatten(name="fl")(x)
    x = Dense(1024, name="d1",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(N_CLASSES, name="d2", activation="softmax")(x)

    m = Model(inputs=x_in, outputs=x)
    return "vgg16-1024-1Dense-norm", m


def get_vgg_6():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = vgg(x)
    x = Flatten(name="fl")(x)
    x = Dense(1024, name="d0", kernel_initializer="he_uniform",
              activation="relu")(x)
    x = BatchNormalization()(x)
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
    return "vgg16-1024-3Dense_he_uniform", m


def get_vgg_7():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = vgg(x)
    x = Flatten(name="fl")(x)
    x = Dense(512, name="d1", kernel_initializer="he_uniform",
              activation="relu")(x)
    x = BatchNormalization(name="bn1")(x)
    x = Dense(256, name="d1.5", kernel_initializer="he_uniform",
              activation="relu")(x)
    x = BatchNormalization(name="bn2")(x)
    x = Dropout(0.5, name="dr1")(x)
    x = Dense(N_CLASSES, name="d2", activation="softmax",
              kernel_initializer="he_uniform")(x)

    m = Model(inputs=x_in, outputs=x)
    return "vgg16-512-2Dense_he_uniform", m


def get_vgg_8():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = vgg(x)
    x = Flatten(name="fl")(x)
    x = Dense(512, name="d1", kernel_initializer="he_uniform",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(256, name="d1.5", kernel_initializer="he_uniform",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(N_CLASSES, name="d2", activation="softmax",
              kernel_initializer="he_uniform")(x)

    m = Model(inputs=x_in, outputs=x)
    return "vgg16-512-2Dense_nodrop_he_uniform", m


def get_vgg_9():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = vgg(x)
    x = Flatten(name="fl")(x)
    x = Dense(1024, name="d1.5", kernel_initializer="he_uniform",
              activation="relu")(x)
    x = Dense(N_CLASSES, name="d2", activation="softmax",
              kernel_initializer="he_uniform")(x)

    m = Model(inputs=x_in, outputs=x)
    return "vgg16-256-2Dense_nodrop_he_uniform", m


def get_vgg_10():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = vgg(x)
    x = Flatten(name="fl")(x)
    x = Dense(1024, name="d1.5", kernel_initializer="he_uniform",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(N_CLASSES, name="d2", activation="softmax",
              kernel_initializer="he_uniform")(x)

    m = Model(inputs=x_in, outputs=x)
    return "vgg16-1024-1Dense-norm_he_uniform", m


def get_vgg_11():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = vgg(x)
    x = Flatten(name="fl")(x)
    x = Dense(1024, name="d0", kernel_initializer="he_normal",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(512, name="d1", kernel_initializer="he_normal",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(256, name="d1.5", kernel_initializer="he_normal",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5, name="dr1")(x)
    x = Dense(N_CLASSES, name="d2", activation="softmax",
              kernel_initializer="he_normal")(x)

    m = Model(inputs=x_in, outputs=x)
    return "vgg16-1024-3Dense_he_normal", m


def get_vgg_12():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = vgg(x)
    x = Flatten(name="fl")(x)
    x = Dense(512, name="d1", kernel_initializer="he_normal",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(256, name="d1.5", kernel_initializer="he_normal",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5, name="dr1")(x)
    x = Dense(N_CLASSES, name="d2", activation="softmax",
              kernel_initializer="he_normal")(x)

    m = Model(inputs=x_in, outputs=x)
    return "vgg16-512-2Dense_he_normal", m


def get_vgg_13():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = vgg(x)
    x = Flatten(name="fl")(x)
    x = Dense(512, name="d1", kernel_initializer="he_normal",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(256, name="d1.5", kernel_initializer="he_normal",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(N_CLASSES, name="d2", activation="softmax",
              kernel_initializer="he_normal")(x)

    m = Model(inputs=x_in, outputs=x)
    return "vgg16-512-2Dense_nodrop_he_normal", m


def get_vgg_14():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = vgg(x)
    x = Flatten(name="fl")(x)
    x = Dense(1024, name="d1.5", kernel_initializer="he_normal",
              activation="relu")(x)
    x = Dense(N_CLASSES, name="d2", activation="softmax",
              kernel_initializer="he_normal")(x)

    m = Model(inputs=x_in, outputs=x)
    return "vgg16-256-2Dense_nodrop_he_normal", m


def get_vgg_15():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = vgg(x)
    x = Flatten(name="fl")(x)
    x = Dense(1024, name="d1.5", kernel_initializer="he_normal",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(N_CLASSES, name="d2", activation="softmax",
              kernel_initializer="he_normal")(x)

    m = Model(inputs=x_in, outputs=x)
    return "vgg16-1024-1Dense-norm_he_normal", m


def get_vgg2_1():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = vgg_avg(x)
    x = Flatten(name="fl")(x)
    x = Dense(1024, name="d0",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(512, name="d1",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(256, name="d1.5",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5, name="dr1")(x)
    x = Dense(N_CLASSES, name="d2", activation="softmax")(x)

    m = Model(inputs=x_in, outputs=x)
    return "vgg16_pool-1024-3Dense", m


def get_vgg2_2():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = vgg_avg(x)
    x = Flatten(name="fl")(x)
    x = Dense(512, name="d1",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(256, name="d1.5",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5, name="dr1")(x)
    x = Dense(N_CLASSES, name="d2", activation="softmax")(x)

    m = Model(inputs=x_in, outputs=x)
    return "vgg16_pool-512-2Dense", m


def get_vgg2_3():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = vgg_avg(x)
    x = Flatten(name="fl")(x)
    x = Dense(512, name="d1",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(256, name="d1.5",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(N_CLASSES, name="d2", activation="softmax")(x)

    m = Model(inputs=x_in, outputs=x)
    return "vgg16_pool-512-2Dense_nodrop", m


def get_vgg2_4():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = vgg_avg(x)
    x = Flatten(name="fl")(x)
    x = Dense(1024, name="d1.5",
              activation="relu")(x)
    x = Dense(N_CLASSES, name="d2", activation="softmax")(x)

    m = Model(inputs=x_in, outputs=x)
    return "vgg16_pool-1024-1Dense", m


def get_vgg2_5():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = vgg_avg(x)
    x = Flatten(name="fl")(x)
    x = Dense(1024, name="d1.5",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(N_CLASSES, name="d2", activation="softmax")(x)

    m = Model(inputs=x_in, outputs=x)
    return "vgg16_pool-1024-1Dense-norm", m


def get_vgg2_6():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = vgg_avg(x)
    x = Flatten(name="fl")(x)
    x = Dense(1024, name="d0", kernel_initializer="he_uniform",
              activation="relu")(x)
    x = BatchNormalization()(x)
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
    return "vgg16_pool-1024-3Dense_he_uniform", m


def get_vgg2_7():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = vgg_avg(x)
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
    return "vgg16_pool-512-2Dense_he_uniform", m


def get_vgg2_8():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = vgg_avg(x)
    x = Flatten(name="fl")(x)
    x = Dense(512, name="d1", kernel_initializer="he_uniform",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(256, name="d1.5", kernel_initializer="he_uniform",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(N_CLASSES, name="d2", activation="softmax",
              kernel_initializer="he_uniform")(x)

    m = Model(inputs=x_in, outputs=x)
    return "vgg16_pool-512-2Dense_nodrop_he_uniform", m


def get_vgg2_9():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = vgg_avg(x)
    x = Flatten(name="fl")(x)
    x = Dense(1024, name="d1", kernel_initializer="he_uniform",
              activation="relu")(x)
    x = Dense(N_CLASSES, name="d2", activation="softmax",
              kernel_initializer="he_uniform")(x)

    m = Model(inputs=x_in, outputs=x)
    return "vgg16_pool-256-2Dense_nodrop_he_uniform", m


def get_vgg2_10():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = vgg_avg(x)
    x = Flatten(name="fl")(x)
    x = Dense(1024, name="d1", kernel_initializer="he_uniform",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(N_CLASSES, name="d2", activation="softmax",
              kernel_initializer="he_uniform")(x)

    m = Model(inputs=x_in, outputs=x)
    return "vgg16_pool-1024-1Dense-norm_he_uniform", m


def get_vgg2_11():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = vgg_avg(x)
    x = Flatten(name="fl")(x)
    x = Dense(1024, name="d0", kernel_initializer="he_normal",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(512, name="d1", kernel_initializer="he_normal",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(256, name="d1.5", kernel_initializer="he_normal",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5, name="dr1")(x)
    x = Dense(N_CLASSES, name="d2", activation="softmax",
              kernel_initializer="he_normal")(x)

    m = Model(inputs=x_in, outputs=x)
    return "vgg16_pool-1024-3Dense_he_normal", m


def get_vgg2_12():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = vgg_avg(x)
    x = Flatten(name="fl")(x)
    x = Dense(512, name="d1", kernel_initializer="he_normal",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(256, name="d1.5", kernel_initializer="he_normal",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5, name="dr1")(x)
    x = Dense(N_CLASSES, name="d2", activation="softmax",
              kernel_initializer="he_normal")(x)

    m = Model(inputs=x_in, outputs=x)
    return "vgg16_pool-512-2Dense_he_normal", m


def get_vgg2_13():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = vgg_avg(x)
    x = Flatten(name="fl")(x)
    x = Dense(512, name="d1", kernel_initializer="he_normal",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(256, name="d1.5", kernel_initializer="he_normal",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(N_CLASSES, name="d2", activation="softmax",
              kernel_initializer="he_normal")(x)

    m = Model(inputs=x_in, outputs=x)
    return "vgg16_pool-512-2Dense_nodrop_he_normal", m


def get_vgg2_14():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = vgg_avg(x)
    x = Flatten(name="fl")(x)
    x = Dense(1024, name="d1", kernel_initializer="he_normal",
              activation="relu")(x)
    x = Dense(N_CLASSES, name="d2", activation="softmax",
              kernel_initializer="he_normal")(x)

    m = Model(inputs=x_in, outputs=x)
    return "vgg16_pool-256-2Dense_nodrop_he_normal", m


def get_vgg2_15():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = vgg_avg(x)
    x = Flatten(name="fl")(x)
    x = Dense(1024, name="d1", kernel_initializer="he_normal",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(N_CLASSES, name="d2", activation="softmax",
              kernel_initializer="he_normal")(x)

    m = Model(inputs=x_in, outputs=x)
    return "vgg16_pool-1024-1Dense-norm_he_normal", m
