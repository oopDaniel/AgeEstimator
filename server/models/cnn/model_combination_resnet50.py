r"""This script provides the model definition and the path
to pre-trained weight(s).

Import this module instead of calling it in CLI.
"""
import os
from tensorflow.keras.layers import Input, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50V2
import pandas as pd

# --------------- Global Vars -------------------

OLD_WEIGHTS_PATH = "old_resnet50_classification_weights.hdf5"
BEST_WEIGHTS_PATH = "best_resnet50_classification_weights.hdf5"

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


res50 = ResNet50V2(weights="imagenet", include_top=False,
                   input_shape=INPUT_SHAPE)
res50_avg = ResNet50V2(weights="imagenet", include_top=False,
                       input_shape=INPUT_SHAPE, pooling="avg")

res50.trainable = False
res50_avg.trainable = False

# ---------------- RES50 start ----------------


def get_res50_1():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = res50(x)
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
    return "res50-1024-3Dense", m


def get_res50_2():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = res50(x)
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
    return "res50-512-2Dense", m


def get_res50_3():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = res50(x)
    x = Flatten(name="fl")(x)
    x = Dense(512, name="d1",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(256, name="d1.5",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(N_CLASSES, name="d2", activation="softmax")(x)

    m = Model(inputs=x_in, outputs=x)
    return "res50-512-2Dense_nodrop", m


def get_res50_4():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = res50(x)
    x = Flatten(name="fl")(x)
    x = Dense(1024, name="d1",
              activation="relu")(x)
    x = Dense(N_CLASSES, name="d2", activation="softmax")(x)

    m = Model(inputs=x_in, outputs=x)
    return "res50-1024-1Dense", m


def get_res50_5():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = res50(x)
    x = Flatten(name="fl")(x)
    x = Dense(1024, name="d1",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(N_CLASSES, name="d2", activation="softmax")(x)

    m = Model(inputs=x_in, outputs=x)
    return "res50-1024-1Dense-norm", m


def get_res50_6():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = res50(x)
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
    return "res50-1024-3Dense_he_uniform", m


def get_res50_7():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = res50(x)
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
    return "res50-512-2Dense_he_uniform", m


def get_res50_8():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = res50(x)
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
    return "res50-512-2Dense_nodrop_he_uniform", m


def get_res50_9():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = res50(x)
    x = Flatten(name="fl")(x)
    x = Dense(1024, name="d1.5", kernel_initializer="he_uniform",
              activation="relu")(x)
    x = Dense(N_CLASSES, name="d2", activation="softmax",
              kernel_initializer="he_uniform")(x)

    m = Model(inputs=x_in, outputs=x)
    return "res50-256-2Dense_nodrop_he_uniform", m


def get_res50_10():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = res50(x)
    x = Flatten(name="fl")(x)
    x = Dense(1024, name="d1.5", kernel_initializer="he_uniform",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(N_CLASSES, name="d2", activation="softmax",
              kernel_initializer="he_uniform")(x)

    m = Model(inputs=x_in, outputs=x)
    return "res50-1024-1Dense-norm_he_uniform", m


def get_res50_11():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = res50(x)
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
    return "res50-1024-3Dense_he_normal", m


def get_res50_12():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = res50(x)
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
    return "res50-512-2Dense_he_normal", m


def get_res50_13():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = res50(x)
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
    return "res50-512-2Dense_nodrop_he_normal", m


def get_res50_14():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = res50(x)
    x = Flatten(name="fl")(x)
    x = Dense(1024, name="d1.5", kernel_initializer="he_normal",
              activation="relu")(x)
    x = Dense(N_CLASSES, name="d2", activation="softmax",
              kernel_initializer="he_normal")(x)

    m = Model(inputs=x_in, outputs=x)
    return "res50-256-2Dense_nodrop_he_normal", m


def get_res50_15():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = res50(x)
    x = Flatten(name="fl")(x)
    x = Dense(1024, name="d1.5", kernel_initializer="he_normal",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(N_CLASSES, name="d2", activation="softmax",
              kernel_initializer="he_normal")(x)

    m = Model(inputs=x_in, outputs=x)
    return "res50-1024-1Dense-norm_he_normal", m


def get_res50_2_1():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = res50_avg(x)
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
    return "res50_avg-1024-3Dense", m


def get_res50_2_2():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = res50_avg(x)
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
    return "res50_avg-512-2Dense", m


def get_res50_2_3():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = res50_avg(x)
    x = Flatten(name="fl")(x)
    x = Dense(512, name="d1",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(256, name="d1.5",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(N_CLASSES, name="d2", activation="softmax")(x)

    m = Model(inputs=x_in, outputs=x)
    return "res50_avg-512-2Dense_nodrop", m


def get_res50_2_4():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = res50_avg(x)
    x = Flatten(name="fl")(x)
    x = Dense(1024, name="d1.5",
              activation="relu")(x)
    x = Dense(N_CLASSES, name="d2", activation="softmax")(x)

    m = Model(inputs=x_in, outputs=x)
    return "res50_avg-1024-1Dense", m


def get_res50_2_5():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = res50_avg(x)
    x = Flatten(name="fl")(x)
    x = Dense(1024, name="d1.5",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(N_CLASSES, name="d2", activation="softmax")(x)

    m = Model(inputs=x_in, outputs=x)
    return "res50_avg-1024-1Dense-norm", m


def get_res50_2_6():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = res50_avg(x)
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
    return "res50_avg-1024-3Dense_he_uniform", m


def get_res50_2_7():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = res50_avg(x)
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
    return "res50_avg-512-2Dense_he_uniform", m


def get_res50_2_8():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = res50_avg(x)
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
    return "res50_avg-512-2Dense_nodrop_he_uniform", m


def get_res50_2_9():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = res50_avg(x)
    x = Flatten(name="fl")(x)
    x = Dense(1024, name="d1", kernel_initializer="he_uniform",
              activation="relu")(x)
    x = Dense(N_CLASSES, name="d2", activation="softmax",
              kernel_initializer="he_uniform")(x)

    m = Model(inputs=x_in, outputs=x)
    return "res50_avg-256-2Dense_nodrop_he_uniform", m


def get_res50_2_10():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = res50_avg(x)
    x = Flatten(name="fl")(x)
    x = Dense(1024, name="d1", kernel_initializer="he_uniform",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(N_CLASSES, name="d2", activation="softmax",
              kernel_initializer="he_uniform")(x)

    m = Model(inputs=x_in, outputs=x)
    return "res50_avg-1024-1Dense-norm_he_uniform", m


def get_res50_2_11():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = res50_avg(x)
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
    return "res50_avg-1024-3Dense_he_normal", m


def get_res50_2_12():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = res50_avg(x)
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
    return "res50_avg-512-2Dense_he_normal", m


def get_res50_2_13():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = res50_avg(x)
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
    return "res50_avg-512-2Dense_nodrop_he_normal", m


def get_res50_2_14():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = res50_avg(x)
    x = Flatten(name="fl")(x)
    x = Dense(1024, name="d1", kernel_initializer="he_normal",
              activation="relu")(x)
    x = Dense(N_CLASSES, name="d2", activation="softmax",
              kernel_initializer="he_normal")(x)

    m = Model(inputs=x_in, outputs=x)
    return "res50_avg-256-2Dense_nodrop_he_normal", m


def get_res50_2_15():
    x = x_in = Input(INPUT_SHAPE, name="input")
    x = res50_avg(x)
    x = Flatten(name="fl")(x)
    x = Dense(1024, name="d1", kernel_initializer="he_normal",
              activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(N_CLASSES, name="d2", activation="softmax",
              kernel_initializer="he_normal")(x)

    m = Model(inputs=x_in, outputs=x)
    return "res50_avg-1024-1Dense-norm_he_normal", m
