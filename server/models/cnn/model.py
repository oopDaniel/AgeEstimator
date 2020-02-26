r"""This script provides the model definition and the path
to pre-trained weight(s).

Import this module instead of calling it in CLI.
"""
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16

OLD_WEIGHTS_PATH = "old_vgg_regression_weights.hdf5"
BEST_WEIGHTS_PATH = "best_vgg_regression_weights.hdf5"


def get_model(summary=True):
    r"""Get the model definition

    @Args:
        summary:    Whether to show the model summary
    @Returns:
                    The model's definition
    """
    #     x = x_in = Input((250, 250, 3), name="input")
    #     x = Conv2D(32, (3,3), padding="valid",  name="fe0")(x)
    #     x = Activation("relu", name="r0")(x)
    #     x = MaxPooling2D(2,2,name="mp0")(x)
    #     x = Conv2D(64, (3,3), padding="valid", name="fe1")(x)
    #     x = Activation("relu", name="r1")(x)
    #     x = MaxPooling2D(2,2,name="mp1")(x)
    #     x = Conv2D(128, (3,3), padding="valid", name="fe2")(x)
    #     x = Activation("relu", name="r2")(x)
    #     x = MaxPooling2D(2,2,name="mp2")(x)
    #     x = Flatten(name="fl")(x)
    #     x = Dropout(0.5, name="d5")(x)
    #     x = Dense(512, name="d1", activation="relu")(x)
    #     x = Dense(1, name="d2")(x)
    #     m = Model(inputs=x_in, outputs=x)

    vgg16 = VGG16(weights="imagenet", include_top=False,
                  input_shape=(250, 250, 3))
    vgg16.trainable = False
    vgg16.summary()

    x = x_in = Input((250, 250, 3), name="input")
    x = vgg16(x)
    x = Flatten(name="fl")(x)
    x = Dropout(0.5, name="dr1")(x)
    x = Dense(512, name="d1", activation="relu")(x)
    x = Dense(1, name="d2")(x)
    m = Model(inputs=x_in, outputs=x)

    if summary:
        m.summary()

    return m
