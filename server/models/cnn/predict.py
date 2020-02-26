r"""This script provides a function to make prediction of given image
based on the pre-trained convolutional neural network model.

Import this module instead of calling it in CLI.
"""

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from server.models.cnn.model import get_model, OLD_WEIGHTS_PATH, BEST_WEIGHTS_PATH


def predict(img):
    r"""Predict the age of the given facial image.

    @Args:
        image:      image matrix
    @Returns:
                    The predicted age in float
    """
    model = get_model(summary=False)

    if os.path.exists(BEST_WEIGHTS_PATH):
        model.load_weights(BEST_WEIGHTS_PATH)
    elif os.path.exists(OLD_WEIGHTS_PATH):
        model.load_weights(OLD_WEIGHTS_PATH)
    else:
        print("No existent weight found. Train the network before using it.")

    # Add batch size as first dimension
    img = img.reshape((1,) + img.shape)

    data_gen = ImageDataGenerator(rescale=1./255)
    test_gen = data_gen.flow(img)

    return model.predict(test_gen).item()
