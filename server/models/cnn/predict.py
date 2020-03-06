r"""This script provides a function to make prediction of given image
based on the pre-trained convolutional neural network model.

Import this module instead of calling it in CLI.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from server.models.cnn.model import get_model, IMAGE_SIZE, OLD_WEIGHTS_PATH, BEST_WEIGHTS_PATH, CURR_DIR


def predict(img):
    r"""Predict the age of the given facial image.

    @Args:
        image:      image matrix
    @Returns:
                    The predicted age in float
    """
    model = get_model(summary=False)
    if os.path.exists(os.path.join(CURR_DIR, BEST_WEIGHTS_PATH)):
        model.load_weights(os.path.join(CURR_DIR, BEST_WEIGHTS_PATH))
    elif os.path.exists(os.path.join(CURR_DIR, OLD_WEIGHTS_PATH)):
        model.load_weights(os.path.join(CURR_DIR, OLD_WEIGHTS_PATH))
    else:
        print("No existent weight found. Train the network before using it.")
        raise FileNotFoundError

    # Add batch size as first dimension
    img = cv2.resize(img, IMAGE_SIZE)
    img = img.reshape((1,) + img.shape)

    try:
        data_gen = ImageDataGenerator(rescale=1./255)
        test_gen = data_gen.flow(img)
        y_hat = model.predict(test_gen)

        # TODO: map class back to age
        possibles_predictions = y_hat.argsort()[:, -5:]
        print("Possible labels:", possibles_predictions)

        result = int(possibles_predictions.flatten()[-1])

    except tf.errors.InvalidArgumentError:
        print("[TF] Invalid Argument Error")
        raise ValueError
    except Exception:
        print("EXP")

    return result
    # return round(result, 1)
