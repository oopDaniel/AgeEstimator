r"""This script provides a function to make prediction of given image
based on the pre-trained convolutional neural network model.

Import this module instead of calling it in CLI.
"""

import os
import cv2
from collections import defaultdict
import numpy as np
import pandas as pd
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

        possibles_predictions = y_hat.argsort()[:, -5:]
        possibles_predictions = list(
            map(get_age_span_from_class, possibles_predictions.flatten()))
        print("Possible labels:", possibles_predictions)

        result = possibles_predictions[-1]

    except tf.errors.InvalidArgumentError:
        print("[TF] Invalid Argument Error")
        raise ValueError

    return result


LABEL_MAPPING = pd.Series.from_csv(
    os.path.join(CURR_DIR, "age_class_mapping.csv"),
    header=0).to_dict()

CLASS_TO_MAPPED_AGE = {i: v for i,
                       v in enumerate(set(LABEL_MAPPING.values()))}

AGE_DICT = defaultdict(list)
for original_age, mapped_age in LABEL_MAPPING.items():
    AGE_DICT[mapped_age].append(original_age)

LABEL_DICT = {}
for mapped_age, ages in AGE_DICT.items():
    max_age = max(ages)
    min_age = min(ages)
    LABEL_DICT[mapped_age] = str(min_age) if min_age == max_age else \
        "%d-%d" % (min_age, max_age)
    if mapped_age == 90:
        LABEL_DICT[mapped_age] = "85+"


def get_age_span_from_class(class_label):
    r"""Map class label to displayed age or age span (see AgeClasses.md for details)

    E.g. 3 -> 12 -> 10-14, return "10-14" given 3
    """
    return LABEL_DICT[CLASS_TO_MAPPED_AGE[class_label]]
