#!/usr/bin/env python
"""The web server providing API for image upload and
make predictions based on pretrained models.
"""

import sys
import os
import random
import numpy as np
import cv2
from flask import Flask, request, jsonify

# Workaround to make packages work in both Jupyter notebook and Python
MODULE_ROOT_NAME = "AgeEstimator"
MODULE_PATHS = [
    os.path.abspath(os.path.join('..')),
    os.path.abspath(os.path.join('../..'))
]
MODULE_PATHS = list(
    filter(lambda x: x.endswith(MODULE_ROOT_NAME), MODULE_PATHS))
MODULE_PATH = MODULE_PATHS[0] if len(MODULE_PATHS) == 1 else ""
if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)

# pylint: disable=wrong-import-position
# from server.models.cnn.predict import predict as cnn_predict    # noqa: F404
from server.data.unify_dimension import resize_rgb_img          # noqa: F404
from server.models.regression.predict import predict as reg_predict

# Messages
INVALID_FORMAT = "Invalid image format"
NO_WEIGHT_FOUND = "The model isn't ready"
VALUE_ERROR = "Value error"
PARTIAL_SUCCESS = "Partially Succeed"
SUCCESS = "Succeed!"

# pylint: disable=invalid-name
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024


@app.route('/api/predict_age', methods=['POST'])
def predict_age():
    r"""Predict the age of the uploaded facial image
        with 3 different models.

    @Returns:
        message:    message of processing
        name:       original file name
        ages:       The predicted age of 3 models in float
    """
    try:
        file = request.files['file']
        file_name = file.filename
        nparr = np.fromstring(file.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as error:
        print(error)
        return jsonify({
            "name": file_name,
            "message": INVALID_FORMAT,
            "ages": None,
        })

    # Resize img
    img = resize_rgb_img(img, (250, 250))

    # Prediction - CNN
    try:
        # age_cnn = cnn_predict(img)
        age_cnn = random.randint(2, 80)
    except ValueError:
        message = VALUE_ERROR
        age_cnn = -1
    except FileNotFoundError:
        message = NO_WEIGHT_FOUND
        age_cnn = -1
    except Exception as error:
        print("CNN Error:", error)
        age_cnn = -1

    # Prediction - Regression
    try:
        age_regression = reg_predict(img)
        # age_regression = random.randint(34, 48)
    except Exception as error:
        print("Regression Error:", error)
        age_regression = -1

    # Prediction - Clustering
    try:
        age_cluster = random.randint(2, 80)
    except Exception as error:
        print("Clustering Error:", error)
        age_cluster = -1

    ages = {
        "cnn": age_cnn,
        "regression": age_regression,
        "clustering": age_cluster,
    }
    message = SUCCESS if all(age != -1 for age in ages.values()) \
        else PARTIAL_SUCCESS

    return jsonify({
        "name": file_name,
        "message": message,
        "ages": ages,
    })


# TODO turn off debug

app.run(port=5000, debug=True)
