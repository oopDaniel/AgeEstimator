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
from server.models.input_shape import INPUT_SHAPE                    # noqa: F404
from server.models.cnn.predict import predict as cnn_predict         # noqa: F404
from server.models.regression.predict import predict as reg_predict  # noqa: F404

# Messages
INVALID_FORMAT = "Invalid image format"
NO_WEIGHT_FOUND = "The model isn't ready"
VALUE_ERROR = "Value error"
PARTIAL_SUCCESS = "Partially Succeed"
SUCCESS = "Succeed!"
FAILURE = "Something wrong with the models"

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
    img = cv2.resize(img, INPUT_SHAPE)

    success_count = 0
    # Prediction - CNN
    try:
        age_cnn = cnn_predict(img)
        success_count += 1
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
        success_count += 1
    except Exception as error:
        print("Regression Error:", error)
        age_regression = -1

    # Prediction - Clustering
    try:
        age_cluster = random.randint(2, 80)
        success_count += 1
    except Exception as error:
        print("Clustering Error:", error)
        age_cluster = -1

    ages = {
        "cnn": age_cnn,
        "regression": str(age_regression),
        "clustering": str(age_cluster),
    }

    if success_count == 3:
        message = SUCCESS
    else:
        message = FAILURE if success_count == 0 else PARTIAL_SUCCESS

    return jsonify({
        "name": file_name,
        "message": message,
        "ages": ages,
    })


# TODO turn off debug

app.run(port=5000, debug=True)
