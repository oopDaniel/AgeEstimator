#!/usr/bin/env python
# coding: utf-8


import shutil
from PIL import Image
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
import numpy as np
from server.models.cnn.model import IMAGE_SIZE, IMAGE_SHAPE, CURR_DIR
from server.data.dataset import DataLoader
import sys
import os

# Workaround to make packages work in both Jupyter notebook and Python
MODULE_ROOT_NAME = "AgeEstimator"
MODULE_PATHS = [
    os.path.abspath(os.path.join('..')),
    os.path.abspath(os.path.join('../..')),
    os.path.abspath(os.path.join('../../..'))
]
MODULE_PATHS = list(
    filter(lambda x: x.endswith(MODULE_ROOT_NAME), MODULE_PATHS))
MODULE_PATH = MODULE_PATHS[0] if len(MODULE_PATHS) == 1 else ""
if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)


# Shared model to create bottleneck features

model = VGGFace(model="resnet50", include_top=False,
                input_shape=IMAGE_SHAPE)

dl = DataLoader()
DATASET_DIR = os.path.join(dl.train_dir, "..", "..", "dataset")
x_train_all, y_train_all = dl.load_train()
x_test_all, y_test_all = dl.load_test()

detector = MTCNN()


def extract_face(filename="", input_image=None, img_size=IMAGE_SIZE):
    r"""Extract a single face from a given photograph"""

    img = plt.imread(filename) if input_image is None else input_image

    # Create the detector, using default weights
    detection = detector.detect_faces(img)

    assert len(detection) > 0, "Probably sth wrong with this image %s" % filename
    # Extract the bounding box from the first face
    x1, y1, width, height = detection[0]['box']
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]

    # Resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(img_size)
    face_array = np.asarray(image)

    return face_array


def preprocess_image(img_in):
    print("Processing image...")
    try:
        img = extract_face(input_image=img_in)
    except AssertionError:
        # Use original image instead
        img = Image.fromarray(img_in)
        img = img.resize(IMAGE_SIZE)
        img = np.asarray(img)
    except Exception as e:
        print(str(e))

    img = img.astype('float32')
    img = np.expand_dims(img, axis=0)

    print("Preprocessing input...")
    processed = preprocess_input(img, version=2)

    print("Creating bottleneck features...")
    y_hat = model.predict(processed)

    print("Done.")
    return y_hat.reshape((y_hat.shape[0], -1))


def preprocess_dataset(idx=0, total_workers=1):

    train_limit = len(x_train_all) // total_workers
    test_limit = len(x_test_all) // total_workers

    x_train = x_train_all[idx * train_limit: (idx + 1) * train_limit]
    y_train = y_train_all[idx * train_limit: (idx + 1) * train_limit]
    x_test = x_test_all[idx * test_limit: (idx + 1) * test_limit]
    y_test = y_test_all[idx * test_limit: (idx + 1) * test_limit]

    print("[Worker%d]: Processing training images..." % idx)
    x_train_labels = []
    x_train_features = []
    x_train_crashed = []
    for i, fname in enumerate(x_train):
        try:
            feature = extract_face(fname)
        except AssertionError as e:
            print(">>>> [Worker%d]: %s" % (idx, str(e)))
            x_train_crashed.append((fname, y_train[i]))
            continue
        except Exception as e:
            print(">>>> [Worker%d]: Exception: %s" % (idx, str(e)))
            print(fname)
            x_train_crashed.append((fname, y_train[i]))
            continue
        x_train_labels.append(y_train[i])
        x_train_features.append(feature)

    print("[Worker%d]: Processing test images..." % idx)
    x_test_labels = []
    x_test_features = []
    x_test_crashed = []
    for i, fname in enumerate(x_test):
        try:
            feature = extract_face(fname)
        except AssertionError as e:
            print(">>>> [Worker%d]: %s" % (idx, str(e)))
            x_test_crashed.append((fname, y_train[i]))
            continue
        except Exception as e:
            print(">>>> [Worker%d]: Exception: %s" % (idx, str(e)))
            print(fname)
            x_test_crashed.append((fname, y_train[i]))
            continue
        x_test_labels.append(y_test[i])
        x_test_features.append(feature)

    crashed = (x_train_crashed, x_test_crashed)

    print("[Worker%d]: Preprocessing inputs..." % idx)
    x_train_processed = preprocess_input(x_train_features, version=2)
    x_test_processed = preprocess_input(x_test_features, version=2)

    print("[Worker%d]: Creating bottleneck features..." % idx)
    p_train = model.predict(x_train_processed)
    p_test = model.predict(x_test_processed)

    try:
        print("[Worker%d]: Removing dataset dir..." % idx)
        shutil.rmtree(DATASET_DIR)
    except:
        pass

    print("[Worker%d]: Saving features..." % idx)
    np.save(os.path.join(CURR_DIR, "features-train-worker%d" % idx), p_train)
    np.save(os.path.join(CURR_DIR, "features-test-worker%d" % idx), p_test)
    np.save(os.path.join(CURR_DIR, "labels-train-worker%d" % idx), x_train_labels)
    np.save(os.path.join(CURR_DIR, "labels-test-worker%d" % idx), x_test_labels)
