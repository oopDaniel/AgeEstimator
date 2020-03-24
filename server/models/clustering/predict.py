import sys
import pickle
import os
import tensorflow as tf
import numpy as np
import pandas
import multiprocessing
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
import cv2

try:
    CURR_DIR = os.path.dirname(os.path.realpath(__file__))
except NameError:
    CURR_DIR = os.getcwd()


def predict(img):
    vgg16_feature_list = []
    img_data = cv2.resize(img, (224, 224))
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    model = VGG16(include_top=False)
    vgg16_feature = model.predict(img_data)
    vgg16_feature_np = np.array(vgg16_feature)
    vgg16_feature_list.append(vgg16_feature_np.flatten())
    vgg16_feature_list_np = np.array(vgg16_feature_list)
    with open(os.path.join(CURR_DIR, "gnb.pickle"), 'rb') as f:
        gnb_load = pickle.load(f)
    res = gnb_load.predict(vgg16_feature_list_np)
    return res[0]
