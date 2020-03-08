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

def predict(img):
 
    img_data = cv2.resize(img, (224, 224))
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    model = VGG16(include_top=True)
    vgg16_feature = model.predict(img_data)
    test = np.array(vgg16_feature)
    with open('.\models\clustering\gnb.pickle', 'rb') as f:
        gnb_load = pickle.load(f)
    res = gnb_load.predict(test)		
    return res[0]
   