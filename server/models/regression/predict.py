r"""This script provides a function to make prediction of given image
based on the pre-trained convolutional neural network model.
Import this module instead of calling it in CLI.
"""

import os
import numpy as np
from PIL import Image, ImageFilter 
import torch
from torch.autograd import Variable

try:
    CURR_DIR = os.path.dirname(os.path.realpath(__file__))
except NameError:
    CURR_DIR = os.getcwd()


class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super().__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out


test1 = Image.open('/Users/xiaoxiao/Desktop/SCU/2020winter/coen281/hw/termPro/AgeEstimator/server/data/dataset/test/26_utk_17029.jpg')

def predict(img):
    r"""Predict the age of the given facial image.
    @Args:
        image:      image matrix
    @Returns:
                    The predicted age
    """

    model = linearRegression(250*250, 121)
    
    if os.path.exists(os.path.join(CURR_DIR, 'param')):
        model.load_state_dict(torch.load(os.path.join(CURR_DIR, 'param')))
    else:
        print("No existent weight found. Train the network before using it.")
        raise FileNotFoundError
    
    img = np.array(img)
    # print(type(img))
    img = Image.fromarray(img, 'RGB')
    image = img.convert('L')
    image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    image = np.array(image).reshape(1, 250*250)
    image_tensor = Variable(torch.Tensor(image))
    output = model(image_tensor)
    _, predicted = torch.max(output.data, 1)

    return predicted.item()

p = predict(test1)
print(p)
