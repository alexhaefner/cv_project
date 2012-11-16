import cv
import cv2
import numpy as np

from gaussian import Gaussian3D

def build_gaussian(image_names):
    """ Builds a gaussian from a number of reference images """
    images = [cv2.imread(name) for name in image_names]
    average_color_channel = lambda image, index: [sum(row) for row in cv2.split(image)[index]]
    rgb_averages = [[sum(average_color_channel(image, index))/(len(image[0])*len(image)) for index in [0, 1, 2]] for image in images ]
    gaussian = Gaussian3D()
    gaussian.calculate_sigma(rgb_averages)
    return gaussian

