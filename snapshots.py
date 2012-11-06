
import math
import numpy as np
import cv
import cv2
from build_gaussian import build_gaussian
from gaussian import KLDistance, Gaussian3D

cv2.namedWindow("w1", cv.CV_WINDOW_AUTOSIZE)
camera_index = 0
count = 0
capture = cv2.VideoCapture(camera_index)

def repeat():
    global capture #declare as globals since we are assigning to them now
    global camera_index
    global count
    image = capture.read()[1] # image.shape to get height, width, depth
    image = cv2.resize(image, (320, 240))
    #image = cv2.integral(image)
    height, width, _ = image.shape
    prob = np.zeros((240, 320))
    cv2.imshow("w1", image)
    c = cv2.waitKey(10)
    if(c == 101):
        exit()
    if(c == 112):
        cv2.imwrite('{0}.jpg'.format(count), image)
        count += 1



while True:
        repeat()
