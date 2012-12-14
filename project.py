
import time
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
image_collection = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
N_o = build_gaussian(['hands/{0}.png'.format(count) for count in image_collection])
N_B = build_gaussian(['background/{0}.png'.format(count) for count in image_collection])
import ipdb; ipdb.set_trace()

def repeat():
    global capture #declare as globals since we are assigning to them now
    global camera_index
    global count
    c = cv2.waitKey(10)
    image = capture.read()[1] # image.shape to get height, width, depth
    cv2.imshow('img', image)
    print c
    if(c != 112):
        return
    then = time.time()
    image = capture.read()[1] # image.shape to get height, width, depth
    image = cv2.resize(image, (320, 240))
    image2 = np.copy(image)
    #image = cv2.integral(image)
    height, width, _ = image.shape
    prob = np.zeros((240, 320))
    r = 3
    w = 3
    for j in xrange(0, height-1):
        if j == 0:
            w_start = 0
        elif j== height-2:
            w_start = j-2
        else:
            w_start = j-1
        for k in xrange(0, width-1):
            if k == 0:
                r_start = 0
            elif k == width-2:
                r_start = k-2
            else:
                r_start = k-1
            N_iw = Gaussian3D()
            rgbs = []
            for index in xrange(w_start, w_start + 2):
                for rindex in xrange(r_start, r_start + 2):
                    rgbs.append(image[(index, rindex)])
            #print "{0}, {1}, w_start: {2}, r_start: {3}".format(j, k, w_start, r_start)
            N_iw.calculate_sigma(rgbs)

            try:
                probability = math.exp(-KLDistance(N_iw, N_o)/ (KLDistance(N_iw, N_o) + KLDistance(N_iw, N_B)))
            except:
                probability = 0
            print probability
            prob[(j, k)]  = probability
            #if probability > 0.3:
             #   image[(j, k)][0] = 250
              #  image[(j, k)][1] = 0
               # image[(j, k)][2] = 0
    threshold = 0.55
    while True:
        result = prob > threshold
        total = np.nonzero(result)
        if len(total[0]) > 4500:
            threshold += 0.01
        else:
            break
        if threshold > 0.75:
            # Too many pixels are getting marked as very very likely to be hand color.  There must be something wrong.
            raise Exception, "BeyondReasonableThresholdError"

    if threshold <= 0.56:
        # It's likely there is not a hand in the frame
        raise Exception, "BelowReasonableThresholdError"

    image2_copy = image2.copy()
    import ipdb; ipdb.set_trace()
    image2_copy[result, :, 0] = 255
    image2_copy[result, :, 1] = 255
    image2_copy[result, :, 2] = 255

    image3 = image2.copy()

    bad_probability = prob <= 0.5
    image3[bad_probability, :, 0] = 0
    image3[bad_probability, :, 1] = 0
    image3[bad_probability, :, 2] = 0

    show_image3 = image3.copy()
    mser = cv2.MSER()
    regions = mser.detect(image3, None)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    cv2.polylines(show_image3, hulls, 1, (0, 255, 0))
    cv2.imshow('show_image3', show_image3)

    show_image2 = image2_copy.copy()
    mser = cv2.MSER()
    regions = mser.detect(image2_copy, None)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    cv2.polylines(show_image2, hulls, 1, (0, 255, 0))
    cv2.imshow('show_image2', show_image2)
    import ipdb; ipdb.set_trace()

    cv2.imwrite("result.jpg", image2)
    cv2.imshow("w1", image2)
    now = time.time()
    print now - then


while True:
    repeat()
