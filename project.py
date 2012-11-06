
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
N_o = build_gaussian(['hands/{0}.png'.format(count) for count in ['0', '1', '2']])
N_B = build_gaussian(['background/{0}.png'.format(count) for count in ['0', '1', '2']])

def repeat():
    global capture #declare as globals since we are assigning to them now
    global camera_index
    global count
    image = capture.read()[1] # image.shape to get height, width, depth
    image = cv2.resize(image, (320, 240))
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
                import ipdb; ipdb.set_trace()
            prob[(j, k)]  = probability
            if probability > 0.3:
                image[(j, k)][0] = 250
                image[(j, k)][1] = 0
                image[(j, k)][2] = 0

    print "done"
    cv2.imshow("w1", image)
    c = cv2.waitKey(10)
    if(c == 101):
        exit()
    if(c == 112):
        cv2.imwrite('{0}.jpg'.format(count), image)
        count += 1



while True:
        repeat()
