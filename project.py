
import time
import math
from threading import Thread

import numpy as np
import cv
import cv2
from build_gaussian import build_gaussian
from gaussian import KLDistance, Gaussian3D


cv2.namedWindow("w1", cv.CV_WINDOW_AUTOSIZE)
camera_index = 0
count = 0
capture = cv2.VideoCapture(camera_index)
image_collection = ['0', '1', '2', '3', '4', '5', '6', '7']
N_o = build_gaussian(['hands/{0}.png'.format(count) for count in image_collection])
N_B = build_gaussian(['background/{0}.png'.format(count) for count in image_collection])
prob = np.zeros((241, 321))

def repeat():
    global capture #declare as globals since we are assigning to them now
    global camera_index
    global count
    image = capture.read()[1] # image.shape to get height, width, depth
    image = cv2.resize(image, (320, 240))
    image2 = np.copy(image)
    image = cv2.integral(image)
    P = np.zeros((3, 241, 321))
    P[(0)] = image[:, :, 0]
    P[(1)] = image[:, :, 1]
    P[(2)] = image[:, :, 2]
    Q = np.zeros((6, 241, 321))
    import ipdb; ipdb.set_trace()
    Q[(0)] = P[0] * P[0]
    Q[(1)] = P[0] * P[1]
    Q[(2)] = P[0] * P[2]
    Q[(3)] = P[1] * P[1]
    Q[(4)] = P[1] * P[2]
    Q[(5)] = P[2] * P[2]
    height, width, _ = image.shape
    p1 = Thread(target=process_image, args=(2, 318, 2, 238, P, Q, image2))
    p1.start()
    #p2 = Thread(target=process_image, args=(161, 318, 2, 238, image, image2)) 
    #p2.start()
    p1.join()
    #p2.join()
    #process_image(2, 318, 2, 238, image, image2)


def process_image(start_width, end_width, start_height, end_height, P, Q, image2):
    global prob
    then = time.time()
    r = 3
    image3 = np.copy(image2)
    height = 240
    width = 320
    w = 3
    dict_to_keys = {(0,0): (0), (0,1): (1), (1,0): (1), (0,2): (2), (2,0): (2), (1,1): (3), (1,2): (4), (2,1): (4), (2,2): (5)}
    for j in xrange(start_height, end_height):
        w_start = j-1
        for k in xrange(start_width, end_width):
            r_start = k-1
            N_iw = Gaussian3D()
            Psub = np.zeros((3))
            ul =  (w_start, r_start)
            ur = (w_start, r_start + 2)
            ll = (w_start+w, r_start)
            lr = (w_start+w, r_start+r)
            for i in xrange(0, 3):
                Psub[(i)] = P[(i)][ul] + P[(i)][lr] - P[(i)][ur] - P[(i)][ll]
            Psub = np.asmatrix(Psub)
            Qsub = np.zeros((3, 3))
            for i in xrange(0, 3):
                for mm in xrange(0,3):
                    Qsub[(i,mm)] = (Q[dict_to_keys[(i, mm)]][ul] + Q[dict_to_keys[(i,mm)]][lr] - Q[dict_to_keys[(i, mm)]][ur] - Q[dict_to_keys[(i,mm)]][ll])/9.0

            Qsub = np.asmatrix(Qsub)
            rgbs = []
            for index in xrange(w_start, w_start + 3):
                for rindex in xrange(r_start, r_start + 3):
                    rgbs.append(image2[(index, rindex)])
            #print "{0}, {1}, w_start: {2}, r_start: {3}".format(j, k, w_start, r_start)
            N_iw.mu = Psub / 9.0
            N_iw.calculate_sigma(rgbs)
            m = float(r*(w))
            cov = (1.0/(m-1.0))*(Qsub - (1.0/m)*(Psub.T*Psub))
            N_iw.sigma = np.reshape(cov, (3,3))

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

    results =  prob > 0.86
    image2[results, 0] = 255
    image2[results, 1] = 0
    image2[results, 2] = 255
    cv2.imshow("w1", image2)
    cv2.imwrite('{0}.jpg'.format('result'), image2)
    c = cv2.waitKey(10)
    if(c == 101):
        exit()
    if(c == 112):
        cv2.imwrite('{0}.jpg'.format(count), image)
        count += 1
    now = time.time()
    print now - then


if __name__ == '__main__':
    repeat()
