import cv
import cv2
import numpy as np

class Gaussian3D:
    sigma = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    mu = np.array([0, 0, 0])
    def calculate_sigma(self, rgbs):
        self.sigma = np.cov(rgbs, rowvar=0)
        self.mu = np.average(rgbs, axis=0)

def KLDistance(gauss1, gauss2):
    """ Calculate the Kullback Lieber distance between two gaussians """
    return 0.5*np.log(np.linalg.det(gauss2.sigma)/np.linalg.det(gauss1.sigma)) + np.trace(np.linalg.pinv(np.asmatrix(gauss2.sigma))*np.asmatrix(gauss1.sigma)) - 3 + np.asmatrix(gauss1.mu - gauss2.mu)*np.asmatrix(np.linalg.pinv(gauss2.sigma))*np.asmatrix(gauss1.mu - gauss2.mu).T
