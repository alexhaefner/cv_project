import cv
import cv2
import numpy as np

class Gaussian3D:
    sigma = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    mu = np.array([0, 0, 0])
    def calculate_sigma(self, rgbs):
        self.sigma = np.matrix(np.cov(rgbs, rowvar=0))
        self.mu = np.matrix(np.average(rgbs, axis=0))

def KLDistance(gauss1, gauss2):
    """ Calculate the Kullback Lieber distance between two gaussians """
    gauss2_inverse = np.linalg.pinv(gauss2.sigma)
    mu1_minus_mu2 = gauss1.mu - gauss2.mu
    res = 0.5*np.log(np.linalg.det(gauss2.sigma)/np.linalg.det(gauss1.sigma)) + np.trace(gauss2_inverse*gauss1.sigma) - 3 + (mu1_minus_mu2)*(gauss2_inverse)*(mu1_minus_mu2).T
    if np.isnan(res):
        return 0.0
    return res
