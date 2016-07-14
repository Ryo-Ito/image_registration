import numpy as np
from rtk import uniform_filter, gradient

np.seterr(all='ignore')


class ZNCC(object):

    def __init__(self, penalty, window_length, window_size):
        self.penalty = penalty
        self.window_length = window_length
        self.window_size = window_size

    def __str__(self):
        return ("Zero-means Normalized Cross Correlation, panalty="
                + str(self.penalty)
                + ", window_length="
                + str(self.window_length))

    def cost(self, J, I):
        return np.sum(self.local_cost(J, I))

    def local_cost(self, J, I):
        Im = uniform_filter(I, self.window_length) / self.window_size
        Jm = uniform_filter(J, self.window_length) / self.window_size
        II = (uniform_filter(I * I, self.window_length)
              - self.window_size * Im * Im)
        JJ = (uniform_filter(J * J, self.window_length)
              - self.window_size * Jm * Jm)
        IJ = (uniform_filter(I * J, self.window_length)
              - self.window_size * Im * Jm)
        cost = -(IJ ** 2) / (II * JJ)
        cost[np.where((II < 1e-5) + (JJ < 1e-5))] = 0
        return cost

    def derivative(self, J, I):
        """
        derivative of cost function of zero means normalized cross correlation

        Parameters
        ----------
        J : ndarray
            Input deformed fixed images.
            eg. 3 dimensional case (len(x), len(y), len(z))
        I : ndarray
            Input deformed moving images.

        Returns
        -------
        momentum : ndarray
            momentum field.
            eg. 3d case (dimension, len(x), len(y), len(z))
        """
        Im = uniform_filter(I, self.window_length) / self.window_size
        Jm = uniform_filter(J, self.window_length) / self.window_size

        Ibar = I - Im
        Jbar = J - Jm

        II = (uniform_filter(I * I, self.window_length)
              - self.window_size * Im * Im)
        JJ = (uniform_filter(J * J, self.window_length)
              - self.window_size * Jm * Jm)
        IJ = (uniform_filter(I * J, self.window_length)
              - self.window_size * Im * Jm)

        denom = II * JJ
        IJoverIIJJ = IJ / denom
        IJoverII = IJ / II
        IJoverIIJJ[np.where(denom < 1e-3)] = 0
        IJoverII[np.where(II < 1e-3)] = 0

        return (2 * gradient(Ibar) * IJoverIIJJ
                * (Jbar - Ibar * IJoverII) / self.penalty)
