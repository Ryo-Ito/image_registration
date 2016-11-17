import numpy as np
from rtk import gradient, uniform_filter

np.seterr(all='ignore')


class ZNCC(object):
    """
    Zero means Normalized Cross Correlation
    similairty = <I,J>^2 / (<I,I> * <J,J>)
    <I,J> = sum{G*(I - Im)(J - Jm)}
          = sum{G*(I * J) - G*(I)*Jm - G*(J)*Im + Im * Jm}
          = sum{G*(I * J)} - Im * Jm
    Im = sum(G*I)
    G is a uniform convolution kernel
    """
    def __init__(self, variance, window_length):
        self.variance = variance
        self.window_length = window_length
        # self.window_size = window_size

    def __str__(self):
        return ("Zero-means Normalized Cross Correlation, panalty="
                + str(self.variance)
                + ", window_length="
                + str(self.window_length))

    def cost(self, J, I):
        return np.sum(self.local_cost(J, I))

    def local_cost(self, J, I):
        Im = uniform_filter(I, self.window_length)
        Jm = uniform_filter(J, self.window_length)
        II = uniform_filter(I * I, self.window_length) - Im * Im
        JJ = uniform_filter(J * J, self.window_length) - Jm * Jm
        IJ = uniform_filter(I * J, self.window_length) - Im * Jm
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
        Im = uniform_filter(I, self.window_length)
        Jm = uniform_filter(J, self.window_length)

        Ibar = I - Im
        Jbar = J - Jm

        II = uniform_filter(I * I, self.window_length) - Im * Im
        JJ = uniform_filter(J * J, self.window_length) - Jm * Jm
        IJ = uniform_filter(I * J, self.window_length) - Im * Jm

        denom = II * JJ
        IJoverIIJJ = IJ / denom
        IJoverII = IJ / II
        IJoverIIJJ[np.where(denom < 1e-3)] = 0
        IJoverII[np.where(II < 1e-3)] = 0

        return (2 * gradient(Ibar) * IJoverIIJJ
                * (Jbar - Ibar * IJoverII) / self.variance)
