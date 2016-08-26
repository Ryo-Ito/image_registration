import numpy as np
from scipy.ndimage.filters import gaussian_filter
from rtk import gradient


class GNCC(object):
    """
    gaussian normalized cross correlation
    similarity = <I, J>^2 / (<I, I> * <J,J>)
    <I,J> = G{G*(I - Im)(J - Jm)}
    Im = mean(G * I)
    G is a gaussian filter
    """
    def __init__(self, variance, sigma):
        self.variance = variance
        self.sigma = sigma

    def __str__(self):
        return ("Gaussian Normalized Cross Correlation"
                + ", variance=" + str(self.variance)
                + ", gaussian kernel's sigma" + str(self.sigma))

    def cost(self, J, I):
        return np.sum(self.local_cost(J, I))

    def local_cost(self, J, I):
        Im = gaussian_filter(I, self.sigma, mode="constant")
        Jm = gaussian_filter(J, self.sigma, mode="constant")
        II = gaussian_filter(I * I, self.sigma, mode="constant") - Im * Im
        JJ = gaussian_filter(J * J, self.sigma, mode="constant") - Jm * Jm
        IJ = gaussian_filter(I * J, self.sigma, mode="constant") - Im * Jm

        cost = -(IJ ** 2) / (II * JJ)
        cost[np.where((II < 1e-5) + (JJ < 1e-5))] = 0
        return cost

    def derivative(self, J, I):
        Im = gaussian_filter(I, self.sigma, mode="constant")
        Jm = gaussian_filter(J, self.sigma, mode="constant")

        Ibar = I - Im
        Jbar = J - Jm

        II = gaussian_filter(I * I, self.sigma, mode="constant") - Im * Im
        JJ = gaussian_filter(J * J, self.sigma, mode="constant") - Jm * Jm
        IJ = gaussian_filter(I * J, self.sigma, mode="constant") - Im * Jm

        denom = II * JJ
        IJoverIIJJ = IJ / denom
        IJoverII = IJ / II
        IJoverIIJJ[np.where(denom < 1e-3)] = 0
        IJoverII[np.where(II < 1e-3)] = 0

        return (2 * gradient(Ibar) * IJoverIIJJ
                * (Jbar - Ibar * IJoverII) / self.variance)
