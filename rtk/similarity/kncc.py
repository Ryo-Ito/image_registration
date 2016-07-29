import numpy as np
from scipy.ndimage.filters import correlate
from rtk import gradient

np.seterr(all='ignore')


class KNCC(object):
    """
    kernel normalized cross correlation
    similairty = <I,J>^2 / (<I,I> * <J,J>)
    <I,J> = sum{G*(I - Im)(J - Jm)}
    Im = mean(G*I)
    G is a general convolution kernel
    """

    def __init__(self, variance, kernel):
        self.variance = variance
        self.kernel = kernel
        self.kernel_size = kernel.size

    def __str__(self):
        return ("Kernel Normalized Cross Correlation"
                + ", variance=" + str(self.variance)
                + ", kernel_size=" + str(self.kernel_size))

    def cost(self, J, I):
        return np.sum(self.local_cost(J, I))

    def local_cost(self, J, I):
        Im = correlate(I, self.kernel, mode='constant') / self.kernel_size
        Jm = correlate(J, self.kernel, mode='constant') / self.kernel_size
        II = (correlate(I * I, self.kernel, mode='constant')
              - self.kernel_size * Im * Im)
        JJ = (correlate(J * J, self.kernel, mode='constant')
              - self.kernel_size * Jm * Jm)
        IJ = (correlate(I * J, self.kernel, mode='constant')
              - self.kernel_size * Im * Jm)

        cost = -(IJ ** 2) / (II * JJ)
        cost[np.where((II < 1e-5) + (JJ < 1e-5))] = 0
        return cost

    def derivative(self, J, I):
        Im = correlate(I, self.kernel, mode='constant') / self.kernel_size
        Jm = correlate(J, self.kernel, mode='constant') / self.kernel_size

        Ibar = I - Im
        Jbar = J - Jm

        II = (correlate(I * I, self.kernel, mode='constant')
              - self.kernel_size * Im * Im)
        JJ = (correlate(J * J, self.kernel, mode='constant')
              - self.kernel_size * Jm * Jm)
        IJ = (correlate(I * J, self.kernel, mode='constant')
              - self.kernel_size * Im * Jm)

        denom = II * JJ
        IJoverIIJJ = IJ / denom
        IJoverII = IJ / II
        IJoverIIJJ[np.where(denom < 1e-3)] = 0
        IJoverII[np.where(II < 1e-3)] = 0

        return (2 * gradient(Ibar) * IJoverIIJJ
                * (Jbar - Ibar * IJoverII) / self.variance)
