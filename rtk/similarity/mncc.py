import numpy as np
from rtk import gradient, sliding_matmul


class MNCC(object):

    def __init__(self, variance, matrix):
        self.variance = variance
        self.matrix = matrix
        self.index = int((len(matrix) - 1) / 2)

    def __str__(self):
        return ("Mahalanobis Normalized Cross Correlation"
                + ", variance=" + str(self.variance)
                + ", window_size" + str(len(self.matrix)))

    def cost(self, J, I):
        return np.sum(self.local_cost(J, I))

    def local_cost(self, J, I):
        Ai = sliding_matmul(I, self.matrix)
        Aj = sliding_matmul(J, self.matrix)

        II = np.einsum('...i,...i->...', Ai, Ai)
        JJ = np.einsum('...i,...i->...', Aj, Aj)
        IJ = np.einsum('...i,...i->...', Ai, Aj)
        IIJJ = II * JJ
        cost = -(IJ ** 2) / IIJJ
        cost[np.where((II < 1e-5) + (JJ < 1e-5))] = 0

        return cost

    def derivative(self, J, I):
        """
        derivative of cost function of mahalanobis normalized cross correlation

        Parameters
        ----------
        J : ndarray
            Input deformed fixed image.
            eg. 3 dimensional case (len(x), len(y), len(z))
        I : ndarray
            Input deformed moving image.
        matrix : ndarray
            metric matrix

        Returns
        -------
        momentum : ndarray
            Unsmoothed vector field
        """
        assert(I.dtype == np.float)
        assert(J.dtype == np.float)
        Ai = sliding_matmul(I, self.matrix)
        Aj = sliding_matmul(J, self.matrix)
        Ibar = np.copy(Ai[..., self.index]).astype(np.float)
        Jbar = np.copy(Aj[..., self.index]).astype(np.float)

        II = np.einsum('...i,...i->...', Ai, Ai)
        JJ = np.einsum('...i,...i->...', Aj, Aj)
        IJ = np.einsum('...i,...i->...', Ai, Aj)
        IIJJ = II * JJ
        IJoverIIJJ = IJ / IIJJ
        IJoverII = IJ / II
        IJoverIIJJ[np.where(IIJJ < 1e-3)] = 0
        IJoverII[np.where(II < 1e-3)] = 0

        return (2 * gradient(Ibar) * IJoverIIJJ
                * (Jbar - Ibar * IJoverII) / self.variance)
