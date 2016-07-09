import numpy as np
from rtk import gradient, sliding_matrix_product as smp


def cost_function_mncc(I, J, matrix):
    Ai = smp(I, matrix)
    Aj = smp(J, matrix)

    II = np.einsum('...i,...i->...', Ai, Ai)
    JJ = np.einsum('...i,...i->...', Aj, Aj)
    IJ = np.einsum('...i,...i->...', Ai, Aj)
    IIJJ = II * JJ
    E = (IJ ** 2) / IIJJ
    E[np.where((II < 1e-5) + (JJ < 1e-5))] = 0

    return - np.sum(E)


def derivative_mncc(J, I, matrix):
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
    index = int((len(matrix) - 1) / 2)
    Ai = smp(I, matrix)
    Aj = smp(J, matrix)
    Ibar = np.copy(Ai[..., index]).astype(np.float)
    Jbar = np.copy(Aj[..., index]).astype(np.float)

    II = np.einsum('...i,...i->...', Ai, Ai)
    JJ = np.einsum('...i,...i->...', Aj, Aj)
    IJ = np.einsum('...i,...i->...', Ai, Aj)
    IIJJ = II * JJ
    IJoverIIJJ = IJ / IIJJ
    IJoverII = IJ / II
    IJoverIIJJ[np.where(IIJJ < 1e-3)] = 0
    IJoverII[np.where(II < 1e-3)] = 0

    return 2 * gradient(Ibar) * IJoverIIJJ * (Jbar - Ibar * IJoverII)
