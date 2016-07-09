import numpy as np
from rtk import uniform_filter, gradient

np.seterr(all='ignore')


def cost_function_zncc(I, J, window_length, window_size):
    Im = uniform_filter(I, window_length) / window_size
    Jm = uniform_filter(J, window_length) / window_size
    II = uniform_filter(I * I, window_length) - window_size * Im * Im
    JJ = uniform_filter(J * J, window_length) - window_size * Jm * Jm
    IJ = uniform_filter(I * J, window_length) - window_size * Im * Jm
    E = (IJ ** 2) / (II * JJ)
    E[np.where((II < 1e-5) + (JJ < 1e-5))] = 0
    return - np.sum(E)


def derivative_zncc(J, I, window_length, window_size):
    """
    derivative of cost function of zero means normalized cross correlation

    Parameters
    ----------
    J : ndarray
        Input deformed fixed images.
        eg. 3 dimensional case (len(x), len(y), len(z))
    I : ndarray
        Input deformed moving images.
    window_length: int
        window_length of window
    window_size: int
        window_size of window

    Returns
    -------
    momentum : ndarray
        momentum field.
        eg. 3d case (dimension, len(x), len(y), len(z))
    """
    Im = uniform_filter(I, window_length) / window_size
    Jm = uniform_filter(J, window_length) / window_size

    Ibar = I - Im
    Jbar = J - Jm

    II = uniform_filter(I * I, window_length) - window_size * Im * Im
    JJ = uniform_filter(J * J, window_length) - window_size * Jm * Jm
    IJ = uniform_filter(I * J, window_length) - window_size * Im * Jm

    denom = II * JJ
    IJoverIIJJ = IJ / denom
    IJoverII = IJ / II
    IJoverIIJJ[np.where(denom < 1e-3)] = 0
    IJoverII[np.where(II < 1e-3)] = 0

    return 2 * gradient(Ibar) * IJoverIIJJ * (Jbar - Ibar * IJoverII)
