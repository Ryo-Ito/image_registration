import numpy as np
from scipy.ndimage.filters import correlate, gradient

np.seterr(all='ignore')


def local_kncc(J, I, kernel):
    Im = correlate(I, kernel, mode='constant') / kernel.size
    Jm = correlate(J, kernel, mode='constant') / kernel.size
    II = correlate(I * I, kernel, mode='constant') - kernel.size * Im * Im
    JJ = correlate(J * J, kernel, mode='constant') - kernel.size * Jm * Jm
    IJ = correlate(I * J, kernel, mode='constant') - kernel.size * Im * Jm

    lkncc = (IJ ** 2) / (II * JJ)
    lkncc[np.where((II < 1e-5) + (JJ < 1e-5))] = 0
    return lkncc


def cost_function_kncc(J, I, kernel):
    return - np.sum(local_kncc(J, I, kernel))


def derivative_kncc(J, I, kernel):
    Im = correlate(I, kernel) / kernel.size
    Jm = correlate(J, kernel) / kernel.size

    Ibar = I - Im
    Jbar = J - Jm

    II = correlate(I * I, kernel) - kernel.size * Im * Im
    JJ = correlate(J * J, kernel) - kernel.size * Jm * Jm
    IJ = correlate(I * J, kernel) - kernel.size * Im * Jm

    denom = II * JJ
    IJoverIIJJ = IJ / denom
    IJoverII = IJ / II
    IJoverIIJJ[np.where(denom < 1e-3)] = 0
    IJoverII[np.where(II < 1e-3)] = 0

    return 2 * gradient(Ibar) * IJoverIIJJ * (Jbar - Ibar * IJoverII)
