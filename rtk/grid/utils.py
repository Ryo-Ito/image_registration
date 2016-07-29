import numpy as np
from rtk import gradient


def identity_mapping(shape):
    ndim = len(shape)

    if ndim == 2:
        return np.mgrid[:shape[0], :shape[1]].astype(np.float)
    elif ndim == 3:
        return np.mgrid[:shape[0], :shape[1], :shape[2]].astype(np.float)


def jacobian_matrix(grid):
    """
    Jacobian matrix at all points

    Parameters
    ----------
    grid : ndarray
        Input grid.
        eg. if 3 dimensional ((dimension, len(x), len(y), len(z)))

    Returns
    -------
    J : ndarray
        Jacobian matrix.
        eg. 3d case (dimension, dimension, len(x), len(y), len(z))
    """
    dimension = grid.ndim - 1

    if dimension == 2:
        return np.array([gradient(grid[0]), gradient(grid[1])])
    elif dimension == 3:
        return np.array(
            [gradient(grid[0]), gradient(grid[1]), gradient(grid[2])])


def determinant(J):
    """
    Determinant of jacobian matrix at all points

    Parameters
    ----------
    J : ndarray
        Input jacobian matrixs.
        eg. 3 dimensional case ((dimension, dimension, len(x), len(y), len(z)))

    Returns
    -------
    D : ndarray
        Determinant of jacobian matrix.
        eg. 3 dimensional case (len(x), len(y), len(z))
    """
    dimension = J.ndim - 2

    if dimension == 2:
        return J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
    elif dimension == 3:
        """
        (0,0) (0,1) (0,2)
        (1,0) (1,1) (1,2)
        (2,0) (2,1) (2,2)
        """
        return (J[0, 0] * J[1, 1] * J[2, 2]
                + J[1, 0] * J[2, 1] * J[0, 2]
                + J[0, 1] * J[1, 2] * J[2, 0]
                - J[0, 0] * J[1, 2] * J[2, 1]
                - J[2, 2] * J[1, 0] * J[0, 1]
                - J[0, 2] * J[1, 1] * J[2, 0])
