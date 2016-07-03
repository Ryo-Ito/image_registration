import numpy as np
import rtk


class DiffeomorphicDeformation(object):

    def __init__(self, n_step, shape):
        self.n_step = n_step
        self.shape = shape
        self.ndim = len(shape)

        self.initial_grid = rtk.identity_mapping(self.shape)

        self.grids = np.ones(
            (self.n_step + 1, self.ndim) + self.shape) * self.initial_grid
        self.jacobian_determinants = np.ones((self.n_step + 1,) + self.shape)
