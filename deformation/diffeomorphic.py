import numpy as np
import rtk


class DiffeomorphicDeformation(object):

    def __init__(self, n_step, shape=None, time_interval=1.):
        self.n_step = n_step
        self.time_interval = time_interval
        self.delta_time = time_interval / n_step

        if shape is not None:
            self.set_shape()

    def set_shape(self, shape):
        self.shape = shape
        self.ndim = len(shape)
        self.init_mappings()

    def init_mappings(self):
        self.initial_grid = rtk.identity_mapping(self.shape)
        self.forward_mappings = np.ones(
            (self.n_step + 1, self.ndim) + self.shape) * self.initial_grid
        self.backward_mappings = np.copy(self.forward_mappings)
        self.forward_jacobian_determinants = np.ones(
            (self.n_step + 1,) + self.shape)
        self.backward_jacobian_determinants = np.ones(
            (self.n_step + 1,) + self.shape)
