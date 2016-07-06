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

    def euler_integration(self, grid, jacobian_matrix, vector_fields):
        return grid - np.einsum('ij...,j...->i...', jacobian_matrix, vector_fields) * self.delta_time

    def update_mappings(self, vector_fields):
        assert len(vector_fields) == self.n_step

        forward_jacobian_matrix = rtk.jacobian_matrix(self.initial_grid)
        backward_jacobian_matrix = np.copy(forward_jacobian_matrix)

        for i in xrange(self.n_step):
            self.forward_mappings[i + 1] = self.euler_integration(
                self.forward_mappings[i],
                forward_jacobian_matrix,
                vector_fields[i])
            self.backward_mappings[i + 1] = self.euler_integration(
                self.backward_mappings[i],
                backward_jacobian_matrix,
                vector_fields[-i - 1])

            forward_jacobian_matrix = rtk.jacobian_matrix(
                self.forward_mappings[i + 1])
            backward_jacobian_matrix = rtk.jacobian_matrix(
                self.backward_mappings[i + 1])

            self.forward_jacobian_determinants[i + 1] = rtk.determinant(
                forward_jacobian_matrix)
            self.backward_jacobian_determinants[i + 1] = rtk.determinant(
                backward_jacobian_matrix)
