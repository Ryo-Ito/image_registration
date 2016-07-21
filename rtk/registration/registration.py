import numpy as np
import rtk
from rtk.deformation import DiffeomorphicDeformation


class Registration(object):

    def __init__(self,
                 n_step,
                 regularizer,
                 similarity,
                 n_iters=(50, 20, 10),
                 resolutions=(4, 2, 1),
                 smoothing_sigmas=(2, 1, 0),
                 delta_phi_threshold=1.,
                 unit_threshold=0.1,
                 learning_rate=0.1,
                 n_jobs=1):
        self.n_step = n_step
        self.deformation = DiffeomorphicDeformation(
            n_step=n_step)
        self.regularizer = regularizer

        self.similarity = similarity

        try:
            self.n_iters = tuple(n_iters)
        except:
            self.n_iters = (n_iters,)

        try:
            self.resolutions = tuple(resolutions)
        except:
            self.resolutions = (resolutions,)
        while len(self.resolutions) < len(self.n_iters):
            self.resolutions += (self.resolutions[-1],)

        try:
            self.smoothing_sigmas = tuple(smoothing_sigmas)
        except:
            self.smoothing_sigmas = (smoothing_sigmas,)
        while len(self.smoothing_sigmas) < len(self.n_iters):
            self.smoothing_sigmas += (self.smoothing_sigmas[-1],)

        self.delta_phi_threshold = delta_phi_threshold
        self.unit_threshold = unit_threshold
        self.learning_rate = learning_rate
        assert isinstance(n_jobs, int)
        self.n_jobs = n_jobs

    def print_settings(self):
        print self.__class__.__name__
        print self.similarity
        print "regularization", self.regularizer.__class__.__name__
        print "iterations", self.n_iters
        print "resolutions", self.resolutions
        print "smoothing sigmas", self.smoothing_sigmas
        print "threshold of displacement update", self.delta_phi_threshold
        print "threshold of grid unit", self.unit_threshold
        print "learning rate", self.learning_rate
        print "number of cpu cores", self.n_jobs

    def set_images(self, fixed, moving):
        assert fixed.ndim == moving.ndim
        assert fixed.shape == moving.shape

        self.fixed = fixed.change_scale(255)
        self.moving = moving.change_scale(255)

        self.ndim = fixed.ndim
        self.shape = fixed.shape

    def zoom_grid(self, grid, resolution):
        shape = grid.shape[1:]
        if resolution != 1:
            interpolated_grid = np.zeros((self.ndim,) + self.shape)
            for i in xrange(self.ndim):
                interpolated_grid[i] = rtk.interpolate_mapping(
                    grid[i], np.array(self.shape, dtype=np.int32)
                ) * (self.shape[i] - 1) / (shape[i] - 1)
            return interpolated_grid
        else:
            return grid

    def check_injectivity(self):
        self.min_unit = np.min(
            self.deformation.forward_dets[-1])
        if self.min_unit < self.unit_threshold:
            self.vector_fields.back_to_previous()
            self.integrate_vector_fields()
            print "reached limit of jacobian determinant %f < %f" % (
                self.min_unit, self.unit_threshold)
        return self.min_unit > self.unit_threshold
