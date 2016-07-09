import functools
import numpy as np
import rtk
from rtk.deformation import DiffeomorphicDeformation


class Registration(object):

    def __init__(self,
                 ndim,
                 n_step,
                 penalty,
                 regularizer,
                 similarity,
                 window_length=None,
                 metric_matrix=None,
                 n_iters=(50, 20, 10),
                 resolutions=(4, 2, 1),
                 smoothing_sigmas=(2, 1, 0),
                 delta_phi_threshold=1.,
                 unit_threshold=0.1,
                 learning_rate=0.1,
                 parallel=False,
                 n_jobs=1):
        self.ndim = ndim
        self.n_step = n_step
        self.penalty = penalty
        self.deformation = DiffeomorphicDeformation(
            n_step=n_step)
        self.regularizer = regularizer

        self.similarity = similarity
        if similarity == 'zncc':
            if window_length is None:
                self.window_length = 5
            else:
                self.window_length = window_length
            self.window_size = self.window_length ** self.ndim
        elif similarity == 'mncc':
            if metric_matrix is None:
                self.similarity = 'zncc'
                self.window_length = 5
                self.window_size = self.window_length ** self.ndim
            else:
                self.window_size = len(metric_matrix)
                self.window_length = int(np.round(
                    self.window_size ** (1. / self.ndim)))
                assert self.window_length ** self.ndim == self.window_size
                assert self.window_length % 2 == 1
                self.metric_matrix = metric_matrix

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
        self.parallel = parallel
        if not parallel:
            n_jobs = 1
        self.n_jobs = n_jobs

        self.set_similarity_functions()
        self.print_settings()

    def print_settings(self):
        print self.__class__.__name__
        print "similarity metric", self.similarity
        if hasattr(self, 'window_length'):
            print "window length", self.window_length
        print "regularization", self.regularizer.__class__.__name__
        print "iterations", self.n_iters
        print "resolutions", self.resolutions
        print "smoothing sigmas", self.smoothing_sigmas
        print "threshold of displacement update", self.delta_phi_threshold
        print "threshold of grid unit", self.unit_threshold
        print "learning rate", self.learning_rate
        print "parallel computation", self.parallel
        print "number of cpu cores", self.n_jobs

    def set_similarity_functions(self):
        all_similarity_metric = ['ssd', 'zncc', 'mncc']
        if self.similarity not in all_similarity_metric:
            raise ValueError("input similarity metric is not valid")
        if self.similarity == 'ssd':
            from rtk.similarity.ssd import cost_function_ssd, derivative_ssd
            self.cost_function = cost_function_ssd
            self.derivative = derivative_ssd
        elif self.similarity == 'zncc':
            from rtk.similarity.zncc import cost_function_zncc, derivative_zncc
            self.cost_function = functools.partial(
                cost_function_zncc,
                window_length=self.window_length,
                window_size=self.window_size)
            self.derivative = functools.partial(
                derivative_zncc,
                window_length=self.window_length,
                window_size=self.window_size)
        elif self.similarity == 'mncc':
            from rtk.similarity.mncc import cost_function_mncc, derivative_mncc
            self.cost_function = functools.partial(
                cost_function_mncc,
                matrix=self.metric_matrix)
            self.derivative = functools.partial(
                derivative_mncc,
                matrix=self.metric_matrix)
        else:
            raise ValueError

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
