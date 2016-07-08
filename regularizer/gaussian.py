import numpy as np
from scipy.ndimage.filters import gaussian_filter
from joblib import Parallel, delayed


class GaussianRegularizer(object):

    def __init__(self, sigma=1., mode='constant', cval=0., truncate=4.):
        self.sigma = sigma
        self.mode = mode
        self.cval = cval
        self.truncate = truncate

    def __call__(self, momentum, n_jobs=None):
        if n_jobs is None:
            vector_field = np.copy(momentum)
            for i in xrange(len(momentum)):
                vector_field[i] = gaussian_filter(
                    momentum[i],
                    self.sigma,
                    mode=self.mode,
                    cval=self.cval,
                    truncate=self.truncate)
            return vector_field
        else:
            return np.asarray(
                Parallel(n_jobs, 'threading')(
                    delayed(gaussian_filter)(
                        flow,
                        self.sigma,
                        mode=self.mode,
                        cval=self.cval,
                        truncate=self.truncate)
                    for flow in momentum))

if __name__ == '__main__':
    r = GaussianRegularizer()
    print r
    print r.__class__.__name__
