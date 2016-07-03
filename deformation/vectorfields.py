class VectorFields(object):

    def __init__(self, n_step, shape):
        self.n_step = n_step
        self.shape = shape
        self.ndim = len(shape)
