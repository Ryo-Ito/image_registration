import numpy as np
from scipy.ndimage.interpolation import zoom


class VectorFields(object):

    def __init__(self, n_step, shape=None, vector_fields=None):
        self.n_step = n_step

        if shape is not None:
            self.set_shape(shape)

        if vector_fields is not None:
            assert n_step + 1 == len(vector_fields)
            self.vector_fields = vector_fields
            self.delta_vector_fields = np.zeros_like(vector_fields)
            self.shape = vector_fields.shape[2:]
            self.ndim = vector_fields.shape[1]

    def __getitem__(self, index):
        return self.vector_fields[index]

    def set_shape(self, shape):
        self.shape = shape
        self.ndim = len(shape)
        self.init_vector_fields()

    def init_vector_fields(self):
        self.vector_fields = np.zeros(
            (self.n_step + 1, self.ndim) + self.shape)
        self.delta_vector_fields = np.copy(self.vector_fields)

    def update(self):
        """
        update vector fields

        v_next = v_now - learning_rate * nabla(Energy)
        """
        self.vector_fields -= self.delta_vector_fields

    def back_to_previous(self):
        """
        get back to previous vector field

        v^(k+1) = v^(k) - learning_rate * nabla(Energy)

        v_before = v_now + learning_rate * nabla(Energy_before)
        """
        self.vector_fields += self.delta_vector_fields

    def change_resolution(self, resolution, order=1):
        """
        change image's resolution

        Parameters
        ----------
        resolution : int
            how much to magnify
            if resolution is 2, the shape of the image will be halved
        sigma : float
            standard deviation of gaussian filter for smoothing
        order : int
            order of interpolation

        Returns
        -------
        vector_fields : VectorFields
            zoomed vector fields
        """
        if resolution == 1:
            return VectorFields(self.n_step,
                                vector_fields=np.copy(self.vector_fields))
        ratio = [1, 1] + [1 / float(resolution)] * self.ndim
        data = zoom(self.vector_fields, ratio, order=order)
        return VectorFields(self.n_step, vector_fields=data)
