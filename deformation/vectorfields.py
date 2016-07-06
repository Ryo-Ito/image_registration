import numpy as np


class VectorFields(object):

    def __init__(self, n_step, shape=None):
        self.n_step = n_step

        if shape is not None:
            self.set_shape(shape)

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
