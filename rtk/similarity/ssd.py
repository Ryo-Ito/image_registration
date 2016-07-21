import numpy as np
from rtk import gradient


class SSD(object):

    def __init__(self, variance):
        self.variance = variance

    def __str__(self):
        return "Sum of Squared Difference, variance=%f" % self.variance

    def cost(self, fixed, moving):
        return np.sum(self.local_cost(fixed, moving))

    def local_cost(self, fixed, moving):
        return np.square(fixed - moving)

    def derivative(self, fixed, moving):
        return 2 * gradient(moving) * (fixed - moving) / self.variance
