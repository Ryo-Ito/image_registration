import numpy as np
from rtk import gradient


class SSD(object):

    def __init__(self, penalty):
        self.penalty = penalty

    def __str__(self):
        return "Sum of Squared Difference, penalty=%f" % self.penalty

    def cost(self, fixed, moving):
        return np.sum(self.local_cost(fixed, moving))

    def local_cost(self, fixed, moving):
        return np.square(fixed - moving)

    def derivative(self, fixed, moving):
        return 2 * gradient(moving) * (fixed - moving) / self.penalty
