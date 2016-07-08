import numpy as np
from rtk import gradient


def cost_function_ssd(data1, data2):
    return np.sum((data1 - data2) ** 2)


def derivative_ssd(fixed, moving):
    return 2 * gradient(moving) * (fixed - moving)
