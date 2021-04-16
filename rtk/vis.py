#!/usr/bin/python3

# VISUALIZATION SCRIPT FOR QUICKLY PLOTTING SLICES OF 3D ARRAY

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import cfl
import sys


def plot_slices(data, *args, **kwargs):

    fig = plt.figure()
    ax = fig.gca()
    p = plt.imshow(np.abs(data[:,:, 0]), *args, **kwargs)
    plt.colorbar()

    ax = fig.add_axes([0.25, 0.1, 0.65, 0.03])

    slc = Slider(ax, 'Slice Number', 0, len(data[0,0,:]) - 1, valinit=0, valstep=1)

    def update(val):
        s = int(slc.val)

        p.set_data(np.abs(data[:, :, s]))
        fig.canvas.draw()

    slc.on_changed(update)

    plt.show()
