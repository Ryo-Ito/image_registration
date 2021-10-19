#!/usr/bin/python3

import numpy as np
import pyvista as pv
import cfl

def dump_cfl(filename):
    """
    function to load cfl file and dump array into vtk file
    """
    fname = filename.split('.')[0]
    data = np.abs(cfl.readcfl(filename))
    wrapped = pv.wrap(data)
    path = f'{fname}.vtk'
    wrapped.save(path)
    return path


def dump_array(array, fname):
    """
    function to dump numpy array into vtk file
    """

    assert(isinstance(array, np.ndarray)), 'Please pass in an array!'
    wrapped = pv.wrap(array)
    wrapped.save(f'{fname}.vtk')
    
    return fname


def read(filename):
    """
    function to read a vtk file into python
    """

    vis = pv.read(filename)
    return vis


def wrap(array):
    """
    wrapper for pv.wrap()
    """


    assert(isinstance(array, np.ndarray)), 'Please pass in an array!'
    return pv.wrap(array)

def save(array, filename):
    """
    Wrap and save filename
    """
    array = np.expand_dims(array, axis=0)
    out = pv.wrap(array)
    out.save(filename)


def plot(vtk_array, volume=True):
    """
    function to plot a vtk array using pyvista
    """

    pv.plot(vtk_array, opacity='linear', volume=volume)

