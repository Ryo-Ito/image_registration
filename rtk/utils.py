from .grid import Deformation
from .image import ScalarImage

import pydicom as dcm
import os
import glob
import numpy as np

def load_slices(path):
    """
    load image from a folder containing
    lots of dicoms

    Parameters
    ----------
    path : str
        path of directory containing dcm files
    dtype : str
        name of the data type of file
        ['scalarimage']

    Returns
    -------
    out : np.ndarray
        loaded image in numpy array
    """

    files = sorted(glob.glob(os.path.join(path, '*.dcm')))

    image_size = dcm.dcmread(os.path.join(path, files[0])).pixel_array.shape

    out = np.zeros((image_size[0], image_size[1], len(files), 1))


    for p in range(len(files)):
        out[:,:, p, 0] = dcm.dcmread(os.path.join(path, files[p])).pixel_array
    return out


def load_mcle(path, num_echoes=6):
    """
    load multicontrast image from a folder
    containing lots of dicoms

    [[Slice 1], [Slice2], Slice3],...] where each slice contains multiple echoes

    Parameters
    ----------
    filepath : str
        path of directory containing dcm files
    dtype : str
        name of the data type of file
        ['scalarimage']

    Returns
    -------
    out : np.ndarray
        loaded image in numpy array
    """

    array = load_slices(path)

    num_slices = len(array[0, 0, :])

    assert(num_slices % num_echoes == 0), f"Number of echoes ({num_echoes}) is not compatible with length of array ({len(array)})"

    mcle_array = np.zeros(np.shape(array))

    temp_array = []
    for j in range(num_echoes):
        temp_array.append([])
        slc = [i for i in range(num_slices) if i % num_echoes == j]
        for s in slc:
            temp_array[j].append(array[:, :, s, 0])

    count = 0
    for t in temp_array:
        block = count*len(t)
        for i in range(len(t)):
            mcle_array[:, :, block + i, 0] = t[i]
        count += 1

    shx, shy, shz, sht = np.shape(mcle_array)

    num_z = int(num_slices / num_echoes)

    high_dim_array = np.zeros((shx, shy, num_z, num_echoes))

    n = 0
    for e in range(num_echoes):
        high_dim_array[:, :, :, e] = mcle_array[:, :, n*e:n*e + num_z, 0]
        n += 1

    return high_dim_array


def load_img(filename, dtype='scalarimage'):
    """
    load image in the file

    Parameters
    ----------
    filename : str
        name of the file
    dtype : str
        name of the data type of file
        ['scalarimage']

    Returns
    -------
    img : rtk.image.ScalarImage
        loaded image
    """
    all_dtypes = ['scalarimage']
    if dtype not in all_dtypes:
        raise ValueError #, "type must be one of", all_dtypes

    if dtype == 'scalarimage':
        return ScalarImage(filename=filename)
    else:
        return None


def load_dicom(filepath, dtype='scalarimage', mcle=False):
    """
    load image contained in a folder
    with lots of dicom files

    Parameters
    ----------
    filepath : str
        name of the file
    dtype : str
        name of the data type of file
        ['scalarimage']

    Returns
    -------
    img : rtk.image.ScalarImage
        loaded image
    """
    all_dtypes = ['scalarimage']
    if dtype not in all_dtypes:
        raise ValueError #, "type must be one of", all_dtypes

    if dtype == 'scalarimage':

        if mcle:
            data = load_mcle(filepath)

        else:
            data = load_slices(filepath)[:,:,:,0]

        return ScalarImage(data=data)
    else:
        return None


def load_warp(filename):
    """
    load deformation field in the file

    Parameters
    ----------
    filename : str
        name of the file containing deformation field

    Returns
    -------
    deformation : rtk.grid.Deformation
        load deformation field
    """
    return Deformation(filename=filename)


def transform(img, warp):
    if not hasattr(img, "apply_transform"):
        raise NotImplementedError
    else:
        if type(warp) == Deformation:
            return img.apply_transform(warp)
        else:
            return img.apply_transform(Deformation(grid=warp))


def show(obj, **args):
    if not hasattr(obj, 'show'):
        raise NotImplementedError
    else:
        obj.show(**args)


def save(obj, filename, **args):
    if not hasattr(obj, 'save'):
        raise NotImplementedError
    else:
        obj.save(filename=filename, **args)
