import numpy as np
import nibabel as nib
from imageprocessing import gradient


def pngtonifti(pngfile, savefile):
    """
    save png image file as nifti file

    Parameters
    ----------
    pngfile : string
        filename of png image
    savefile : string
        name of nifti file
    """
    from scipy.ndimage import imread
    img_data = imread(pngfile, mode='L')

    nib.save(nib.Nifti1Image(img_data, np.identity(4)), savefile)


def brain_extraction(file):
    from dipy.segment.mask import median_otsu
    from os.path import splitext
    img = nib.load(file)
    data = img.get_data()
    masked_data, mask = median_otsu(data, 2, 1)
    mask_img = nib.Nifti1Image(mask.astype(np.int), img.get_affine())
    masked_img = nib.Nifti1Image(masked_data, img.get_affine())

    root, ext = splitext(file)

    nib.save(mask_img, root + '_binary_mask.nii')
    nib.save(masked_img, root + '_masked.nii')


def identity_mapping(shape):
    ndim = len(shape)

    if ndim == 2:
        return np.mgrid[:shape[0], :shape[1]].astype(np.float)
    elif ndim == 3:
        return np.mgrid[:shape[0], :shape[1], :shape[2]].astype(np.float)


def jacobian_matrix(grid):
    """
    Jacobian matrix at all points

    Parameters
    ----------
    grid : ndarray
        Input grid.
        eg. if 3 dimensional ((dimension, len(x), len(y), len(z)))

    Returns
    -------
    J : ndarray
        Jacobian matrix.
        eg. 3d case (dimension, dimension, len(x), len(y), len(z))
    """
    dimension = grid.ndim - 1

    if dimension == 2:
        return np.array([gradient(grid[0]), gradient(grid[1])])
    elif dimension == 3:
        return np.array(
            [gradient(grid[0]), gradient(grid[1]), gradient(grid[2])])


def determinant(J):
    """
    Determinant of jacobian matrix at all points

    Parameters
    ----------
    J : ndarray
        Input jacobian matrixs.
        eg. 3 dimensional case ((dimension, dimension, len(x), len(y), len(z)))
    Returns
    -------
    D : ndarray
        Determinant of jacobian matrix.
        eg. 3 dimensional case (len(x), len(y), len(z))
    """
    dimension = J.ndim - 2

    if dimension == 2:
        return J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
    elif dimension == 3:
        """
        (0,0) (0,1) (0,2)
        (1,0) (1,1) (1,2)
        (2,0) (2,1) (2,2)
        """
        return (J[0, 0] * J[1, 1] * J[2, 2] +
                J[1, 0] * J[2, 1] * J[0, 2] +
                J[0, 1] * J[1, 2] * J[2, 0] -
                J[0, 0] * J[1, 2] * J[2, 1] -
                J[2, 2] * J[1, 0] * J[0, 1] -
                J[0, 2] * J[1, 1] * J[2, 0])
