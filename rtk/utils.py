from grid import Deformation
from image import ScalarImage


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
        raise ValueError, "type must be one of", all_dtypes

    if dtype == 'scalarimage':
        return ScalarImage(filename=filename)
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
