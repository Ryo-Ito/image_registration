from grid import Deformation
from image import ScalarImage


def load(filename, dtype='scalarimage'):
    """
    load image in the file

    Parameters
    ----------
    filename : str
        name of the file
    dtype : str
        name of the data type of file
        ['scalarimage', 'deformation']

    Returns
    -------
    img : rtk.image.ScalarImage
        loaded image
    """
    all_dtypes = ['scalarimage', 'deformation']
    if dtype not in all_dtypes:
        raise ValueError, "type must be one of", all_dtypes

    if dtype == 'scalarimage':
        return ScalarImage(filename=filename)
    elif dtype == 'deformation':
        return Deformation(filename=filename)
    else:
        return None


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
