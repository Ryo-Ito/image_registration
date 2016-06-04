import numpy as np
from cythonTool import uniformFilter

def segmentation_dissimilarity(img1, img2, window_length):
    """
    calculate patchwise dissimilarity of segmentation labels at every voxel

    Parameters
    ----------
    img1 : ScalarImage
        an image with segmentation label defined on each voxel
    img2 : ScalarImage
        another image defining segmentation label
    window_length : int
        length of patch side

    Returns
    -------
    dissimilarity : ndarray
        dissimilarity ratio of two labeled images
    """
    data1 = img1.get_data()
    data2 = img2.get_data()
    ndim = img1.get_ndim()
    window_size = window_length ** ndim

    label_difference = np.abs(data1 - data2)
    label_difference[np.where(label_difference > 0)] = 1.

    dissimilarity = uniformFilter(label_difference, window_length) / window_size

    return dissimilarity

def test():
    from os.path import expanduser, join
    from images import ScalarImage

    home = expanduser('~')
    dname = join(home, 'registration/img/IBSR/from02to01')
    fixed_img_file = join(dname, 'fixed_image/IBSR_01_segTRI_ana.nii.gz')
    moving_img_file = join(dname, 'LDDMM/IBSR_02_segTRI_warped.nii.gz')

    fixed_img = ScalarImage(fixed_img_file)
    moving_img = ScalarImage(moving_img_file)

    dissimilarity = segmentation_dissimilarity(fixed_img, moving_img, 5)

    dissimilarity_img = ScalarImage(data=dissimilarity, affine=fixed_img.get_affine())

    dissimilarity_img.save(join(dname, 'LDDMM/dissimilarity.nii.gz'))

if __name__ == '__main__':
    test()