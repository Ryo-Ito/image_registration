import numpy as np
from image import ScalarImage
from imageprocessing import uniform_filter


def local_label_dissimilarity(img1, img2, window_length):
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

    dissimilarity = uniform_filter(
        label_difference, window_length) / window_size

    return dissimilarity


def label_dissimilarity(label1, label2):
    data1 = label1.get_data()
    data2 = label2.get_data()
    similarity = (data1 == data2).astype(np.int)
    difference = 1 - similarity
    return np.sum(difference)


def test():
    from os.path import expanduser, join
    from images import ScalarImage

    home = expanduser('~')
    dname = join(home, 'registration/img/IBSR/from02to01')
    fixed_img_file = join(dname, 'fixed_image/IBSR_01_segTRI_ana.nii.gz')
    moving_img_file = join(dname, 'LDDMM/IBSR_02_segTRI_warped.nii.gz')

    fixed_img = ScalarImage(fixed_img_file)
    moving_img = ScalarImage(moving_img_file)

    dissimilarity = local_label_dissimilarity(fixed_img, moving_img, 5)

    dissimilarity_img = ScalarImage(data=dissimilarity,
                                    affine=fixed_img.get_affine())

    dissimilarity_img.save(join(dname, 'LDDMM/dissimilarity.nii.gz'))


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='calculate dissimilarity',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--labels',
                        type=str,
                        nargs=2,
                        action='store',
                        help='two label images')

    args = parser.parse_args()
    label1 = ScalarImage(args.labels[0])
    label2 = ScalarImage(args.labels[1])
    print label_dissimilarity(label1, label2)

if __name__ == '__main__':
    main()
