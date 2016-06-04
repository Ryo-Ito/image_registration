import numpy as np
import nibabel as nib

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