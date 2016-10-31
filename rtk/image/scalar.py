import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import zoom
from skimage.transform import warp
import nibabel as nib
from image import Image


class ScalarImage(Image):

    def __init__(self, filename=None, data=None, affine=None):
        if filename:
            img = nib.load(filename)
            data = img.get_data()
            data = np.squeeze(data)
            data = data.astype(np.float).astype(np.float)
            self.data = np.copy(data, order='C')
            self.ndim = self.data.ndim
            self.shape = self.data.shape
            self.affine = img.get_affine()
        elif data is not None:
            self.data = np.squeeze(data)
            self.ndim = self.data.ndim
            self.shape = self.data.shape
            if affine is None:
                self.affine = np.identity(4)
            else:
                self.affine = affine

    def change_resolution(self, resolution, sigma, order=1):
        """
        change image's resolution

        Parameters
        ----------
        resolution : int
            how much to magnify
            if resolution is 2, the shape of the image will be halved
        sigma : float
            standard deviation of gaussian filter for smoothing
        order : int
            order of interpolation

        Returns
        -------
        img : ScalarImage
            zoomed scalar image
        """
        if resolution != 1:
            blurred_data = gaussian_filter(self.data, sigma)
            ratio = [1 / float(resolution)] * self.ndim
            data = zoom(blurred_data, ratio, order=order)
        elif resolution == 1:
            data = gaussian_filter(self.data, sigma)
        img = ScalarImage(data=data)
        return img

    def change_scale(self, maximum_value):
        data = maximum_value * self.data / np.max(self.data)
        img = ScalarImage(data=data, affine=self.affine)

        return img

    def apply_transform(self, deformation, order=1):
        """
        apply transform and warps image

        Parameters
        ----------
        deformation : Deformation
            deformation applying to this image
        order : int
            order of interpolation

        Returns
        -------
        warped_img : ScalarImage
            the result of the deformation applied to this image.
        """
        warped_data = warp(self.data, deformation.grid, order=order)
        warped_img = ScalarImage(data=warped_data, affine=self.affine)
        return warped_img

    def show(self, show_axis=False):
        import matplotlib.pyplot as plt
        if self.ndim == 2:
            if show_axis is False:
                plt.axis('off')
            plt.imshow(self.data, cmap='gray')
            plt.show()
