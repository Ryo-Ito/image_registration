import nibabel as nib
import numpy as np
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from skimage.transform import warp
from joblib import Parallel, delayed


class ScalarImage(object):
    "a class containing information of scalar image"

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
        zoom in or out this image and change the shape of the image to the input shape.

        Parameters
        ----------
        shape : tuple
            shape of the image after zooming

        Returns
        -------
        img : ScalarImage
            zoomed scalar image
        """
        blurred_data = gaussian_filter(self.data, sigma)
        ratio = [1 / float(resolution)] * self.ndim
        # if shrink_factor is not None:
        #     ratio = [1 / float(shrink_factor)] * self.ndim
        # ratio = [len_new / float(len_now) for len_new, len_now in zip(shape, self.shape)]
        data = zoom(blurred_data, ratio, order=order)
        img = ScalarImage(data=data)
        return img

    def change_scale(self, maximum_value):
        data = maximum_value * self.data / np.max(self.data)
        img = ScalarImage(data=data, affine=self.affine)
        return img

    def get_data(self):
        return np.copy(self.data)

    def get_affine(self):
        return np.copy(self.affine)

    def get_ndim(self):
        return self.ndim

    def get_shape(self):
        return self.shape

    def save(self, filename):
        nib.save(nib.Nifti1Image(self.data, self.affine), filename)
        print "saved image: %s" % filename

    def show(self):
        if self.ndim == 2:
            plt.imshow(self.data, cmap='gray')
            plt.show()
        elif self.ndim == 3:
            shape = np.array(self.shape)
            center = (shape / 2).astype(np.int)
            plt.imshow(self.data[:, :, center[2]], cmap='gray')
            plt.imshow(self.data[:, center[1], :], cmap='gray')
            plt.imshow(self.data[center[0], :, :], cmap='gray')
            plt.show()

    def apply_transform(self, transformation, order=1):
        """
        applying input transformation

        Parameters
        ----------
        transformation : Transformation
            transformation applying to this image
        order : int
            order of interpolation

        Returns
        -------
        warped_img : ScalarImage
            the result of the transformation applied to this image.
        """
        warped_data = warp(self.data, transformation.mapping, order=order)
        warped_img = ScalarImage(data=warped_data, affine=self.affine)
        return warped_img

class SequentialScalarImages(object):
    # a class containing sequential scalar images for diffeomorphic registration
    # contains deformed fixed or moving images by the diffeomorphic mappings

    def __init__(self, img, deformation_step):
        """
        Parameters
        ----------
        data : ndarray
            original fixed or moving image
        """
        self.deformation_step = deformation_step
        self.ndim = img.ndim
        self.shape = img.shape
        self.sequential_data = np.ones((deformation_step + 1,) + self.shape) * img.get_data()

    def __getitem__(self, index):
        return self.sequential_data[index]

    def apply_transforms(self, mappings):
        for i in xrange(self.deformation_step + 1):
            self.sequential_data[i] = warp(self.sequential_data[0], mappings[i])

    def apply_transforms_parallel(self, mappings, n_jobs=-1):
        self.sequential_data = np.asarray(Parallel(n_jobs=n_jobs, backend='threading')(delayed(warp)(self.sequential_data[0], mappings[i]) for i in range(self.deformation_step + 1)))