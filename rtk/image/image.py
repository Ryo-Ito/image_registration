import nibabel as nib
import numpy as np
from ..pv_wrapper import save as savevtk


class Image(object):

    def get_data(self):
        return self.data

    def get_affine(self):
        return self.affine

    def get_shape(self):
        return self.shape

    def get_ndim(self):
        return self.ndim

    def save(self, filename):

        if '.vtk' in filename:
            savevtk(self.data, filename)

        elif '.nii.gz' in filename or '.nii' in filename:
            nib.save(nib.Nifti1Image(np.expand_dims(self.data, axis=0), self.affine), filename)

        else:
            print('Please input a valid filetype')
            raise ValueError 

        print(f"saved image: {filename}")
