import nibabel as nib


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
        nib.save(nib.Nifti1Image(self.data, self.affine), filename)
        print "saved image: %s" % filename
