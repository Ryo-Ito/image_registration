from skimage.transform import warp
from joblib import Parallel, delayed


class SequentialScalarImages(object):

    def __init__(self, img, deformation_step):
        """
        container for sequentially deforming images

        Parameters
        ----------
        img : Image
            original fixed of moving image
        """
        self.deformation_step = deformation_step
        self.ndim = img.ndim
        self.shape = img.shape

        self.original = img.data

        self.data = [img.data for _ in range(deformation_step + 1)]

    def __getitem__(self, index):
        return self.data[index]

    def apply_transforms(self, mappings):
        self.data = [warp(self.original, mapping) for mapping in mappings]

    def apply_transforms_parallel(self, mappings, n_jobs=-1):
        self.data = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(warp)(self.original, mapping) for mapping in mappings)
