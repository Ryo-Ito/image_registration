import numpy as np
from images import SequentialScalarImages
import deformations
from imageprocessing import interpolate_mapping

class Registration(object):

    def __init__(self, energy_threshold=0., unit_threshold=0., learning_rate=0.1):
        self.energy_threshold = energy_threshold
        self.unit_threshold = unit_threshold
        self.energy = 0.
        self.learning_rate = learning_rate
        print "threshold value for update ratio of cost function", energy_threshold
        print "threshold value for jacobian determinant of mapping function", unit_threshold
        print "leaning rate", learning_rate

    def set_deformation(self, deformation):
        self.deformation = deformation
        self.deformation_step = deformation.deformation_step

        print "dimensionality", deformation.ndim
        print "number of deformation step", deformation.deformation_step
        print "interval of time", deformation.time_interval
        print "penalty to vector field", deformation.penalty
        print "penalty(alpha) = ", deformation.alpha
        print "penalty(gamma) = ", deformation.gamma
        print "penalty(beta) = ", deformation.beta
        print "similarity metric", deformation.similarity_metric

    def set_maximum_iterations(self, maximum_iterations):
        try:
            self.maximum_iterations = tuple(maximum_iterations)
        except:
            self.maximum_iterations = (maximum_iterations,)

        self.multi_resolutions = len(self.maximum_iterations)

        print "maximum iterations", self.maximum_iterations

    def set_resolution_level(self, resolution_level):
        try:
            self.resolution_level = tuple(resolution_level)
        except:
            self.resolution_level = (resolution_level,)

        while len(self.resolution_level) < self.multi_resolutions:
            self.resolution_level += (self.resolution_level[-1],)

        assert len(self.resolution_level) == self.multi_resolutions

        print "resolutions", self.resolution_level

    def set_smoothing_sigma(self, smoothing_sigma):
        try:
            self.smoothing_sigma = tuple(smoothing_sigma)
        except:
            self.smoothing_sigma = (smoothing_sigma,)

        while len(self.smoothing_sigma) < self.multi_resolutions:
            self.smoothing_sigma += (self.smoothing_sigma[-1],)

        print "smoothing sigma", self.smoothing_sigma

    def set_images(self, fixed_img, moving_img):
        if not(fixed_img.ndim == moving_img.ndim):
            print "dimensionality of fixed image", fixed_img.ndim
            print "dimensionality of moving image", moving_img.ndim
            raise ValueError("the dimension of the both images have to be the same.")
        if not(fixed_img.shape == moving_img.shape):
            print "shape of fixed image", fixed_img.shape
            print "shape of moving image", moving_img.shape
            raise ValueError("the shape of the two images are different.")

        self.fixed_img = fixed_img.change_scale(255)
        self.moving_img = moving_img.change_scale(255)

        self.ndim = fixed_img.get_ndim()
        self.shape = fixed_img.get_shape()

    def zoom_mapping(self, mapping, resolution):
        shape = mapping.shape[1:]
        if resolution != 1:
            interpolated_mapping = np.zeros((self.ndim,) + self.shape)
            for i in xrange(self.ndim):
                interpolated_mapping[i] = interpolate_mapping(mapping[i], np.array(self.shape, dtype=np.int32)) * (self.shape[i] - 1) / (shape[i] - 1)
        else:
            interpolated_mapping = np.copy(mapping)

        return interpolated_mapping

    def check_one_to_one(self):
        min_unit = self.deformation.get_minimum_unit()
        if min_unit < self.unit_threshold:
            self.deformation.back_to_previous_deformation()
            print "reached limit of jacobian determinant"
            # self.learning_rate *= 0.1
        else:
            print "minimum unit", min_unit
        return min_unit > self.unit_threshold

    def check_energy_update(self, I, J):
        energy = self.deformation.get_energy(I, J)
        finish_update = 2 * abs(self.energy - energy) / abs(self.energy + energy) < self.energy_threshold
        self.energy = energy
        return finish_update

    def execute(self):
        if isinstance(self.deformation, deformations.SyN):
            return self.two_way_registration()
        else:
            return self.one_way_registration()

    def one_way_registration(self):
        transform = deformations.Transformation(shape=self.shape)
        for max_iter, resolution, sigma in zip(self.maximum_iterations, self.resolution_level, self.smoothing_sigma):
            print "============================"
            print "resolution", resolution
            warped_moving_img = self.moving_img.apply_transform(transform)

            moving_img = warped_moving_img.change_resolution(resolution, sigma)
            fixed_img = self.fixed_img.change_resolution(resolution, sigma)
            shape = moving_img.get_shape()

            self.deformation.set_grid(shape, resolution)

            deformed_moving_images = SequentialScalarImages(moving_img, self.deformation_step)
            deformed_fixed_images = SequentialScalarImages(fixed_img, self.deformation_step)

            self.energy = self.deformation.get_energy(deformed_fixed_images[0], deformed_moving_images[-1])

            print "iteration   0, Energy %f" % (self.energy)

            for i in xrange(max_iter):
                self.deformation.update_new(deformed_fixed_images, deformed_moving_images, self.learning_rate)

                if not self.check_one_to_one():
                    break

                deformed_moving_images.apply_transforms(self.deformation.forward_mappings)
                deformed_fixed_images.apply_transforms(self.deformation.backward_mappings)

                finish_update = self.check_energy_update(deformed_fixed_images[0], deformed_moving_images[-1])
                print "iteration%4d, Energy %f" % (i + 1, self.energy)
                if finish_update:
                    print "|update ratio of cost function| < threshold value = %f" % self.energy_threshold
                    break

            mapping = self.zoom_mapping(self.deformation.get_forward_mapping(), resolution)

            transform.add_pullback_mapping(mapping)

        return transform

    def two_way_registration(self):
        forward_transform = deformations.Transformation(shape=self.shape)
        forward_transform_inverse = deformations.Transformation(shape=self.shape)
        backward_transform = deformations.Transformation(shape=self.shape)
        backward_transform_inverse = deformations.Transformation(shape=self.shape)
        index_half = int(self.deformation_step / 2)

        for max_iter, resolution, sigma in zip(self.maximum_iterations, self.resolution_level, self.smoothing_sigma):
            print "============================"
            print "resolution", resolution
            moving_img_warped = self.moving_img.apply_transform(forward_transform)
            fixed_img_warped = self.fixed_img.apply_transform(backward_transform)

            moving_img = moving_img_warped.change_resolution(resolution, sigma)
            fixed_img = fixed_img_warped.change_resolution(resolution, sigma)
            shape = moving_img.get_shape()

            self.deformation.set_grid(shape, resolution)

            deformed_moving_images = SequentialScalarImages(moving_img, self.deformation_step)
            deformed_fixed_images = SequentialScalarImages(fixed_img, self.deformation_step)

            self.energy = self.deformation.get_energy(moving_img.data, fixed_img.data)

            print "iteration   0, Energy %f" % (self.energy)

            for i in xrange(max_iter):
                self.deformation.update_new(deformed_fixed_images, deformed_moving_images, self.learning_rate)

                if not self.check_one_to_one():
                    break

                deformed_moving_images.apply_transforms(self.deformation.forward_mappings)
                deformed_fixed_images.apply_transforms(self.deformation.backward_mappings)

                finish_update = self.check_energy_update(deformed_fixed_images[index_half], deformed_moving_images[index_half])
                print "iteration%4d, Energy %f" % (i + 1, self.energy)
                if finish_update:
                    print "|delta E| < threshold value = %f" % self.energy_threshold
                    break

            forward_mapping = self.zoom_mapping(self.deformation.get_forward_mapping(), resolution)
            forward_mapping_inverse = self.zoom_mapping(self.deformation.get_forward_mapping_inverse(), resolution)
            backward_mapping = self.zoom_mapping(self.deformation.get_backward_mapping(), resolution)
            backward_mapping_inverse = self.zoom_mapping(self.deformation.get_backward_mapping_inverse(), resolution)

            forward_transform.add_pullback_mapping(forward_mapping)
            forward_transform_inverse.add_pushforward_mapping(forward_mapping_inverse)
            backward_transform.add_pullback_mapping(backward_mapping)
            backward_transform_inverse.add_pushforward_mapping(backward_mapping_inverse)

        # forward_transform.add_pullback_mapping(backward_transform.get_inverse_transform().mapping)
        # backward_transform.add_pullback_mapping(forward_transform.get_inverse_transform().mapping)
        forward_transform.add_pullback_mapping(backward_transform_inverse.get_mapping())
        backward_transform.add_pullback_mapping(forward_transform_inverse.get_mapping())

        return forward_transform, backward_transform

def main():
    import argparse
    parser = argparse.ArgumentParser(description='estimating transformation from a moving image to a template image')

    parser.add_argument('--dimension', '-d', type=int, help='dimension of the space: 2 or 3')
    parser.add_argument('--metric', '-m', default='cc', type=str, help='metric to calculate image difference or similarity: ssd, cc')
    parser.add_argument('--transformation', default='SyN', type=str, help='transformation type: LDDMM, SyN')
    parser.add_argument('--target', '-t', type=str, help='fixed image file')
    parser.add_argument('--input', '-i', type=str, help='moving image file')
    parser.add_argument('--output', '-o', type=str, help='output file without extension')

    args = parser.parse_args()

if __name__ == '__main__':
    main()