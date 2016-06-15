import numpy as np
from images import ScalarImage, SequentialScalarImages
from deformations import Transformation, LDDMM, SyN
from imageprocessing import interpolate_mapping

class Registration(object):

    def __init__(self, energy_threshold=0., unit_threshold=0., learning_rate=0.1):
        self.energy_threshold = energy_threshold
        self.unit_threshold = unit_threshold
        self.energy = 0.
        self.learning_rate = learning_rate
        print "threshold value for update ratio of cost function", energy_threshold
        print "threshold value for jacobian determinant of mapping function", unit_threshold
        print "learning rate", learning_rate

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
        self.min_unit = self.deformation.get_minimum_unit()
        if self.min_unit < self.unit_threshold:
            self.deformation.back_to_previous_deformation()
            print "reached limit of jacobian determinant"
            # self.learning_rate *= 0.1
        return self.min_unit > self.unit_threshold

    def check_energy_update(self, I, J):
        energy = self.deformation.get_energy(I, J)
        finish_update = 2 * abs(self.energy - energy) / abs(self.energy + energy) < self.energy_threshold
        self.energy = energy
        return finish_update

    def check_convergence(self, max_iter):
        """
        returns whether it reached convergence or not

        Parameters
        ----------
        max_iter : int
            number of maximum iteration

        Returns
        -------
        finish_update : bool
            finished updates if True else not.
        """
        v_norm = self.deformation.delta_norm()
        return (v_norm * self.deformation_step * max_iter < 1.)

    def execute(self):
        if isinstance(self.deformation, SyN):
            return self.two_way_registration()
        else:
            return self.one_way_registration()

    def one_way_registration(self):
        transform = Transformation(shape=self.shape)
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
                self.deformation.update(deformed_fixed_images, deformed_moving_images, self.learning_rate)

                if not self.check_one_to_one():
                    break

                deformed_moving_images.apply_transforms(self.deformation.forward_mappings)
                deformed_fixed_images.apply_transforms(self.deformation.backward_mappings)

                print "iteration%4d, Energy %f" % (i + 1, self.deformation.get_energy(deformed_fixed_images[0], deformed_moving_images[-1]))
                v_norm = self.deformation.delta_norm()
                print 14 * ' ', "minimum unit", self.min_unit
                print 14 * ' ', "v_norm", v_norm
                if v_norm * self.deformation_step * (max_iter - i) < 1:
                    print "|maximum norm of displacement| x iteration < 1 voxel"
                    break

            mapping = self.zoom_mapping(self.deformation.get_forward_mapping(), resolution)

            transform.add_pullback_mapping(mapping)

        return transform

    def two_way_registration(self):
        forward_transform = Transformation(shape=self.shape)
        forward_transform_inverse = Transformation(shape=self.shape)
        backward_transform = Transformation(shape=self.shape)
        backward_transform_inverse = Transformation(shape=self.shape)
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
                self.deformation.update(deformed_fixed_images, deformed_moving_images, self.learning_rate)

                if not self.check_one_to_one():
                    break

                deformed_moving_images.apply_transforms(self.deformation.forward_mappings)
                deformed_fixed_images.apply_transforms(self.deformation.backward_mappings)

                print "iteration%4d, Energy %f" % (i + 1, self.deformation.get_energy(deformed_fixed_images[0], deformed_moving_images[-1]))
                v_norm = self.deformation.delta_norm()
                print 14 * ' ', "minimum unit", self.min_unit
                print 14 * ' ', "v_norm", v_norm
                if v_norm * self.deformation_step * (max_iter - i) < 1:
                    print "|maximum norm of displacement| x iteration < 1 voxel"
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

def estimate_transform_LDDMM(args):
    fixed_img = ScalarImage(args.fixed)
    moving_img = ScalarImage(args.moving)

    deformation = LDDMM(ndim=fixed_img.ndim, deformation_step=args.deformation_step, penalty=args.penalty, time_interval=args.time_interval)
    deformation.set_prior_parameter(alpha=args.alpha, gamma=args.gamma, beta=args.beta)
    deformation.set_similarity_metric(args.similarity_metric, args.window_length)

    reg = Registration(energy_threshold=args.energy_threshold,
                       unit_threshold=args.unit_threshold,
                       learning_rate=args.learning_rate)
    reg.set_deformation(deformation=deformation)
    reg.set_maximum_iterations(maximum_iterations=args.maximum_iterations)
    reg.set_resolution_level(resolution_level=args.resolution_level)
    reg.set_smoothing_sigma(smoothing_sigma=args.smoothing_sigma)
    reg.set_images(fixed_img=fixed_img, moving_img=moving_img)

    transform = reg.execute()

    transform.save(filename=args.output, affine=fixed_img.get_affine())

def estimate_transform_SyN(args):
    fixed_img = ScalarImage(args.fixed)
    moving_img = ScalarImage(args.moving)

    deformation = SyN(ndim=fixed_img.ndim, deformation_step=args.deformation_step, penalty=args.penalty, time_interval=args.time_interval)
    deformation.set_prior_parameter(alpha=args.alpha, gamma=args.gamma, beta=args.beta)
    deformation.set_similarity_metric(args.similarity_metric, args.window_length)

    reg = Registration(energy_threshold=args.energy_threshold,
                       unit_threshold=args.unit_threshold,
                       learning_rate=args.learning_rate)
    reg.set_deformation(deformation=deformation)
    reg.set_maximum_iterations(maximum_iterations=args.maximum_iterations)
    reg.set_resolution_level(resolution_level=args.resolution_level)
    reg.set_smoothing_sigma(smoothing_sigma=args.smoothing_sigma)
    reg.set_images(fixed_img=fixed_img, moving_img=moving_img)

    transform, inverse_transform = reg.execute()

    transform.save(filename=args.output, affine=fixed_img.get_affine())

def main():
    import argparse

    parser = argparse.ArgumentParser(description='estimating transformation from a moivng image to a fixed image', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-m', '--moving',
                        type=str,
                        help='moving image file')
    parser.add_argument('-f', '--fixed',
                        type=str,
                        help='fixed image file')
    parser.add_argument('-s', '--similarity_metric',
                        choices=['cc', 'ssd', 'mc'],
                        default='cc',
                        help='similarity metric to evaluate how similar two images are.\nChoose one of the following similarity metric\n    cc: zero means normalized Cross Correlation\n    ssd: Sum of Squared Difference\n    mc: mahalanobis cosine similarity\nDefault: cc')
    parser.add_argument('--window_length',
                        type=int,
                        help='length of window when calculating cross correlation\nDefault: 5(cc), None(ssd)')
    parser.add_argument('-t', '--transformation',
                        choices=['LDDMM', 'SyN'],
                        default='SyN',
                        type=str,
                        help='transformation type\nDefault: SyN')
    parser.add_argument('-o', '--output',
                        default='output_warp.nii.gz',
                        type=str,
                        help='output transformation file\nDefault: output_warp.nii.gz')
    parser.add_argument('--deformation_step',
                        default=32,
                        type=int,
                        help='number of steps to deform images\nDefault: 32')
    parser.add_argument('--time_interval',
                        default=1.,
                        type=float,
                        help='length of time interval of transformation\nDefault: 1.')
    parser.add_argument('--penalty',
                        type=float,
                        help='penalty coefficient for vector field\nDefault: 0.0001(cc), 1000(ssd)')
    parser.add_argument('--alpha',
                        default=1.,
                        type=float,
                        help='penalty coefficient for convexity of vector field\nDefault: 1.')
    parser.add_argument('--beta',
                        default=2,
                        type=int,
                        help='as this number increases, higher the derivatives be\nDefault: 2.')
    parser.add_argument('--gamma',
                        default=1.,
                        type=float,
                        help='penalty coefficient for norm of vector\nDefault: 1.')
    parser.add_argument('--energy_threshold',
                        default=0.0001,
                        type=float,
                        help='threshold value of update ratio of cost function\nDefault: 0.0001')
    parser.add_argument('--unit_threshold',
                        default=0.2,
                        type=float,
                        help='threshold value of jacobian determinant of mapping function\nDefault: 0.2')
    parser.add_argument('--learning_rate',
                        type=float,
                        help='learning rate of updating estimate of vector field\nDefault: 0.01(cc), 0.1(ssd)')
    parser.add_argument('--maximum_iterations',
                        default=[50, 20, 10],
                        type=int,
                        nargs='*',
                        action='store',
                        help='maximum number of updating estimate of vector field\nDefault: [50, 20, 10]')
    parser.add_argument('--resolution_level',
                        default=[4,2,1],
                        type=int,
                        nargs='*',
                        action='store',
                        help='resolution at each level\nDefault: [4, 2, 1]')
    parser.add_argument('--smoothing_sigma',
                        type=int,
                        nargs='*',
                        action='store',
                        help='values of smoothing sigma at each level\nDefault: [2, 1, 1](cc), [2, 1, 0](ssd)')

    args = parser.parse_args()

    if args.similarity_metric == 'cc':
        if args.window_length is None:
            args.window_length = 5
        if args.penalty is None:
            args.penalty = 0.0001
        if args.learning_rate is None:
            args.learning_rate = 0.01
        if args.smoothing_sigma is None:
            args.smoothing_sigma = [2, 1, 1]
    elif args.similarity_metric == 'ssd':
        if args.penalty is None:
            args.penalty = 1000
        if args.learning_rate is None:
            args.learning_rate = 0.1
        if args.smoothing_sigma is None:
            args.smoothing_sigma = [2, 1, 0]

    if args.transformation == 'LDDMM':
        estimate_transform_LDDMM(args)
    elif args.transformation == 'SyN':
        estimate_transform_SyN(args)

if __name__ == '__main__':
    main()