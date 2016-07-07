import numpy as np
from registration import Registration
from rtk.deformation import Deformation, VectorFields
from rtk.image import SequentialScalarImages


class LDDMM(Registration):

    def set_vector_fields(self, shape):
        self.vector_fields = VectorFields(self.n_step, shape)

    def update(self, fixed, moving):
        for i in xrange(self.n_step + 1):
            j = - i - 1
            momentum = (self.derivative(fixed[j], moving[i])
                        * self.deformation.backward_jacobian_determinants[j]
                        / self.penalty)
            grad = 2 * self.vector_fields[i] + self.regularizer(momentum)
            delta = self.learning_rate * grad
            self.vector_fields.delta_vector_fields[i] = np.copy(delta)
        self.vector_fields.update()
        self.integrate_vector_fields()

    def update_parallel(self, fixed, moving):
        raise ValueError

    def integrate_vector_fields(self):
        v = 0.5 * (self.vector_fields[:-1] + self.vector_fields[1:])
        forward_mapping_before = np.copy(self.deformation.forward_mappings[-1])
        self.deformation.update_mappings(v)
        forward_mapping_after = np.copy(self.deformation.forward_mappings[-1])
        self.delta_phi = np.max(
            np.abs(forward_mapping_after - forward_mapping_before))

    def check_injectivity(self):
        self.min_unit = np.min(
            self.deformation.forward_jacobian_determinants[-1])
        if self.min_unit < self.unit_threshold:
            self.vector_fields.back_to_previous()
            self.integrate_vector_fields()
            print "reached limit of jacobian determinant %f" % self.unit_threshold
        return self.min_unit > self.unit_threshold

    def execute(self):
        warp = Deformation(shape=self.shape)

        for n_iter, resolution, sigma in zip(self.n_iters,
                                             self.resolutions,
                                             self.smoothing_sigmas):
            print "======================================="
            print "resolution", resolution
            warped_moving = self.moving.apply_transform(warp)

            moving = warped_moving.change_resolution(resolution, sigma)
            fixed = self.fixed.change_resolution(resolution, sigma)
            shape = moving.get_shape()
            self.deformation.set_shape(shape)
            self.set_vector_fields(shape)

            grid = self.optimization(fixed, moving, n_iter, resolution)
            warp += Deformation(grid=grid)

        return warp

    def optimization(self, fixed, moving, max_iter, resolution):
        moving_images = SequentialScalarImages(moving, self.n_step + 1)
        fixed_images = SequentialScalarImages(fixed, self.n_step + 1)

        print "iteration   0, Energy %f" % (
            self.cost_function(fixed.data, moving.data))

        for i in xrange(max_iter):
            if self.parallel:
                self.update_parallel(fixed_images, moving_images)
            else:
                self.update(fixed_images, moving_images)

            if not self.check_injectivity():
                break

            if self.parallel:
                moving_images.apply_transforms_parallel(
                    self.deformation.forward_mappings, self.n_jobs)
                fixed_images.apply_transforms_parallel(
                    self.deformation.backward_mappings, self.n_jobs)
            else:
                moving_images.apply_transforms(
                    self.deformation.forward_mappings)
                fixed_images.apply_transforms(
                    self.deformation.backward_mappings)

            print "iteration%4d, Energy %f" % (
                i + 1,
                self.cost_function(fixed_images[0], moving_images[-1]))
            print 14 * ' ', "minimum unit", self.min_unit
            print 14 * ' ', "delta phi", self.delta_phi
            print 14 * ' ', "maximum delta phi", self.delta_phi * (max_iter - i)
            if self.delta_phi * (max_iter - i) < self.delta_phi_threshold / resolution:
                print "|maximum norm of displacement| x iteration < %f voxel" % (self.delta_phi_threshold / resolution)
                break

        return self.zoom_grid(self.deformation.forward_mappings[-1],
                              resolution)


if __name__ == '__main__':
    a = LDDMM(similarity='ssd')
