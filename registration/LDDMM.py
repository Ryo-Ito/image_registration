import numpy as np
from joblib import Parallel, delayed
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
                        * self.deformation.backward_dets[j]
                        / self.penalty)
            grad = 2 * self.vector_fields[i] + self.regularizer(momentum)
            delta = self.learning_rate * grad
            self.vector_fields.delta_vector_fields[i] = np.copy(delta)
        self.vector_fields.update()
        self.integrate_vector_fields()

    def update_parallel(self, fixed, moving):
        if hasattr(self.regularizer, "set_operator"):
            self.regularizer.set_operator(shape=fixed.shape)
        self.vector_fields.delta_vector_fields = np.array(
            Parallel(self.n_jobs)(
                delayed(derivative)(self.derivative,
                                    fixed[-i - 1],
                                    moving[i],
                                    self.deformation.backward_dets[-i - 1],
                                    self.penalty,
                                    self.vector_fields[i],
                                    self.regularizer,
                                    self.learning_rate)
                for i in xrange(self.n_step + 1)
                )
            )
        self.vector_fields.update()
        self.integrate_vector_fields()

    def integrate_vector_fields(self):
        v = 0.5 * (self.vector_fields[:-1] + self.vector_fields[1:])
        forward_mapping_before = np.copy(self.deformation.forward_mappings[-1])
        self.deformation.update_mappings(v)
        forward_mapping_after = np.copy(self.deformation.forward_mappings[-1])
        self.delta_phi = np.max(
            np.abs(forward_mapping_after - forward_mapping_before))

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

    def execute_coarse_to_fine(self):
        vector_fields = VectorFields(self.n_step, shape=self.shape)

        for n_iter, resolution, sigma in zip(self.n_iters,
                                             self.resolutions,
                                             self.smoothing_sigmas):
            print "======================================="
            print "resolution", resolution
            fixed = self.fixed.change_resolution(resolution, sigma)
            moving = self.moving.change_resolution(resolution, sigma)
            shape = fixed.get_shape()
            self.vector_fields = vector_fields.change_resolution(resolution)
            self.deformation.set_shape(shape)
            v = 0.5 * (self.vector_fields[:-1] + self.vector_fields[1:])
            self.deformation.update_mappings(v)

            vector_fields = self.optimization_coarse_to_fine(fixed, moving, n_iter, resolution)

        return self.deformation

    def optimization_coarse_to_fine(self, fixed, moving, max_iter, resolution):
        fixed_images = SequentialScalarImages(fixed, self.n_step)
        moving_images = SequentialScalarImages(moving, self.n_step)
        fixed_images.apply_transforms(self.deformation.backward_mappings)
        moving_images.apply_transforms(self.deformation.forward_mappings)
        print "iteration   0, Energy %f" % (
            self.cost_function(fixed_images[0], moving_images[-1]))

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

        return self.vector_fields.change_resolution(resolution=1. / resolution)


def derivative(func,
               fixed,
               moving,
               Dphi,
               penalty,
               vector_field,
               regularizer,
               learning_rate):
    momentum = (func(fixed, moving)
                * Dphi
                / penalty)
    grad = 2 * vector_field + regularizer(momentum)
    return learning_rate * grad


if __name__ == '__main__':
    a = LDDMM(similarity='ssd')
