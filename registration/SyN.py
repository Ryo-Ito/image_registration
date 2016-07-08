import numpy as np
from joblib import Parallel, delayed
from registration import Registration
from rtk import identity_mapping, jacobian_matrix
from rtk.deformation import Deformation, VectorFields
from rtk.image import SequentialScalarImages


class SyN(Registration):

    def set_vector_fields(self, shape):
        assert self.n_step % 2 == 0
        self.n_step_half = self.n_step / 2

        # moving -> midpoint
        self.forward_vector_fields = VectorFields(self.n_step_half, shape)
        # midpoint <- fixed
        self.backward_vector_fields = VectorFields(self.n_step_half, shape)

    def get_forward_mapping_inverse(self):
        inverse_mapping = identity_mapping(self.forward_vector_fields.shape)
        v = - 0.5 * (self.forward_vector_fields[-2::-1]
                     + self.forward_vector_fields[:0:-1])
        for i in xrange(self.n_step_half):
            inverse_mapping = inverse_mapping - np.einsum(
                'ij...,j...->i...',
                jacobian_matrix(inverse_mapping),
                v[i]) / self.n_step

        return inverse_mapping

    def get_backward_mapping_inverse(self):
        inverse_mapping = identity_mapping(self.backward_vector_fields.shape)
        v = - 0.5 * (self.backward_vector_fields[-2::-1]
                     + self.backward_vector_fields[:0:-1])
        for i in xrange(self.n_step_half):
            inverse_mapping = inverse_mapping - np.einsum(
                'ij...,j...->i...',
                jacobian_matrix(inverse_mapping),
                v[i]) / self.n_step

        return inverse_mapping

    def update(self, fixed, moving):
        for i in xrange(self.n_step_half + 1):
            j = -i - 1

            # moving -> midpoint
            momentum = (self.derivative(fixed[j], moving[i])
                        * self.deformation.backward_dets[j]
                        / self.penalty)
            grad = 2 * self.forward_vector_fields[i] + self.regularizer(momentum)
            grad *= self.learning_rate
            self.forward_vector_fields.delta_vector_fields[i] = np.copy(grad)

            # midpoint <- fixed
            momentum = (self.derivative(moving[j], fixed[i])
                        * self.deformation.forward_dets[j]
                        / self.penalty)
            grad = 2 * self.backward_vector_fields[i] + self.regularizer(momentum)
            grad *= self.learning_rate
            self.backward_vector_fields.delta_vector_fields[i] = np.copy(grad)

        self.forward_vector_fields.update()
        self.backward_vector_fields.update()

        self.integrate_vector_fields()

    def update_parallel(self, fixed, moving):
        if hasattr(self.regularizer, "set_operator"):
            self.regularizer.set_operator(shape=fixed.shape)
        self.forward_vector_fields.delta_vector_fields = np.array(
            Parallel(self.n_jobs)(
                delayed(derivative)(
                    self.derivative,
                    fixed[-i - 1],
                    moving[i],
                    self.deformation.backward_dets[-i - 1],
                    self.penalty,
                    self.forward_vector_fields[i],
                    self.regularizer,
                    self.learning_rate)
                for i in xrange(self.n_step_half + 1)
                )
            )
        self.backward_vector_fields.delta_vector_fields = np.array(
            Parallel(self.n_jobs)(
                delayed(derivative)(
                    self.derivative,
                    moving[-i - 1],
                    fixed[i],
                    self.deformation.forward_dets[-i - 1],
                    self.penalty,
                    self.backward_vector_fields[i],
                    self.regularizer,
                    self.learning_rate)
                for i in xrange(self.n_step_half + 1)
                )
            )

        self.forward_vector_fields.update()
        self.backward_vector_fields.update()

        self.integrate_vector_fields()

    def integrate_vector_fields(self):
        v_forward = 0.5 * (self.forward_vector_fields[:-1] + self.forward_vector_fields[1:])
        v_backward = 0.5 * (self.backward_vector_fields[:-1] + self.backward_vector_fields[1:])
        v = np.vstack((v_forward, -v_backward))

        forward_mapping_before = np.copy(self.deformation.forward_mappings[self.n_step_half])
        backward_mapping_before = np.copy(self.deformation.backward_mappings[self.n_step_half])

        self.deformation.update_mappings(v)

        forward_mapping_after = self.deformation.forward_mappings[self.n_step_half]
        backward_mapping_after = self.deformation.backward_mappings[self.n_step_half]

        delta_phi_forward = np.max(np.abs(forward_mapping_after - forward_mapping_before))
        delta_phi_backward = np.max(np.abs(backward_mapping_after - backward_mapping_before))
        self.delta_phi = max(delta_phi_forward, delta_phi_backward)

    def execute(self):
        forward_warp = Deformation(shape=self.shape)
        backward_warp = Deformation(shape=self.shape)
        forward_warp_inverse = Deformation(shape=self.shape)
        backward_warp_inverse = Deformation(shape=self.shape)

        for n_iter, resolution, sigma in zip(self.n_iters,
                                             self.resolutions,
                                             self.smoothing_sigmas):
            print "======================================="
            print "resolution", resolution
            warped_moving = self.moving.apply_transform(forward_warp)
            warped_fixed = self.fixed.apply_transform(backward_warp)
            moving = warped_moving.change_resolution(resolution, sigma)
            fixed = warped_fixed.change_resolution(resolution, sigma)
            shape = moving.get_shape()
            self.deformation.set_shape(shape)
            self.set_vector_fields(shape)

            mappings = self.optimization(fixed, moving, n_iter, resolution)
            forward_warp += Deformation(grid=mappings[0])
            backward_warp += Deformation(grid=mappings[1])
            forward_warp_inverse = Deformation(grid=mappings[2]) + forward_warp_inverse
            backward_warp_inverse = Deformation(grid=mappings[3]) + backward_warp_inverse

        return forward_warp + backward_warp_inverse

    def optimization(self, fixed, moving, max_iter, resolution):
        fixed_images = SequentialScalarImages(fixed, self.n_step + 1)
        moving_images = SequentialScalarImages(moving, self.n_step + 1)

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
                self.cost_function(fixed_images[self.n_step_half],
                                   moving_images[self.n_step_half])
                )
            print 14 * ' ', "minimum unit", self.min_unit
            print 14 * ' ', "delta phi", self.delta_phi
            print 14 * ' ', "maximum delta phi", self.delta_phi * (max_iter - i)
            if self.delta_phi * (max_iter - i) < self.delta_phi_threshold / resolution:
                print "|maximum norm of displacement| x iteration < %f voxel" % (self.delta_phi_threshold / resolution)
                break

        forward_mapping = self.zoom_grid(self.deformation.forward_mappings[self.n_step_half], resolution)
        backward_mapping = self.zoom_grid(self.deformation.backward_mappings[self.n_step_half], resolution)
        forward_mapping_inverse = self.zoom_grid(self.get_forward_mapping_inverse(), resolution)
        backward_mapping_inverse = self.zoom_grid(self.get_backward_mapping_inverse(), resolution)

        return (forward_mapping,
                backward_mapping,
                forward_mapping_inverse,
                backward_mapping_inverse)


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
