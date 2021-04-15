import numpy as np
from joblib import Parallel, delayed
from .registration import Registration
from ..grid import (identity_mapping, jacobian_matrix,
                      Deformation, VectorFields)
from ..image import SequentialScalarImages


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
        if self.n_jobs != 1:
            self.update_parallel(fixed, moving)
        else:
            self.update_sequential(fixed, moving)

    def update_sequential(self, fixed, moving):
        for i in xrange(self.n_step_half + 1):
            j = -i - 1

            # moving -> midpoint
            momentum = (self.similarity.derivative(fixed[j], moving[i])
                        * self.deformation.backward_dets[j])
            grad = self.learning_rate * (
                2*self.forward_vector_fields[i] + self.regularizer(momentum)
            )
            self.forward_vector_fields.delta_vector_fields[i] = np.copy(grad)

            # midpoint <- fixed
            momentum = (self.similarity.derivative(moving[j], fixed[i])
                        * self.deformation.forward_dets[j])
            grad = self.learning_rate * (
                2*self.backward_vector_fields[i] + self.regularizer(momentum)
            )
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
                    self.similarity,
                    fixed[-i - 1],
                    moving[i],
                    self.deformation.backward_dets[-i - 1],
                    self.forward_vector_fields[i],
                    self.regularizer,
                    self.learning_rate)
                for i in xrange(self.n_step_half + 1)
            )
        )
        self.backward_vector_fields.delta_vector_fields = np.array(
            Parallel(self.n_jobs)(
                delayed(derivative)(
                    self.similarity,
                    moving[-i - 1],
                    fixed[i],
                    self.deformation.forward_dets[-i - 1],
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
        v_forward = 0.5 * (self.forward_vector_fields[:-1]
                           + self.forward_vector_fields[1:])
        v_backward = 0.5 * (self.backward_vector_fields[:-1]
                            + self.backward_vector_fields[1:])
        v = np.vstack((v_forward, -v_backward))

        forward_mapping_before = np.copy(
            self.deformation.forward_mappings[self.n_step_half])
        backward_mapping_before = np.copy(
            self.deformation.backward_mappings[self.n_step_half])

        self.deformation.update_mappings(v)

        forward_mapping = self.deformation.forward_mappings[self.n_step_half]
        backward_mapping = self.deformation.backward_mappings[self.n_step_half]

        delta_phi_forward = np.max(np.abs(
            forward_mapping - forward_mapping_before))
        delta_phi_backward = np.max(np.abs(
            backward_mapping - backward_mapping_before))
        self.delta_phi = max(delta_phi_forward, delta_phi_backward)

    def execute(self):
        forward_warp = Deformation(shape=self.shape)
        backward_warp = Deformation(shape=self.shape)
        forward_warp_inv = Deformation(shape=self.shape)
        backward_warp_inv = Deformation(shape=self.shape)

        for n_iter, resolution, sigma in zip(self.n_iters,
                                             self.resolutions,
                                             self.smoothing_sigmas):
            print("=======================================")
            print("resolution", resolution)
            warped_moving = self.moving.apply_transform(forward_warp)
            warped_fixed = self.fixed.apply_transform(backward_warp)
            moving = warped_moving.change_resolution(resolution, sigma)
            fixed = warped_fixed.change_resolution(resolution, sigma)
            shape = moving.get_shape()
            self.deformation.set_shape(shape)
            self.set_vector_fields(shape)

            mappings = self.optimization(fixed, moving, n_iter, resolution)
            forward_warp += mappings[0]
            backward_warp += mappings[1]
            forward_warp_inv = mappings[2] + forward_warp_inv
            backward_warp_inv = mappings[3] + backward_warp_inv

        return forward_warp + backward_warp_inv

    def optimization(self, fixed, moving, max_iter, resolution):
        fixed_images = SequentialScalarImages(fixed, self.n_step + 1)
        moving_images = SequentialScalarImages(moving, self.n_step + 1)

        print(f"iteration   0, Energy {self.similarity.cost(fixed.data, moving.data)}")

        for i in xrange(max_iter):
            self.update(fixed_images, moving_images)

            if not self.check_injectivity():
                break

            if self.n_jobs != 1:
                moving_images.apply_transforms_parallel(
                    self.deformation.forward_mappings, self.n_jobs)
                fixed_images.apply_transforms_parallel(
                    self.deformation.backward_mappings, self.n_jobs)
            else:
                moving_images.apply_transforms(
                    self.deformation.forward_mappings)
                fixed_images.apply_transforms(
                    self.deformation.backward_mappings)

            max_delta_phi = self.delta_phi * (max_iter - i)
            print(f"iteration {i + 1}, Energy {self.similarity.cost(fixed_images[self.n_step_half], moving_images[self.n_step_half])}")
            print(14 * ' ', "minimum unit", self.min_unit)
            print(14 * ' ', "delta phi", self.delta_phi)
            print(14 * ' ', "maximum delta phi", max_delta_phi)
            if max_delta_phi < self.delta_phi_threshold / resolution:
                print("|L_inf norm of displacement| x iter < {self.delta_phi_threshold / resolution}")
                break

        forward_mapping = self.zoom_grid(
            self.deformation.forward_mappings[self.n_step_half], resolution)
        backward_mapping = self.zoom_grid(
            self.deformation.backward_mappings[self.n_step_half], resolution)
        forward_mapping_inverse = self.zoom_grid(
            self.get_forward_mapping_inverse(), resolution)
        backward_mapping_inverse = self.zoom_grid(
            self.get_backward_mapping_inverse(), resolution)

        return (Deformation(grid=forward_mapping),
                Deformation(grid=backward_mapping),
                Deformation(grid=forward_mapping_inverse),
                Deformation(grid=backward_mapping_inverse))


def derivative(similarity,
               fixed,
               moving,
               Dphi,
               vector_field,
               regularizer,
               learning_rate):
    momentum = similarity.derivative(fixed, moving) * Dphi
    grad = 2 * vector_field + regularizer(momentum)
    return learning_rate * grad
