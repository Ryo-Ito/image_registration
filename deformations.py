from __future__ import division
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.ndimage.filters import correlate
from scipy.ndimage.interpolation import map_coordinates
from math import pi
try:
    from pyfftw.interfaces.scipy_fftpack import fftn, ifftn
except ImportError:
    from scipy.fftpack import fftn, ifftn
from imageprocessing import gradient, uniform_filter

np.seterr(all='ignore')


class Transformation(object):
    "a class for transformation of grid or space"

    def __init__(self, filename=None, mapping=None, displacement=None, shape=None):
        if filename is not None:
            t = nib.load(filename)
            self.ndim = t.shape[-1]
            self.shape = t.shape[:-1]
            data = t.get_data()
            transposing_axis = (self.ndim,)
            for i in xrange(self.ndim):
                transposing_axis = transposing_axis + (i,)
            data = np.transpose(data, transposing_axis)
            self.displacement = np.copy(data, order='C').astype(np.float)
        elif mapping is not None:
            self.shape = mapping.shape[1:]
            self.ndim = mapping.shape[0]
            self.mapping = np.copy(mapping).astype(np.float)
        elif displacement is not None:
            self.shape = displacement.shape[1:]
            self.ndim = displacement.shape[0]
            self.displacement = np.copy(displacement).astype(np.float)
        elif shape is not None:
            self.ndim = len(shape)
            self.shape = shape
            self.displacement = np.zeros((self.ndim,) + self.shape)

        self.identity_mapping = self.get_identity_mapping()

        if not hasattr(self, "mapping") and hasattr(self, "displacement"):
            self.mapping = self.displacement + self.identity_mapping

        if not hasattr(self, "displacement") and hasattr(self, "mapping"):
            self.displacement = self.mapping - self.identity_mapping

    def get_identity_mapping(self, shape=None):
        if shape is None:
            shape = self.shape
            ndim = self.ndim
        else:
            ndim = len(shape)

        if ndim == 2:
            return np.mgrid[:shape[0], :shape[1]].astype(np.float)
        elif ndim == 3:
            return np.mgrid[:shape[0], :shape[1], :shape[2]].astype(np.float)

    def add_pullback_mapping(self, mapping):
        """
        adding another mapping function.
        phi_1^{-1}(x) : the original mapping function
        phi_2^{-1}(x) : new mapping function added
        the resulting mapping function will be
        phi_1^{-1}(phi_2^{-1}(x))

        An image I will be transformed like
        phi_2(phi_1(I))
        or
        I(phi_1^{-1}(phi_2^{-1}(x)))

        Parameters
        ----------
        mapping : ndarray
            new mapping function added
        """
        # for i in xrange(self.ndim):
        #     # order of spline interpolation is 3
        #     # points outside the boundary are filled by nearest mode
        #     self.mapping[i] = map_coordinates(self.mapping[i], mapping, mode='nearest')
        self.mapping = warp_grid(self.mapping, mapping)
        self.displacement = self.mapping - self.identity_mapping

    def add_pushforward_mapping(self, mapping):
        """
        adding another mapping function.
        phi_1^{-1}(x) : the original mapping function
        phi_2^{-1}(x) : new mapping funciton added
        the resulting mapping function will be
        phi_2^{-1}(phi_1^{-1}(x))

        Parameters
        ----------
        mapping : ndarray
            mapping fucntion added
        """
        self.mapping = warp_grid(mapping, np.copy(self.mapping))
        self.displacement = self.mapping - self.identity_mapping

    def jacobian_matrix(self, mapping=None):
        """
        calculate jacobian matrix of mapping function

        Returns
        -------
        J : ndarray
            jacobain matrix of mapping function
            eg. 3-d case: (3, 3, len(x), len(y), len(z))
        """
        if mapping is None:
            mapping = np.copy(self.mapping)

        if self.ndim == 2:
            J = np.array([gradient(mapping[0]), gradient(mapping[1])])
        elif self.ndim == 3:
            J = np.array([gradient(mapping[0]), gradient(mapping[1]), gradient(mapping[2])])
        return J

    def jacobian(self):
        """
        calculate jacobian (determinant of jacobian matrix) of mapping function

        Returns
        -------
        D : ndarray
            determinant of jacobian matrix which indicates area or volume of the point.
        """
        J = self.jacobian_matrix()
        if self.ndim == 2:
            return J[0,0] * J[1,1] - J[0,1] * J[1,0]
        elif self.ndim == 3:
            """
            (0,0) (0,1) (0,2)
            (1,0) (1,1) (1,2)
            (2,0) (2,1) (2,2)
            """
            return (J[0,0] * J[1,1] * J[2,2] +
                    J[1,0] * J[2,1] * J[0,2] +
                    J[0,1] * J[1,2] * J[2,0] -
                    J[0,0] * J[1,2] * J[2,1] -
                    J[2,2] * J[1,0] * J[0,1] -
                    J[0,2] * J[1,1] * J[2,0])

    def minimum_unit(self):
        """
        calculate minimum unit(area, volume) of this transformation

        Returns
        -------
        minimum_unit_value : float
            value of minimum unit of this transformation
            if this value is larger than 0, one-to-one mapping is preserved.
        """
        return np.min(self.jacobian())

    def get_inverse_transform(self, initial_mapping=None, eps=1., maximum_iteration=1000, steps=8):
        """
        calculate inverse of this transform

        Parameters
        ----------
        initial_mapping : ndarray
            initial value of inverse mapping

        Returns
        -------
        inverse_transform : Transformation
            inverse of this transform
        """
        inverse_mappings = np.zeros((steps + 1, self.ndim) + self.shape)

        if initial_mapping is None:
            inverse_mappings += self.identity_mapping
        else:
            inverse_mappings += initial_mapping

        Id_tilde = np.copy(self.mapping)

        diff = Id_tilde - self.identity_mapping
        diff_max = np.max(np.abs(diff))

        eps = 1.

        for _ in xrange(maximum_iteration):
            gamma = eps / diff_max
            diff *= gamma
            for step in xrange(steps):
                inverse_mappings[step + 1] = inverse_mappings[step] - np.einsum('ij...,j...->i...', self.jacobian_matrix(inverse_mappings[step]), diff)

            Id_tilde = warp_grid(self.mapping, inverse_mappings[-1])

            diff_temp = Id_tilde - self.identity_mapping
            diff_max_temp = np.max(np.abs(diff_temp))
            if diff_max_temp > diff_max:
                eps *= 0.1
                diff *= 1 / gamma
                continue
            else:
                diff_max = diff_max_temp
                diff = diff_temp * 1.
                inverse_mappings[0] = inverse_mappings[-1] * 1.
            if diff_max < 0.01:
                break
            if _ == maximum_iteration - 1:
                raise RuntimeError("failed to converge after %d iterations" % maximum_iteration)

        inverse_transform = Transformation(mapping=inverse_mappings[-1] * 1.)

        return inverse_transform

    def get_inverse_transform_2d(self):
        from scipy.interpolate import griddata

        negative_displacement = self.identity_mapping - self.mapping

        if self.dimension == 2:
            negative_displacement = negative_displacement.transpose(1,2,0).reshape(-1,2)
            points = self.mapping.transpose(1,2,0).reshape(-1,2)
            negative_displacement = griddata(points, negative_displacement, (self.identity_mapping[0], self.identity_mapping[1]), method='linear').transpose(2,0,1)

        return negative_displacement + self.identity_mapping

    def get_mapping(self):
        return self.mapping * 1.

    def get_displacement(self):
        return self.displacement * 1.

    def save(self, filename, affine=np.identity(4)):
        transposing_axis = ()
        for i in xrange(1, self.ndim + 1):
            transposing_axis = transposing_axis + (i,)
        transposing_axis += (0,)

        displacement = np.transpose(self.displacement, transposing_axis)

        nib.save(nib.Nifti1Image(displacement, affine), filename)

        print "saved transformation: %s" % filename

    def show(self, interval=1, limit_axis=True, show_axis=False):
        if self.ndim == 2:
            # if limit_axis:
            #     plt.xlim(0, self.shape[1])
            #     plt.ylim(0, self.shape[0])

            if not show_axis:
                plt.axis('off')

            ax = plt.gca()
            ax.invert_yaxis()
            ax.set_aspect('equal')
            for x in xrange(0, self.shape[0], interval):
                plt.plot(self.mapping[1,x,:], self.mapping[0,x,:], 'k')
            for y in xrange(0, self.shape[1], interval):
                plt.plot(self.mapping[1,:,y], self.mapping[0,:,y], 'k')

            plt.show()


class DiffeormorphicDeformation(object):

    def __init__(self, ndim, deformation_step, penalty, time_interval=1.):
        self.ndim = ndim
        self.deformation_step = deformation_step
        self.penalty = penalty
        self.time_interval = time_interval
        self.delta_time = time_interval / deformation_step

        self.similarity_metric = 'ssd'

    def set_prior_parameter(self, alpha=1., gamma=1., beta=2):
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta

    def set_metric_kernel(self):
        # dx_sqinv = 1 / (self.resolution ** 2)
        # dx_sqinv = self.resolution ** 2
        dx_sqinv = 1.

        if self.ndim == 2:
            dy_sqinv = dx_sqinv
            kernel = np.array([
                [0., - dx_sqinv, 0.],
                [- dy_sqinv, 2 * (dx_sqinv + dy_sqinv), - dy_sqinv],
                [0., - dx_sqinv, 0.]]) * self.alpha
            kernel[1,1] += self.gamma
        elif self.ndim == 3:
            dy_sqinv = dx_sqinv
            dz_sqinv = dx_sqinv
            kernel = np.array(
                [[[0., 0., 0.],
                [0., - dx_sqinv, 0.],
                [0., 0., 0.]],
                [[0., - dy_sqinv, 0.],
                [- dz_sqinv, 2 * (dx_sqinv + dy_sqinv + dz_sqinv), - dz_sqinv],
                [0, - dy_sqinv, 0]],
                [[0., 0., 0.],
                [0., - dx_sqinv, 0.],
                [0., 0., 0.]]]) * self.alpha
            kernel[1,1,1] += self.gamma

        self.metric_kernel = np.copy(kernel)

        for _ in xrange(self.beta):
            self.metric_kernel = convolve(self.metric_kernel, kernel)

    def set_vectorize_operator(self):
        # dx_sqinv = 1 / (self.resolution ** 2)
        # dx_sqinv = self.resolution ** self.ndim
        dx_sqinv = 1.

        A = self.gamma * np.ones(self.shape)

        if self.ndim == 2:
            self.identity_mapping = np.mgrid[:self.shape[0], :self.shape[1]]
        elif self.ndim == 3:
            self.identity_mapping = np.mgrid[:self.shape[0], :self.shape[1], :self.shape[2]]

        for i in xrange(self.ndim):
            A += 2 * self.alpha * (1 - np.cos(2. * pi * self.identity_mapping[i] / self.shape[i])) * dx_sqinv

        self.vectorize_operator = 1 / (A ** self.beta)

    def set_grid(self, shape, resolution):
        assert len(shape) == self.ndim
        self.shape = shape
        self.resolution = resolution
        self.initialize_vector_fields()
        self.initialize_mappings()
        self.set_metric_kernel()
        self.set_vectorize_operator()

    def initialize_mappings_old(self):
        self.identity_mapping = self.get_identity_mapping()

        self.forward_mappings = np.ones((self.deformation_step + 1, self.ndim) + self.shape) * self.identity_mapping
        self.backward_mappings = np.copy(self.forward_mappings)

        self.forward_jacobian_matrixs = np.ones((self.deformation_step + 1, self.ndim, self.ndim) + self.shape) * jacobian_matrix(self.identity_mapping)
        self.backward_jacobian_matrixs = np.copy(self.forward_jacobian_matrixs)

        self.forward_jacobian_determinants = np.ones((self.deformation_step + 1,) + self.shape)
        self.backward_jacobian_determinants = np.copy(self.forward_jacobian_determinants)

    def initialize_mappings(self):
        self.identity_mapping = self.get_identity_mapping()

        self.forward_mappings = np.ones((self.deformation_step + 1, self.ndim) + self.shape) * self.identity_mapping
        self.backward_mappings = np.copy(self.forward_mappings)

        # self.forward_jacobian_matrixs = np.ones((self.deformation_step + 1, self.ndim, self.ndim) + self.shape) * jacobian_matrix(self.identity_mapping)
        # self.backward_jacobian_matrixs = np.copy(self.forward_jacobian_matrixs)

        self.forward_jacobian_determinants = np.ones((self.deformation_step + 1,) + self.shape)
        self.backward_jacobian_determinants = np.copy(self.forward_jacobian_determinants)

    def set_similarity_metric(self, similarity_metric, *args):
        """
        designate which similarity metric to use to calculate vector fields

        Parameters
        ----------
        similarity_metric : str
            choose among one of these:
            'ssd': Sum of Squared Difference
            'cc': Cross Correlation
        """
        all_similarity_metric = ['ssd', 'cc']
        if similarity_metric in all_similarity_metric:
            self.similarity_metric = similarity_metric
        else:
            raise ValueError("input similarity metric is not valid.")

        if self.similarity_metric == 'cc':
            try:
                self.window_length = args[0]
            except:
                self.window_length = 5

            self.window_size = self.window_length ** self.ndim

    def get_identity_mapping(self, shape=None):
        if shape is None:
            shape = self.shape
            ndim = self.ndim
        else:
            ndim = len(shape)

        if ndim == 2:
            return np.mgrid[:shape[0], :shape[1]].astype(np.float)
        elif ndim == 3:
            return np.mgrid[:shape[0], :shape[1], :shape[2]].astype(np.float)

    def vectorize(self, momentum):
        """
        apply vectorizing operator to momentum which induces smoothness

        Parameters
        ----------
        momentum : ndarray
            Input momentum field.
            eg. 3 dimensional case (dimension, len(x), len(y), len(z))

        Returns
        -------
        vectorField : ndarray
            Vectorized momentum field.
            eg. 3d case (dimension, len(x), len(y), len(z))

        input
        vector_field: the vector field to be regularized.

        returns
        f: regularized vector field

        operator L = - alpha * laplacian + gamma * Id
        K = (LL)^{-1}
        f = Kg
        or
        g = (LL)f
        where f is what we want to calculate
        Fourier transform
        G = A^2 * F
        where
        A = gamma + 2 * alpha sum^3_{i=1} (1 - cos(2pi*dxi*ki))/dxi^2
        where
        ki is frequency and
        dxi is the discritization of the image domain which will be shape[i]
        Therefore

        f = inverse of Fourier transform of (G / A^2)
        """
        G = np.zeros(momentum.shape, dtype=np.complex128)
        for i in xrange(self.ndim):
            try:
                G[i] = fftn(momentum[i], threads=5)
            except:
                G[i] = fftn(momentum[i])
            # G[i] = fftn(momentum[i])
            # G[i] = fftn(momentum[i], threads=5)

        F = G * self.vectorize_operator

        vector_field = np.zeros_like(momentum)
        for i in xrange(self.ndim):
            try:
                vector_field[i] = np.real(ifftn(F[i], threads=5))
            except:
                vector_field[i] = np.real(ifftn(F[i]))
            # vector_field[i] = np.real(ifftn(F[i]))
            # vector_field[i] = np.real(ifftn(F[i], threads=5))

        return vector_field

    def momentum_ssd(self, fixed, moving, Dphi):
        return 2 * gradient(moving) * (fixed - moving) * Dphi / self.penalty

    def momentum_cc(self, J, I, Dphi):
        """
        Convolution of vector field with a kernel
        Parameters
        ----------
        J : ndarray
            Input deformed fixed image.
            eg. 3 dimensional case (len(x), len(y), len(z))
        I : ndarray
            Input deformed moving image.
        jacobian : ndarray
            jacobian determinant of backward mapping
            eg. 3d case (dimension, len(x), len(y), len(z))
        penalty : double
            penalty for vector field
        Returns
        -------
        momentum : ndarray
            momentum field.
            eg. 3d case (dimension, len(x), len(y), len(z))
        """
        Im = uniform_filter(I, self.window_length) / self.window_size
        Jm = uniform_filter(J, self.window_length) / self.window_size

        Ibar = I - Im
        Jbar = J - Jm

        II = uniform_filter(I * I, self.window_length) - self.window_size * Im * Im
        JJ = uniform_filter(J * J, self.window_length) - self.window_size * Jm * Jm
        IJ = uniform_filter(I * J, self.window_length) - self.window_size * Im * Jm

        denom = II * JJ
        IJoverIIJJ = IJ / denom
        IJoverII = IJ / II
        IJoverIIJJ[np.where(denom < 1e-3)] = 0
        IJoverII[np.where(II < 1e-3)] = 0

        f = gradient(Ibar) * IJoverIIJJ * (Jbar - Ibar * IJoverII) * Dphi

        return 2 * f / self.penalty

    def momentum_mi(self, fixed, moving, Dphi):
        pass

    def momentum(self, fixed, moving, Dphi):
        """
        mere derivation of cost function of similarity term.
        vectorization process is usually followed after this calculation to smooth vector field

        Parameters
        ----------
        fixed_data : ndarray
            ndarray data of fixed image
        moving_data : ndarray
            ndarray data of moving image
        Dphi : ndarray
            jacobian determinant of backward mapping which warped original fixed image to the input fixed_data

        Returns
        -------
        momentum : ndarray
            update of vector field before smoothing
        """
        if self.similarity_metric is 'ssd':
            return self.momentum_ssd(fixed, moving, Dphi)
        elif self.similarity_metric is 'cc':
            return self.momentum_cc(fixed, moving, Dphi)
        else:
            raise ValueError("this similarity metric is not valid, %s" % self.similarity_metric)

    def grad_E_similarity(self, fixed_data, moving_data, Dphi):
        """
        returns gradient of cost function of similarity term

        Parameters
        ----------
        fixed_data : ndarray
            ndarray data of fixed image
        moving_data : ndarray
            ndarray data of moving image
        Dphi : ndarray
            jacobian determinant of backward mapping which warped original fixed image to the input fixed_data

        Returns
        -------
        gradE : ndarray
            gradient of cost function of similarity term
        """
        return self.vectorize(self.momentum(fixed_data, moving_data, Dphi))

    def euler_integration(self, mapping, jacobian_matrix, vector_field):
        return mapping - np.einsum('ij...,j...->i...', jacobian_matrix, vector_field) * self.delta_time

    def get_similarity_energy(self, data1, data2):
        if self.similarity_metric == 'ssd':
            return similarity_energy_ssd(data1, data2)
        elif self.similarity_metric == 'cc':
            return similarity_energy_cc(data1, data2, self.window_length, self.window_size)


class LDDMM(DiffeormorphicDeformation):

    def initialize_vector_fields(self):
        self.vector_fields = np.zeros((self.deformation_step + 1, self.ndim) + self.shape)
        self.delta_vector_fields = np.copy(self.vector_fields)

    # def update(self, fixed_images, moving_images):
    #     """
    #     update deformation using gradient descent method

    #     Parameter
    #     ----------
    #     fixed_images : SequentialScalarImages
    #         deformed fixed images
    #     moving_images : SequentialScalarImages
    #         deformed moving images
    #     """
    #     for i in xrange(self.deformation_step + 1):
    #         j = - i - 1
    #         self.delta_vector_fields[i] = self.learning_rate * (2 * self.vector_fields[i] + self.grad_E_similarity(fixed_images[j], moving_images[i], self.backward_jacobian_determinants[j]))

    #     self.vector_fields -= self.delta_vector_fields

    #     self.integrate_vector_fields()

    def update(self, fixed_images, moving_images, learning_rate):
        """
        update deformation using gradient descent method

        Parameters
        ----------
        fixed_images : SequentialScalarImages
            deformed fixed images
        moving_images : SequentialScalarImages
            deformed moving images
        """
        for i in xrange(self.deformation_step + 1):
            j = - i - 1
            self.delta_vector_fields[i] = learning_rate * (2 * self.vector_fields[i] + self.grad_E_similarity(fixed_images[j], moving_images[i], self.backward_jacobian_determinants[j]))

        self.vector_fields -= self.delta_vector_fields

        self.integrate_vector_fields()

    def integrate_vector_fields_old(self):
        """
        integrate vector fields and produce diffeomorphic mappings
        """

        for i in xrange(self.deformation_step):
            v = 0.5 * (self.vector_fields[i] + self.vector_fields[i + 1])
            self.forward_mappings[i + 1] = self.euler_integration(self.forward_mappings[i], self.forward_jacobian_matrixs[i], v)

            v = - 0.5 * (self.vector_fields[-i - 1] + self.vector_fields[-i - 2])
            self.backward_mappings[i + 1] = self.euler_integration(self.backward_mappings[i], self.backward_jacobian_matrixs[i], v)

            self.forward_jacobian_matrixs[i + 1] = jacobian_matrix(self.forward_mappings[i + 1])
            self.backward_jacobian_matrixs[i + 1] = jacobian_matrix(self.backward_mappings[i + 1])

            self.forward_jacobian_determinants[i + 1] = determinant(self.forward_jacobian_matrixs[i + 1])
            self.backward_jacobian_determinants[i + 1] = determinant(self.backward_jacobian_matrixs[i + 1])

    def integrate_vector_fields(self):
        """
        integrate vector fields and produce diffeomorphic mappings
        """

        forward_jacobian_matrix = jacobian_matrix(self.forward_mappings[0])
        backward_jacobian_matrix = np.copy(forward_jacobian_matrix)

        for i in xrange(self.deformation_step):
            v = 0.5 * (self.vector_fields[i] + self.vector_fields[i + 1])
            self.forward_mappings[i + 1] = self.euler_integration(self.forward_mappings[i], forward_jacobian_matrix, v)

            v = - 0.5 * (self.vector_fields[-i - 1] + self.vector_fields[-i - 2])
            self.backward_mappings[i + 1] = self.euler_integration(self.backward_mappings[i], backward_jacobian_matrix, v)

            forward_jacobian_matrix = jacobian_matrix(self.forward_mappings[i + 1])
            backward_jacobian_matrix = jacobian_matrix(self.backward_mappings[i + 1])

            self.forward_jacobian_determinants[i + 1] = determinant(forward_jacobian_matrix)
            self.backward_jacobian_determinants[i + 1] = determinant(backward_jacobian_matrix)

    def get_energy(self, data1, data2):
        # E_v = 0
        # for i in xrange(self.deformation_step):
        #     E_v += np.sum(convolve_vector(self.vector_fields[i], self.metric_kernel) * self.vector_fields[i]) * self.delta_time
        E_simi = self.get_similarity_energy(data1, data2)
        # return E_v * self.penalty + E_simi
        return E_simi

    def back_to_previous_deformation(self):
        self.vector_fields += self.delta_vector_fields
        self.integrate_vector_fields()

    def get_forward_mapping(self):
        return self.forward_mappings[-1]

    def get_backward_mapping(self):
        return self.backward_mappings[-1]

    def get_minimum_unit(self):
        return np.min(self.forward_jacobian_determinants[-1])


class SyN(DiffeormorphicDeformation):

    def initialize_vector_fields(self):
        assert(self.deformation_step % 2 == 0)

        self.half_deformation_step = int(self.deformation_step / 2)

        self.former_vector_fields = np.zeros((self.half_deformation_step + 1, self.ndim) + self.shape)
        self.latter_vector_fields = np.copy(self.former_vector_fields)

        self.former_delta_vector_fields = np.copy(self.former_vector_fields)
        self.latter_delta_vector_fields = np.copy(self.former_vector_fields)

    # def update(self, fixed_images, moving_images):
    #     """
    #     update deformation using gradient descent method

    #     Parameters
    #     ----------
    #     fixed_images : SequentialScalarImages
    #         deformed fixed images
    #     moving_images : SequentialScalarImages
    #         deformed moving images
    #     """
    #     for i in xrange(self.half_deformation_step + 1):
    #         j = -i - 1
    #         self.former_delta_vector_fields[i] = self.learning_rate * (2 * self.former_vector_fields[i] + self.grad_E_similarity(fixed_data=fixed_images[j], moving_data=moving_images[i], Dphi=self.backward_jacobian_determinants[j]))
    #         self.latter_delta_vector_fields[i] = self.learning_rate * (2 * self.latter_vector_fields[i] + self.grad_E_similarity(fixed_data=moving_images[j], moving_data=fixed_images[i], Dphi=self.forward_jacobian_determinants[i]))

    #     self.former_vector_fields -= self.former_delta_vector_fields
    #     self.latter_vector_fields -= self.latter_delta_vector_fields

    #     self.integrate_vector_fields()

    def update(self, fixed_images, moving_images, learning_rate):
        """
        update deformation using gradient descent method

        Parameters
        ----------
        fixed_images : SequentialScalarImages
            deformed fixed images
        moving_images : SequentialScalarImages
            deformed moving images
        """
        for i in xrange(self.half_deformation_step + 1):
            j = -i - 1
            self.former_delta_vector_fields[i] = learning_rate * (2 * self.former_vector_fields[i] + self.grad_E_similarity(fixed_data=fixed_images[j], moving_data=moving_images[i], Dphi=self.backward_jacobian_determinants[j]))
            self.latter_delta_vector_fields[i] = learning_rate * (2 * self.latter_vector_fields[i] + self.grad_E_similarity(fixed_data=moving_images[j], moving_data=fixed_images[i], Dphi=self.forward_jacobian_determinants[i]))

        self.former_vector_fields -= self.former_delta_vector_fields
        self.latter_vector_fields -= self.latter_delta_vector_fields

        self.integrate_vector_fields()

    def integrate_vector_fields_old(self):
        for i in xrange(self.half_deformation_step):
            v = 0.5 * (self.former_vector_fields[i] + self.former_vector_fields[i + 1])
            self.forward_mappings[i + 1] = self.euler_integration(self.forward_mappings[i], self.forward_jacobian_matrixs[i], v)

            v = 0.5 * (self.latter_vector_fields[-i - 1] + self.latter_vector_fields[-i - 2])
            self.backward_mappings[i + 1] = self.euler_integration(self.backward_mappings[i], self.backward_jacobian_matrixs[i], v)

            self.forward_jacobian_matrixs[i + 1] = jacobian_matrix(self.forward_mappings[i + 1])
            self.backward_jacobian_matrixs[i + 1] = jacobian_matrix(self.backward_mappings[i + 1])

            self.forward_jacobian_determinants[i + 1] = determinant(self.forward_jacobian_matrixs[i + 1])
            self.backward_jacobian_determinants[i + 1] = determinant(self.backward_jacobian_matrixs[i + 1])

        for i in xrange(self.half_deformation_step):
            j = i + self.half_deformation_step
            v = - 0.5 * (self.latter_vector_fields[-i] + self.latter_vector_fields[-i-1])
            self.forward_mappings[j+1] = self.euler_integration(self.forward_mappings[j], self.forward_jacobian_matrixs[j], v)

            v = - 0.5 * (self.former_vector_fields[-i] + self.former_vector_fields[-i-1])
            self.backward_mappings[j+1] = self.euler_integration(self.backward_mappings[j], self.backward_jacobian_matrixs[j], v)

            self.forward_jacobian_matrixs[j+1] = jacobian_matrix(self.forward_mappings[j+1])
            self.backward_jacobian_matrixs[j+1] = jacobian_matrix(self.backward_mappings[j+1])

            self.forward_jacobian_determinants[j+1] = determinant(self.forward_jacobian_matrixs[j+1])
            self.backward_jacobian_determinants[j+1] = determinant(self.backward_jacobian_matrixs[j+1])

    def integrate_vector_fields(self):
        forward_jacobian_matrix = jacobian_matrix(self.identity_mapping)
        backward_jacobian_matrix = np.copy(forward_jacobian_matrix)
        for i in xrange(self.half_deformation_step):
            v = 0.5 * (self.former_vector_fields[i] + self.former_vector_fields[i + 1])
            self.forward_mappings[i + 1] = self.euler_integration(self.forward_mappings[i], forward_jacobian_matrix, v)

            v = 0.5 * (self.latter_vector_fields[-i - 1] + self.latter_vector_fields[-i - 2])
            self.backward_mappings[i + 1] = self.euler_integration(self.backward_mappings[i], backward_jacobian_matrix, v)

            forward_jacobian_matrix = jacobian_matrix(self.forward_mappings[i + 1])
            backward_jacobian_matrix = jacobian_matrix(self.backward_mappings[i + 1])

            self.forward_jacobian_determinants[i + 1] = determinant(forward_jacobian_matrix)
            self.backward_jacobian_determinants[i + 1] = determinant(backward_jacobian_matrix)

        for i in xrange(self.half_deformation_step):
            j = i + self.half_deformation_step
            v = - 0.5 * (self.latter_vector_fields[-i] + self.latter_vector_fields[-i-1])
            self.forward_mappings[j+1] = self.euler_integration(self.forward_mappings[j], forward_jacobian_matrix, v)

            v = - 0.5 * (self.former_vector_fields[-i] + self.former_vector_fields[-i-1])
            self.backward_mappings[j+1] = self.euler_integration(self.backward_mappings[j], backward_jacobian_matrix, v)

            forward_jacobian_matrix = jacobian_matrix(self.forward_mappings[j+1])
            backward_jacobian_matrix = jacobian_matrix(self.backward_mappings[j+1])

            self.forward_jacobian_determinants[j+1] = determinant(forward_jacobian_matrix)
            self.backward_jacobian_determinants[j+1] = determinant(backward_jacobian_matrix)

    def get_energy(self, data1, data2):
        E_simi = self.get_similarity_energy(data1, data2)
        return E_simi

    def back_to_previous_deformation(self):
        self.former_vector_fields += self.former_delta_vector_fields
        self.latter_vector_fields += self.latter_delta_vector_fields

        self.integrate_vector_fields()

    def get_forward_mapping(self):
        return self.forward_mappings[self.half_deformation_step]

    def get_forward_mapping_inverse(self):
        inverse_mapping = np.copy(self.identity_mapping)
        for i in xrange(self.half_deformation_step):
            v = - 0.5 * (self.former_vector_fields[-i-1] + self.former_vector_fields[-i-2])
            inverse_mapping = self.euler_integration(inverse_mapping, jacobian_matrix(inverse_mapping), v)

        return inverse_mapping

    def get_backward_mapping(self):
        return self.backward_mappings[self.half_deformation_step]

    def get_backward_mapping_inverse(self):
        inverse_mapping = np.copy(self.identity_mapping)
        for i in xrange(self.half_deformation_step):
            v = - 0.5 * (self.latter_vector_fields[-i-1] + self.latter_vector_fields[-i-2])
            inverse_mapping = self.euler_integration(inverse_mapping, jacobian_matrix(inverse_mapping), v)
        return inverse_mapping

    def get_minimum_unit(self):
        return np.min(self.forward_jacobian_determinants[self.half_deformation_step])


def jacobian_matrix(mapping):
    """
    Jacobian matrix at all points

    Parameters
    ----------
    mapping : ndarray
        Input mapping.
        eg. if 3 dimensional ((dimension, len(x), len(y), len(z)))

    Returns
    -------
    J : ndarray
        Jacobian matrix.
        eg. 3d case (dimension, dimension, len(x), len(y), len(z))
    """
    dimension = mapping.ndim - 1

    if dimension == 2:
        return np.array([gradient(mapping[0]), gradient(mapping[1])])
    elif dimension == 3:
        return np.array([gradient(mapping[0]), gradient(mapping[1]), gradient(mapping[2])])

def determinant(J):
    """
    Determinant of jacobian matrix at all points

    Parameters
    ----------
    J : ndarray
        Input jacobian matrixs.
        eg. 3 dimensional case ((dimension, dimension, len(x), len(y), len(z)))
    Returns
    -------
    D : ndarray
        Determinant of jacobian matrix.
        eg. 3 dimensional case (len(x), len(y), len(z))
    """
    dimension = J.ndim - 2

    if dimension == 2:
        return J[0,0] * J[1,1] - J[0,1] * J[1,0]
    elif dimension == 3:
        """
        (0,0) (0,1) (0,2)
        (1,0) (1,1) (1,2)
        (2,0) (2,1) (2,2)
        """
        return (J[0,0] * J[1,1] * J[2,2] +
                J[1,0] * J[2,1] * J[0,2] +
                J[0,1] * J[1,2] * J[2,0] -
                J[0,0] * J[1,2] * J[2,1] -
                J[2,2] * J[1,0] * J[0,1] -
                J[0,2] * J[1,1] * J[2,0])

def convolve_vector(vector_field, kernel):
    """
    Convolution of vector field with a kernel

    Parameters
    ----------
    vectorField : ndarray
        Input vector field.
        eg. 3 dimensional case (dimension, len(x), len(y), len(z))
    kernel : ndarray
        kernel to convolve with.
    Returns
    -------
    convolved : ndarray
        convolved vector field.
        eg. 3 dimensional case (dimension, len(x), len(y), len(z))
    """
    convolved = np.zeros_like(vector_field)

    for i in xrange(vector_field.ndim - 1):
        convolved[i] = correlate(vector_field[i], kernel, mode='wrap')

    return convolved

def warp_grid(grid, mapping_function, order=3, mode='nearest'):
    """
    warp grid with a mapping function
    phi_1 = grid
    phi_2 = mapping_function
    the result is
    phi_1(phi_2)

    Parameters
    ----------
    grid : ndarray
        a grid which is going to be warped
    mapping_function : ndarray
        the grid will be deformed by this mapping function

    Returns
    -------
    warped_grid : ndarray
        grid deformed by the mapping function
    """
    if len(grid) == len(mapping_function):
        ndim = len(grid)
    else:
        raise ValueError('the dimension of the two inputs are the same')

    warped_grid = np.zeros_like(grid)
    for i in xrange(ndim):
        warped_grid[i] = map_coordinates(grid[i], mapping_function, order=order, mode=mode)

    return warped_grid

def similarity_energy_ssd(data1, data2):
    return np.sum((data1 - data2) ** 2)

def similarity_energy_cc(I, J, window_length, window_size):
    Im = uniform_filter(I, window_length) / window_size
    Jm = uniform_filter(J, window_length) / window_size
    II = uniform_filter(I * I, window_length) - window_size * Im * Im
    JJ = uniform_filter(J * J, window_length) - window_size * Jm * Jm
    IJ = uniform_filter(I * J, window_length) - window_size * Im * Jm
    E = (IJ ** 2) / (II * JJ)
    E[np.where((II < 1e-5) + (JJ < 1e-5))] = 0
    return - np.sum(E)