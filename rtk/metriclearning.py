import numpy as np
from imageprocessing import uniform_filter

class CosineSimilarityMetricLearning(object):

    def __init__(self, window_length, initial_metric_matrix):
        self.window_length = window_length
        self.initial_metric_matrix = initial_metric_matrix
        self.alpha = 1.

    def set_validation_images(self, validation_img1, validation_img2):
        if not(validation_img1.ndim == validation_img2.ndim):
            print "dimensionality of img1", validation_img1.ndim
            print "dimensionality of img2", validation_img2.ndim
            raise ValueError("the dimensionality of the both images have to be the same")
        if not(validation_img1.ndim == validation_img2.ndim):
            print "shape of img1", validation_img1.shape
            print "shape of img2", validation_img2.shape
            raise ValueError("the shape of the two images are different.")

        self.validation_img1 = validation_img1
        self.validation_img2 = validation_img2

        if hasattr(self, 'ndim'):
            assert(self.ndim == validation_img1.ndim)
        else:
            self.ndim = validation_img1.get_ndim()

        if hasattr(self, 'shape'):
            assert(self.shape == validation_img1.shape)
        else:
            self.shape = validation_img1.get_shape()

        self.set_dissimilarity()

    def set_input_images(self, img1, img2):
        if not(img1.ndim == img2.ndim):
            print "dimensionality of img1", img1.ndim
            print "dimensionality of img2", img2.ndim
            raise ValueError("the dimensionality of the both images have to be the same")
        if not(img1.ndim == img2.ndim):
            print "shape of img1", img1.shape
            print "shape of img2", img2.shape
            raise ValueError("the shape of the two images are different.")

        self.img1 = img1
        self.img2 = img2

        if hasattr(self, 'ndim'):
            assert(self.ndim == img1.ndim)
        else:
            self.ndim = img1.get_ndim()

        if hasattr(self, 'shape'):
            assert(self.shape == img1.shape)
        else:
            self.shape = img1.get_shape()

    def set_dissimilarity(self):

        self.dissimilarity = self.label_dissimilarity()

    def label_dissimilarity(self):
        data1 = self.img1.get_data()
        data2 = self.img2.get_data()
        window_size = self.window_length ** self.ndim

        label_difference = np.abs(data1 - data2)
        label_difference[np.where(label_difference > 0)] = 1.

        return uniform_filter(label_difference, self.window_length) / window_size

    def update(self):
        pass

    def stochastic_gradient_descent(self, x, y):
        U = np.dot(np.dot(self.metric_matrix, x), np.dot(self.metric_matrix, y))
        V = np.linalg.norm(np.dot(self.metric_matrix, x)) * np.linalg.norm(np.dot(self.metric_matrix, x))
        gradU = np.dot(self.metric_matrix,np.outer(x, y) + np.outer(y, x))
        gradV = (np.linalg.norm(np.dot(self.metric_matrix, y)) / np.linalg.norm(np.dot(self.metric_matrix, x))) * np.dot(self.metric_matrix, np.outer(x, x)) - (np.linalg.norm(np.dot(self.metric_matrix, x)) / np.linalg.norm(np.dot(self.metric_matrix, y))) * np.dot(self.metric_matrix, np.outer(y, y))

        grad = gradU / V - gradV * U / (V ** 2)

        return grad
