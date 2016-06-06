import numpy as np
from scipy.ndimage.filters import uniform_filter
from deformations import LDDMM
from imageprocessing import sliding_matrix_product


def check_vector_field_smoothing():
    vector_fields = np.zeros((2,10,10))
    vector_fields[:,5,5] = 10.
    print np.round(vector_fields[0], decimals=1)

    deformation = LDDMM(ndim=2, deformation_step=1, penalty=1000, learning_rate=0.1, time_interval=1.)

    deformation.set_prior_parameter(alpha=1)
    deformation.set_grid(shape=(10,10), resolution=1)
    A = deformation.vectorize(vector_fields)
    print np.round(A[0], decimals=1)

    deformation.set_prior_parameter(alpha=2)
    deformation.set_grid(shape=(10,10), resolution=1)
    A = deformation.vectorize(vector_fields)
    print np.round(A[0], decimals=1)

    vector_fields = np.zeros((2,20,20))
    vector_fields[:,10,10] = 10.

    deformation.set_prior_parameter(alpha=1.)
    deformation.set_grid(shape=(20,20), resolution=1)
    A = deformation.vectorize(vector_fields)
    print np.round(A[0], decimals=1)

def mutual_information(data1, data2, bins=32):
    if not data1.shape == data2.shape:
        raise ValueError("the two input must have the same shape")

    data1_range = __range(data1, bins)
    data2_range = __range(data2, bins)

    data12_hist, _, _ = np.histogram2d(data1.flatten(), data2.flatten(), bins=bins, range=[data1_range, data2_range])
    data1_hist, _ = np.histogram(data1.flatten(), bins=bins, range=data1_range)
    data2_hist, _ = np.histogram(data2.flatten(), bins=bins, range=data2_range)

    data12_entropy = __entropy(data12_hist)
    data1_entropy = __entropy(data1_hist)
    data2_entropy = __entropy(data2_hist)

    return data1_entropy + data2_entropy - data12_entropy

def __range(a, bins):
    """
    calculate appropriate range of intensity of an input

    Parameters
    ----------
    a : ndarray
        sequence of intensities

    Returns
    -------
    (range_min, range_max) : tuple
        appropriate range of intesity
    """
    a_max = a.max()
    a_min = a.min()
    s = 0.5 * (a_max - a_min) / float(bins - 1)
    return (a_min - s, a_max + s)

def __entropy(data):
    """
    computes entropy of the flattened data set

    Parameters
    ----------
    data : ndarray
        flattened data

    Returns
    -------
    entropy : float
        the entropy of the input data
    """
    data = data / float(np.sum(data))

    data = data[np.nonzero(data)]

    return -1. * np.sum(data * np.log2(data))

def patchwise_mutual_information(img1, img2):
    MI = np.zeros(img1.shape)
    padded1 = np.pad(img1, 4, mode='constant')
    padded2 = np.pad(img2, 4, mode='constant')
    for i in xrange(img1.shape[0]):
        for j in xrange(img1.shape[1]):
            MI[i,j] = mutual_information(padded1[i:i + 9, j:j + 9], padded2[i:i + 9,j:j + 9])
    return MI

def test_MI():
    # from image import ScalarImage
    # img1 = ScalarImage(filename="original.nii")
    # img2 = ScalarImage(filename="warped_original.nii")
    # mi = patchwise_mutual_information(img1.data, img2.data)
    # MI = ScalarImage(data=mi)
    # MI.show()
    A = np.ones((100,100)) * 50.
    A[45:55,45:55] = np.random.randint(0,100,(10,10))
    from image import ScalarImage
    img = ScalarImage(data=A)
    img.show()
    data_range = __range(img.data, bins=32)
    data = img.get_data()
    entropy = np.zeros(data.shape)
    for i in xrange(data.shape[0]):
        for j in xrange(data.shape[1]):
            try:
                hist = np.histogram(data[i-4:i+5,j-4:j+5], bins=32, range=data_range)
                entropy[i,j] = __entropy(hist)
            except IndexError:
                entropy[i,j] = 0.
    print entropy[50,50]
    I = ScalarImage(data=entropy)
    I.show()

def test_uniform_filter():
    A = np.random.randint(0,200,(100,100)).astype(np.float)
    print np.allclose(uniform_filter(A, 5)[10:90,10:90], uniformFilter(A, 5)[10:90,10:90]/25)

def test_cost_function():
    from os.path import expanduser, join
    from deformation import similarity_energy_cc
    from images import ScalarImage
    home = expanduser('~')
    dname = join(home, 'registration/img/IBSR/from02to01')
    fixed_img_file = join(dname, 'IBSR01.nii')
    moving_img_file = join(dname, 'from02to01affine.nii.gz')

    fixed_img = ScalarImage(fixed_img_file)
    moving_img = ScalarImage(moving_img_file)

    fixed_data = fixed_img.get_data()
    moving_data = moving_img.get_data()

    print fixed_data[100,60,100]

    print similarity_energy_cc(fixed_data, moving_data, 5, 125)

def test_module():
    from time import time
    from imageprocessing import uniformFilter
    A = np.random.normal(size=(200,100,200))
    t = time()
    B = uniformFilter(A, 5)
    print time() - t
    t = time()
    B = uniform_filter(A, 5)
    print time() - t
    # print np.allclose(A, B)
    # print np.allclose(A, B.transpose(2,1,0))

def test_slide_dot_matrix():
    A = np.random.randint(0,200,(200,100, 200)).astype(np.float)
    matrix = np.identity(27)
    result = sliding_matrix_product(A, matrix)
    print result.shape
    print A[0:3,0:3,0:3]
    print result[0,0,0]

if __name__ == '__main__':
    test_slide_dot_matrix()