import numpy as np
cimport numpy as cnp
from libc.math cimport ceil, floor, round
from libc.stdlib cimport malloc, free
 
ctypedef cnp.float64_t DOUBLE_t

def gradient(cnp.ndarray func):
    cdef int n = func.ndim
    if n == 1:
        return grad1d(<double*> func.data, func.shape[0])
    elif n == 2:
        return grad2d(<double*> func.data, func.shape[0], func.shape[1])
    elif n == 3:
        return grad3d(<double*> func.data, func.shape[0], func.shape[1], func.shape[2])
    else:
        raise ValueError('dimension of the input has to be 2 or 3')

cdef inline cnp.ndarray[DOUBLE_t, ndim=1] grad1d(double* func, int xlen):
    cdef cnp.ndarray[DOUBLE_t, ndim=1] grad

    grad = np.zeros(xlen)

    cdef int x, y

    for x in range(xlen):
        if x == 0:
            grad[x] = func[1] - func[0]
        elif x == xlen - 1:
            grad[x] = func[x] - func[x-1]
        else:
            grad[x] = 0.5 * (func[x+1] - func[x-1])

    return grad

cdef inline cnp.ndarray[DOUBLE_t, ndim=3] grad2d(double* func, int xlen, int ylen):
    cdef cnp.ndarray[DOUBLE_t, ndim=3] grad

    grad = np.zeros((2,xlen,ylen))

    cdef int x, y

    for x in range(xlen):
        for y in range(ylen):
            if x == 0:
                grad[0,x,y] = func[ylen + y] - func[y]
            elif x == xlen - 1:
                grad[0,x,y] = func[x * ylen + y] - func[(x-1) * ylen + y]
            else:
                grad[0,x,y] = 0.5 * (func[(x+1) * ylen + y] - func[(x-1) * ylen + y])

            if y == 0:
                grad[1,x,y] = func[x * ylen + 1] - func[x * ylen]
            elif y == ylen - 1:
                grad[1,x,y] = func[x * ylen + y] - func[x * ylen + y-1]
            else:
                grad[1,x,y] = 0.5 * (func[x * ylen + y+1] - func[x * ylen + y-1])

    return grad

cdef inline cnp.ndarray[DOUBLE_t, ndim=4] grad3d(double* func, int xlen, int ylen, int zlen):
    cdef cnp.ndarray[DOUBLE_t, ndim=4] grad

    grad = np.zeros((3, xlen, ylen, zlen))

    cdef int x, y, z

    for x in range(xlen):
        for y in range(ylen):
            for z in range(zlen):
                if x == 0:
                    grad[0,x,y,z] = func[(ylen + y) * zlen + z] - func[y * zlen + z]
                elif x == xlen - 1:
                    grad[0,x,y,z] = func[(x * ylen + y) * zlen + z] - func[((x-1) * ylen + y) * zlen + z]
                else:
                    grad[0,x,y,z] = 0.5 * (func[((x+1) * ylen + y) * zlen + z] - func[((x-1) * ylen + y) * zlen + z])

                if y == 0:
                    grad[1,x,y,z] = func[(x * ylen + 1) * zlen + z] - func[x * ylen * zlen + z]
                elif y == ylen - 1:
                    grad[1,x,y,z] = func[(x * ylen + y) * zlen + z] - func[(x * ylen + y - 1) * zlen + z]
                else:
                    grad[1,x,y,z] = 0.5 * (func[(x * ylen + y+1) * zlen + z] - func[(x * ylen + y-1) * zlen + z])

                if z == 0:
                    grad[2,x,y,z] = func[(x * ylen + y) * zlen + 1] - func[(x * ylen + y) * zlen]
                elif z == zlen - 1:
                    grad[2,x,y,z] = func[(x * ylen + y) * zlen + z] - func[(x * ylen + y) * zlen + z - 1]
                else:
                    grad[2,x,y,z] = 0.5 * (func[(x * ylen + y) * zlen + z + 1] - func[(x * ylen + y) * zlen + z - 1])

    return grad

def uniform_filter(cnp.ndarray func, int length):
    if length % 2 == 0:
        raise ValueError('length must be odd number')
    r = (length - 1) / 2
    if func.ndim == 1:
        return uniform_filter1d(<double*> func.data, func.shape[0], r)
    elif func.ndim == 2:
        return uniform_filter2d(<double*> func.data, func.shape[0], func.shape[1], r)
    elif func.ndim == 3:
        return uniform_filter3d(<double*> func.data, func.shape[0], func.shape[1], func.shape[2], r)
    else:
        raise ValueError('mismatch of the dimension sizes')

cdef inline cnp.ndarray[DOUBLE_t, ndim=1] uniform_filter1d(double* func, int xlen, int r):
    cdef cnp.ndarray[DOUBLE_t, ndim=1] convolved = np.zeros(xlen)
    cdef int x
    cdef double value

    value = 0.
    for x in range(r):
        value += func[x]
    for x in range(xlen):
        if x > r:
            value -= func[x - r - 1]
        if x + r < xlen:
            value += func[x + r]
        convolved[x] = value

    return convolved

cdef inline cnp.ndarray[DOUBLE_t, ndim=2] uniform_filter2d(double* func, int xlen, int ylen, int r):
    cdef cnp.ndarray[DOUBLE_t, ndim=2] convolved = np.zeros((xlen, ylen))
    cdef int x, y, i
    cdef double* temp = <double*>malloc(xlen * ylen * sizeof(double))
    cdef double value

    for x in range(xlen):
        value = 0.
        for i in range(r):
            value += func[x * ylen + i]
        for y in range(ylen):
            if y > r:
                value -= func[x * ylen + y - r - 1]
            if y + r < ylen:
                value += func[x * ylen + y + r]
            temp[x * ylen + y] = value

    for y in range(ylen):
        value = 0.
        for i in range(r):
            value += temp[i * ylen + y]
        for x in range(xlen):
            if x > r:
                value -= temp[(x - r - 1) * ylen + y]
            if x + r < xlen:
                value += temp[(x + r) * ylen + y]
            convolved[x,y] = value

    free(temp)

    return convolved

cdef inline cnp.ndarray[DOUBLE_t, ndim=3] uniform_filter3d(double* func, int xlen, int ylen, int zlen, int r):
    cdef cnp.ndarray[DOUBLE_t, ndim=3] convolved = np.zeros((xlen, ylen, zlen))
    cdef int x, y, z, i
    cdef double* temp1 = <double*>malloc(xlen * ylen * zlen * sizeof(double))
    cdef double* temp2 = <double*>malloc(xlen * ylen * zlen * sizeof(double))
    cdef double value

    for x in range(xlen):
        for y in range(ylen):
            value = 0.
            for i in range(r):
                value += func[(x * ylen + y) * zlen + i]
            for z in range(zlen):
                if z > r:
                    value -= func[(x * ylen + y) * zlen + z - r - 1]
                if z + r < zlen:
                    value += func[(x * ylen + y) * zlen + z + r]
                temp1[(x * ylen + y) * zlen + z] = value

    for x in range(xlen):
        for z in range(zlen):
            value = 0.
            for i in range(r):
                value += temp1[(x * ylen + i) * zlen + z]
            for y in range(ylen):
                if y > r:
                    value -= temp1[(x * ylen + y - r - 1) * zlen + z]
                if y + r < ylen:
                    value += temp1[(x * ylen + y + r) * zlen + z]
                temp2[(x * ylen + y) * zlen + z] = value

    for y in range(ylen):
        for z in range(zlen):
            value = 0.
            for i in range(r):
                value += temp2[(i * ylen + y) * zlen + z]
            for x in range(xlen):
                if x > r:
                    value -= temp2[((x - r - 1) * ylen + y) * zlen + z]
                if x + r < xlen:
                    value += temp2[((x + r) * ylen + y) * zlen + z]
                convolved[x,y,z] = value

    free(temp1)
    free(temp2)

    return convolved

def sliding_matrix_product(cnp.ndarray img, cnp.ndarray matrix):
    assert(matrix.shape[0] == matrix.shape[1])
    if img.ndim == 2:
        return sliding_matrix_product_2d(<double*> img.data, img.shape[0], img.shape[1], <double*> matrix.data, matrix.shape[0])
    elif img.ndim == 3:
        return sliding_matrix_product_3d(<double*> img.data, img.shape[0], img.shape[1], img.shape[2], <double*> matrix.data, matrix.shape[0])
    else:
        raise ValueError('the dimensionality of the input img has to be 2 or 3')

cdef inline cnp.ndarray[DOUBLE_t, ndim=3] sliding_matrix_product_2d(double* img, int xlen, int ylen, double* matrix, int mat_len):

    cdef int i, j, column_index, row_index, x, y
    cdef double c

    cdef int window_len = <int>round(mat_len ** 0.5)

    assert(window_len ** 2 == mat_len)
    assert window_len % 2 == 1, "length of matrix must be odd number"
    cdef int radius = (window_len - 1) / 2

    cdef cnp.ndarray[DOUBLE_t, ndim=3] product
    product = np.zeros((xlen, ylen, mat_len))

    for x in range(xlen):
        for y in range(ylen):
            for column_index in range(mat_len):
                c = 0
                row_index = 0
                for i in range(-radius, radius + 1):
                    for j in range(-radius, radius + 1):
                        c += getValue2d(img, x+i, y+j, xlen, ylen, 'C') * matrix[column_index * mat_len + row_index]
                        row_index += 1
                product[x,y,column_index] = c

    return product

cdef inline cnp.ndarray[DOUBLE_t, ndim=4] sliding_matrix_product_3d(double* img, int xlen, int ylen, int zlen, double* matrix, int mat_len):

    cdef int i, j, k, column_index, row_index, x, y, z
    cdef double c

    cdef int window_len = <int>round(mat_len ** (1. / 3))

    assert(window_len ** 3 == mat_len)
    assert window_len % 2 == 1, "length of matrix must be odd number"
    cdef int radius = (window_len - 1) / 2

    cdef cnp.ndarray[DOUBLE_t, ndim=4] product
    product = np.zeros((xlen, ylen, zlen, mat_len))

    for x in range(xlen):
        for y in range(ylen):
            for z in range(zlen):
                for column_index in range(mat_len):
                    c = 0
                    row_index = 0
                    for i in range(-radius, radius + 1):
                        for j in range(-radius, radius + 1):
                            for k in range(-radius, radius + 1):
                                c += getValue3d(img, x+i, y+j, z+k, xlen, ylen, zlen, 'C') * matrix[column_index * mat_len + row_index]
                                row_index += 1
                    product[x,y,z,column_index] = c

    return product

def interpolate_mapping(cnp.ndarray func, int[:] target_shape):
    if func.ndim == 2:
        return interpolate2d(<double*> func.data, func.shape[0], func.shape[1], target_shape)
    elif func.ndim == 3:
        return interpolate3d(<double*> func.data, func.shape[0], func.shape[1], func.shape[2], target_shape)
    else:
        raise ValueError('mismatch of the dimension size')

cdef inline cnp.ndarray[DOUBLE_t, ndim=2] interpolate2d(double* func, int xlen_now, int ylen_now, int[:] target_shape):
    cdef int xlen_target = target_shape[0]
    cdef int ylen_target = target_shape[1]

    cdef cnp.ndarray[DOUBLE_t, ndim=2] interpolated
    interpolated = np.zeros((xlen_target, ylen_target))

    cdef double xi, yi
    cdef int x, y
    for x in range(xlen_target):
        xi = x * (xlen_now - 1) / (xlen_target - 1.)
        for y in range(ylen_target):
            yi = y * (ylen_now - 1) / (ylen_target - 1.)
            interpolated[x,y] = bilinear_interpolation(func, xi, yi, xlen_now, ylen_now)

    return interpolated

cdef inline double bilinear_interpolation(double* func, double x, double y, int xlen, int ylen):
    """
    Bilinear interpolation at a given position in the image.
    Parameters
    ----------
    func : double array
        Input function.
    x, y : double
        Position at which to interpolate.
    Returns
    -------
    value : double
        Interpolated value.
    """

    cdef double dx, dy, f0, f1
    cdef int x0, x1, y0, y1

    x0 = <int>floor(x)
    x1 = <int>ceil(x)
    y0 = <int>floor(y)
    y1 = <int>ceil(y)

    dx = x - x0
    dy = y - y0

    f0 = (1 - dy) * getValue2d(func, x0, y0, xlen, ylen, 'N') + dy * getValue2d(func, x0, y1, xlen, ylen, 'N')
    f1 = (1 - dy) * getValue2d(func, x1, y0, xlen, ylen, 'N') + dy * getValue2d(func, x1, y1, xlen, ylen, 'N')

    return (1 - dx) * f0 + dx * f1

cdef inline double getValue2d(double* func, int x, int y, int xlen, int ylen, char mode='N'):

    if mode == 'N':
        if x < 0:
            x = 0
        elif x > xlen - 1:
            x = xlen - 1

        if y < 0:
            y = 0
        elif y > ylen - 1:
            y = ylen - 1
    elif mode == 'C':
        if x < 0 or x > xlen - 1 or y < 0 or y > ylen - 1:
            return 0

    return func[x * ylen + y]

cdef inline cnp.ndarray[DOUBLE_t, ndim=3] interpolate3d(double* func, int xlen_now, int ylen_now, int zlen_now, int[:] target_shape):
    cdef int xlen_target = target_shape[0]
    cdef int ylen_target = target_shape[1]
    cdef int zlen_target = target_shape[2]

    cdef cnp.ndarray[DOUBLE_t, ndim=3] interpolated
    interpolated = np.zeros((xlen_target, ylen_target, zlen_target))

    cdef double xi, yi, zi
    cdef int x, y, z
    for x in range(xlen_target):
        xi = x * (xlen_now - 1) / (xlen_target - 1.)
        for y in range(ylen_target):
            yi = y * (ylen_now - 1) / (ylen_target - 1.)
            for z in range(zlen_target):
                zi = z * (zlen_now - 1) / (zlen_target - 1.)
                interpolated[x,y,z] = trilinear_interpolation(func, xi, yi, zi, xlen_now, ylen_now, zlen_now)

    return interpolated
    
cdef inline double trilinear_interpolation(double* func, double x, double y, double z, int xlen, int ylen, int zlen):
    """
    trilinear interpolation at a given position in the function.
    Parameters
    ----------
    func : double array
        Input function.
    x, y : double
        Position at which to interpolate.
    Returns
    -------
    value : double
        Interpolated value.
    """

    cdef double dx, dy, dz, value
    cdef int x0, x1, y0, y1, z0, z1

    x0 = <int>floor(x)
    x1 = <int>ceil(x)
    y0 = <int>floor(y)
    y1 = <int>ceil(y)
    z0 = <int>floor(z)
    z1 = <int>ceil(z)

    dx = x - x0
    dy = y - y0
    dz = z - z0

    value = (1 - dx) * (1 - dy) * (1 - dz) * getValue3d(func, x0, y0, z0, xlen, ylen, zlen, 'N')
    value += (1 - dx) * (1 - dy) * dz * getValue3d(func, x0, y0, z1, xlen, ylen, zlen, 'N')
    value += (1 - dx) * dy * (1 - dz) * getValue3d(func, x0, y1, z0, xlen, ylen, zlen, 'N')
    value += (1 - dx) * dy * dz * getValue3d(func, x0, y1, z1, xlen, ylen, zlen, 'N')
    value += dx * (1 - dy) * (1 - dz) * getValue3d(func, x1, y0, z0, xlen, ylen, zlen, 'N')
    value += dx * (1 - dy) * dz * getValue3d(func, x1, y0, z1, xlen, ylen, zlen, 'N')
    value += dx * dy * (1 - dz) * getValue3d(func, x1, y1, z0, xlen, ylen, zlen, 'N')
    value += dx * dy * dz * getValue3d(func, x1, y1, z1, xlen, ylen, zlen, 'N')

    return value

cdef inline double getValue3d(double* func, int x, int y, int z, int xlen, int ylen, int zlen, char mode='N'):
    # if -1 < x < xlen and -1 < y < ylen and -1 < z < zlen:
    #     return func[(x * ylen + y) * zlen + z]
    # else:
    #     return 0

    if mode == 'C':
        if x < 0 or x > xlen - 1 or y < 0 or y > ylen - 1 or z < 0 or z > zlen - 1:
            return 0
    elif mode == 'N':
        if x < 0:
            x = 0
        elif x > xlen - 1:
            x = xlen - 1

        if y < 0:
            y = 0
        elif y > ylen - 1:
            y = ylen - 1

        if z < 0:
            z = 0
        elif z > zlen - 1:
            z = zlen - 1

    return func[(x * ylen + y) * zlen + z]
