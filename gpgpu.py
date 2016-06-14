import pyopencl
from pyopencl import mem_flags
import numpy as np

source_code = '''
__kernel void sliding_dot2d(
    __global const float *img,
    __global const float *matrix,
    __global float *product,
    const int xlen,
    const int ylen,
    const int mat_len,
    const int r
)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int column_index = get_global_id(2);
    const int index = (x * ylen + y) * mat_len + column_index;
    int xi, yj;

    product[index] = 0.;
    int row_index = 0;
    for(int i = - r; i < r + 1; i++){
        for(int j = -r; j < r + 1; j++){
            xi = x + i;
            yj = y + j;
            if(xi > -1 && xi < xlen && yj > -1 && yj < ylen){
                product[index] += img[xi * ylen + yj] * matrix[column_index * mat_len + row_index];
            }
            row_index += 1;
        }
    }
}

__kernel void sliding_dot3d(
    __global const float *img,
    __global const float *matrix,
    __global float *product,
    const int xlen,
    const int ylen,
    const int zlen,
    const int mat_len,
    const int r
)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);
    int index = ((x * ylen + y) * zlen + z) * mat_len;
    int xi, yj, zk, row_index;

    for(int col_index = 0; col_index < mat_len; col_index++){
        product[index] = 0;
        row_index = 0;
        for(int i = -r; i < r + 1; i++){
            for(int j = -r; j < r + 1; j++){
                for(int k = -r; k < r + 1; k++){
                    xi = x + i;
                    yj = y + j;
                    zk = z + k;
                    if(xi > -1 && xi < xlen && yj > -1 && yj < ylen && zk > -1 && zk < zlen){
                        product[index] += img[(xi * ylen + yj) * zlen + zk] * matrix[col_index * mat_len + row_index];
                    }
                    row_index += 1;
                }
            }
        }
        index += 1;
    }
}
'''

context = pyopencl.create_some_context(interactive=False)
queue = pyopencl.CommandQueue(context)

program = pyopencl.Program(context, source_code).build()

def gradient(image):
    if image.ndim == 2:
        return grad2d(image)
    elif image.ndim == 3:
        return grad3d(image)

def grad2d(image):
    image = image.astype(np.float32)
    gradx = np.zeros_like(image)
    grady = np.zeros_like(image)
    context = pyopencl.create_some_context(interactive=False)
    queue = pyopencl.CommandQueue(context)
    image_buffer = pyopencl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=image)
    gradx_buffer = pyopencl.Buffer(context, mem_flags.WRITE_ONLY, gradx.nbytes)
    grady_buffer = pyopencl.Buffer(context, mem_flags.WRITE_ONLY, grady.nbytes)

    program = pyopencl.Program(context, '''
        __kernel void grad(
            __global const float *image,
            __global float *gradx,
            __global float *grady,
            const int xlen,
            const int ylen)
            {
                const int x = get_global_id(0);
                const int y = get_global_id(1);
                const int index = x * ylen + y;

                if(x == 0){
                    gradx[index] = image[ylen+y] - image[y];
                }else if(x == xlen-1){
                    gradx[index] = image[x * ylen + y] - image[(x - 1) * ylen + y];
                }else{
                    gradx[index] = 0.5 * (image[(x+1)*ylen+y] - image[(x-1)*ylen+y]);
                }
                if(y == 0){
                    grady[index] = image[x * ylen + 1] - image[x * ylen];
                }else if(y == ylen - 1){
                    grady[index] = image[x * ylen + y] - image[x * ylen + y-1];
                }else{
                    grady[index] = 0.5 * (image[x * ylen + y + 1] - image[x * ylen + y - 1]);
                }
            }
        ''').build()

    xlen = np.int32(image.shape[0])
    ylen = np.int32(image.shape[1])

    e = program.grad(queue, image.shape, None, image_buffer, gradx_buffer, grady_buffer, xlen, ylen)
    e.wait()

    pyopencl.enqueue_copy(queue, gradx, gradx_buffer)
    pyopencl.enqueue_copy(queue, grady, grady_buffer)

    return gradx, grady

def grad3d(image):
    image = image.astype(np.float32)
    gradx = np.zeros_like(image)
    grady = np.zeros_like(image)
    gradz = np.zeros_like(image)
    context = pyopencl.create_some_context(interactive=False)
    queue = pyopencl.CommandQueue(context)
    image_buffer = pyopencl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=image)
    gradx_buffer = pyopencl.Buffer(context, mem_flags.WRITE_ONLY, gradx.nbytes)
    grady_buffer = pyopencl.Buffer(context, mem_flags.WRITE_ONLY, grady.nbytes)
    gradz_buffer = pyopencl.Buffer(context, mem_flags.WRITE_ONLY, gradz.nbytes)

    program = pyopencl.Program(context, '''
        __kernel void grad(
            __global const float *image,
            __global float *gradx,
            __global float *grady,
            __global float *gradz,
            const int xlen,
            const int ylen,
            const int zlen)
            {
                const int x = get_global_id(2);
                const int y = get_global_id(1);
                const int z = get_global_id(0);
                const int result_index = (x * ylen + y) * zlen + z;

                if(x == 0){
                    gradx[result_index] = image[(ylen+y)*zlen+z] - image[y*zlen+z];
                }else if(x == xlen - 1){
                    gradx[result_index] = image[(x * ylen + y) * zlen + z] - image[((x - 1) * ylen + y) * zlen + z];
                }else{
                    gradx[result_index] = 0.5 * (image[((x+1)*ylen+y)*zlen+z] - image[((x-1)*ylen+y)*zlen+z]);
                }
                if(y == 0){
                    grady[result_index] = image[(x * ylen + 1) * zlen + z] - image[x * ylen * zlen + z];
                }else if(y == ylen - 1){
                    grady[result_index] = image[(x * ylen + y) * zlen + z] - image[(x * ylen + y-1) * zlen + z];
                }else{
                    grady[result_index] = 0.5 * (image[(x * ylen + y + 1) * zlen + z] - image[(x * ylen + y - 1) * zlen + z]);
                }
                if(z == 0){
                    gradz[result_index] = image[(x * ylen + y) * zlen + 1] - image[(x * ylen + y) * zlen];
                }else if(z == zlen - 1){
                    gradz[result_index] = image[(x * ylen + y) * zlen + z] - image[(x * ylen + y) * zlen + z - 1];
                }else{
                    gradz[result_index] = 0.5 * (image[(x * ylen + y) * zlen + z + 1] - image[(x * ylen + y) * zlen + z - 1]);
                }
            }
        ''').build()

    xlen = np.int32(image.shape[0])
    ylen = np.int32(image.shape[1])
    zlen = np.int32(image.shape[2])

    e = program.grad(queue, image.shape, None, image_buffer, gradx_buffer, grady_buffer, gradz_buffer, xlen, ylen, zlen)
    e.wait()

    pyopencl.enqueue_copy(queue, gradx, gradx_buffer)
    pyopencl.enqueue_copy(queue, grady, grady_buffer)
    pyopencl.enqueue_copy(queue, gradz, gradz_buffer)

    return gradx, grady, gradz

def sliding_matrix_multiply(img, matrix):
    assert(matrix.shape[0] == matrix.shape[1])
    if img.ndim == 2:
        return sliding_matrix_multiply2d(img, matrix)
    elif img.ndim == 3:
        return sliding_matrix_multiply3d(img, matrix)

def sliding_matrix_multiply2d(img, matrix):
    img = img.astype(np.float32)
    matrix = matrix.astype(np.float32)

    mat_len = np.int32(len(matrix))
    window_len = int(np.round(mat_len ** 0.5))
    assert(window_len ** 2 == mat_len)
    radius = np.int32((window_len - 1) / 2)

    product = np.empty(shape=img.shape + (mat_len,), dtype=np.float32)

    context = pyopencl.create_some_context(interactive=False)
    queue = pyopencl.CommandQueue(context)

    program = pyopencl.Program(context, source_code).build()

    img_buffer = pyopencl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=img)
    matrix_buffer = pyopencl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=matrix)
    product_buffer = pyopencl.Buffer(context, mem_flags.WRITE_ONLY, product.nbytes)

    xlen = np.int32(img.shape[0])
    ylen = np.int32(img.shape[1])

    e = program.sliding_dot2d(queue, product.shape, None, img_buffer, matrix_buffer, product_buffer, xlen, ylen, mat_len, radius)
    e.wait()

    pyopencl.enqueue_copy(queue, product, product_buffer)

    return product

def sliding_matrix_multiply3d(img, matrix):
    img = img.astype(np.float32)
    matrix = matrix.astype(np.float32)

    xlen = np.int32(img.shape[0])
    ylen = np.int32(img.shape[1])
    zlen = np.int32(img.shape[2])
    mat_len = np.int32(len(matrix))
    window_len = int(np.round(mat_len ** 0.33333333333))
    assert(window_len ** 3 == mat_len)
    assert(window_len % 2 == 1)
    radius = np.int32((window_len - 1) / 2)

    product = np.empty(shape=(xlen, ylen, zlen, mat_len), dtype=np.float32)

    img_buffer = pyopencl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=img)
    matrix_buffer = pyopencl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=matrix)
    product_buffer = pyopencl.Buffer(context, mem_flags.WRITE_ONLY, product.nbytes)

    e = program.sliding_dot3d(queue, product.shape[:-1], None, img_buffer, matrix_buffer, product_buffer, xlen, ylen, zlen, mat_len, radius)
    e.wait()

    pyopencl.enqueue_copy(queue, product, product_buffer)

    return product

def multiply():
    import time
    for size in range(100,200, 300):
        print "matrix : ", size, " x ", size
        a = np.random.randint(0, 256, (size,size)).astype(np.int32)
        b = np.random.randint(0, 256, (size,size)).astype(np.int32)
        dest = np.empty_like(a)
     
        start = time.time()
        dest1 = np.dot(a, b)
        end = time.time()
        print "compute with CPU (np) : ", end - start  , " sec"
     
        dest1 = np.empty_like(a)
     
        context = pyopencl.create_some_context(interactive=False)
        queue = pyopencl.CommandQueue(context)
        a_buf = pyopencl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=a)
        b_buf = pyopencl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=b)
        dest_buf = pyopencl.Buffer(context, mem_flags.WRITE_ONLY, dest1.nbytes)

        print dest1.flags
     
        program = pyopencl.Program(context, '''
        __kernel void matrix_mul(
            __global const int* a,
            __global const int* b,
            __global int* dest,
            const int n
        )
        {
            const int i = get_global_id(0);
            const int j = get_global_id(1);
            const int dest_index = j * n + i;
     
            dest[dest_index] = 0;
            for(int k = 0; k < n; k++){
                dest[dest_index] += a[j * n + k] * b[k * n + i];
            }
        }
        ''').build()
     
        n = np.int32(size)
        start = time.time()
        e = program.matrix_mul(queue, a.shape, None, a_buf, b_buf, dest_buf, n)
        e.wait()
        stop = time.time()
     
        pyopencl.enqueue_copy(queue, dest1, dest_buf)
        print dest1[0:5,0:5]
        print "compute with GPGPU : ", stop - start, " sec"

def test_sliding_dot():
    from imageprocessing import sliding_matrix_product as smp
    from time import time
    I = np.random.randint(0,10,(100, 100)).astype(np.float)
    # print I[0:7, 0:7]
    matrix = np.identity(25)
    start = time()
    product_cyt = smp(I, matrix)
    print "cython", time() - start
    start = time()
    product_gpu = sliding_matrix_multiply(I, matrix)
    print "gpu", time() - start
    print np.allclose(product_gpu, product_cyt)
    # print "(0,0,0) component gpu", product_gpu[0,0,0]
    # print "(0,0,0) component cyt", product_cyt[0,0,0]
    # print "(1,2,3) component gpu", product_gpu[1,2,3]
    # print "(1,2,3) component cyt", product_cyt[1,2,3]
    # print "(1,2) component", product_gpu[1,2]
    # print "(3,4) component", product_gpu[3,4]

def test_gradient():
    from time import time
    from imageprocessing import gradient
    I = np.random.randint(0,10,(200,100,200)).astype(np.float)
    npgradI = np.asarray(np.gradient(I))
    start = time()
    cygradI = gradient(I)
    print time() - start
    start = time()
    gpugradI = np.asarray(grad3d(I))
    print time() - start
    print np.allclose(npgradI, cygradI)
    print np.allclose(cygradI, gpugradI)

def main():
    test_sliding_dot()

if __name__ == '__main__':
    main()