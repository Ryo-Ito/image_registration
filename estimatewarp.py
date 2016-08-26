import argparse
import numpy as np
import rtk


def main():
    parser = argparse.ArgumentParser(description="""
estimating transformation from a moivng image to a fixed image
        """, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-m', '--moving',
                        type=str,
                        help="""
moving image file\n
                        """)
    parser.add_argument('-f', '--fixed',
                        type=str,
                        help="""
fixed image file\n
                        """)
    parser.add_argument('-s', '--similarity_metric',
                        type=str,
                        choices=['ssd', 'zncc', 'mncc', 'kncc', 'gncc'],
                        default='ssd',
                        help="""
similarity metric to evaluate how similar two images are.
Choose one of the following similarity metric
    ssd: Sum of Squared Difference
    zncc: zero means normalized cross correlation
    mncc: mahalanobis cosine similarity
    kncc: kernel normalized cross correlation
    gncc: gaussian normalized cross correlation
Default: ssd\n
                        """)
    parser.add_argument('--window_length',
                        type=int,
                        default=5,
                        help="""
length of window when calculating cross correlation
Default: 5\n
                        """)
    parser.add_argument('--metric_matrix',
                        type=str,
                        help="""
file containing metric matrix for mncc\n
                        """)
    parser.add_argument('--convolution_kernel',
                        type=str,
                        help="""
file containing convolution kernel for kncc\n
                        """)
    parser.add_argument('--filter_sigma',
                        type=float,
                        default=1.,
                        help="""
standard deviation of gaussian function for gncc
Default: 1.\n
                        """)
    parser.add_argument('-t', '--transformation',
                        choices=['LDDMM', 'SyN'],
                        default='LDDMM',
                        type=str,
                        help="""
transformation type
Default: LDDMM\n
                        """)
    parser.add_argument('-o', '--output',
                        default='output_warp.nii.gz',
                        type=str,
                        help="""
output transformation file
Default: output_warp.nii.gz\n
                        """)
    parser.add_argument('--deformation_step',
                        default=32,
                        type=int,
                        help="""
number of steps to deform images
Default: 32\n
                        """)
    parser.add_argument('--time_interval',
                        default=1.,
                        type=float,
                        help="""
length of time interval of transformation
Default: 1.\n
                        """)
    parser.add_argument('--penalty',
                        type=float,
                        help="""
penalty coefficient for vector field
Default: 1000(ssd), 0.0001(zncc)\n
                        """)
    parser.add_argument('-r', '--regularizer',
                        choices=['biharmonic', 'gaussian'],
                        default='biharmonic',
                        type=str,
                        help="""
regularation of vector field
Default: biharmonic\n
                        """)
    parser.add_argument('--convexity_penalty',
                        default=1.,
                        type=float,
                        help="""
penalty coefficient for convexity of vector field
Default: 1.\n
                        """)
    parser.add_argument('--norm_penalty',
                        default=1.,
                        type=float,
                        help="""
penalty coefficient for norm of vector
Default: 1.\n
                        """)
    parser.add_argument('--gaussian_vector_smooth',
                        default=1.,
                        type=float,
                        help="""
gaussian smoothing of vector fields
Default: 1.\n
                        """)
    parser.add_argument('--delta_phi_threshold',
                        default=1.,
                        type=float,
                        help="""
threshold value for maximum update of displacement
Default: 1\n
                        """)
    parser.add_argument('--unit_threshold',
                        default=0.2,
                        type=float,
                        help="""
threshold value of jacobian determinant of mapping function
Default: 0.2\n
                        """)
    parser.add_argument('--learning_rate',
                        type=float,
                        help="""
learning rate of updating estimate of vector field
Default: 0.1(ssd), 0.01(zncc)\n
                        """)
    parser.add_argument('--maximum_iterations',
                        default=[50, 20, 10],
                        type=int,
                        nargs='*',
                        action='store',
                        help="""
maximum number of updating estimate of vector field
Default: [50, 20, 10]\n
                        """)
    parser.add_argument('--resolution_level',
                        default=[4, 2, 1],
                        type=int,
                        nargs='*',
                        action='store',
                        help="""
resolution at each level
Default: [4, 2, 1]\n
                        """)
    parser.add_argument('--smoothing_sigma',
                        type=int,
                        nargs='*',
                        action='store',
                        help="""
values of smoothing sigma at each level
Default: [2, 1, 0](ssd), [2, 1, 1](zncc)\n
                        """)
    parser.add_argument('--n_jobs',
                        type=int,
                        default=1,
                        help="""
number of cpu cores to use
Default: 1
                        """)

    args = parser.parse_args()

    if args.similarity_metric == 'ssd':
        if args.penalty is None:
            args.penalty = 1000
        if args.learning_rate is None:
            args.learning_rate = 0.1
        if args.smoothing_sigma is None:
            args.smoothing_sigma = [2, 1, 0]
    if args.similarity_metric == 'zncc':
        if args.penalty is None:
            args.penalty = 0.0001
        if args.learning_rate is None:
            args.learning_rate = 0.01
        if args.smoothing_sigma is None:
            args.smoothing_sigma = [2, 1, 1]
    elif args.similarity_metric == 'mncc':
        if args.metric_matrix is None:
            args.similarity_metric == 'zncc'
        if args.penalty is None:
            args.penalty = 0.0001
        if args.learning_rate is None:
            args.learning_rate = 0.01
        if args.smoothing_sigma is None:
            args.smoothing_sigma = [2, 1, 1]
    elif args.similarity_metric == 'gncc':
        if args.penalty is None:
            args.penalty = 0.01
        if args.learning_rate is None:
            args.learning_rate = 0.01
        if args.smoothing_sigma is None:
            args.smoothing_sigma = [2, 1, 1]

    fixed = rtk.load(filename=args.fixed, dtype='scalarimage')
    moving = rtk.load(filename=args.moving, dtype='scalarimage')

    if args.similarity_metric == 'ssd':
        similarity = rtk.similarity.SSD(args.penalty)
    elif args.similarity_metric == 'zncc':
        similarity = rtk.similarity.ZNCC(
            args.penalty, args.window_length, args.window_length ** fixed.ndim)
    elif args.similarity_metric == 'mncc':
        similarity = rtk.similarity.MNCC(
            args.penalty, np.load(args.metric_matrix))
    elif args.similarity_metric == 'kncc':
        similarity = rtk.similarity.KNCC(
            args.penalty, np.load(args.convolution_kernel))
    elif args.similarity_metric == 'gncc':
        similarity = rtk.similarity.GNCC(
            args.penalty, args.filter_sigma)

    if args.regularizer == 'biharmonic':
        regularizer = rtk.regularizer.BiharmonicRegularizer(
            args.convexity_penalty, args.norm_penalty)
    elif args.regularizer == 'gaussian':
        regularizer = rtk.regularizer.GaussianRegularizer(
            args.gaussian_vector_smooth)

    if args.transformation == 'LDDMM':
        import rtk.registration.LDDMM as Registration
    elif args.transformation == 'SyN':
        import rtk.registration.SyN as Registration

    reg = Registration(
        n_step=args.deformation_step,
        regularizer=regularizer,
        similarity=similarity,
        n_iters=args.maximum_iterations,
        resolutions=args.resolution_level,
        smoothing_sigmas=args.smoothing_sigma,
        delta_phi_threshold=args.delta_phi_threshold,
        unit_threshold=args.unit_threshold,
        learning_rate=args.learning_rate,
        n_jobs=args.n_jobs)
    print "fixed image:", args.fixed
    print "moving image:", args.moving
    reg.print_settings()
    reg.set_images(fixed, moving)
    warp = reg.execute()
    rtk.save(warp, filename=args.output, affine=fixed.get_affine())


if __name__ == '__main__':
    main()
