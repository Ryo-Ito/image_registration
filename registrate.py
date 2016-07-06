import argparse
import rtk


def main():
    parser = argparse.ArgumentParser(description="""
        estimating transformation from a moivng image to a fixed image
        """,
        formatter_class=argparse.RawTextHelpFormatter)
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
                        choices=['ssd', 'zncc', 'mncc'],
                        default='zncc',
                        help="""
similarity metric to evaluate how similar two images are.
Choose one of the following similarity metric
    ssd: Sum of Squared Difference
    zncc: zero means normalized cross correlation
    mncc: mahalanobis cosine similarity
Default: cc\n
                        """)
    parser.add_argument('--window_length',
                        type=int,
                        help="""
length of window when calculating cross correlation
Default: None(ssd), 5(zncc)\n
                        """)
    parser.add_argument('--metric_matrix',
                        type=str,
                        help="""
file containing metric matrix for mncc
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
                        default=[4,2,1],
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
    parser.add_argument('--parallel',
                        dest='parallel',
                        action='store_true',
                        help="""
enable parallel computation\n
                        """)
    parser.add_argument('--no-parallel',
                        dest='parallel',
                        action='store_false',
                        help="""
disables parallel computation\n
                        """)
    parser.set_defaults(parallel=True)

    args = parser.parse_args()

    if args.similarity_metric == 'ssd':
        if args.penalty is None:
            args.penalty = 1000
        if args.learning_rate is None:
            args.learning_rate = 0.1
        if args.smoothing_sigma is None:
            args.smoothing_sigma = [2, 1, 0]
    if args.similarity_metric == 'zncc':
        if args.window_length is None:
            args.window_length = 5
        if args.penalty is None:
            args.penalty = 0.0001
        if args.learning_rate is None:
            args.learning_rate = 0.01
        if args.smoothing_sigma is None:
            args.smoothing_sigma = [2, 1, 1]
    elif args.similarity_metric == 'mncc':
        if args.metric_matrix is None:
            args.similarity_metric == 'zncc'
            args.window_length = 5
        if args.penalty is None:
            args.penalty = 0.0001
        if args.learning_rate is None:
            args.learning_rate = 0.01
        if args.smoothing_sigma is None:
            args.smoothing_sigma = [2, 1, 1]

if __name__ == '__main__':
    main()
