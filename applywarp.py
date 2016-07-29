import rtk


def apply_warp(moving_img_file,
               fixed_img_file,
               transformation_file,
               output_img_file,
               order):
    """
    apply transform and save the output image

    Parameters
    ----------
    moving_img_file : str
        file name of input image
    fixed_img_file : str
        file name of reference image
    transformation_file : str
        file name of transformation to apply
    output_img_file : str
        file name of warped input image
    order : int
        order of interpolation
    """
    moving_img = rtk.load(filename=moving_img_file, dtype='scalarimage')
    fixed_img = rtk.load(filename=fixed_img_file, dtype='scalarimage')
    transform = rtk.load(filename=transformation_file, dtype='deformation')

    warped_img = moving_img.apply_transform(transform, order=order)
    warped_img.affine = fixed_img.get_affine()
    warped_img.save(filename=output_img_file)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='applying transformation to an input image')
    parser.add_argument('-i', '--input',
                        type=str,
                        help="""
moving image file\n
                        """)
    parser.add_argument('-r', '--reference',
                        type=str,
                        help="""
reference image file\n
                        """)
    parser.add_argument('-t', '--transformation',
                        type=str,
                        help="""
transformation file\n
                        """)
    parser.add_argument('-o', '--output',
                        type=str,
                        help="""
output file name\n
                        """)
    parser.add_argument('--order',
                        type=int,
                        default=1,
                        help="""
order of interpolation\n
                        """)

    args = parser.parse_args()
    print "warp to apply:", args.transformation
    print "apply warp to this image:", args.input
    apply_warp(args.input,
               args.reference,
               args.transformation,
               args.output,
               args.order)
