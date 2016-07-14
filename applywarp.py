import rtk


def apply_warp(moving_img_file,
               transformation_file,
               output_img_file,
               order):
    """
    apply transform and save the output image

    Parameters
    ----------
    moving_img_file : str
        file name of input image
    transformation_file : str
        file name of transformation applying
    output_img_file : str
        file name of warped input image
    order : int
        order of interpolation
    """
    moving_img = rtk.image.ScalarImage(filename=moving_img_file)
    transform = rtk.deformation.Deformation(filename=transformation_file)

    warped_img = moving_img.apply_transform(transform, order=order)

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
    apply_warp(args.input, args.transformation, args.output, args.order)
