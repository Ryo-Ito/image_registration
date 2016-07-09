from image import ScalarImage
from deformation import Deformation


def apply_warp(moving_img_file,
               transformation_file,
               output_img_file,
               fixed_img_file=None):
    """
    apply transform and save the output image

    Parameters
    ----------
    moving_img_file : str
        file name of input image
    fixed_img_file : str
        file name of template image
    transformation_file : str
        file name of transformation applying
    output_img_file : str
        file name of warped input image
    """
    moving_img = ScalarImage(filename=moving_img_file)
    fixed_img = ScalarImage(filename=fixed_img_file)
    transform = Deformation(filename=transformation_file)

    warped_img = moving_img.apply_transform(transform)

    if fixed_img_file is not None:
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
    parser.add_argument('-t', '--transformation',
                        type=str,
                        help="""
transformation file\n
                        """)
    parser.add_argument('-o', '--output',
                        type=str,
                        help="""
output file without extension\n
                        """)

    args = parser.parse_args()
    apply_warp(args.input, args.transformation, args.output)
