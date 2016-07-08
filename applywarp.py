from image import ScalarImage
from deformation import Deformation

def apply_transform(moving_img_file, transformation_file, output_img_file, fixed_img_file=None):
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

    parser = argparse.ArgumentParser(description='applying transformation to an input image')
    parser.add_argument('--input', '-i', type=str, help='moving image file')
    parser.add_argument('--transformation', '-t', type=str, help='transformation file')
    parser.add_argument('--output', '-o', type=str, help='output file without extension')

    args = parser.parse_args()
    apply_transform(args.input, args.transformation, args.output)
