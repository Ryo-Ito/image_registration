from os.path import expanduser, join
import numpy as np
from registration import Registration
from images import ScalarImage
from deformations import LDDMM

def test1():
    home = expanduser('~')
    dname = join(home, 'registration/img/artificial_data_2d')
    fixed_img_file = join(dname, 'square.nii')
    moving_img_file = join(dname, 'circle.nii')

    fixed_img = ScalarImage(filename=fixed_img_file)
    moving_img = ScalarImage(filename=moving_img_file)

    deformation = LDDMM(ndim=fixed_img.ndim, deformation_step=32, penalty=1000, time_interval=1.)
    deformation.set_prior_parameter(alpha=1.)
    deformation.set_similarity_metric('ssd')
    reg = Registration(energy_threshold=0.001, unit_threshold=0.1)
    reg.set_deformation(deformation)
    reg.set_maximum_iterations((50,20,10))
    reg.set_resolution_level((4,2,1))
    reg.set_smoothing_sigma((2,1,0))
    reg.set_images(fixed_img, moving_img)
    transform = reg.execute()

    transform.save(filename=join(dname, 'warp.nii'))

    transform.show(interval=2)
    moving_img.show()
    fixed_img.show()
    warped_img = moving_img.apply_transform(transform)
    warped_img.show()

def test2():
    home = expanduser('~')
    dname = join(home, 'registration/img/brain_slice_2d')
    fixed_img_file = join(dname, 'warped_original.nii')
    moving_img_file = join(dname, 'original.nii')

    fixed_img = ScalarImage(filename=fixed_img_file)
    moving_img = ScalarImage(filename=moving_img_file)

    deformation = LDDMM(ndim=fixed_img.ndim, deformation_step=32, penalty=0.0001, time_interval=1.)
    deformation.set_prior_parameter(alpha=4.)
    deformation.set_similarity_metric('cc')
    reg = Registration(energy_threshold=0., unit_threshold=0.2, learning_rate=0.003)
    reg.set_deformation(deformation)
    reg.set_maximum_iterations((50,20,10))
    reg.set_resolution_level((4,2,1))
    reg.set_smoothing_sigma((2,1,0))
    reg.set_images(fixed_img, moving_img)
    transform = reg.execute()

    transform.save(join(dname, 'LDDMM_deformation.nii'))

    warped_img = moving_img.apply_transform(transform)
    warped_img.save(join(dname, 'LDDMM_warped.nii'))

def test3():
    home = expanduser('~')
    dname = join(home, 'registration/img/IBSR/from02to01')
    fixed_img_file = join(dname, 'fixed_image/IBSR_01.nii')
    moving_img_file = join(dname, 'moving_image/IBSR_02_affine.nii.gz')

    fixed_img = ScalarImage(filename=fixed_img_file)
    moving_img = ScalarImage(filename=moving_img_file)

    deformation = LDDMM(ndim=fixed_img.ndim, deformation_step=32, penalty=0.0001, time_interval=1.)
    deformation.set_prior_parameter(alpha=2.)
    deformation.set_similarity_metric('cc')

    reg = Registration(energy_threshold=0.0001, unit_threshold=0.2, learning_rate=0.001)
    reg.set_deformation(deformation)
    reg.set_maximum_iterations((500,200,100))
    reg.set_resolution_level((4,2,1))
    reg.set_smoothing_sigma((2,1,1))
    reg.set_images(fixed_img, moving_img)
    transform = reg.execute()

    transform.save(filename=join(dname, 'LDDMM/from02to01warp.nii'), affine=fixed_img.get_affine())
    warped_img = moving_img.apply_transform(transform)
    warped_img.save(filename=join(dname, 'LDDMM/02warpedto01.nii'), affine=fixed_img.get_affine())

def test4():
    home = expanduser('~')
    dname = join(home, 'registration/img/IBSR/from02to01')
    fixed_img_file = join(dname, 'fixed_image/IBSR_01.nii')
    moving_img_file = join(dname, 'moving_image/IBSR_02_affine.nii.gz')

    fixed_img = ScalarImage(filename=fixed_img_file)
    moving_img = ScalarImage(filename=moving_img_file)
    metric_matrix = np.identity(125) - np.ones((125,125)) / 125

    deformation = LDDMM(ndim=fixed_img.ndim, deformation_step=32, penalty=0.0001, time_interval=1.)
    deformation.set_prior_parameter(alpha=2.)
    deformation.set_similarity_metric('mc', metric_matrix)

    reg = Registration(energy_threshold=0.0001, unit_threshold=0.2, learning_rate=0.001)
    reg.set_deformation(deformation)
    reg.set_maximum_iterations((500,200,100))
    reg.set_resolution_level((4,2,1))
    reg.set_smoothing_sigma((2,1,1))
    reg.set_images(fixed_img, moving_img)
    transform = reg.execute()

    transform.save(filename=join(dname, 'LDDMM/from02to01warp_mc.nii'), affine=fixed_img.get_affine())
    warped_img = moving_img.apply_transform(transform)
    warped_img.save(filename=join(dname, 'LDDMM/IBSR_02_warped_mc.nii'), affine=fixed_img.get_affine())

if __name__ == '__main__':
    test4()
