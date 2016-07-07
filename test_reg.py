import rtk


def test():
    fixed_img_file = 'data/square.nii'
    moving_img_file = 'data/circle.nii'

    fixed_img = rtk.image.ScalarImage(filename=fixed_img_file)
    moving_img = rtk.image.ScalarImage(filename=moving_img_file)

    reg = rtk.registration.LDDMM(
        ndim=fixed_img.ndim,
        n_step=32,
        penalty=1000,
        regularizer=rtk.regularizer.BiharmonicRegularizer(convexity_penalty=1., norm_penalty=1.),
        similarity='ssd',
        n_iters=(50, 20, 10),
        resolutions=(4, 2, 1),
        smoothing_sigmas=(2, 1, 0),
        delta_phi_threshold=1.,
        unit_threshold=0.1,
        learning_rate=0.1,
        parallel=False)
    reg.set_images(fixed_img, moving_img)
    reg.execute()

if __name__ == '__main__':
    test()
