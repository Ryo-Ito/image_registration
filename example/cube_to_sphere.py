import rtk


fixed_img = rtk.load_img("data/sphere.nii")
moving_img = rtk.load_img("data/cube.nii")
similarity = rtk.similarity.SSD(1000.)
regularizer = rtk.regularizer.BiharmonicRegularizer(convexity_penalty=1., norm_penalty=1.)
model = rtk.registration.LDDMM(
    n_step=32,
    regularizer=regularizer,
    similarity=similarity,
    n_iters=(50, 20, 10),
    resolution=(4, 2, 1),
    smoothing_sigmas=(2, 1, 0),
    delta_phi_threshold=1.,
    unit_threshold=0.1,
    learning_rate=0.1,
    n_jobs=1)
model.set_images(fixed_img, moving_img)
warp = model.execute()
warped_img = moving_img.apply_transform(warp)
rtk.save(warped_img, "cube_to_sphere.nii")
