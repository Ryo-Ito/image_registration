#!/usr/bin/env python3
import rtk
import numpy as np


# fixed_img = rtk.ScalarImage(data=fixed_image) # isinstance(fixed_image, np.ndarray) = True
# moving_img = rtk.ScalarImage(data=moving_image) # isinstance(moving_image, np.ndarray) = True

fixed_img = rtk.load_img("data/sphere.nii")
moving_img = rtk.load_img("data/cube.nii")
similarity = rtk.similarity.SSD(1000.)
regularizer = rtk.regularizer.BiharmonicRegularizer(convexity_penalty=1., norm_penalty=1.)
model = rtk.registration.LDDMM(
    n_step=32,
    regularizer=regularizer,
    similarity=similarity,
    n_iters=(50, 20, 10),
    resolutions=(4, 2, 1),
    smoothing_sigmas=(2, 1, 0),
    delta_phi_threshold=1.,
    unit_threshold=0.1,
    learning_rate=0.1,
    n_jobs=4)
model.set_images(fixed_img, moving_img)
warp = model.execute()
warped_img = moving_img.apply_transform(warp)

print('model', type(model))
print(dir(model))
print('warp', type(warp))
print(dir(warp))
print('warp grid', type(warp.grid))
np.save('warp_grid.npy', warp.grid)
print('warped_img', type(warped_img))
print(dir(warped_img))
rtk.save(warped_img, "cube_to_sphere.nii")
