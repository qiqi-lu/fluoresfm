"""
The Ground truth image may have some noise.
Use denoising algorithm to remove the noise.
"""

import os, tqdm
import numpy as np
from skimage import io
from skimage.restoration import denoise_nl_means, estimate_sigma
from utils.data import win2linux, rolling_ball_approximation

path_images = (
    # "E:\qiqilu\datasets\BioTISR\\transformed\CCPs\\test\channel_0\SIM_2d",
    # "E:\qiqilu\datasets\BioTISR\\transformed\F-actin_nonlinear\\test\channel_0\SIM_2d",
    # "E:\qiqilu\datasets\VMSIM\\transformed\SFig7_488\\test\channel_0\SIM_p512_s256_2d",
    # "E:\qiqilu\datasets\VMSIM\\transformed\SFig7_568\\test\channel_0\SIM_p512_s256_2d",
    # "E:\qiqilu\datasets\VMSIM\\transformed\SFig7_647\\test\channel_0\SIM_p512_s256_2d",
    # "E:\qiqilu\datasets\SIMActin\\transformed\\test\channel_0\SIM_2d_proj_p1024_s420_2d",
    # "E:\qiqilu\datasets\BioTISR\\transformed\Lysosomes\\test\channel_0\SIM_2d",
    "E:\qiqilu\datasets\BioTISR\\transformed\CCPs\\train\channel_0\SIM_2d",
)

path_images = win2linux(path_images[0])
path_save_to = path_images + "-dn-rb"
os.makedirs(path_save_to, exist_ok=True)

# get all the tif image  from the folder
image_names = [f for f in os.listdir(path_images) if f.endswith(".tif")]
# sort the image
image_names.sort()
print("Number of images:", len(image_names))

radius, sf = 5, 4
patch_kw = dict(patch_size=5, patch_distance=6)  # 5x5 patches  # 13x13 search area
hw = 5  # CCPs


# for each image, read the image and perform denoising
pbar = tqdm.tqdm(total=len(image_names), desc="Denoising")
for i, image_name in enumerate(image_names):
    image = io.imread(os.path.join(path_images, image_name))[0]
    image = np.clip(image, 0, None)
    sigma_est = np.mean(estimate_sigma(image))
    image_denoised = denoise_nl_means(
        image, h=hw * sigma_est, fast_mode=True, **patch_kw
    )
    image_denoised = np.clip(image_denoised, 0, None)

    # remove background noise
    image_denoised_rb, bg = rolling_ball_approximation(
        image_denoised, radius=radius, sf=sf
    )
    image_denoised_rb = np.clip(image_denoised_rb, 0, None)

    # save the image_denoised to the folder path_save_to
    io.imsave(
        os.path.join(path_save_to, image_name), image_denoised_rb, check_contrast=False
    )
    pbar.update(1)
pbar.close()
