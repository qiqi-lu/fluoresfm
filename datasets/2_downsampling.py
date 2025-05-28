"""
Downsampling image by a scale factor using average pooling.
Only for 2D images with a shape of
- [B,C,H,W]
- [C,H,W]
- [H,W]
"""

import os, tqdm
import numpy as np
from skimage import io
from utils_data import ave_pooling_2d

path_imges = "BioTISR\\transformed\Mitochondria\\test\channel_0\SIM_2d"
scale_factor = 2

# ------------------------------------------------------------------------------
path_save_to = path_imges + "_ave" + str(scale_factor)
os.makedirs(path_save_to, exist_ok=True)

# get all the filenames of tif images in the folder
sample_names = os.listdir(path_imges)
sample_names = [x for x in sample_names if x.endswith(".tif")]
sample_names.sort()
num_samples = len(sample_names)

print("-" * 80)
print("load image from:", path_imges)
print("save image to:", path_save_to)
print("Numb of samples:", num_samples)

# ------------------------------------------------------------------------------
# downsampling by a scale factor using average pooling
pbar = tqdm.tqdm(total=num_samples, desc="Downsampling", ncols=80)
for i_sample in range(num_samples):
    # load image
    img = io.imread(os.path.join(path_imges, sample_names[i_sample]))
    img = img.astype(np.float32)

    # downsampling by a scale factor using average pooling
    img_ave = ave_pooling_2d(img, scale_factor=scale_factor)

    # save image
    io.imsave(
        os.path.join(path_save_to, sample_names[i_sample]),
        img_ave,
        check_contrast=False,
    )
    pbar.update(1)
pbar.close()
