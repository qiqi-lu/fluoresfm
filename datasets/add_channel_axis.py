"""
Add additional channel axis as the first axis.
"""

import skimage.io as io
import numpy as np
from utils_data import read_txt
import os, tqdm

# path_txt = "SIMMicrotubule\\transformed\\test_2d_proj.txt"
# path_image = "SIMMicrotubule\\transformed\\test\\channel_0\\SIM_2d_proj"

path_txt = "BIG-CElegans\\transformed\\test.txt"
path_image = "BIG-CElegans\\transformed\\test\FITC\WF"

# ------------------------------------------------------------------------------
print("-" * 100)
files = read_txt(path_txt)
print(len(files))

print(path_image)
pbar = tqdm.tqdm(total=len(files), desc="add axis", ncols=100)
for name in files:
    img = io.imread(os.path.join(path_image, name))
    img = img.astype(np.float32)
    if len(img.shape) == 2:
        img = img[None]
    io.imsave(fname=os.path.join(path_image, name), arr=img, check_contrast=False)
    pbar.update(1)
pbar.close()
