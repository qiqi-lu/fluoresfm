"""
Convert a 3D image to 2D images. (iso)
"""

import skimage.io as io
import numpy as np
from utils_data import read_txt
import os, tqdm

path_txt = "Scaffold-A549\\transformed\\test.txt"
path_image = "Scaffold-A549\\transformed\\test\channel_0\masks"

# ------------------------------------------------------------------------------
print("-" * 80)
print(path_image)
# save to
path_save = os.path.join(path_image + "_2d")
os.makedirs(path_save, exist_ok=True)

# load all file names in the folder
files = read_txt(path_txt)
print("Number of files:", len(files))

pbar = tqdm.tqdm(total=len(files), desc="3to2D", ncols=100)
for name in files:
    img = io.imread(os.path.join(path_image, name))
    img = img.astype(np.float32)
    # print(img.shape)
    num_slice = img.shape[0]

    pbar_single = tqdm.tqdm(total=num_slice, desc=name, ncols=100, leave=False)
    for i_slice in range(num_slice):
        img2d = img[i_slice][None]
        io.imsave(
            fname=os.path.join(path_save, name.split(".")[0] + f"_{i_slice}.tif"),
            arr=img2d,
            check_contrast=False,
        )
        pbar_single.update(1)
    pbar_single.close()
    pbar.update(1)
pbar.close()
