"""
Convert a 3D image to 2D images. (iso)
"""

import skimage.io as io
import numpy as np
from utils_data import read_txt
import os, tqdm

path_txt = "3DSIM\\transformed\FixedPFA_GAL\\test.txt"
path_image = "3DSIM\\transformed\FixedPFA_GAL\\test\channel_0\\3DWF"
# ------------------------------------------------------------------------------
print("-" * 80)
print(path_image)
# save to
path_save = os.path.join(path_image + "_maxproj")
os.makedirs(path_save, exist_ok=True)

# load all file names in the folder
files = read_txt(path_txt)
print("Number of files:", len(files))

pbar = tqdm.tqdm(total=len(files), desc="maxproj", ncols=100)
for name in files:
    img = io.imread(os.path.join(path_image, name))
    img = img.astype(np.float32)
    img2d = img.max(axis=0)[None]
    io.imsave(fname=os.path.join(path_save, name), arr=img2d, check_contrast=False)
    pbar.update(1)
pbar.close()
