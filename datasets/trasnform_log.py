"""
Log trasnform of the images.
"""

import os, tqdm
import skimage.io as io
import numpy as np
from utils_data import normalization

path_img_folder = "RCAN3D\\transformed\DN_Tomm20_Mito\\test\channel_0\GT_2d"
path_save_to = path_img_folder + "_log"
os.makedirs(path_save_to, exist_ok=True)

# get all the tif file in the folder
filenames = os.listdir(path_img_folder)
filenames = [f for f in filenames if f.endswith(".tif")]
filenmaes = sorted(filenames)

norm = lambda x: normalization(x, 0.03, 0.995)[0]

pbar = tqdm.tqdm(total=len(filenames), desc="Processing")
for filename in filenames:
    path_img = os.path.join(path_img_folder, filename)
    img = io.imread(path_img)
    img = np.clip(img, 0, None)
    img = norm(img)
    img_log = np.log(img + 1)
    # save
    path_save = os.path.join(path_save_to, filename)
    io.imsave(path_save, img_log, check_contrast=False)
    pbar.update(1)
pbar.close()
