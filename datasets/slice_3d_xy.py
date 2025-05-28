"""
Convert a 3D image to 2D images.
"""

import skimage.io as io
import numpy as np
from utils_data import read_txt
import os, tqdm

path_images = [
    "GranulocytesSeg\\transformed\high_noise\\test\channel_0\image",
    "GranulocytesSeg\\transformed\high_noise\\test\channel_0\label",
    "GranulocytesSeg\\transformed\high_noise\\train\channel_0\image",
    "GranulocytesSeg\\transformed\high_noise\\train\channel_0\label",
    "GranulocytesSeg\\transformed\low_noise\\test\channel_0\image",
    "GranulocytesSeg\\transformed\low_noise\\test\channel_0\label",
    "GranulocytesSeg\\transformed\low_noise\\train\channel_0\image",
    "GranulocytesSeg\\transformed\low_noise\\train\channel_0\label",
]

# ------------------------------------------------------------------------------
for path_image in path_images:
    print("-" * 50)
    print(path_image)
    # save to
    path_save = os.path.join(path_image + "_2d")
    os.makedirs(path_save, exist_ok=True)

    # load all file names in the folder
    files = os.listdir(path_image)
    print("Number of files:", len(files))
    # only process tif files
    files = [x for x in files if ".tif" in x]

    pbar = tqdm.tqdm(total=len(files), desc="3to2D", ncols=50)
    for name in files:
        img = io.imread(os.path.join(path_image, name)).astype(np.float32)
        img = np.clip(img, a_min=0.0, a_max=None)
        # print(img.shape)
        num_slice = img.shape[0]

        pbar_single = tqdm.tqdm(total=num_slice, desc=name, ncols=50, leave=False)
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
