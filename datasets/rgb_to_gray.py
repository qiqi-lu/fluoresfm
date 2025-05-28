"""
Convert RGB images to grayscale images.
"""

import os, tqdm
import skimage.io as io
import numpy as np

path_images = "Cellpose\\transformed\mouse-cortical-neuron\\train\channel_0\images"

path_save_to = path_images + "_gray"
os.makedirs(path_save_to, exist_ok=True)

channel_sep = (0, 1)
for c in channel_sep:
    os.makedirs(path_images + f"_c{c}", exist_ok=True)

# get all the tif file in the folder
filenames = os.listdir(path_images)
filenames = [f for f in filenames if f.endswith(".tif") or f.endswith(".png")]
filenames = sorted(filenames)

pbar = tqdm.tqdm(total=len(filenames), desc="Processing")
for filename in filenames:
    img = io.imread(os.path.join(path_images, filename))
    img_gray = np.sum(img, axis=-1).astype(np.float32)
    # save
    io.imsave(
        fname=os.path.join(path_save_to, filename.split(".")[0] + ".tif"),
        arr=img_gray[None],
        check_contrast=False,
    )
    for c in channel_sep:
        img_c = img[..., c]  # select the c-th channel
        # save
        io.imsave(
            fname=os.path.join(path_images + f"_c{c}", filename.split(".")[0] + ".tif"),
            arr=img_c[None],
            check_contrast=False,
        )
    pbar.update(1)
