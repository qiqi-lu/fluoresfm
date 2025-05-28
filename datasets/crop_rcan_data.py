"""
The ground truth of RCAN dataset has a shape of (1024, 1023) / (1023, 1022).
Crop it to (1024, 1022) / (1022, 1022).
"""

import os, tqdm
import skimage.io as io

path_images = "RCAN3D\\transformed\C2S_MT\\test\channel_0\STED_2d"
path_save_to = path_images + "_crop"
os.makedirs(path_save_to, exist_ok=True)


# get all the tif file in the folder
filenames = os.listdir(path_images)
filenames = [f for f in filenames if f.endswith(".tif")]
filenames = sorted(filenames)

pbar = tqdm.tqdm(total=len(filenames), desc="Processing")
for filename in filenames:
    img = io.imread(os.path.join(path_images, filename))
    h, w = img.shape[1:]
    if h % 2 == 1:
        h -= 1
    if w % 2 == 1:
        w -= 1
    img = img[:, :h, :w]  # crop the image
    # save
    io.imsave(fname=os.path.join(path_save_to, filename), arr=img, check_contrast=False)
    pbar.update(1)
pbar.close()
