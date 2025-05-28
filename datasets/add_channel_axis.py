"""
Add additional channel axis as the first axis.
"""

import skimage.io as io
import numpy as np
import os, tqdm

path_image = "DeepBacs\\transformed\Seg_Saureus\\test\FM\\masks"

# ------------------------------------------------------------------------------
print("-" * 100)
# get the list of files in the path_image, and filter out the non-tif files
print(path_image)
list_image = os.listdir(path_image)
list_image = [
    x for x in list_image if x.endswith(".tif")
]  # filter out the non-tif files
print(len(list_image))

pbar = tqdm.tqdm(total=len(list_image), desc="ADD AXIS", ncols=100)
for name in list_image:
    img = io.imread(os.path.join(path_image, name))
    img = img.astype(np.float32)
    if len(img.shape) == 2:
        img = img[None]
    io.imsave(fname=os.path.join(path_image, name), arr=img, check_contrast=False)
    pbar.update(1)
pbar.close()
