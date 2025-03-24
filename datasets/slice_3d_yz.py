"""
Convert a 3D image to 2D images.
"""

import skimage.io as io
import numpy as np
from utils_data import read_txt, interp_sf
import os
import tqdm

# path_txt = "CARE_Isotropic_Drosophila\\transformed\\test.txt"
# path_image = "CARE_Isotropic_Drosophila\\transformed\\test\channel_0\condition_0"
# scale_factor = 5

# path_txt = "CARE_Isotropic_Liver\\transformed\\test.txt"
# path_image = "CARE_Isotropic_Liver\\transformed\\test\channel_0\condition_0"
# path_image = "CARE_Isotropic_Liver\\transformed\\test\channel_0\gt"
# scale_factor = 1

path_txt = "CARE_Isotropic_Retina\\transformed\\test.txt"
path_image = "CARE_Isotropic_Retina\\transformed\\test\channel_0\condition_0"
# path_image = "CARE_Isotropic_Retina\\transformed\\test\channel_1\condition_0"
scale_factor = 10.2

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
    print(img.shape)
    num_slice = img.shape[-1]

    pbar_single = tqdm.tqdm(total=num_slice, desc=name, ncols=100, leave=False)
    for i_slice in range(num_slice):
        img2d = img[:, :, i_slice]
        img2d = interp_sf(img2d, sf=(scale_factor, 1), mode="bilinear")
        img2d = np.transpose(img2d, axes=(1, 0))
        img2d = img2d[None]
        io.imsave(
            fname=os.path.join(path_save, name.split(".")[0] + f"_{i_slice}.tif"),
            arr=img2d,
            check_contrast=False,
        )
        pbar_single.update(1)
    pbar_single.close()
    pbar.update(1)
pbar.close()
