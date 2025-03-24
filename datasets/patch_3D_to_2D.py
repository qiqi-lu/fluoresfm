# transfor 3D pathces into 2D patches by treating each slice as a single 2D image
import numpy as np
import os
import skimage.io as io
import tqdm
from utils_data import read_txt


# ------------------------------------------------------------------------------
# read file names of all 3D patches
# filenmaes = read_txt("CARE_Denoising_Planaria\\transformed\\train.txt")
# path_patches_3d = "CARE_Denoising_Planaria\\transformed\\train\channel_0\condition_123"
# path_patches_3d = "CARE_Denoising_Planaria\\transformed\\train\channel_0\gt"

filenmaes = read_txt("CARE_Denoising_Tribolium\\transformed\\train.txt")
# path_patches_3d = "CARE_Denoising_Tribolium\\transformed\\train\channel_0\condition_123"
path_patches_3d = "CARE_Denoising_Tribolium\\transformed\\train\channel_0\gt"

path_patches_2d = path_patches_3d + "_patch_64_2d"
os.makedirs(path_patches_2d, exist_ok=True)

print("Number of 3D patches:", len(filenmaes))
pbar = tqdm.tqdm(total=len(filenmaes), desc="3D to 2 D")

for i in range(len(filenmaes)):
    patch_3d = io.imread(fname=os.path.join(path_patches_3d, filenmaes[i]))
    # print(i, patch_3d.shape)
    num_slice = patch_3d.shape[0]
    fn = filenmaes[i].split(".")[-2]
    for j in range(num_slice):
        io.imsave(
            fname=os.path.join(path_patches_2d, f"{fn}_{j}.tif"),
            arr=np.clip(patch_3d[j][None], a_min=0.0, a_max=None),
            check_contrast=False,
        )
    pbar.update(1)
pbar.close()
