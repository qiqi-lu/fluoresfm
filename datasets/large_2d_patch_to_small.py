import skimage.io as io
import numpy as np
import os
from tqdm import tqdm
from utils_data import read_txt, normalization


# ------------------------------------------------------------------------------
path_data = (
    # "CARE_Isotropic_Drosophila\\transformed\\train\channel_0\condition_0_patch_128_2d"
    # "CARE_Isotropic_Drosophila\\transformed\\train\channel_0\gt_patch_128_2d"
    # "CARE_Isotropic_Liver\\transformed\\train\channel_0\condition_0_patch_128_2d"
    # "CARE_Isotropic_Liver\\transformed\\train\channel_0\gt_patch_128_2d"
    # "CARE_Isotropic_Retina\\transformed\\train\channel_0\condition_0_patch_128_2d"
    # "CARE_Isotropic_Retina\\transformed\\train\channel_0\gt_patch_128_2d"
    # "CARE_Isotropic_Retina\\transformed\\train\channel_1\condition_0_patch_128_2d"
    # "CARE_Isotropic_Retina\\transformed\\train\channel_1\gt_patch_128_2d"
    # "CARE_Synthetic_tubulin_gfp\\transformed\\train\channel_0\condition_0_patch_128_2d"
    # "CARE_Synthetic_tubulin_gfp\\transformed\\train\channel_0\gt_patch_128_2d"
    # "CARE_Synthetic_tubulin_granules\\transformed\\train\granules\condition_0_patch_128_2d"
    # "CARE_Synthetic_tubulin_granules\\transformed\\train\granules\gt_patch_128_2d"
    # "CARE_Synthetic_tubulin_granules\\transformed\\train\\tubulin\condition_0_patch_128_2d"
    # "CARE_Synthetic_tubulin_granules\\transformed\\train\\tubulin\gt_patch_128_2d"
    # "SR-CACO-2\\transformed\\test\channel_h2b\LowRes128"
    # "SR-CACO-2\\transformed\\test\channel_h2b\LowRes256"
    # "SR-CACO-2\\transformed\\test\channel_h2b\LowRes512"
    # "SR-CACO-2\\transformed\\test\channel_h2b\HighRes1024"
    # "SR-CACO-2\\transformed\\test\channel_survivin\LowRes128"
    # "SR-CACO-2\\transformed\\test\channel_survivin\LowRes256"
    # "SR-CACO-2\\transformed\\test\channel_survivin\LowRes512"
    # "SR-CACO-2\\transformed\\test\channel_survivin\HighRes1024"
    # "SR-CACO-2\\transformed\\test\channel_tubulin\LowRes128"
    # "SR-CACO-2\\transformed\\test\channel_tubulin\LowRes256"
    # "SR-CACO-2\\transformed\\test\channel_tubulin\LowRes512"
    "SR-CACO-2\\transformed\\test\channel_tubulin\HighRes1024"
)

path_index = (
    # "CARE_Isotropic_Drosophila\\transformed\\train_patch_128_2d.txt"
    # "CARE_Isotropic_Liver\\transformed\\train_patch_128_2d.txt"
    # "CARE_Isotropic_Retina\\transformed\\train_patch_128_2d.txt"
    # "CARE_Synthetic_tubulin_gfp\\transformed\\train_patch_128_2d.txt"
    # "CARE_Synthetic_tubulin_granules\\transformed\\train_patch_128_2d_granules.txt"
    # "CARE_Synthetic_tubulin_granules\\transformed\\train_patch_128_2d_tubulin.txt"
    "SR-CACO-2\\transformed\\test.txt"
)


# patch_size, step_size = 64, 64
# patch_size, step_size = 128, 128
# patch_size, step_size = 256, 256
# patch_size, step_size = 512, 512
patch_size, step_size = 1024, 1024

# load names of samples
sample_names = read_txt(path_index)
num_samples = len(sample_names)
print("Number of samples:", num_samples)

# create the folder to save patches
save_to = path_data + f"_patch_{patch_size}_2d"
os.makedirs(save_to, exist_ok=True)
print("save to", save_to)

# ------------------------------------------------------------------------------
pbar = tqdm(total=num_samples, desc="Patching", ncols=100)

# loop for patching
for idx_sample in range(num_samples):
    img = io.imread(os.path.join(path_data, sample_names[idx_sample]))
    img = img.astype(np.float32)
    img = np.clip(img, a_min=0.0, a_max=None)
    img = normalization(img, 0.0, 0.9999)
    # --------------------------------------------------------------------------
    if len(img.shape) == 2:
        Ny, Nx = img.shape
    if len(img.shape) == 3:
        Nz, Ny, Nx = img.shape
        img = img[0]
    # --------------------------------------------------------------------------

    num_y = (Ny - patch_size) // step_size + 1
    num_x = (Nx - patch_size) // step_size + 1
    for i in range(num_y):
        for j in range(num_x):
            patch = img[
                step_size * i : step_size * i + patch_size,
                step_size * j : step_size * j + patch_size,
            ]
            io.imsave(
                os.path.join(
                    save_to,
                    sample_names[idx_sample].split(sep=".")[0] + f"_{i}_{j}.tif",
                ),
                patch[None],
                check_contrast=False,
            )
    pbar.update(1)
pbar.close()
