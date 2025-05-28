"""
convert large 2D/3D images into 2D patches.
3D images will be converted into 2D patches by slicing along the z-axis.
"""

import skimage.io as io
import numpy as np
import os, tqdm
from utils_data import normalization, read_txt


# ------------------------------------------------------------------------------
path_dataset = ["VMSIM\\transformed\Fig5\\train\mito\SIM"]
path_index_file = "VMSIM\\transformed\Fig5\\train.txt"

# ------------------------------------------------------------------------------
suffix = ""
log_patch_name = True  # log the patch name to the txt file

pl, ph = 0.03, 0.995  # normalization

# patch_size, step_size = 8, 8
# patch_size, step_size = 16, 16
# patch_size, step_size = 32, 32
# patch_size, step_size = 64, 64
patch_size, step_size = 128, 128
# patch_size, step_size = 192, 192
# patch_size, step_size = 256, 256
# patch_size, step_size = 512, 512
# patch_size, step_size = 512, 256
# patch_size, step_size = 1024, 420

print("-" * 100)
print("patch size:", patch_size, ", step size:", step_size)

# ------------------------------------------------------------------------------
for i_dataset in range(len(path_dataset)):
    print("-" * 100)
    path_data = path_dataset[i_dataset]
    path_index = path_index_file

    print("Data from:      ", path_data)
    print("Data index from:", path_index)

    # load names of samples
    sample_names = read_txt(path_index)
    num_samples = len(sample_names)
    print("Numb of samples:", num_samples)

    # create the foldF_actin_nonlinear to save patches
    save_to = path_data + f"_p{patch_size}_s{step_size}_2d" + suffix
    os.makedirs(save_to, exist_ok=True)
    print("save image to:", save_to)

    # output txt file
    if log_patch_name:
        save_to_txt = (
            path_index.split(".")[0]
            + f"_p{patch_size}_s{step_size}_2d"
            + suffix
            + ".txt"
        )
        out_file = open(save_to_txt, "w")
        print("save filenames of patches into:", save_to_txt)

    # --------------------------------------------------------------------------
    pbar = tqdm.tqdm(total=num_samples, desc="Patching", ncols=100)
    for i_sample in range(num_samples):
        # load image
        img = io.imread(os.path.join(path_data, sample_names[i_sample])).astype(
            np.float32
        )
        img = np.clip(img, a_min=0.0, a_max=None)  # clip negative values to 0

        # normalization
        img, _, _ = normalization(img, p_low=pl, p_high=ph)  # normalize to [0, 1]

        dim = len(img.shape)
        # 2D raw data
        if (dim == 2) or (dim == 3 and img.shape[0] == 1):
            if dim == 3:
                img = img[0]

            Ny, Nx = img.shape
            num_y = (Ny - patch_size) // step_size + 1
            num_x = (Nx - patch_size) // step_size + 1

            # patching
            for i in range(num_y):
                for j in range(num_x):
                    patch = img[
                        step_size * i : step_size * i + patch_size,
                        step_size * j : step_size * j + patch_size,
                    ]
                    # save patch
                    patch_name = sample_names[i_sample].split(".")[0] + f"_{i}_{j}.tif"
                    io.imsave(
                        os.path.join(save_to, patch_name),
                        patch[None],
                        check_contrast=False,
                    )
                    # write filename
                    if log_patch_name:
                        out_file.write(patch_name + "\n")

        # --------------------------------------------------------------------------
        # 3D raw data
        elif (dim == 3) and (img.shape[0] != 1):
            Nz, Ny, Nx = img.shape
            num_y = (Ny - patch_size) // step_size + 1
            num_x = (Nx - patch_size) // step_size + 1
            for k in range(Nz):
                for i in range(num_y):
                    for j in range(num_x):
                        patch = img[
                            k,
                            step_size * i : step_size * i + patch_size,
                            step_size * j : step_size * j + patch_size,
                        ]
                        # save patch
                        patch_name = (
                            sample_names[i_sample].split(".")[0] + f"_{k}_{i}_{j}.tif"
                        )
                        io.imsave(
                            os.path.join(save_to, patch_name),
                            patch[None],
                            check_contrast=False,
                        )
                        # write filename
                        if log_patch_name:
                            out_file.write(patch_name + "\n")
        else:
            print("invalid image shape", img.shape)
        pbar.update(1)
    pbar.close()
    if log_patch_name:
        out_file.close()
