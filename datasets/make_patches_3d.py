import skimage.io as io
import numpy as np
import os
from tqdm import tqdm
from utils_data import normalization, read_txt
import pandas


# ------------------------------------------------------------------------------
path_dataset = [
    "E:\qiqilu\datasets\FMD\\transformed\Confocal_MICE\\train\channel_0\\avg16\\repeat_37",
]
path_index_file = "E:\qiqilu\datasets\FMD\\transformed\Confocal_MICE\\train.txt"

suffix = ""

# patch_size, step_size = 8, 8
# patch_size, step_size = 16, 16
# patch_size, step_size = 32, 32
patch_size, step_size = 64, 64
# patch_size, step_size = 128, 128
# patch_size, step_size = 192, 192
# patch_size, step_size = 256, 256
# patch_size, step_size = 512, 512

convert_3d_to_2d = True  # whether convert 3d to 2d pathces

print("patch size:", patch_size, ", step size:", step_size)
print("Convert 3D images into 2D pathces:", convert_3d_to_2d)

# ------------------------------------------------------------------------------
for i in range(len(path_dataset)):
    print("-" * 100)
    path_data = path_dataset[i]
    path_index = path_index_file

    print("load data from:      ", path_data)
    print("load data index from:", path_index)

    # load names of samples
    sample_names = read_txt(path_index)
    num_samples = len(sample_names)
    print("Number of samples:", num_samples)

    # create the folder to save patches
    save_to = path_data + f"_p{patch_size}_s{step_size}"
    if convert_3d_to_2d:
        save_to = save_to + "_2d" + suffix
    else:
        save_to = save_to + "_3d"

    os.makedirs(save_to, exist_ok=True)
    print("save to", save_to)

    # processing bar
    pbar = tqdm(total=num_samples, desc="Patching", ncols=100)
    # output filenames
    out_file = open(save_to + ".txt", "w")

    # patching loop
    for idx_sample in range(num_samples):
        # load original image
        img = io.imread(os.path.join(path_data, sample_names[idx_sample]))
        img = img.astype(np.float32)
        pbar.set_postfix({"image size": img.shape})

        # normalization
        pl, ph = 0.0, 0.9999
        img = normalization(img, p_low=pl, p_high=ph)

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
                    io.imsave(
                        os.path.join(
                            save_to,
                            sample_names[idx_sample].split(sep=".")[0]
                            + f"_{i}_{j}.tif",
                        ),
                        patch[None],
                        check_contrast=False,
                    )
                    # write filename
                    out_file.write(
                        sample_names[idx_sample].split(sep=".")[0] + f"_{i}_{j}.tif\n"
                    )

        # --------------------------------------------------------------------------
        # 3D raw data
        elif (dim == 3) and (img.shape[0] != 1):
            Nz, Ny, Nx = img.shape

            if convert_3d_to_2d == False:
                if Nz <= patch_size:
                    _, Ny, Nx = img.shape
                    num_y = (Ny - patch_size) // step_size + 1
                    num_x = (Nx - patch_size) // step_size + 1
                    for i in range(num_y):
                        for j in range(num_x):
                            patch = img[
                                :,
                                step_size * i : step_size * i + patch_size,
                                step_size * j : step_size * j + patch_size,
                            ]
                            io.imsave(
                                os.path.join(
                                    save_to,
                                    sample_names[idx_sample].split(sep=".")[0]
                                    + f"_0_{i}_{j}.tif",
                                ),
                                patch,
                                check_contrast=False,
                            )
                            # write filename
                            out_file.write(
                                sample_names[idx_sample].split(sep=".")[0]
                                + f"_0_{i}_{j}.tif\n"
                            )

                if Nz > patch_size:
                    num_z = (Nz - patch_size) // step_size + 1
                    num_y = (Ny - patch_size) // step_size + 1
                    num_x = (Nx - patch_size) // step_size + 1
                    for k in range(num_z):
                        for i in range(num_y):
                            for j in range(num_x):
                                patch = img[
                                    step_size * k : step_size * k + patch_size,
                                    step_size * i : step_size * i + patch_size,
                                    step_size * j : step_size * j + patch_size,
                                ]
                                io.imsave(
                                    os.path.join(
                                        save_to,
                                        sample_names[idx_sample].split(sep=".")[0]
                                        + f"_{k}_{i}_{j}.tif",
                                    ),
                                    patch,
                                    check_contrast=False,
                                )
                                # write filename
                                out_file.write(
                                    sample_names[idx_sample].split(sep=".")[0]
                                    + f"_{k}_{i}_{j}.tif\n"
                                )

            if convert_3d_to_2d == True:
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
                            io.imsave(
                                os.path.join(
                                    save_to,
                                    sample_names[idx_sample].split(sep=".")[0]
                                    + f"_{k}_{i}_{j}.tif",
                                ),
                                patch[None],
                                check_contrast=False,
                            )
                            # write filename
                            out_file.write(
                                sample_names[idx_sample].split(sep=".")[0]
                                + f"_{k}_{i}_{j}.tif\n"
                            )
        pbar.update(1)
    pbar.close()
    out_file.close()
