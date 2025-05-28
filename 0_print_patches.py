"""
print patches used in each datsets.
Randomly selelct 5 patches from each dataset.
"""

import matplotlib.pyplot as plt
import os, pandas, tqdm
import skimage.io as io
import numpy as np
from utils.data import interp_sf, read_txt, win2linux
from utils.plot import add_scale_bar

path_dataset_xlx = "dataset_train_transformer-v2.xlsx"
datasets_frame = pandas.read_excel(path_dataset_xlx, sheet_name="64x64")

path_save_to = os.path.join("results", "figures", "datasets", "train_patch")
use_clean = True
file_type = "png"

direction = "ver"
# direction = 'hor'

# ------------------------------------------------------------------------------
num_datasets = len(datasets_frame)
print("num of datasets:", num_datasets)

if direction == "ver":
    fig, axes = plt.subplots(2, 1, figsize=(1, 2), dpi=300, constrained_layout=True)
if direction == "hor":
    fig, axes = plt.subplots(1, 2, figsize=(2, 1), dpi=300, constrained_layout=True)

pbar = tqdm.tqdm(desc="show patches", total=num_datasets, ncols=80)
for i_dataset in range(num_datasets):
    ds = datasets_frame.iloc[i_dataset]

    ds_index, ds_name = ds["index"], ds["id"]
    ds_sf_lr, ds_sf_hr = ds["sf_lr"], ds["sf_hr"]
    ds_path_lr, ds_path_hr = ds["path_lr"], ds["path_hr"]
    ds_path_index = ds["path_index"]
    ps_lr, ps_hr = ds["input pixel size"], ds["target pixel size"]
    ps_hr = float(ps_hr.split("x")[0]) / 1000.0

    if use_clean:
        ds_path_index = ds_path_index.split(".")[0] + "_clean.txt"
    if os.name == "posix":
        ds_path_hr = win2linux(ds_path_hr)
        ds_path_lr = win2linux(ds_path_lr)
        ds_path_index = win2linux(ds_path_index)

    filenames = read_txt(ds_path_index)
    # idx_patches = np.random.choice(len(filenames), size=5, replace=False)
    count = 0
    # for idx_patch in idx_patches:
    for idx_patch in range(len(filenames)):
        filename = filenames[idx_patch]
        img_hr = io.imread(os.path.join(ds_path_hr, filename))
        if np.mean(img_hr) < 0.05:
            continue
        count += 1
        if count > 5:
            break
        img_hr = interp_sf(img_hr, ds_sf_hr)

        img_lr = io.imread(os.path.join(ds_path_lr, filename))
        img_lr = interp_sf(img_lr, ds_sf_lr)

        if file_type == "tif":
            # connect images
            img = np.concatenate([img_lr, img_hr], axis=2)
            io.imsave(
                os.path.join(path_save_to, "tif", f"{ds_index}_{ds_name}_{filename}"),
                img,
                check_contrast=False,
            )
        elif file_type == "png":
            # show images
            axes[0].imshow(img_lr[0], vmin=0, vmax=1.0, cmap="hot")
            axes[1].imshow(img_hr[0], vmin=0, vmax=1.0, cmap="hot")
            add_scale_bar(
                axes[1], img_lr[0], pixel_size=ps_hr, bar_length=0.5, pos=(5, 59)
            )
            axes[0].axis("off")
            axes[1].axis("off")
            plt.savefig(
                os.path.join(
                    path_save_to,
                    "png_" + direction,
                    f"{ds_index}_{ds_name}_{filename.split('.')[0]}.png",
                )
            )
            axes[0].clear()
            axes[1].clear()
        else:
            raise ValueError("file_type must be 'tif' or 'png'")

    pbar.update(1)
pbar.close()
