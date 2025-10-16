"""
Display patches used in each datsets.
Selelct `num_patch_display` patches from each dataset.
"""

import matplotlib.pyplot as plt
import os, pandas, tqdm
import skimage.io as io
import numpy as np
from utils.data import interp_sf, read_txt, win2linux
from utils.plot import add_scale_bar


file_type = "png"  # file type to save the patches
# file_type = "tif"
direction = "ver"  # direction of the patches displayed in the figure
# direction = 'hor'
num_patch_disply = 5  # the number of pathces to display

use_clean = True
threshold = 0.05

path_dataset_xlx = "dataset_train_transformer-v2.xlsx"
datasets_frame = pandas.read_excel(path_dataset_xlx, sheet_name="64x64")
path_save_to = os.path.join("results", "figures", "datasets", "train_patch")
os.makedirs(os.path.join(path_save_to, file_type + "_" + direction), exist_ok=True)

# ------------------------------------------------------------------------------
num_datasets = len(datasets_frame)
dict_fig = {"dpi": 300, "constrained_layout": True}
print("-" * 80)
print("Number of datasets:", num_datasets)

if direction == "ver":
    fig, axes = plt.subplots(2, 1, figsize=(1, 2), **dict_fig)
if direction == "hor":
    fig, axes = plt.subplots(1, 2, figsize=(2, 1), **dict_fig)

pbar = tqdm.tqdm(desc="Display patches", total=num_datasets, ncols=80)
for i_dataset in range(num_datasets):
    ds = datasets_frame.iloc[i_dataset]

    ds_index, ds_name = ds["index"], ds["id"]
    ds_sf_lr, ds_sf_hr = ds["sf_lr"], ds["sf_hr"]
    ds_path_lr, ds_path_hr = win2linux(ds["path_lr"]), win2linux(ds["path_hr"])
    ds_path_index = win2linux(ds["path_index"])
    ps_hr = float(ds["target pixel size"].split("x")[0]) / 1000.0

    if use_clean:
        ds_path_index = ds_path_index.split(".")[0] + "_clean.txt"

    filenames = read_txt(ds_path_index)
    # idx_patches = np.random.choice(len(filenames), size=5, replace=False)
    count = 0
    # for idx_patch in idx_patches:
    for idx_patch in range(len(filenames)):
        filename = filenames[idx_patch]
        img_hr = io.imread(os.path.join(ds_path_hr, filename))
        if np.mean(img_hr) < threshold:
            continue
        count += 1
        if count > num_patch_disply:
            break
        img_hr = interp_sf(img_hr, ds_sf_hr)
        img_lr = io.imread(os.path.join(ds_path_lr, filename))
        img_lr = interp_sf(img_lr, ds_sf_lr)

        if file_type == "tif":
            # connect images
            if direction == "hor":
                img = np.concatenate([img_lr, img_hr], axis=2)
            elif direction == "ver":
                img = np.concatenate([img_lr, img_hr], axis=1)
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
