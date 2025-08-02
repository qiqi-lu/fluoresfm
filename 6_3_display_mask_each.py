"""
Display estimated mask of each sample.
"""

import os, pandas, tqdm
import numpy as np
import skimage.io as skio
import skimage.measure as skm
import matplotlib.pyplot as plt
from dataset_analysis import datasets_seg_show
from utils.data import win2linux, read_txt, normalization, interp_sf
from utils.plot import get_outlines

# from cellpose import utils as cp_utils


def preprocess(img, scale_factor=1):
    img = np.clip(img, 0, None)
    img = normalization(img, p_low=0.03, p_high=0.995)
    if scale_factor != 1:
        img = interp_sf(img, scale_factor)
    return img


# ------------------------------------------------------------------------------
path_fig_save_to = os.path.join("results", "figures", "masks", "each_sample-2")
os.makedirs(path_fig_save_to, exist_ok=True)

# ------------------------------------------------------------------------------
path_results = os.path.join("results", "predictions")

path_test_xlsx = "dataset_test-v2.xlsx"
df_test = pandas.read_excel(path_test_xlsx)
num_sample_eva = 8
methods = ["raw", "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16"]
num_methods = len(methods)

fig, axes = plt.subplots(1, 3, figsize=(9, 3), dpi=300, constrained_layout=True)
[ax.set_axis_off() for ax in axes.ravel()]
dict_img = {"vmin": 0.0, "vmax": 1.0, "cmap": "gray"}
dict_outline_gt = {"linewidth": 0.5, "color": "magenta"}
dict_outline_est = {"linewidth": 0.5, "color": "yellow"}

# ------------------------------------------------------------------------------
pbar = tqdm.tqdm(total=len(datasets_seg_show), ncols=80)
for id_dataset in datasets_seg_show:
    info = df_test[df_test["id"] == id_dataset].iloc[0]

    seg_model = info["seg_model"]
    path_img_lr = win2linux(info["path_lr"])
    path_img_hr = win2linux(info["path_hr"])
    path_index = win2linux(info["path_index"])
    path_mask_gt = win2linux(info["path_mask"])
    sf_lr = info["sf_lr"]
    sf_hr = info["sf_hr"]

    # get all the filenames used for test
    filenames = read_txt(path_index)[:num_sample_eva]
    pbar_sample = tqdm.tqdm(total=len(filenames), ncols=80, leave=False)
    for i_file, filename in enumerate(filenames):
        imgs_est, masks_est = [], []
        for meth in methods:
            path_method = os.path.join(path_results, id_dataset, meth)

            # raw/estimated image
            if meth == "raw":
                img = skio.imread(os.path.join(path_img_lr, filename))
                img = preprocess(img, sf_lr)
                imgs_est.append(img[0])
            else:
                img = skio.imread(os.path.join(path_method, filename))
                img = preprocess(img)
                imgs_est.append(img[0])

            # estimated mask
            path_mask = path_method + "_mask_" + seg_model
            mask = skio.imread(os.path.join(path_mask, filename))
            masks_est.append(mask[0].astype(np.uint16))

        # high-quality image
        if path_img_hr != "Unknown":
            img_hr = skio.imread(os.path.join(path_img_hr, filename))
            img_hr = preprocess(img_hr, sf_hr)[0]
        else:
            # img_hr = np.zeros_like(imgs_est[0])
            img_hr = imgs_est[0]

        # ground truth mask
        if path_mask_gt == "Unknown":
            path_mask_gt = os.path.join(path_results, id_dataset, "gt_mask")
            path_mask_gt = path_mask_gt + "_" + seg_model
        mask_gt = skio.imread(os.path.join(path_mask_gt, filename))
        mask_gt = mask_gt[0].astype(np.uint16)

        # ----------------------------------------------------------------------
        axes[0].imshow(img_hr, **dict_img)
        mask_gt = skm.label(mask_gt)

        # mask_gt_outlines = cp_utils.outlines_list(mask_gt)
        mask_gt_outlines = get_outlines(mask_gt)
        for o in mask_gt_outlines:
            axes[0].plot(o[:, 1], o[:, 0], **dict_outline_gt)

        axes[0].set_title("GT")

        for i_method in range(num_methods):
            img_est = imgs_est[i_method]
            mask_est = masks_est[i_method]

            axes[i_method + 1].imshow(img_est, **dict_img)

            # mask_est_outlines = cp_utils.outlines_list(mask_est)
            mask_est_outlines = get_outlines(mask_est)
            for o in mask_est_outlines:
                axes[i_method + 1].plot(o[:, 1], o[:, 0], **dict_outline_est)

            axes[i_method + 1].set_title(methods[i_method])

        # save figure
        path_save_to = os.path.join(path_fig_save_to, f"{id_dataset}_{i_file}.png")
        plt.savefig(path_save_to)
        for ax in axes:
            ax.clear()
            ax.set_axis_off()
        del mask_gt_outlines, mask_est_outlines
        pbar_sample.update(1)
    pbar_sample.close()
    pbar.update(1)
pbar.close()
