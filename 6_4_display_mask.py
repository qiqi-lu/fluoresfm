"""
Dispaly the segmentation results.
Single sample.
"""

import os, pandas
import numpy as np
import skimage.io as skio
import skimage.measure as skm
import matplotlib.pyplot as plt

# from cellpose import utils as cp_utils
from utils.data import win2linux, read_txt, normalization, interp_sf
from utils.evaluation import average_precision, IoU
from utils.plot import add_scale_bar, get_outlines

path_results = os.path.join("results", "predictions")
dataset_info = [
    # id_dataset | sample_id | region_show (Y,X,H,W)
    # --------------------------------------------------------------------------
    ("biotisr-ccp-dcv-1", 3, (0, 125, 250, 250)),
    ("biotisr-lysosome-dcv-3", 3, (118, 163, 300, 300)),
    ("rcan3d-dn-mixtrixmito-dn", 3, (370, 643, 760, 760)),
    ("biosr-er-dcv-2", 1, None),
    ("biotisr-mt-dn-2", 3, None),
    ("hl60-high-noise-c00", 6, (0, 0, 500, 500)),
    ("care-tribolium-dn-2", 7, (0, 160, 626, 626)),
]

crop_region = True
# crop_region = False

methods = ["raw", "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16"]

# ------------------------------------------------------------------------------
dataset_frame = pandas.read_excel("dataset_test-v2.xlsx")
path_save_to = os.path.join("results", "figures", "masks")
os.makedirs(path_save_to, exist_ok=True)


def preprocess(img, scale_factor=1):
    img = np.clip(img, 0, None)
    img = normalization(img, p_low=0.03, p_high=0.995)
    if scale_factor != 1:
        img = interp_sf(img, scale_factor)
    return img


# ------------------------------------------------------------------------------
num_datasets = len(dataset_info)
num_methods = len(methods)

nc, nr = num_datasets, num_methods + 1
fig, axes = plt.subplots(
    nr, nc, figsize=(nc * 3, nr * 3), dpi=300, constrained_layout=True
)
[ax.set_axis_off() for ax in axes.ravel()]
plt.rcParams["svg.fonttype"] = "none"


for i_dataset in range(num_datasets):
    print(dataset_info[i_dataset])
    id_dataset, sample_id, region = dataset_info[i_dataset]
    df = dataset_frame[dataset_frame["id"] == id_dataset].iloc[0]

    seg_model = df["seg_model"]
    path_img_lr = win2linux(df["path_lr"])
    path_img_hr = win2linux(df["path_hr"])
    path_index = win2linux(df["path_index"])
    path_mask_gt = win2linux(df["path_mask"])
    sf_lr = df["sf_lr"]
    sf_hr = df["sf_hr"]
    dataset_name = df["dataset-name"]
    structure = df["structure"]
    pixel_size = float(df["target pixel size"].split("x")[0]) / 1000.0  # um

    # get all the filenames used for test
    filenames = read_txt(path_index)

    # --------------------------------------------------------------------------
    # load images and masks
    imgs_est, masks_est = [], []
    for meth in methods:
        path_method = os.path.join(path_results, id_dataset, meth)

        # raw/estimated image
        if meth == "raw":
            img = skio.imread(os.path.join(path_img_lr, filenames[sample_id]))
            img = preprocess(img, sf_lr)
            imgs_est.append(img[0])
        else:
            img = skio.imread(os.path.join(path_method, filenames[sample_id]))
            img = preprocess(img)
            imgs_est.append(img[0])

        # estimated mask
        path_mask = path_method + "_mask_" + seg_model
        mask = skio.imread(os.path.join(path_mask, filenames[sample_id]))
        masks_est.append(mask[0].astype(np.uint16))

    # high-quality image
    if path_img_hr != "Unknown":
        img_hr = skio.imread(os.path.join(path_img_hr, filenames[sample_id]))
        img_hr = preprocess(img_hr, sf_hr)[0]
    else:
        # img_hr = np.zeros_like(imgs_est[0])
        img_hr = imgs_est[0]

    # ground truth mask
    if path_mask_gt == "Unknown":
        path_mask_gt = os.path.join(path_results, id_dataset, "gt_mask")
        path_mask_gt = path_mask_gt + "_" + seg_model
    mask_gt = skio.imread(os.path.join(path_mask_gt, filenames[sample_id]))
    mask_gt = mask_gt[0].astype(np.uint16)

    # --------------------------------------------------------------------------
    # crop the region of interest to show
    if crop_region and region is not None:
        y, x, h, w = region
        crop = lambda img: img[y : y + h, x : x + w]
        img_hr = crop(img_hr)
        mask_gt = crop(mask_gt)
        for i in range(num_methods):
            imgs_est[i] = crop(imgs_est[i])
            masks_est[i] = crop(masks_est[i])

    # --------------------------------------------------------------------------
    # display
    img_shape = img_hr.shape
    text_pos = 0.05
    fontsize = 14

    dict_img = {"vmin": 0.0, "vmax": 1.5, "cmap": "gray"}
    dict_outline_gt = {"linewidth": 0.5, "color": "magenta"}
    dict_outline_est = {"linewidth": 0.5, "color": "yellow"}
    dict_metrics = {
        "x": int(img_shape[1] * text_pos),
        "y": int(img_shape[0] * (1 - text_pos)),
        "color": "white",
        "fontsize": fontsize,
        "ha": "left",
        "va": "bottom",
    }
    dict_method = {
        "x": int(img_shape[1] * (1 - text_pos)),
        "y": int(img_shape[0] * text_pos),
        "color": "white",
        "fontsize": fontsize,
        "ha": "right",
        "va": "top",
    }
    dict_dataset_name = {
        "x": int(img_shape[1] * text_pos),
        "y": int(img_shape[0] * text_pos),
        "color": "white",
        "fontsize": fontsize,
        "ha": "left",
        "va": "top",
    }

    dict_scale_bar = {
        "pixel_size": pixel_size,
        "bar_length": 5,  # um
        "bar_height": 0.01,
        "bar_color": "white",
        "pos": (int(img_shape[1] * text_pos), int(img_shape[0] * (1 - text_pos))),
    }

    # display gt image and mask
    axes[0, i_dataset].imshow(img_hr, **dict_img)

    mask_gt = skm.label(mask_gt)
    # mask_gt_outlines = cp_utils.outlines_list(mask_gt)
    mask_gt_outlines = get_outlines(mask_gt)
    for o in mask_gt_outlines:
        # axes[0, i_dataset].plot(o[:, 0], o[:, 1], **dict_outline_gt)
        axes[0, i_dataset].plot(o[:, 1], o[:, 0], **dict_outline_gt)
    # mask_gt_contours = skm.find_contours(mask_gt, 0.5)
    # for contour in mask_gt_contours:
    #     axes[0, i_dataset].plot(contour[:, 1], contour[:, 0], **dict_outline_gt)

    # if i_dataset == 0:
    #     axes[0, i_dataset].text(s="GT", **dict_method)

    axes[0, i_dataset].text(s=structure, **dict_dataset_name)
    add_scale_bar(axes[0, i_dataset], image=img_hr, **dict_scale_bar)

    # display estmated image
    for i_method in range(num_methods):
        img_est = imgs_est[i_method]
        mask_est = masks_est[i_method]
        mask_est = skm.label(mask_est)

        axes[i_method + 1, i_dataset].imshow(img_est, **dict_img)

        # mask_est_outlines = cp_utils.outlines_list(mask_est)
        mask_est_outlines = get_outlines(mask_est)

        for o in mask_est_outlines:
            axes[i_method + 1, i_dataset].plot(o[:, 1], o[:, 0], **dict_outline_est)

        ap = average_precision(mask_gt[None, None], mask_est[None, None])[0][0]
        iou = IoU(mask_gt[None, None], mask_est[None, None])[0]

        # add text
        if i_dataset == 0:
            axes[i_method + 1, i_dataset].text(
                s=f"AP@0.5: {ap:.3f} | IoU: {iou:.3f}", **dict_metrics
            )
        else:
            axes[i_method + 1, i_dataset].text(
                s=f"{ap:.3f} | {iou:.3f}", **dict_metrics
            )

        # if i_dataset == 0:
        #     axes[i_method + 1, i_dataset].text(s=titles[i_method], **dict_method)

    # save figure
plt.savefig(os.path.join(path_save_to, f"mask_dataset_method.png"))
plt.savefig(os.path.join(path_save_to, f"mask_dataset_method.svg"))
