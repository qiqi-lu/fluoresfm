"""
Display the predicted image of each sample from different methods.
"""

import numpy as np
import pandas, os, tqdm
import matplotlib.pyplot as plt
import utils.data as utils_data
import utils.evaluation as eva
from utils.plot import colorize
from dataset_analysis import dataset_names_all, datasets_need_bkg_sub
import skimage.io as io

show_error = True
suffix = "-error-2"

methods = [
    ("UniFMIR", "unifmir_all-newnorm-v2"),
    ("FluoResFM-bs16", "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16"),
    ("FluoResFM (w/o text)", "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-crossx"),
    ("FluoResFM-T", "unet_sd_c_all_newnorm-ALL-v2-small-bs16-T77"),
    ("FluoResFM-TS", "unet_sd_c_all_newnorm-ALL-v2-small-bs16-TS77"),
    # ("FluoResFM-TSpixel", "unet_sd_c_all_newnorm-ALL-v2-small-bs16-TSpixel77"),
    # ("FluoResFM-TSmicro", "unet_sd_c_all_newnorm-ALL-v2-small-bs16-TSmicro77"),
    ("FluoResFM-T (in)", "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-in-T"),
    ("FluoResFM-TS (in)", "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-in-TS"),
    # ("FluoResFM-TSpixel (in)", "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-in-TSpixel"),
    # ("FluoResFM-TSmicro (in)", "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-in-TSmicro"),
]

# ------------------------------------------------------------------------------
dataset_groups = ["internal_dataset", "external_dataset"]
tasks = ["sr", "dn", "dcv"]
path_predictions = os.path.join("results", "predictions")
path_save_fig_to = os.path.join("results", "figures", "images", "each_sample")
if show_error:
    path_save_fig_to = path_save_fig_to + suffix
os.makedirs(path_save_fig_to, exist_ok=True)

datasets_show = []
for dataset_group in dataset_groups:
    for task in tasks:
        datasets_show.extend(dataset_names_all[dataset_group][task])

# datasets_show = [
# "biosr+-er-dn-1",
# "biosr+-ccp-dn-1",
# "biosr+-ccp-dn-2",
# "biosr+-ccp-dn-3",
# "biosr+-ccp-dn-4",
# "biosr+-ccp-dn-5",
# "biosr+-ccp-dn-6",
# "biosr+-ccp-dn-7",
# "biosr+-ccp-dn-8",
# "w2s-c0-sr-1",
# "w2s-c0-sr-2",
# "w2s-c0-sr-3",
# "w2s-c0-sr-4",
# "w2s-c0-sr-5",
# "w2s-c0-sr-6",
# "w2s-c0-sr-7",
# "w2s-c1-sr-1",
# "w2s-c1-sr-2",
# "w2s-c1-sr-3",
# "w2s-c1-sr-4",
# "w2s-c1-sr-5",
# "w2s-c1-sr-6",
# "w2s-c1-sr-7",
# "w2s-c2-sr-1",
# "w2s-c2-sr-2",
# "w2s-c2-sr-3",
# "w2s-c2-sr-4",
# "w2s-c2-sr-5",
# "w2s-c2-sr-6",
# "w2s-c2-sr-7",
# "w2s-c0-dcv-1",
# "w2s-c0-dcv-2",
# "w2s-c0-dcv-3",
# "w2s-c0-dcv-4",
# "w2s-c0-dcv-5",
# "w2s-c0-dcv-6",
# "w2s-c0-dcv-7",
# "w2s-c1-dcv-1",
# "w2s-c1-dcv-2",
# "w2s-c1-dcv-3",
# "w2s-c1-dcv-4",
# "w2s-c1-dcv-5",
# "w2s-c1-dcv-6",
# "w2s-c1-dcv-7",
# "w2s-c2-dcv-1",
# "w2s-c2-dcv-2",
# "w2s-c2-dcv-3",
# "w2s-c2-dcv-4",
# "w2s-c2-dcv-5",
# "w2s-c2-dcv-6",
# "w2s-c2-dcv-7",
# "vmsim3-mito-sr-crop",
# "vmsim3-mito-dcv-crop",
# "vmsim5-mito-sr-crop",
# "vmsim5-mito-dcv-crop",
# "biotisr-ccp-sr-1",
# "biotisr-ccp-sr-2",
# "biotisr-ccp-sr-3",
# "biotisr-factin-nonlinear-dcv-1",
# "biotisr-factin-nonlinear-dcv-2",
# "biotisr-factin-nonlinear-dcv-3",
# "biotisr-factin-nonlinear-sr-1",
# "biotisr-factin-nonlinear-sr-2",
# "biotisr-factin-nonlinear-sr-3",
# "biotisr-factin-nonlinear-dn-1",
# "biotisr-factin-nonlinear-dn-2",
# "vmsim488-bead-patch-dcv",
# "vmsim568-bead-patch-dcv",
# "vmsim647-bead-patch-dcv",
# "sim-actin-3d-dcv",
# "sim-actin-2d-patch-dcv",
#     "biotisr-ccp-dcv-1",
#     "biotisr-ccp-dcv-2",
#     "biotisr-ccp-dcv-3",
#     "biotisr-factin-dcv-1",
#     "biotisr-factin-dcv-2",
#     "biotisr-factin-dcv-3",
#     "biotisr-factin-nonlinear-dcv-1",
#     "biotisr-factin-nonlinear-dcv-2",
#     "biotisr-factin-nonlinear-dcv-3",
#     "biotisr-lysosome-dcv-1",
#     "biotisr-lysosome-dcv-2",
#     "biotisr-lysosome-dcv-3",
#     "biotisr-mt-dcv-1",
#     "biotisr-mt-dcv-2",
#     "biotisr-mt-dcv-3",
#     "biotisr-mito-dcv-1",  # live
#     "biotisr-mito-dcv-2",  # live
#     "biotisr-mito-dcv-3",  # live
# ]

num_methods = len(methods)
methods_title = [method[0] for method in methods]
methods_id = [method[1] for method in methods]
num_sample_show = 8

normalizer = lambda image: utils_data.normalization(image, p_low=0.03, p_high=0.995)


def bkg_subtraction(image):
    radius, sf = 25, 16
    image, bg = utils_data.rolling_ball_approximation(image, radius=radius, sf=sf)
    image = np.clip(image, 0, None)
    return image


# ------------------------------------------------------------------------------
data_frame = pandas.read_excel("dataset_test-v2.xlsx")

fig_title = ["Raw"] + [meth for meth in methods_title] + ["GT"]
dict_fig = {"dpi": 300, "constrained_layout": True}

if not show_error:
    nr, nc = (num_methods + 2) // 4 + 1, 4
else:
    nr, nc = 2, num_methods + 2

fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(nc * 3, nr * 3), **dict_fig)
[ax.set_axis_off() for ax in axes.ravel()]

if not show_error:
    axes = axes.ravel()

# ------------------------------------------------------------------------------
pbar = tqdm.tqdm(total=len(datasets_show), ncols=80, desc="Displaying")
for dataset_name in datasets_show:
    path_results = os.path.join(path_predictions, dataset_name)
    ds = data_frame[data_frame["id"] == dataset_name].iloc[0]
    path_txt, path_lr, path_hr = (
        utils_data.win2linux(ds["path_index"]),
        utils_data.win2linux(ds["path_lr"]),
        utils_data.win2linux(ds["path_hr"]),
    )
    filenames = utils_data.read_txt(path_txt)[:num_sample_show]

    if dataset_name in datasets_need_bkg_sub:
        sub_bkg = True
    else:
        sub_bkg = False
    # --------------------------------------------------------------------------
    pbar_tmp = tqdm.tqdm(total=len(filenames), ncols=80, desc="Displaying", leave=False)
    for i_file, filename in enumerate(filenames):
        imgs = []
        # load gt image
        if path_hr != "Unknown":
            img_gt = io.imread(os.path.join(path_hr, filename))
            img_gt = utils_data.interp_sf(img_gt, sf=ds["sf_hr"])[0]
            img_gt = normalizer(img_gt)
            img_gt = np.clip(img_gt, 0.0, 2.5)

        # load raw image
        img_raw = io.imread(os.path.join(path_lr, filename))
        img_raw = utils_data.interp_sf(img_raw, sf=ds["sf_lr"])[0]
        if path_hr == "Unknown":
            img_raw = normalizer(img_raw)
        else:
            # img_raw = eva.linear_transform(img_true=img_gt, img_test=img_raw)
            img_raw = normalizer(img_raw)
        img_raw = np.clip(img_raw, 0.0, 2.5)
        imgs.append(img_raw)

        if path_hr == "Unknown":
            img_gt = img_raw

        # load results
        for meth in methods_id:
            img_meth = io.imread(os.path.join(path_results, meth, filename))[0]
            if sub_bkg:
                img_meth = bkg_subtraction(img_meth)[0]
            # img_meth = eva.linear_transform(img_true=img_gt, img_test=img_meth)
            img_meth = normalizer(img_meth)
            img_meth = np.clip(img_meth, 0.0, 2.5)
            imgs.append(img_meth)

        imgs.append(img_gt)

        # display
        for i, img in enumerate(imgs):
            img_color = colorize(img, vmin=0.0, vmax=0.9, color=(0, 255, 0))
            if not show_error:
                ax = axes[i]
                ax.imshow(img_color)
                ax.set_title(fig_title[i])
            else:
                axes[0, i].imshow(img_color)
                axes[0, i].set_title(fig_title[i])
                # shwo error
                axes[1, i].imshow(img - img_gt, cmap="seismic", vmin=-0.25, vmax=0.25)

        # save
        fig.savefig(os.path.join(path_save_fig_to, f"{dataset_name}_{i_file}.png"))
        pbar_tmp.update(1)
    pbar_tmp.close()

    for ax in axes.ravel():
        ax.clear()
        ax.set_axis_off()
    pbar.update(1)
pbar.close()
