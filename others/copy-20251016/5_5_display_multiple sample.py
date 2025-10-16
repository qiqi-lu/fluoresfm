"""
Display multiple samples form different datasets but only the FluoResFM model.
-----------------------------------
|raw/gt  |    |    |    |    |    |
-----------------------------------
|restored|    |    |    |    |    |
-----------------------------------
"""

import os, tqdm, math
import numpy as np
import pandas
from utils.data import win2linux, read_txt, normalization, interp_sf
import skimage.io as io
import matplotlib.pyplot as plt
from utils.plot import colorize, add_scale_bar, image_combine_2d

path_predictions = os.path.join("results", "predictions")
path_save_fig_to = os.path.join("results", "figures", "images")
methods = ("FluoResFM", "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16")

samples_show = (
    ("rcan3d-c2s-npc-sr", 2, None),  # 0.03
    ("rcan3d-c2s-mt-dcv", 4, None),  # 0.03
    # ("biosr-mt-sr-1", 4, None),  # 0.03
    ("biosr-cpp-sr-1", 3, None),  # 0.03
    ("deepbacs-ecoli2-dn", 1, None),  # 0.04
    ("deepbacs-sim-ecoli-sr", 4, None),  # 0.04
    ("biosr-er-dcv-2", 4, None),  # 0.06
    ("biosr+-myosin-dn-1", 4, None),  # 0.06
    ("deepbacs-sim-saureus-dcv", 4, None),  # 0.08
    ("fmd-wf-bpae-r-avg2", 5, None),  # 0.17
    ("fmd-wf-bpae-g-avg2", 3, None),  # 0.17
    ("fmd-twophoton-mice-avg2", 0, None),  # 0.30
    ("fmd-confocal-fish-avg2", 2, None),  # 0.30
    # ("srcaco2-survivin-sr-2", 4, None),  # 0.50
    ("srcaco2-tubulin-sr-2", 4, None),  # 0.50
    ("care-tribolium-dn-1", 4, None),  # 0.50
    # ("biosr-actin-sr-1", 4, None),  # 0.03
    # ("fmd-confocal-bpae-b-avg2", 4, None),
    # ("deepbacs-ecoli-dn", 0, None),
)

print("[INFO] Number of samples:", len(samples_show))

normalizer = lambda image: normalization(image, p_low=0.03, p_high=0.995)
dict_clip = dict(a_min=0.0, a_max=2.5)

# load data
data_frame = pandas.read_excel("dataset_test-v2.xlsx")

# ------------------------------------------------------------------------------
nr, nc = 4, int(math.ceil(len(samples_show) / 2))
dict_fig = {"dpi": 600, "constrained_layout": True}
dict_colorize = dict(vmin=0, vmax=0.9, color=(0, 255, 0))
dict_text = dict(color="white", fontsize=16, ha="left", va="top")
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(nr, nc, figsize=(nc * 3, nr * 3), **dict_fig)
# turn off axis
[ax.set_axis_off() for ax in axes.flatten()]

pbar = tqdm.tqdm(total=len(samples_show), ncols=80, desc="[INFO] PLOT")
for i, sample in enumerate(samples_show):
    if i % 2 == 0:
        ax1, ax2 = axes[0, i // 2], axes[1, i // 2]
    if i % 2 == 1:
        ax1, ax2 = axes[2, i // 2], axes[3, i // 2]
    # load data ----------------------------------------------------------------
    dataset_name, i_sample, crop = sample
    ds = data_frame[data_frame["id"] == dataset_name].iloc[0]
    path_txt, path_lr, path_hr, pixel_size, structure, task = (
        win2linux(ds["path_index"]),
        win2linux(ds["path_lr"]),
        win2linux(ds["path_hr"]),
        float(ds["target pixel size"].split("x")[0]) / 1000.0,
        ds["structure"],
        ds["task"].upper(),
    )
    filename = read_txt(path_txt)[i_sample]

    path_img_lr = os.path.join(path_lr, filename)
    path_img_gt = os.path.join(path_hr, filename)
    path_img_pred = os.path.join(path_predictions, dataset_name, methods[1], filename)

    img_gt = io.imread(path_img_gt)
    img_gt = interp_sf(img_gt, sf=ds["sf_hr"])[0]
    img_gt = normalizer(img_gt)
    img_gt = np.clip(img_gt, **dict_clip)

    img_lr = io.imread(path_img_lr)
    img_lr = interp_sf(img_lr, sf=ds["sf_lr"])[0]
    img_lr = normalizer(img_lr)
    img_lr = np.clip(img_lr, **dict_clip)

    img_pred = io.imread(path_img_pred)[0]
    img_pred = normalizer(img_pred)
    img_pred = np.clip(img_pred, **dict_clip)

    # if image is not a square, crop center of image
    size = abs(img_gt.shape[0] - img_gt.shape[1])
    if img_gt.shape[0] < img_gt.shape[1]:
        sli = slice(size // 2, img_gt.shape[1] - size // 2)
        img_gt = img_gt[:, sli]
        img_lr = img_lr[:, sli]
        img_pred = img_pred[:, sli]
    elif img_gt.shape[0] > img_gt.shape[1]:
        sli = slice(size // 2, img_gt.shape[0] - size // 2)
        img_gt = img_gt[sli, :]
        img_lr = img_lr[sli, :]
    else:
        pass

    # crop image
    if crop is not None:
        img_gt = img_gt[crop[1] : crop[1] + crop[2], crop[0] : crop[0] + crop[2]]
        img_lr = img_lr[crop[1] : crop[1] + crop[2], crop[0] : crop[0] + crop[2]]
        img_pred = img_pred[crop[1] : crop[1] + crop[2], crop[0] : crop[0] + crop[2]]

    # plot ---------------------------------------------------------------------
    img_lr_hr = image_combine_2d(img_lr, img_gt)

    img_lr_hr = colorize(img_lr_hr, **dict_colorize)
    img_pred = colorize(img_pred, **dict_colorize)

    ax1.imshow(img_lr_hr)
    ax2.imshow(img_pred)

    # add scale bar ------------------------------------------------------------
    img_shape = img_pred.shape
    tp = 0.05  # percent of image
    dict_scale_bar = {
        "pixel_size": pixel_size,
        "bar_length": 5,  # um
        "bar_height": 0.01,
        "bar_color": "white",
        "pos": (int(img_shape[1] * tp), int(img_shape[0] * (1 - tp))),
    }
    add_scale_bar(ax2, image=img_pred, **dict_scale_bar)

    # add dashed line from bottom-left to top-right ----------------------------
    ax1.plot(
        [0, img_shape[1] - 1],
        [img_shape[0] - 1, 0],
        color="white",
        linestyle="--",
        linewidth=1,
    )

    # add text -----------------------------------------------------------------
    ax1.text(
        int(img_shape[1] * tp), int(img_shape[0] * tp), f"{structure}", **dict_text
    )
    ax2.text(int(img_shape[1] * tp), int(img_shape[0] * tp), f"{task}", **dict_text)
    # add pixel size at bottom-right
    ax2.text(
        int(img_shape[1] * (1 - 0.5)),
        int(img_shape[0] * (1 - 0.5)),
        f"{pixel_size:.4f} µm",
        **dict_text,
    )

    pbar.update(1)
pbar.close()

print("[INFO] Save figure to:", os.path.join(path_save_fig_to, "multiple_samples.png"))
plt.savefig(os.path.join(path_save_fig_to, "multiple_samples.png"))
plt.rcParams["svg.fonttype"] = "none"
plt.savefig(os.path.join(path_save_fig_to, "multiple_samples.svg"))
