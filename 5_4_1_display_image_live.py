"""
Show the restoration results of live cell imaging datasets.
- images
- metrics across time points
"""

import numpy as np
import os, pandas, colorcet
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import seaborn as sns
from skimage.measure import profile_line
from utils.data import win2linux, read_txt, normalization, interp_sf
from utils.plot import colorize, add_scale_bar, get_outlines
from utils.evaluation import PSNR, ZNCC, MSSSIM
from utils.analysis import pit_segmentation, lysosome_segmentation
from skimage.measure import regionprops_table
import logging

logging.getLogger().setLevel(logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

cmap_glasbey = [(0, 0, 0)] + list(colorcet.cm.glasbey_dark.colors)
cmap_glasbey = ListedColormap(cmap_glasbey)


plt.rcParams["svg.fonttype"] = "none"
GREEN, BLUE, RED, YELLOW = (0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)

show_images = False
show_images = True
time_step = 0.25
# ------------------------------------------------------------------------------
#              | filename | heatmap | profile | patch | figure direction
# ------------------------------------------------------------------------------
# fig_params = ("_live_ccp_in", False, False, True)
fig_params = ("_live_lyso_in", False, False, True)

print(f"[INFO] figure:{fig_params[0]}")
# ------------------------------------------------------------------------------
figure_suffix, plot_heatmap, show_profile, show_patch = fig_params
pos_live_ccp_in = ((305, 392), 62, GREEN, (93, 345, 157, 269), 2.0)
pos_live_lysosome_in = ((130, 431), 91, YELLOW, (93, 345, 157, 269), 2.0)


# ------------------------------------------------------------------------------
#        dataset_name | sample_id | win_pos (x, y) | win_size | color | line_pos | ylim (profile)
# ------------------------------------------------------------------------------
timepoint_show_dict = {
    "_live_ccp_in": (
        ("biotisr-ccp-sr-1-live-in", 0) + pos_live_ccp_in,
        ("biotisr-ccp-sr-1-live-in", 4) + pos_live_ccp_in,
        ("biotisr-ccp-sr-1-live-in", 8) + pos_live_ccp_in,
        ("biotisr-ccp-sr-1-live-in", 12) + pos_live_ccp_in,
        ("biotisr-ccp-sr-1-live-in", 16) + pos_live_ccp_in,
        # ----------------------------------------------------------------------
        # ("biotisr-ccp-sr-1-live-in", 0) + pos_live_ccp_in,
        # ("biotisr-ccp-sr-1-live-in", 1) + pos_live_ccp_in,
        # ("biotisr-ccp-sr-1-live-in", 2) + pos_live_ccp_in,
        # ("biotisr-ccp-sr-1-live-in", 3) + pos_live_ccp_in,
        # ("biotisr-ccp-sr-1-live-in", 4) + pos_live_ccp_in,
        # ("biotisr-ccp-sr-1-live-in", 5) + pos_live_ccp_in,
        # ("biotisr-ccp-sr-1-live-in", 6) + pos_live_ccp_in,
        # ("biotisr-ccp-sr-1-live-in", 7) + pos_live_ccp_in,
        # ("biotisr-ccp-sr-1-live-in", 8) + pos_live_ccp_in,
        # ("biotisr-ccp-sr-1-live-in", 9) + pos_live_ccp_in,
        # ("biotisr-ccp-sr-1-live-in", 10) + pos_live_ccp_in,
        # ("biotisr-ccp-sr-1-live-in", 11) + pos_live_ccp_in,
        # ("biotisr-ccp-sr-1-live-in", 12) + pos_live_ccp_in,
        # ("biotisr-ccp-sr-1-live-in", 13) + pos_live_ccp_in,
        # ("biotisr-ccp-sr-1-live-in", 14) + pos_live_ccp_in,
        # ("biotisr-ccp-sr-1-live-in", 15) + pos_live_ccp_in,
        # ("biotisr-ccp-sr-1-live-in", 16) + pos_live_ccp_in,
        # ("biotisr-ccp-sr-1-live-in", 17) + pos_live_ccp_in,
        # ("biotisr-ccp-sr-1-live-in", 18) + pos_live_ccp_in,
        # ("biotisr-ccp-sr-1-live-in", 19) + pos_live_ccp_in,
    ),
    "_live_lyso_in": (  # train on the first sample of training data, test all on training data
        ("biotisr-lysosome-sr-3-live-in", 0) + pos_live_lysosome_in,
        ("biotisr-lysosome-sr-3-live-in", 4) + pos_live_lysosome_in,
        ("biotisr-lysosome-sr-3-live-in", 8) + pos_live_lysosome_in,
        ("biotisr-lysosome-sr-3-live-in", 12) + pos_live_lysosome_in,
        ("biotisr-lysosome-sr-3-live-in", 16) + pos_live_lysosome_in,
        # ----------------------------------------------------------------------
        # ("biotisr-lysosome-sr-3-live-in", 0) + pos_live_lysosome_in,
        # ("biotisr-lysosome-sr-3-live-in", 1) + pos_live_lysosome_in,
        # ("biotisr-lysosome-sr-3-live-in", 2) + pos_live_lysosome_in,
        # ("biotisr-lysosome-sr-3-live-in", 3) + pos_live_lysosome_in,
        # ("biotisr-lysosome-sr-3-live-in", 4) + pos_live_lysosome_in,
        # ("biotisr-lysosome-sr-3-live-in", 5) + pos_live_lysosome_in,
        # ("biotisr-lysosome-sr-3-live-in", 6) + pos_live_lysosome_in,
        # ("biotisr-lysosome-sr-3-live-in", 7) + pos_live_lysosome_in,
        # ("biotisr-lysosome-sr-3-live-in", 8) + pos_live_lysosome_in,
        # ("biotisr-lysosome-sr-3-live-in", 9) + pos_live_lysosome_in,
        # ("biotisr-lysosome-sr-3-live-in", 10) + pos_live_lysosome_in,
        # ("biotisr-lysosome-sr-3-live-in", 11) + pos_live_lysosome_in,
        # ("biotisr-lysosome-sr-3-live-in", 12) + pos_live_lysosome_in,
        # ("biotisr-lysosome-sr-3-live-in", 13) + pos_live_lysosome_in,
        # ("biotisr-lysosome-sr-3-live-in", 14) + pos_live_lysosome_in,
        # ("biotisr-lysosome-sr-3-live-in", 15) + pos_live_lysosome_in,
        # ("biotisr-lysosome-sr-3-live-in", 16) + pos_live_lysosome_in,
        # ("biotisr-lysosome-sr-3-live-in", 17) + pos_live_lysosome_in,
        # ("biotisr-lysosome-sr-3-live-in", 18) + pos_live_lysosome_in,
        # ("biotisr-lysosome-sr-3-live-in", 19) + pos_live_lysosome_in,
    ),
}


methods_show_dict = {
    "_live_mito": (
        (
            "FluoResFM",
            "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-ft-inout-biotisr-mito-sr-3",
        ),
    ),
    "_live_lyso": (
        (
            "FluoResFM",
            "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-ft-inout-biotisr-lysosome-sr-3",
        ),
    ),
    "_live_ccp_in": (
        (
            "FluoResFM",
            "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-ft-inout-biotisr-ccp-sr-1",
        ),
    ),
    "_live_lyso_in": (
        (
            "FluoResFM",
            "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-ft-inout-biotisr-lysosome-sr-3",
        ),
    ),
}

timepoints_show = timepoint_show_dict[figure_suffix]
methods_show = methods_show_dict[figure_suffix]

assert len(timepoints_show) > 0, "No time point to show!"
assert len(methods_show) > 0, "No methods to show!"

# ------------------------------------------------------------------------------
num_timepoint, num_methods = len(timepoints_show), len(methods_show)

data_frame = pandas.read_excel("dataset_test-v2.xlsx")
# normalizer = lambda image: normalization(image, p_low=0.001, p_high=0.999)
normalizer = lambda image: normalization(image, p_low=0.03, p_high=0.995)
path_save_fig = os.path.join("results", "figures", "images", "live")
os.makedirs(path_save_fig, exist_ok=True)
path_prediction = os.path.join("results", "predictions")

print("-" * 80)
print("[INFO] Number of time points:", num_timepoint)
print("[INFO] Number of methods:", num_methods)
print("[INFO] Save figures to:", path_save_fig)
print("-" * 80)

# ------------------------------------------------------------------------------
# collect images
# ------------------------------------------------------------------------------
print("[INFO] Collect images...")
pixel_size_tpt = []
imgs_tpt = []
for i_tpt in range(num_timepoint):
    id_dataset, i_sample, win_pos, win_size, color_dataset, line_pos, ylim_prof = (
        timepoints_show[i_tpt]
    )
    path_results = os.path.join(path_prediction, id_dataset)

    ds = data_frame[data_frame["id"] == id_dataset].iloc[0]
    path_txt, path_lr, path_hr, pixel_size = (
        win2linux(ds["path_index"]),
        win2linux(ds["path_lr"]),
        win2linux(ds["path_hr"]),
        float(ds["target pixel size"].split("x")[0]) / 1000.0,
    )
    pixel_size_tpt.append(pixel_size)

    # read test txt file
    filename = read_txt(path_txt)[i_sample]
    # --------------------------------------------------------------------------
    dict_clip = {"a_min": 0.0, "a_max": 2.5}
    imgs = []
    # load gt image
    img_gt = io.imread(os.path.join(path_hr, filename))
    img_gt = interp_sf(img_gt, sf=ds["sf_hr"])[0]
    img_gt = normalizer(img_gt)
    img_gt = np.clip(img_gt, **dict_clip)

    # load raw image
    img_raw = io.imread(os.path.join(path_lr, filename))
    img_raw = interp_sf(img_raw, sf=ds["sf_lr"])[0]
    # img_raw = linear_transform(img_true=img_gt, img_test=img_raw)
    img_raw = normalizer(img_raw)
    img_raw = np.clip(img_raw, **dict_clip)
    imgs.append(img_raw)

    # load results
    for meth in methods_show:
        img_meth = io.imread(os.path.join(path_results, meth[1], filename))[0]
        # img_meth = linear_transform(img_true=img_gt, img_test=img_meth)
        img_meth = normalizer(img_meth)
        img_meth = np.clip(img_meth, **dict_clip)
        imgs.append(img_meth)

    imgs.append(img_gt)
    imgs_tpt.append(imgs)

# ------------------------------------------------------------------------------
# calculate metrics
# ------------------------------------------------------------------------------
print("[INFO] Calculate metrics...")
df_metrics = pandas.DataFrame(
    columns=[
        "timepoint",
        "method",
        "psnr",
        "ssim",
        "zncc",
        "counts",
        "diameter-mean",
        "diameter-std",
        "diameter-median",
    ]
)
methods_title = ["Raw"] + [meth[0] for meth in methods_show] + ["GT"]

lables_tpt = []
for i_tpt in range(num_timepoint):
    img_true = imgs_tpt[i_tpt][-1]
    pixel_size = pixel_size_tpt[i_tpt]
    labels_meth = []
    for i_meth in range(num_methods + 2):
        if i_meth < num_methods + 1:
            img_test = imgs_tpt[i_tpt][i_meth]
            data_range = dict_clip["a_max"] - dict_clip["a_min"]
            dict_img = {"img_true": img_gt, "img_test": img_test}
            psnr = PSNR(data_range=data_range, **dict_img)
            ssim = float(MSSSIM(data_range=data_range, **dict_img))
            zncc = ZNCC(**dict_img)
        else:
            psnr, ssim, zncc = 0, 0, 0

        # calculate counts and diameter
        img_test = imgs_tpt[i_tpt][i_meth]
        if "ccp" in figure_suffix:
            labels = pit_segmentation(
                image=img_test,
                gaussian_sigma=1.0,
                norm_range=(0.03, 0.995),
                clip_range=(0, 2.0),
                min_area_px=3,
                hole_area_px=8,
                min_peak_distance_px=3,
                otsu_thr_factor=1.0,
            )
        if "lyso" in figure_suffix:
            labels = pit_segmentation(
                image=img_test,
                gaussian_sigma=1.0,
                norm_range=(0.03, 0.995),
                clip_range=(0, 2.0),
                min_area_px=25,
                hole_area_px=8,
                min_peak_distance_px=20,
                otsu_thr_factor=1.0,
            )
            # labels = lysosome_segmentation(
            #     image=img_test,
            #     gaussian_sigma=1.0,
            #     norm_range=(0.03, 0.995),
            #     clip_range=(0, 2.0),
            # )

        labels_meth.append(labels)
        props = regionprops_table(
            labels,
            intensity_image=img_test,
            properties=["label", "area", "equivalent_diameter", "centroid"],
        )
        df = pandas.DataFrame(props)
        df["equivalent_diameter_px"] = df["equivalent_diameter"]
        df["equivalent_diameter_um"] = df["equivalent_diameter_px"] * pixel_size
        counts = len(df)
        d_mean = df["equivalent_diameter_um"].mean()
        d_std = df["equivalent_diameter_um"].std()
        d_median = df["equivalent_diameter_um"].median()

        # save metrics
        df_metrics.loc[len(df_metrics)] = [
            i_tpt * time_step,
            methods_title[i_meth],
            psnr,
            ssim,
            zncc,
            counts,
            d_mean,
            d_std,
            d_median,
        ]
    lables_tpt.append(labels_meth)

# save metrics
# path_save_metrics = os.path.join(path_save_fig, f"metrics{figure_suffix}.xlsx")

# ------------------------------------------------------------------------------
# show image
# ------------------------------------------------------------------------------
dict_fig = {"dpi": 300, "constrained_layout": True}
dict_text_struc = {"fontsize": 14, "color": "white", "ha": "left", "va": "top"}
dict_text_meth = {"fontsize": 14, "color": "white", "ha": "right", "va": "top"}
dict_text_metric = {"fontsize": 14, "color": "white", "ha": "right", "va": "bottom"}
dict_line = {"linewidth": 1, "color": "white"}
dict_rect = {"linewidth": 1, "edgecolor": "white", "facecolor": "none"}
dict_bound = {"linewidth": 1, "fill": False}

# ------------------------------------------------------------------------------
if show_images:
    print("[INFO] Plot image results of each timepoint.")
    nr, nc = num_methods + 2, num_timepoint

    fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(nc * 3, nr * 3), **dict_fig)
    fig_labels, axes_labels = plt.subplots(
        nrows=nr, ncols=nc, figsize=(nc * 3, nr * 3), **dict_fig
    )

    for i_tpt in range(num_timepoint):
        id_dataset, i_sample, win_pos, win_size, color_dataset, line_pos, ylim_prof = (
            timepoints_show[i_tpt]
        )
        imgs = imgs_tpt[i_tpt]
        pixel_size = pixel_size_tpt[i_tpt]
        num_meth = len(imgs)
        labels_meth = lables_tpt[i_tpt]

        # --------------------------------------------------------------------------
        for i_meth in range(num_meth):
            ax, img = axes[i_meth, i_tpt], imgs[i_meth]
            ax_labels, labels = axes_labels[i_meth, i_tpt], labels_meth[i_meth]

            ax.set_axis_off()
            ax_labels.set_axis_off()

            metr = df_metrics[
                (df_metrics["timepoint"] == i_tpt * time_step)
                & (df_metrics["method"] == methods_title[i_meth])
            ].iloc[0]

            # crop image to square shape
            if img.shape[1] > img.shape[0]:
                img = img[:, : img.shape[0]]
                labels = labels[:, : img.shape[0]]

            # ------------------------------------------------------------------
            img_color = colorize(img, vmin=0.0, vmax=0.9, color=color_dataset)
            ax.imshow(img_color)
            ax_labels.imshow(labels, cmap=cmap_glasbey)

            # shwo outlines
            outlines = get_outlines(labels)
            for lines in outlines:
                ax.plot(lines[:, 1], lines[:, 0], color="magenta", linewidth=0.5)

            # ------------------------------------------------------------------
            if i_meth == num_meth - 1 and i_tpt == num_timepoint - 1:
                img_shape = img.shape
                tp = 0.05
                dict_scale_bar = {
                    "pixel_size": pixel_size,
                    "bar_length": 5,  # um
                    "bar_height": 0.01,
                    "bar_color": "white",
                    "pos": (int(img_shape[1] * tp), int(img_shape[0] * (1 - tp))),
                }
                add_scale_bar(ax, image=img, **dict_scale_bar)

            # ------------------------------------------------------------------
            #  add metrics
            if "ccp" in figure_suffix:
                # show mean diameter
                diam = metr["diameter-median"]
                ax.text(
                    0.95,
                    0.05,
                    f"{diam:.4f} µm",
                    transform=ax.transAxes,
                    **dict_text_metric,
                )
            if "lyso" in figure_suffix:
                # show counts
                counts = metr["counts"]
                ax.text(
                    0.95, 0.05, f"{counts}", transform=ax.transAxes, **dict_text_metric
                )
            if i_meth == 0:
                # show time point
                ax.text(
                    0.95,
                    0.95,
                    f"{time_step*timepoints_show[i_tpt][1]} s",
                    transform=ax.transAxes,
                    **dict_text_meth,
                )
            if i_tpt == 0:
                # show method
                ax.text(
                    0.05,
                    0.95,
                    methods_title[i_meth],
                    transform=ax.transAxes,
                    **dict_text_struc,
                )

    fig.savefig(os.path.join(path_save_fig, f"timepoint_method{figure_suffix}.png"))
    fig.savefig(os.path.join(path_save_fig, f"timepoint_method{figure_suffix}.svg"))
    fig_labels.savefig(
        os.path.join(path_save_fig, f"timepoint_method{figure_suffix}_labels.png")
    )
    fig_labels.savefig(
        os.path.join(path_save_fig, f"timepoint_method{figure_suffix}_labels.svg")
    )


# ------------------------------------------------------------------------------
# show metrics change over time
# ------------------------------------------------------------------------------
print("[INFO] Plot metrics change over time ...")
colors_meth = ["#8E99AB", "#D95D5B", "#212C3E"]
nr, nc = 2, 3
fig, axes = plt.subplots(
    nrows=nr, ncols=nc, figsize=(nc * 3 / 0.75, nr * 3), **dict_fig
)
methods_wo_gt = methods_title[:-1]

dict_line = dict(
    palette=colors_meth,
    marker="o",
    hue="method",
)

for i_metric, metric in enumerate(["psnr", "ssim", "zncc"]):
    ax = axes[0, i_metric]
    data_show = df_metrics[df_metrics["method"].isin(methods_wo_gt)]
    sns.lineplot(data=data_show, x="timepoint", y=metric, ax=ax, **dict_line)

# show counts
data_show = df_metrics
sns.lineplot(data=data_show, x="timepoint", y="counts", ax=axes[1, 0], **dict_line)
sns.lineplot(
    data=data_show, x="timepoint", y="diameter-mean", ax=axes[1, 1], **dict_line
)
axes[1, 1].set_ylabel("Mean diameter (µm)")
sns.lineplot(
    data=data_show, x="timepoint", y="diameter-median", ax=axes[1, 2], **dict_line
)
axes[1, 2].set_ylabel("Median diameter (µm)")

for ax in axes.ravel():
    ax.set_xlabel("timepoint (s)")
    ax.legend()
    ax.set_box_aspect(0.75)


# save figure
plt.savefig(os.path.join(path_save_fig, f"metrics{figure_suffix}.png"))
plt.savefig(os.path.join(path_save_fig, f"metrics{figure_suffix}.svg"))

# ------------------------------------------------------------------------------
# save source data
# ------------------------------------------------------------------------------
print("[INFO] Save source data ...")
df_source_data = pandas.DataFrame(columns=["timepoint"] + methods_title)
tpt_list = np.array([tpt[1] for tpt in timepoints_show])
df_source_data["timepoint"] = tpt_list * time_step
for i_meth, meth in enumerate(methods_title):
    if "lyso" in figure_suffix:
        df_source_data[meth] = df_metrics[df_metrics["method"] == meth]["counts"].values
    if "ccp" in figure_suffix:
        df_source_data[meth] = df_metrics[df_metrics["method"] == meth][
            "diameter-median"
        ].values
path_source_data = os.path.join(path_save_fig, f"metrics{figure_suffix}.xlsx")
df_source_data.to_excel(path_source_data, index=False)
