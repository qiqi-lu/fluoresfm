"""
Show the reconstructed image of specific sample from specific dataset.

Parameters:
- `fig_direction`: The arrangement direction of different datasets in the figure. `vertical` or `horizontal`.
"""

import numpy as np
import os, pandas
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from skimage.measure import profile_line
from utils.data import win2linux, read_txt, normalization, interp_sf
from utils.evaluation import linear_transform
from utils.plot import colorize, add_scale_bar
from utils.evaluation import PSNR, SSIM, ZNCC, MSSSIM

plt.rcParams["svg.fonttype"] = "none"
GREEN, BLUE, RED, YELLOW = (0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)

# ------------------------------------------------------------------------------
#              | filename | heatmap | profile | patch | figure direction
# ------------------------------------------------------------------------------
fig_params = ("_in_distribution_bias", True, False, True, "vertical")
# fig_params = ("_in", False, True, True, "vertical")
# fig_params = ("_ex", False, True, True, "vertical")
# fig_params = ("_in_text_image_control", False, False, True, "vertical")
# fig_params = ("_live_mito", False, False, True, "horizontal")
# fig_params = ("_live_lyso", False, False, True, "horizontal")
# fig_params = ("_live_ccp_in", False, False, True, "horizontal")
# fig_params = ("_live_lyso_in", False, False, True, "horizontal")
# ------------------------------------------------------------------------------

figure_suffix, plot_heatmap, show_profile, show_patch, fig_direction = fig_params

pos_live_mito = ((770, 170), 140, GREEN, (93, 345, 157, 269))
pos_live_lysosome = ((534, 290), 175, YELLOW, (93, 345, 157, 269))
pos_live_ccp_in = ((305, 392), 62, GREEN, (93, 345, 157, 269))
pos_live_lysosome_in = ((130, 431), 91, YELLOW, (93, 345, 157, 269))


# ------------------------------------------------------------------------------
#        dataset_name | sample_id | win_pos (x, y) | win_size | color | line_pos
# ------------------------------------------------------------------------------
datasets_show_dict = {
    # --------------------- data distribution bias (sr) ------------------------
    "_in_distribution_bias": (
        ("biosr-cpp-sr-6", 0, (256, 500), 128, GREEN, (472, 508, 524, 452)),
        ("biosr-er-sr-2", 2, (84, 374), 124, GREEN, (93, 345, 157, 269)),
        ("biosr-mt-sr-1", 7, (257, 651), 133, GREEN, (99, 407, 142, 355)),
    ),
    # --------------------- internal datasets (dn-dcv-sr) ----------------------
    "_in": (
        ("biosr-cpp-dn-1", 1, (228, 398), 60, GREEN, (136, 312, 163, 291), 1.5),
        ("biosr-actin-dn-1", 0, (122, 169), 70, GREEN, (232, 324, 271, 308), 1),
        ("biosr-mt-dcv-1", 0, (256, 255), 47, GREEN, (299, 233, 347, 244), 1.0),
        ("biosr-er-dcv-2", 0, (286, 227), 53, GREEN, (245, 187, 271, 220), 1.5),
        ("rcan3d-c2s-npc-sr", 4, (231, 598), 70, GREEN, (153, 467, 199, 489), 2),
        ("rcan3d-c2s-mt-sr", 1, (146, 429), 93, GREEN, (440, 565, 492, 596), 0.8),
    ),
    # --------------------- external datasets (dn-dcv-dr) ----------------------
    "_ex": (
        ("biotisr-ccp-dn-1", 1, (174, 159), 67, GREEN, (335, 182, 404, 241), 1.5),
        ("biotisr-factin-dn-1", 3, (293, 132), 71, GREEN, (233, 156, 235, 202), 0.75),
        ("biotisr-lysosome-dcv-2", 0, (150, 128), 64, GREEN, (186, 261, 189, 308), 1.5),
        (
            "biotisr-factin-nonlinear-dcv-1",
            0,
            (231, 107),
            50,
            GREEN,
            (292, 265, 334, 275),
            1.5,
        ),
        ("biotisr-mt-sr-1", 3, (654, 175), 129, GREEN, (335, 789, 349, 870), 1.5),
        ("biotisr-ccp-sr-3", 6, (540, 264), 70, GREEN, (694, 521, 738, 475), 1.5),
    ),
    # ---------------------- text-image control --------------------------------
    "_in_text_image_control": (
        ("biosr-er-dn-2", 1, (300, 90), 60, GREEN, (93, 345, 157, 269)),
        ("biosr-er-dcv-2", 1, (300, 90), 60, GREEN, (93, 345, 157, 269)),
        ("biosr-er-sr-2-in-ccp", 1, (600, 180), 120, GREEN, (93, 345, 157, 269)),
        ("biosr-er-sr-2-in-mt", 1, (600, 180), 120, GREEN, (93, 345, 157, 269)),
        ("biosr-er-sr-2", 1, (600, 180), 120, GREEN, (93, 345, 157, 269)),
    ),
    # ---------------------- live cell imaging ---------------------------------
    "_live_mito": (
        ("biotisr-mito-sr-3-live", 0) + pos_live_mito,
        ("biotisr-mito-sr-3-live", 4) + pos_live_mito,
        ("biotisr-mito-sr-3-live", 8) + pos_live_mito,
        ("biotisr-mito-sr-3-live", 12) + pos_live_mito,
        ("biotisr-mito-sr-3-live", 16) + pos_live_mito,
    ),
    "_live_lyso": (
        ("biotisr-lysosome-sr-3-live", 0) + pos_live_lysosome,
        ("biotisr-lysosome-sr-3-live", 4) + pos_live_lysosome,
        ("biotisr-lysosome-sr-3-live", 8) + pos_live_lysosome,
        ("biotisr-lysosome-sr-3-live", 12) + pos_live_lysosome,
        ("biotisr-lysosome-sr-3-live", 16) + pos_live_lysosome,
    ),
    "_live_ccp_in": (
        ("biotisr-ccp-sr-1-live-in", 0) + pos_live_ccp_in,
        ("biotisr-ccp-sr-1-live-in", 4) + pos_live_ccp_in,
        ("biotisr-ccp-sr-1-live-in", 8) + pos_live_ccp_in,
        ("biotisr-ccp-sr-1-live-in", 12) + pos_live_ccp_in,
        ("biotisr-ccp-sr-1-live-in", 16) + pos_live_ccp_in,
    ),
    "_live_lyso_in": (
        ("biotisr-lysosome-sr-3-live-in", 0) + pos_live_lysosome_in,
        ("biotisr-lysosome-sr-3-live-in", 4) + pos_live_lysosome_in,
        ("biotisr-lysosome-sr-3-live-in", 8) + pos_live_lysosome_in,
        ("biotisr-lysosome-sr-3-live-in", 12) + pos_live_lysosome_in,
        ("biotisr-lysosome-sr-3-live-in", 16) + pos_live_lysosome_in,
    ),
}


methods_show_dict = {
    # --------------------- data distribution bias -----------------------------
    "_in_distribution_bias": (
        ("CARE:CCP", "care_biosr_sr_cpp-v2-newnorm", "CARE:CCP"),
        ("CARE:ER", "care_biosr_sr_er-v2-newnorm", "CARE:ER"),
        ("CARE:MT", "care_biosr_sr_mt-v2-newnorm", "CARE:MT"),
        ("CARE:Mix", "care_biosr_sr_mix-v2-newnorm", "CARE:Mix"),
        (
            "FluoResFM",
            "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16",
            "UNet-c:all-newnorm-ALL-v2-160-small-bs16",
        ),
    ),
    # --------------------- different methods ----------------------------------
    "_in": (
        ("UniFMIR", "unifmir_all-newnorm-v2", "UniFMIR:all-v2"),
        (
            "FluoResFM (w/o text)",
            "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-crossx",
            "UNet-c:all-newnorm-ALL-v2-160-small-bs16",
        ),
        (
            "FluoResFM",
            "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16",
            "UNet-c:all-newnorm-ALL-v2-160-small-bs16",
        ),
    ),
    "_ex": (
        ("UniFMIR", "unifmir_all-newnorm-v2", "UniFMIR:all-v2"),
        (
            "FluoResFM (w/o text)",
            "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-crossx",
            "UNet-c:all-newnorm-ALL-v2-160-small-bs16",
        ),
        (
            "FluoResFM",
            "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16",
            "UNet-c:all-newnorm-ALL-v2-160-small-bs16",
        ),
    ),
    # ------------------ text-image control ------------------------------------
    "_in_text_image_control": (
        (
            "FluoResFM",
            "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16",
            "UNet-c:all-newnorm-ALL-v2-160-small-bs16",
        ),
    ),
    # ------------------ live cell imaging -------------------------------------
    "_live_mito": (
        (
            "FluoResFM",
            "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-ft-inout-biotisr-mito-sr-3",
            "UNet-c:all-newnorm-ALL-v2-160-small-bs16-ft",
        ),
    ),
    "_live_lyso": (
        (
            "FluoResFM",
            "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-ft-inout-biotisr-lysosome-sr-3",
            "UNet-c:all-newnorm-ALL-v2-160-small-bs16-ft",
        ),
    ),
    "_live_ccp_in": (
        (
            "FluoResFM",
            "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-ft-inout-biotisr-ccp-sr-1",
            "UNet-c:all-newnorm-ALL-v2-160-small-bs16-ft",
        ),
    ),
    "_live_lyso_in": (
        (
            "FluoResFM",
            "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-ft-inout-biotisr-lysosome-sr-3",
            "UNet-c:all-newnorm-ALL-v2-160-small-bs16-ft",
        ),
    ),
}

datasets_show = datasets_show_dict[figure_suffix]
methods_show = methods_show_dict[figure_suffix]

assert len(datasets_show) > 0, "No datasets to show!"
assert len(methods_show) > 0, "No methods to show!"

# ------------------------------------------------------------------------------
num_datasets, num_methods = len(datasets_show), len(methods_show)

data_frame = pandas.read_excel("dataset_test-v2.xlsx")
# normalizer = lambda image: normalization(image, p_low=0.001, p_high=0.999)
normalizer = lambda image: normalization(image, p_low=0.03, p_high=0.995)
path_save_fig = os.path.join("results", "figures", "images")
path_prediction = os.path.join("results", "predictions")
os.makedirs(path_save_fig, exist_ok=True)

print("-" * 50)
print("Number of datasets:", num_datasets)
print("Number of methods:", num_methods)
print("Save figures to:", path_save_fig)
print("-" * 50)

dict_fig = {"dpi": 300, "constrained_layout": True}
dict_text_struc = {"fontsize": 14, "color": "white", "ha": "left", "va": "top"}
dict_text_meth = {"fontsize": 14, "color": "white", "ha": "right", "va": "top"}
dict_text_metric = {"fontsize": 14, "color": "white", "ha": "left", "va": "bottom"}
dict_line = {"linewidth": 1, "color": "white"}
dict_rect = {"linewidth": 1, "edgecolor": "white", "facecolor": "none"}
dict_bound = {"linewidth": 1, "fill": False}

# ------------------------------------------------------------------------------
print("Plot image results of each dataset.")

if fig_direction == "vertical":
    nr, nc = num_datasets, num_methods + 2
elif fig_direction == "horizontal":
    nr, nc = num_methods + 2, num_datasets
else:
    raise ValueError("Invalid figure direction.")

fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(nc * 3, nr * 3), **dict_fig)

for i_dataset in range(num_datasets):
    if len(datasets_show[i_dataset]) == 6:
        id_dataset, i_sample, win_pos, win_size, color_dataset, line_pos = (
            datasets_show[i_dataset]
        )
        ylim_prof = 2.0
    elif len(datasets_show[i_dataset]) == 7:
        id_dataset, i_sample, win_pos, win_size, color_dataset, line_pos, ylim_prof = (
            datasets_show[i_dataset]
        )
    else:
        raise ValueError("Invalid dataset information.")

    print("-" * 80)
    print("Dataset:", id_dataset)
    # --------------------------------------------------------------------------
    path_results = os.path.join(path_prediction, id_dataset)

    ds = data_frame[data_frame["id"] == id_dataset].iloc[0]
    path_txt, path_lr, path_hr, pixel_size = (
        win2linux(ds["path_index"]),
        win2linux(ds["path_lr"]),
        win2linux(ds["path_hr"]),
        float(ds["target pixel size"].split("x")[0]) / 1000.0,
    )

    # read test txt file
    filename = read_txt(path_txt)[i_sample]

    print("Path text:", path_txt)
    print("Path LR:", path_lr)
    print("Path HR:", path_hr)
    print("Pixel size:", f"{pixel_size:.4f}")
    print("Filename:", filename)
    print("-" * 80)
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

    # --------------------------------------------------------------------------
    num_img = len(imgs)
    methods_title = ["Raw"] + [meth[0] for meth in methods_show] + ["GT"]
    for i_img in range(num_img):

        if fig_direction == "vertical":
            ax, img = axes[i_dataset, i_img], imgs[i_img]
        elif fig_direction == "horizontal":
            ax, img = axes[i_img, i_dataset], imgs[i_img]
        else:
            raise ValueError("Invalid figure direction.")

        ax.set_axis_off()

        # crop image to square shape
        if img.shape[1] > img.shape[0]:
            img = img[:, : img.shape[0]]
            img_gt = img_gt[:, : img.shape[0]]

        # ----------------------------------------------------------------------
        img_color = colorize(img, vmin=0.0, vmax=0.9, color=color_dataset)
        ax.imshow(img_color)
        # ----------------------------------------------------------------------

        if i_img == num_img - 1:
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

        # ----------------------------------------------------------------------
        #  add metrics
        pos_text = (int(img.shape[0] * 0.04), int(img.shape[1] * 0.04))

        if i_img != num_img - 1:
            data_range = dict_clip["a_max"] - dict_clip["a_min"]
            dict_img = {"img_true": img_gt, "img_test": img}
            psnr = PSNR(data_range=data_range, **dict_img)
            # ssim = SSIM(data_range=data_range, **dict_img)
            ssim = MSSSIM(data_range=data_range, **dict_img)
            zncc = ZNCC(**dict_img)
            ax.text(
                pos_text[1],
                img.shape[0] - pos_text[0],
                f"{psnr:.2f} | {ssim*100:.2f}",
                # f"{psnr:.2f} | {ssim*100:.2f} | {zncc*100:.2f}",
                **dict_text_metric,
            )

        if i_dataset == 0:
            # right aligned text
            ax.text(
                img.shape[1] - pos_text[1],
                pos_text[0],
                methods_title[i_img],
                **dict_text_meth,
            )
        if i_img == 0:
            ax.text(pos_text[1], pos_text[0], f'{ds["structure"]}', **dict_text_struc)

        # ----------------------------------------------------------------------
        if show_patch:
            # rectangle in the image
            if i_img == num_img - 1:
                rect = plt.Rectangle(
                    (win_pos[0], win_pos[1]), win_size, win_size, **dict_rect
                )
                ax.add_patch(rect)

            # ------------------------------------------------------------------
            # cropped patch at the bottom right of image
            img_shape = img_color.shape
            patch = img_color[
                win_pos[1] : win_pos[1] + win_size, win_pos[0] : win_pos[0] + win_size
            ]
            # flip patch
            patch = np.flipud(patch)
            w_box, h_box = (max(img_shape[1] // 3, img_shape[0] // 3),) * 2
            patch_ax = ax.inset_axes(
                [img_shape[1] - w_box, img_shape[0] - h_box, w_box, h_box],
                transform=ax.transData,
            )
            patch_ax.imshow(patch)
            patch_ax.set_xlim((0, win_size - 1))
            patch_ax.set_ylim((0, win_size - 1))
            # del ticks and tick labels
            patch_ax.set_yticks([])
            patch_ax.set_yticklabels([])
            patch_ax.set_xticks([])
            patch_ax.set_xticklabels([])
            # set the color of axes to white and width to 1
            patch_ax.spines["top"].set_color("white")
            patch_ax.spines["left"].set_color("white")
            patch_ax.spines["top"].set_linewidth(1)
            patch_ax.spines["left"].set_linewidth(1)
            # del the left and bottom spines
            patch_ax.spines["right"].set_visible(False)
            patch_ax.spines["bottom"].set_visible(False)

        # ----------------------------------------------------------------------
        if show_profile:
            # profile line
            x1, y1, x2, y2 = line_pos
            if i_img == num_img - 1:
                # plot the line in the image
                ax.plot([x1, x2], [y1, y2], linestyle="--", **dict_line)
            profile = profile_line(
                # np.mean(img_color, axis=-1), (y1, x1), (y2, x2), linewidth=1
                img,
                (y1, x1),
                (y2, x2),
                linewidth=1,
            )
            profile_ax = ax.inset_axes(
                [img_shape[1] - w_box, img_shape[0] - h_box * 1.5, w_box, h_box * 0.5],
                transform=ax.transData,
            )
            profile_ax.plot(profile, **dict_line)
            profile_ax.set_ylim((0, ylim_prof))
            profile_ax.set_axis_off()

plt.savefig(os.path.join(path_save_fig, f"structure_method{figure_suffix}.png"))
plt.savefig(os.path.join(path_save_fig, f"structure_method{figure_suffix}.svg"))


# ------------------------------------------------------------------------------
# show heatmap
# load reaults and calculate the mean of each methods
if plot_heatmap:
    print("-" * 50)
    print("plot heatmap...")

    metrics_heatmap = ["PSNR", "MSSSIM", "ZNCC"]
    methods_heatmap = ["raw"] + [x[2] for x in methods_show]
    datsets_heatmap = [x[0] for x in datasets_show]
    x_ticks = ["raw"] + [x[0] for x in methods_show]
    y_ticks = []
    for id_dataset in datsets_heatmap:
        ds = data_frame[data_frame["id"] == id_dataset].iloc[0]
        y_ticks.append(f"{ds['structure']}")

    results_map = []
    for metric in metrics_heatmap:
        metric_mean = []
        for id_dataset in datsets_heatmap:

            # load the result excel
            path_results = os.path.join(path_prediction, id_dataset, "metrics.xlsx")
            df = pandas.read_excel(path_results, sheet_name=metric)
            # calculate the mean of each methods
            tmp = []
            for meth in methods_heatmap:
                tmp.append(df[meth].mean())
            metric_mean.append(tmp)
        results_map.append(metric_mean)
    results_map = np.array(results_map)  # (metrics,datasets, methods)
    print(results_map.shape)

    # --------------------------------------------------------------------------
    # plot the heatmap
    nc, nr = 1, len(metrics_heatmap)
    fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(3 * nc, 3 * nr), **dict_fig)
    for i_metric in range(len(metrics_heatmap)):
        ax = axes[i_metric]

        sns.heatmap(
            results_map[i_metric],
            ax=ax,
            square=True,
            cbar_kws={"shrink": 0.4},
            xticklabels=x_ticks,
            yticklabels=y_ticks,
            cmap="rocket",
            # cmap="plasma",
            # cmap=cmap_heatmap,
        )
        ax.set_title(metrics_heatmap[i_metric])

    # plt.tight_layout(h_pad=1)
    plt.savefig(
        os.path.join(path_save_fig, f"structure_method{figure_suffix}_heatmap.png")
    )
    plt.savefig(
        os.path.join(path_save_fig, f"structure_method{figure_suffix}_heatmap.svg")
    )

    # save source data
    writer = pandas.ExcelWriter(
        os.path.join(path_save_fig, f"structure_method{figure_suffix}_heatmap.xlsx")
    )
    for i_metric, metric in enumerate(metrics_heatmap):
        df_save = pandas.DataFrame(
            results_map[i_metric], columns=x_ticks, index=y_ticks
        )
        df_save.to_excel(writer, sheet_name=metric)
    writer.close()
