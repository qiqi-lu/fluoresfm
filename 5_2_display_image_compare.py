"""
Show the reconstructed image of specific sample from specific dataset.
Each row of the figure corresponds to a dataset,
and each column corresponds to a method.
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
# colors = ["#F9C9C7", "#EA9A9D", "#D95D5B", "#C23637", "#912322", "#691815"]
colors = ["#F9C9C7", "#EA9A9D", "#D95D5B", "#C23637", "#912322"]
cmap_heatmap = LinearSegmentedColormap.from_list("my_cmap", colors)

# ------------------------------------------------------------------------------
#              | filename | heatmap | profile | patch | figure direction
# ------------------------------------------------------------------------------
# fig_params = ("_in_distribution_bias", True, False, True, "vertical")
# fig_params = ("_in", False, True, True, "vertical")
# fig_params = ("_ex", False, True, True, "vertical")
# fig_params = ("_in_text_image_control", False, False, True, "vertical")
# fig_params = ("_live_mito", False, False, True, "horizontal")
# fig_params = ("_live_lyso", False, False, True, "horizontal")
fig_params = ("_live_ccp", False, False, True, "horizontal")
# ------------------------------------------------------------------------------

figure_suffix, plot_heatmap, show_profile, show_patch, fig_direction = fig_params

pos_live_ccp = ((770, 170), 140, (0, 255, 0), (93, 345, 157, 269))
pos_live_lysosome = ((534, 290), 175, (255, 255, 0), (93, 345, 157, 269))

datasets_show = (
    # --------------------------------------------------------------------------
    # dataset_name | sample_id | win_pos (x, y) | win_size | color | line_pos
    # --------------------- data distribution bias (sr) ------------------------
    # ("biosr-cpp-sr-6", 0, (256, 500), 128, (0, 255, 0), (472, 508, 524, 452)),
    # ("biosr-er-sr-2", 0, (256, 256), 128, (0, 255, 0), (93, 345, 157, 269)),
    # ("biosr-mt-sr-1", 0, (241, 606), 100, (0, 255, 0), (99, 407, 142, 355)),
    # --------------------- internal datasets (dn-dcv-sr) ----------------------
    # ("biosr-actinnl-dn-1", 4, (55, 129), 100, (0, 255, 0), (295, 171, 356, 111)),
    # ("biosr+-myosin-dn-1", 4, (180, 240), 70, (255, 0, 0), (293, 221, 365, 176)),
    # ("rcan3d-c2s-npc-dcv", 0, (200, 500), 125, (255, 0, 0), (331, 407, 381, 365)),
    # ("rcan3d-c2s-mt-dcv", 0, (256, 256), 125, (255, 0, 0), (828, 73, 874, 129)),
    # ("biosr-er-sr-2", 1, (220, 150), 125, (0, 255, 0), (646, 403, 697, 421)),
    # ("deepbacs-sim-saureus-sr", 0, (193, 550), 125, (255, 0, 0), (631, 406, 748, 457)),
    # --------------------- external datasets (dn-dcv-dr) ----------------------
    # ("rcan3d-dn-mt-dn", 1, (1119, 654), 180, (0, 255, 0), (420, 330, 447, 441)),
    # ("rcan3d-dn-er-dn", 6, (694, 772), 154, (0, 255, 0), (1164, 744, 1188, 840)),
    # ("biotisr-lysosome-dcv-2", 0, (150, 128), 64, (255, 255, 0), (186, 261, 189, 308)),
    # ("sim-actin-3d-dcv", 0, (850, 300), 180, (0, 255, 0), (1380, 200, 1380, 500)),
    # ("biotisr-mt-sr-2", 1, (256, 256), 128, (0, 255, 0), (706, 308, 786, 214)),
    # ("biotisr-ccp-sr-1", 1, (256, 256), 128, (0, 255, 0), (363, 633, 384, 597)),
    # ---------------------- text-image control --------------------------------
    # ("biosr-er-dn-2", 1, (300, 90), 60, (0, 255, 0), (93, 345, 157, 269)),
    # ("biosr-er-dcv-2", 1, (300, 90), 60, (0, 255, 0), (93, 345, 157, 269)),
    # ("biosr-er-sr-2-in-ccp", 1, (600, 180), 120, (0, 255, 0), (93, 345, 157, 269)),
    # ("biosr-er-sr-2-in-mt", 1, (600, 180), 120, (0, 255, 0), (93, 345, 157, 269)),
    # ("biosr-er-sr-2", 1, (600, 180), 120, (0, 255, 0), (93, 345, 157, 269)),
    # ---------------------- live cell imaging ---------------------------------
    # ("biotisr-mito-sr-3-live", 0) + pos_live_ccp,
    # ("biotisr-mito-sr-3-live", 4) + pos_live_ccp,
    # ("biotisr-mito-sr-3-live", 8) + pos_live_ccp,
    # ("biotisr-mito-sr-3-live", 12) + pos_live_ccp,
    # ("biotisr-mito-sr-3-live", 16) + pos_live_ccp,
    # ("biotisr-lysosome-sr-3-live", 0) + pos_live_lysosome,
    # ("biotisr-lysosome-sr-3-live", 4) + pos_live_lysosome,
    # ("biotisr-lysosome-sr-3-live", 8) + pos_live_lysosome,
    # ("biotisr-lysosome-sr-3-live", 12) + pos_live_lysosome,
    # ("biotisr-lysosome-sr-3-live", 16) + pos_live_lysosome,
    ("biotisr-ccp-sr-3-live", 0) + pos_live_ccp,
    ("biotisr-ccp-sr-3-live", 4) + pos_live_ccp,
    ("biotisr-ccp-sr-3-live", 8) + pos_live_ccp,
    ("biotisr-ccp-sr-3-live", 12) + pos_live_ccp,
    ("biotisr-ccp-sr-3-live", 16) + pos_live_ccp,
)

methods_show = [
    # --------------------- data distribution bias -----------------------------
    # ("CARE:CCP", "care_biosr_sr_cpp-v2-newnorm", "CARE:CCP"),
    # ("CARE:ER", "care_biosr_sr_er-v2-newnorm", "CARE:ER"),
    # ("CARE:MT", "care_biosr_sr_mt-v2-newnorm", "CARE:MT"),
    # ("CARE:Mix", "care_biosr_sr_mix-v2-newnorm", "CARE:Mix"),
    # (
    #     "FluoResFM",
    #     "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16",
    #     "UNet-c:all-newnorm-ALL-v2-160-small-bs16",
    # ),
    # --------------------- different methods ----------------------------------
    # ("UniFMIR", "unifmir_all-newnorm-v2", "UniFMIR:all-v2"),
    # (
    #     "FluoResFM (w/o text)",
    #     "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-crossx",
    #     "UNet-c:all-newnorm-ALL-v2-160-small-bs16",
    # ),
    # (
    #     "FluoResFM",
    #     "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16",
    #     "UNet-c:all-newnorm-ALL-v2-160-small-bs16",
    # ),
    # ------------------ text-image control ------------------------------------
    # (
    #     "FluoResFM",
    #     "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16",
    #     "UNet-c:all-newnorm-ALL-v2-160-small-bs16",
    # ),
    # ------------------ live cell imaging -------------------------------------
    (
        "FluoResFM",
        "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-ft-inout-biotisr-mito-sr-3",
        "UNet-c:all-newnorm-ALL-v2-160-small-bs16-ft",
    ),
    # (
    #     "FluoResFM",
    #     "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-ft-inout-biotisr-lysosome-sr-3",
    #     "UNet-c:all-newnorm-ALL-v2-160-small-bs16-ft",
    # ),
]

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
dict_line = {"linewidth": 0.5, "color": "white"}
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
    id_dataset, i_sample, win_pos, win_size, color_dataset, line_pos = datasets_show[
        i_dataset
    ]
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
            profile = profile_line(img, (y1, x1), (y2, x2), linewidth=1)
            profile_ax = ax.inset_axes(
                [img_shape[1] - w_box, img_shape[0] - h_box * 1.5, w_box, h_box * 0.5],
                transform=ax.transData,
            )
            profile_ax.plot(profile, **dict_line)
            profile_ax.set_ylim((0, 2.0))
            # profile_ax.set_xlim((0, win_size - 1))
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
