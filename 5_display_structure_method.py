import numpy as np
import os, pandas
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.measure import profile_line
from utils.data import win2linux, read_txt, normalization, interp_sf
from utils.evaluation import linear_transform, linear_transform_threshold
from utils.plot import colorize, colorbar, add_scale_bar
from utils.evaluation import PSNR, SSIM, ZNCC
import seaborn as sns

plt.rcParams["svg.fonttype"] = "none"

plot_heatmap = False
# figure_suffix = "_ex_bad_text"
figure_suffix, plot_heatmap = "_in_distribution_bias", True
# figure_suffix = "_in"
# figure_suffix = "_ex"
# figure_suffix = "_ex_sr"
# figure_suffix = "_ex_dcv"
# figure_suffix = "_ex_dn"

show_small_patch = True
# show_small_patch = False
show_profile = True
# show_profile = False


datasets_show = (
    # dataset_name, sample_id, win_pos, win_size, color, line_pos
    # --------------------------------------------------------------------------
    # distribution bias (sr)
    # ("biosr-cpp-sr-2", 0, (256, 500), 128, (0, 255, 0), (472, 508, 524, 452)),
    # ("biosr-cpp-sr-1", 0, (256, 500), 128, (0, 255, 0), (472, 508, 524, 452)),
    ("biosr-cpp-sr-6", 0, (256, 500), 128, (0, 255, 0), (472, 508, 524, 452)),
    ("biosr-er-sr-2", 0, (256, 256), 128, (0, 255, 0), (93, 345, 157, 269)),
    ("biosr-mt-sr-1", 0, (241, 606), 100, (0, 255, 0), (99, 407, 142, 355)),
    # --------------------------------------------------------------------------
    # internal datasets (dn-dcv-sr)
    # ("biosr+-mt-dn-1", 0, (104, 247), 75, (0, 255, 0), (295, 171, 356, 111)),
    # ("biosr+-myosin-dn-1", 4, (180, 240), 70, (255, 0, 0), (293, 221, 365, 176)),
    # ("rcan3d-c2s-npc-dcv", 0, (200, 500), 125, (255, 0, 0), (331, 407, 381, 365)),
    # ("rcan3d-c2s-mt-dcv", 0, (256, 256), 125, (255, 0, 0), (828, 73, 874, 129)),
    # ("deepbacs-sim-ecoli-sr", 0, (114, 72), 125, (0, 255, 0), (646, 503, 697, 421)),
    # ("deepbacs-sim-saureus-sr", 0, (193, 550), 125, (255, 0, 0), (631, 406, 748, 457)),
    # --------------------------------------------------------------------------
    # external datasets (dn-dcv-dr)
    # ("rcan3d-dn-mt-dn", 1, (1119, 654), 180, (0, 255, 0), (420, 330, 447, 441)),
    # ("rcan3d-dn-er-dn", 6, (694, 772), 154, (0, 255, 0), (1164, 744, 1188, 840)),
    # ("biotisr-lysosome-dcv-2", 0, (150, 128), 64, (255, 255, 0), (186, 261, 189, 308)),
    # ("sim-actin-3d-dcv", 0, (850, 300), 180, (0, 255, 0), (1380, 200, 1380, 500)),
    # ("biotisr-mt-sr-1", 1, (256, 256), 128, (0, 255, 0), (706, 308, 786, 214)),
    # ("biotisr-mt-sr-2", 1, (256, 256), 128, (0, 255, 0), (706, 308, 786, 214)),
    # ("biotisr-mt-sr-3", 1, (256, 256), 128, (0, 255, 0), (706, 308, 786, 214)),
    # ("biotisr-ccp-sr-1", 1, (256, 256), 128, (0, 255, 0), (363, 633, 384, 597)),
    # --------------------------------------------------------------------------
    # ("bpae-dn", 2, (120, 120), 50, (0, 255, 0), (100, 200, 100, 250)),
    # ("biotisr-mt-dcv-1", 1, (128, 128), 64, (0, 255, 0), (250, 250, 250, 300)),
    # ("biotisr-lysosome-sr-1", 0, (300, 256), 128, (255, 255, 0), (250, 250, 250, 300)),
    # ("biosr-actin-sr-1", 1, (462, 429), 100, (0, 255, 0), (611, 505, 656, 590)),
    # ("biosr-actin-sr-1", 0, (462, 429), 100, (0, 255, 0), (611, 505, 656, 590)),
    # ("biotisr-ccp-sr-1", 1, (256, 256), 128, (0, 255, 0), (611, 505, 656, 590)),
    # ("biotisr-factin-sr-1", 0, (256, 256), 128, (0, 255, 0)),
    # ("biotisr-ccp-dcv-1", 1, (128, 128), 64, (0, 255, 0), (250, 250, 250, 300)),
    # ("biotisr-mito-dcv-1", 1, (128, 128), 64, (0, 255, 0), (250, 250, 250, 300)),
    # ("biotisr-factin-dcv-1", 0, (128, 128), 64, (0, 255, 0), (128, 128, 192, 192)),
    # ("sim-actin-2d-patch-dcv", 0, (650, 450), 100, (0, 255, 0), (260, 290, 260, 375)),
    # ("biotisr-ccp-dn-1", 1, (256, 256), 128, (0, 255, 0)),
    # ("biotisr-mt-dn-1", 1, (256, 256), 128, (0, 255, 0)),
    # ("biotisr-mito-dn-1", 1, (256, 256), 128, (0, 255, 0)),
    # ("biotisr-lysosome-dn-1", 0, (300, 256), 128, (255, 255, 0)),
    # ("biotisr-mito-sr-1", 1, (256, 256), 128, (0, 255, 0), (706, 308, 786, 214)),
    # ("biotisr-mito-sr-2", 1, (256, 256), 128, (0, 255, 0), (706, 308, 786, 214)),
    # ("biotisr-mito-sr-3", 1, (256, 256), 128, (0, 255, 0), (706, 308, 786, 214)),
    # ("biotisr-factin-dn-1", 0, (256, 256), 128, (0, 255, 0)),
    # ("rcan3d-dn-er-dn", 1, (750, 550), 150, (0, 255, 0)),
    # ("rcan3d-dn-golgi-dn", 1, (700, 300), 150, (0, 255, 0)),
)

methods_show = [
    ("CARE:CCP", "care_biosr_sr_cpp", "CARE:CCP"),
    ("CARE:ER", "care_biosr_sr_er", "CARE:ER"),
    ("CARE:MT", "care_biosr_sr_mt", "CARE:MT"),
    ("CARE:Mix", "care_sr", "CARE:Mix"),
    # ("CARE:sr-biosr-actin", "care_biosr_sr_actin"),
    # ("DFCAN:SR", "dfcan_sr_2"),
    # ("UniFMIR", "unifmir_all"),
    # ("FluoResFM (w/o text)", "unet_sd_c_all_cross"),
    # ("FluoResFM", "unet_sd_c_all"),
    # ("FluoResFM (w/o text)", "unet_sd_c_all_crossx_newnorm"),
    # ("FluoResFM:TS", "unet_sd_c_all_newnorm_TS"),
    # ("FluoResFM:TSmicro", "unet_sd_c_all_newnorm_TSmicro"),
    # ("FluoResFM:TSpixel", "unet_sd_c_all_newnorm_TSpixel"),
    # ("FluoResFM:v2", "unet_sd_c_all_newnorm-ALL-v2"),
    # ("FluoResFM-1", "unet_sd_c_all_newnorm"),
    # ("FluoResFM:v2-s-bs8", "unet_sd_c_all_newnorm-ALL-v2-clip-160-small-bs8"),
    (
        "FluoResFM",
        "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16",
        "UNet-c:all-newnorm-ALL-v2-160-small-bs16",
    ),
    # ("FluoResFM-3", "unet_sd_c_all_newnorm-ALL-v2-160-s123-bs16"),
]

# ------------------------------------------------------------------------------
num_datasets = len(datasets_show)
num_methods = len(methods_show)
print("-" * 50)
print("Number of datasets:", num_datasets)
print("Number of methods:", num_methods)
print("-" * 50)

data_frame = pandas.read_excel("dataset_test-v2.xlsx")
# normalizer = lambda image: normalization(image, p_low=0.001, p_high=0.999)
normalizer = lambda image: normalization(image, p_low=0.03, p_high=0.995)
path_save_fig = os.path.join("results", "figures", "images")
os.makedirs(path_save_fig, exist_ok=True)
print("Save figures to:", path_save_fig)

# ------------------------------------------------------------------------------
nr, nc = num_datasets, num_methods + 2
fig, axes = plt.subplots(
    nrows=nr, ncols=nc, dpi=300, figsize=(nc * 3, nr * 3), constrained_layout=True
)

for i_dataset in range(num_datasets):
    id_dataset, i_sample, win_pos, win_size, color_dataset, line_pos = datasets_show[
        i_dataset
    ]
    print("Dataset:", id_dataset)
    # --------------------------------------------------------------------------
    path_results = os.path.join("results", "predictions", id_dataset)
    ds = data_frame[data_frame["id"] == id_dataset].iloc[0]
    if os.name == "posix":
        path_txt, path_lr, path_hr, pixel_size = (
            win2linux(ds["path_index"]),
            win2linux(ds["path_lr"]),
            win2linux(ds["path_hr"]),
            float(ds["target pixel size"].split("x")[0]) / 1000.0,
        )

    # read test txt file
    filenames = read_txt(path_txt)
    filename = filenames[i_sample]

    imgs = []
    # load gt image
    img_gt = io.imread(os.path.join(path_hr, filename))
    img_gt = interp_sf(img_gt, sf=ds["sf_hr"])[0]
    img_gt = normalizer(img_gt)
    # img_gt = np.clip(img_gt, 0.0, 1.0)
    img_gt = np.clip(img_gt, 0.0, 2.5)

    # load raw image
    img_raw = io.imread(os.path.join(path_lr, filename))
    img_raw = interp_sf(img_raw, sf=ds["sf_lr"])[0]
    img_raw = linear_transform(img_true=img_gt, img_test=img_raw)
    # img_raw = linear_transform_threshold(img_gt, img_raw, threshold=0.5, side="left")
    # img_raw = np.clip(img_raw, 0.0, 1.0)
    img_raw = np.clip(img_raw, 0.0, 2.5)
    imgs.append(img_raw)

    # load results
    for meth in methods_show:
        img_meth = io.imread(os.path.join(path_results, meth[1], filename))[0]
        img_meth = linear_transform(img_true=img_gt, img_test=img_meth)
        # img_meth = linear_transform_threshold(
        #     img_gt, img_meth, threshold=0.5, side="left"
        # )
        # img_meth = np.clip(img_meth, 0.0, 1.0)
        img_meth = np.clip(img_meth, 0.0, 2.5)
        imgs.append(img_meth)

    imgs.append(img_gt)

    # --------------------------------------------------------------------------
    num_img = len(imgs)
    methods_title = ["Raw"] + [meth[0] for meth in methods_show] + ["GT"]
    for i_img in range(num_img):
        ax = axes[i_dataset, i_img]
        ax.set_axis_off()
        img = imgs[i_img]

        if img.shape[1] > img.shape[0]:
            # crop image to square shape
            img = img[:, : img.shape[0]]
            img_gt = img_gt[:, : img.shape[0]]

        pos_text = (int(img.shape[0] * 0.04), int(img.shape[1] * 0.04))
        dict_text_struc = {"fontsize": 14, "color": "white", "ha": "left", "va": "top"}
        dict_text_meth = {"fontsize": 14, "color": "white", "ha": "right", "va": "top"}
        dict_text_metric = {
            "fontsize": 12,
            "color": "white",
            "ha": "left",
            "va": "bottom",
        }
        dict_line = {"linewidth": 0.5, "color": "white"}
        dict_rect = {"linewidth": 1, "edgecolor": "white", "facecolor": "none"}
        dict_bound = {"linewidth": 1, "fill": False}

        # ----------------------------------------------------------------------
        img_color = colorize(img, vmin=0.0, vmax=0.9, color=color_dataset)
        ax.imshow(img_color)

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
        if i_img != num_img - 1:
            data_range = 2.5
            # data_range = 1.0
            psnr = PSNR(img_true=img_gt, img_test=img, data_range=data_range)
            ssim = SSIM(img_true=img_gt, img_test=img, data_range=data_range)
            zncc = ZNCC(img_true=img_gt, img_test=img)
            ax.text(
                pos_text[1],
                img.shape[0] - pos_text[0],
                f"{psnr:.2f} | {ssim*100:.2f} | {zncc*100:.2f}",
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
        if show_small_patch:
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

    metrics_heatmap = ["PSNR", "SSIM", "ZNCC"]
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
            path_results = os.path.join(
                "results", "predictions", id_dataset, "metrics.xlsx"
            )
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
    fig, axes = plt.subplots(
        dpi=300, nrows=nr, ncols=nc, figsize=(3 * nc, 3 * nr), constrained_layout=False
    )

    # dict_xticks = {"rotation": 45, "rotation_mode": "anchor", "ha": "right"}

    for i_metric in range(len(metrics_heatmap)):
        ax = axes[i_metric]

        # im = ax.imshow(results_map[i_metric], cmap="magma")
        # colorbar(im)
        # ax.set_xticks(np.arange(len(x_ticks)), labels=x_ticks, **dict_xticks)
        # ax.set_yticks(np.arange(len(y_ticks)), labels=y_ticks)

        sns.heatmap(
            results_map[i_metric],
            ax=ax,
            square=True,
            cbar_kws={"shrink": 0.4},
            xticklabels=x_ticks,
            yticklabels=y_ticks,
        )
        ax.set_title(metrics_heatmap[i_metric])

    plt.tight_layout(h_pad=1)
    plt.savefig(
        os.path.join(path_save_fig, f"structure_method{figure_suffix}_heatmap.png")
    )
    plt.savefig(
        os.path.join(path_save_fig, f"structure_method{figure_suffix}_heatmap.svg")
    )
