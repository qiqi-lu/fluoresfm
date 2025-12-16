"""
Display the results using finetuend model to do super-resolution with a scale
factor of 3, 4, and 8.
"""

import pandas, os, tqdm, seaborn
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

from utils.data import normalization, win2linux, read_txt, interp_sf
from utils.plot import colorize, add_scale_bar
from utils.evaluation import PSNR, MSSSIM, ZNCC, SQUIRREL, decorrelation_analysis

# set task to show
# task = "srx3"
# task = "srx4"
task = "srx8"


# GLOBAL SETTINGS --------------------------------------------------------------
plt.rcParams["svg.fonttype"] = "none"
GREEN, BlUE, RED, YELLOW = (0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)
# fig_direction = "vertical"  # [methods x 1]
fig_direction = "horizontal"  # [1 x methods]

# datsets and methods to show --------------------------------------------------
#               dataset name, id sample, color, patch (x, y, size)
if task == "srx3":
    dataset_show = ("biosr-factinnl-sr3-9", 0, GREEN, (938, 870, 215))
    methods_show = (
        (
            "FluoResFM",
            "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-ft-inout-biosr-factinnl-sr3-9-0.001",
            "#005D6E",
        ),
    )
    methods_colors = ["#8E99AB"] + [m[2] for m in methods_show] + ["#003245"]
    methods_name = ["WF"] + [m[0] for m in methods_show] + ["SIM"]
    bar_length_patch = 1
    dataset_id, id_sample, dataset_color, patch_pos = dataset_show
    dict_colorize = {"vmin": 0.0, "vmax": 0.9, "color": dataset_color}
    dict_colorize_patch = {"vmin": 0.0, "vmax": 1.5, "color": dataset_color}
    metrics_ticks = (
        np.linspace(0, 40, 20, endpoint=False),
        np.linspace(0, 1, 20, endpoint=False),
        np.linspace(0, 1, 10, endpoint=False),
        np.linspace(0, 1, 50, endpoint=False),
        np.linspace(0, 1, 10, endpoint=False),
        np.linspace(0, 1000, 25, endpoint=False),
    )

if task == "srx4":
    dataset_show = ("synprot-channe-0-reg", 5, GREEN, (473, 701, 72))
    methods_show = (
        (
            "FluoResFM",
            "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-ft-inout-synprot-channe-0-64-reg-0.0001",
            "#4D8FCB",
        ),
    )
    methods_colors = ["#8E99AB"] + [m[2] for m in methods_show] + ["#003245"]
    methods_name = ["confocal"] + [m[0] for m in methods_show] + ["STED"]
    bar_length_patch = 0.2
    dataset_id, id_sample, dataset_color, patch_pos = dataset_show
    dict_colorize = {"vmin": 0.0, "vmax": 0.9, "color": dataset_color}
    dict_colorize_patch = {"vmin": 0.0, "vmax": 0.9, "color": dataset_color}
    metrics_ticks = (
        np.linspace(0, 40, 40, endpoint=False),
        np.linspace(0, 1, 50, endpoint=False),
        np.linspace(0, 1, 50, endpoint=False),
        np.linspace(0, 1, 200, endpoint=False),
        np.linspace(0, 1, 50, endpoint=False),
        np.linspace(0, 1000, 10, endpoint=False),
    )


if task == "srx8":
    dataset_show = ("dl-smlm-microtubule", 1, GREEN, (598, 1508, 194))
    methods_show = (
        (
            "FluoResFM",
            "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-ft-inout-dl-smlm-microtubule-64-0.001",
            "#004586",
        ),
    )
    methods_colors = ["#8E99AB"] + [m[2] for m in methods_show] + ["#002752"]
    methods_name = ["WF"] + [m[0] for m in methods_show] + ["STORM"]
    bar_length_patch = 0.5
    dataset_id, id_sample, dataset_color, patch_pos = dataset_show
    dict_colorize = {"vmin": 0.0, "vmax": 0.9, "color": dataset_color}
    dict_colorize_patch = {"vmin": 0.0, "vmax": 0.9, "color": dataset_color}
    dict_colorize_gt = {"vmin": 0.0, "vmax": 0.4, "color": dataset_color}
    metrics_ticks = (
        np.linspace(0, 40, 20, endpoint=False),
        np.linspace(0, 1, 10, endpoint=False),
        np.linspace(0, 1, 10, endpoint=False),
        np.linspace(0, 1, 100, endpoint=False),
        np.linspace(0, 1, 10, endpoint=False),
        np.linspace(0, 1000, 10, endpoint=False),
    )

# ------------------------------------------------------------------------------
num_methods_show = len(methods_show)
num_sample_show = 8

# dataset infomation -----------------------------------------------------------
# load dataset info
path_info_excel = "dataset_test-v2.xlsx"
path_save_fig = os.path.join("results", "figures", "images", dataset_id)
os.makedirs(path_save_fig, exist_ok=True)
path_prediction = os.path.join("results", "predictions", dataset_id)

df_info = pandas.read_excel(path_info_excel)
info = df_info.loc[df_info["id"] == dataset_id].iloc[0]

path_lr = win2linux(info["path_lr"])
path_hr = win2linux(info["path_hr"])
path_index = win2linux(info["path_index"])

pixel_size_xy_input = float(info["input pixel size"].split("x")[0]) / 1000.0
pixel_size_xy = float(info["target pixel size"].split("x")[0]) / 1000.0

sf_hr = int(info["sf_hr"])
sf_lr = int(info["sf_lr"])

structure_type = info["structure"]

# print dataset info
print("-" * 80)
print(f"[INFO] Dataset:   {dataset_id}")
print(f"[INFO] Path text: {path_index}")
print(f"[INFO] Path LR:   {path_lr}")
print(f"[INFO] Path HR:   {path_hr}")
print(f"[INFO] Pixel size (xy) (in): {pixel_size_xy_input} x {pixel_size_xy_input} um")
print(f"[INFO] Pixel size (xy): {pixel_size_xy} x {pixel_size_xy} um")
print(f"[INFO] Scale factor (HR): {sf_hr}")
print(f"[INFO] Scale factor (LR): {sf_lr}")

# preprocessing settings -------------------------------------------------------
normalizer = lambda image: normalization(image, p_low=0.03, p_high=0.995)
dict_clip = {"a_min": 0.0, "a_max": 2.5}

# ------------------------------------------------------------------------------
# load image
# ------------------------------------------------------------------------------
# load all the image in the datasets
filenames = read_txt(path_index)
print("-" * 80)
print(f"[INFO] Number of samples: {len(filenames)}")
num_samples = min(len(filenames), num_sample_show)

imgs = []
for i_sample in range(num_samples):
    filename = filenames[i_sample]

    imgs_one = []
    # load the ground truth image
    img_gt = io.imread(os.path.join(path_hr, filename)).astype(np.float32)
    img_gt = interp_sf(img_gt, sf=sf_hr)
    img_gt = normalizer(img_gt)
    img_gt = np.clip(img_gt, **dict_clip)

    # load the raw image
    img_raw = io.imread(os.path.join(path_lr, filename)).astype(np.float32)
    img_raw = interp_sf(img_raw, sf=sf_lr)
    img_raw = normalizer(img_raw)
    img_raw = np.clip(img_raw, **dict_clip)
    imgs_one.append(img_raw[0])

    # load the prediction image
    for i_method in range(num_methods_show):
        method_title, method_id, method_color = methods_show[i_method]
        img_pred = io.imread(os.path.join(path_prediction, method_id, filename))
        img_pred = normalizer(img_pred)
        img_pred = np.clip(img_pred, **dict_clip)
        imgs_one.append(img_pred[0])

    imgs_one.append(img_gt[0])
    imgs.append(imgs_one)

# ------------------------------------------------------------------------------
# show image
# ------------------------------------------------------------------------------
dict_fig = {"dpi": 300, "constrained_layout": True}
dict_text_lt = {"fontsize": 14, "color": "white", "ha": "left", "va": "top"}
dict_text_rt = {"fontsize": 14, "color": "white", "ha": "right", "va": "top"}
dict_text_lb = {"fontsize": 14, "color": "white", "ha": "left", "va": "bottom"}
dict_text_rb = {"fontsize": 14, "color": "white", "ha": "right", "va": "bottom"}
dict_line = {"linewidth": 1, "color": "magenta", "linestyle": "--"}

# ------------------------------------------------------------------------------
if fig_direction == "vertical":
    nr, nc = num_methods_show + 2, 1
elif fig_direction == "horizontal":
    nr, nc = 1, num_methods_show + 2
else:
    raise ValueError(
        f"[ERROR] `fig_direction` must be 'vertical' or 'horizontal', but got {fig_direction}"
    )

fig, axes = plt.subplots(nr, nc, figsize=(nc * 3, nr * 3), **dict_fig)
[ax.set_axis_off() for ax in axes.ravel()]

fig_patch, axes_patch = plt.subplots(nc, nr, figsize=(nr * 3, nc * 3), **dict_fig)
[ax.set_axis_off() for ax in axes_patch.ravel()]

imgs_one = imgs[id_sample]

for i_method in range(num_methods_show + 2):
    ax = axes[i_method]
    ax_patch = axes_patch[i_method]

    img = imgs_one[i_method]

    # crop the image into a square shape
    min_size = min(img.shape)
    img = img[:min_size, :min_size]

    if (task == "srx8") and (i_method == num_methods_show + 1):
        # do special processing for the ground truth image in srx8 task
        img_color = colorize(img, **dict_colorize_gt)
    else:
        img_color = colorize(img, **dict_colorize)
    ax.imshow(img_color)

    img_color_patch = colorize(img, **dict_colorize_patch)
    patch_color = img_color_patch[
        patch_pos[1] : patch_pos[1] + patch_pos[2],
        patch_pos[0] : patch_pos[0] + patch_pos[2],
    ]
    ax_patch.imshow(patch_color)

    img_shape = img.shape
    patch_shape = (patch_pos[2], patch_pos[2])

    # add box of the patch in the image
    if i_method == num_methods_show + 1:
        # add the box of the patch in the image
        rect = plt.Rectangle(
            (patch_pos[0], patch_pos[1]),
            patch_pos[2],
            patch_pos[2],
            fill=False,
            edgecolor="magenta",
            linewidth=1,
        )
        ax.add_patch(rect)

    # add text -----------------------------------------------------------------
    # method name
    pos_text = (int(img_shape[1] * 0.96), int(img_shape[0] * 0.04))
    ax.text(pos_text[0], pos_text[1], methods_name[i_method], **dict_text_rt)
    # patch name
    pos_text = (int(patch_shape[1] * 0.96), int(patch_shape[0] * 0.04))
    ax_patch.text(pos_text[0], pos_text[1], methods_name[i_method], **dict_text_rt)

    # add scale bar ------------------------------------------------------------
    if i_method == num_methods_show + 1:
        tp = 0.05
        dict_scale_bar = {
            "pixel_size": pixel_size_xy,
            "bar_length": 5,  # um
            "bar_height": 0.01,
            "bar_color": "white",
            "pos": (int(img_shape[1] * tp), int(img_shape[0] * (1 - tp))),
        }
        add_scale_bar(ax, image=img, **dict_scale_bar)

        dict_scale_bar_pathch = {
            "pixel_size": pixel_size_xy,
            "bar_length": bar_length_patch,  # um
            "bar_height": 0.01,
            "bar_color": "white",
            "pos": (int(patch_pos[2] * tp), int(patch_pos[2] * (1 - tp))),
        }

        add_scale_bar(ax_patch, image=patch_color, **dict_scale_bar_pathch)


# save the figure
fig.savefig(os.path.join(path_save_fig, f"sample_{id_sample}.svg"))
fig.savefig(os.path.join(path_save_fig, f"sample_{id_sample}.png"))

fig_patch.savefig(os.path.join(path_save_fig, f"patch_{id_sample}.svg"))
fig_patch.savefig(os.path.join(path_save_fig, f"patch_{id_sample}.png"))


# ------------------------------------------------------------------------------
# quantitative evaluation
# ------------------------------------------------------------------------------
# calculate the metrics of each sample
path_source_data_npy = os.path.join(path_save_fig, "metrics.npy")
path_source_data_npy_res_gt = os.path.join(path_save_fig, "res_gt.npy")
path_source_data_xlsx = os.path.join(path_save_fig, "metrics.xlsx")

# load exsisting metric values
load_exist_metric = True
# load_exist_metric = False

if load_exist_metric:
    print("-" * 80)
    print(f"[INFO] Load existing metrics from: {path_source_data_npy}")
    metric_values = np.load(path_source_data_npy)
    res_gt = np.load(path_source_data_npy_res_gt)
else:
    print("-" * 80)
    pbar = tqdm.tqdm(total=num_samples, desc="[INFO] Calculate metrics", ncols=80)

    metric_values = []
    res_gt = []

    for i_sample in range(num_samples):
        pbar.update(1)
        imgs_one = imgs[i_sample]
        img_gt = imgs_one[-1]

        metric_methods = []

        for i_method in range(num_methods_show + 1):
            img_pred = imgs_one[i_method]
            dict_img = dict(img_true=img_gt, img_test=img_pred)
            data_range = dict_clip["a_max"] - dict_clip["a_min"]

            psnr = PSNR(data_range=data_range, **dict_img)
            msssim = MSSSIM(data_range=data_range, **dict_img)
            zncc = ZNCC(**dict_img)
            rse, rsp, emap = SQUIRREL(img=img_pred, img_ref=img_gt)
            if i_method == 0:
                pxs = pixel_size_xy_input
                if task == "srx8":
                    img_pred = interp_sf(img_pred[None], sf=-8)[0]
                elif task == "srx4":
                    img_pred = interp_sf(img_pred[None], sf=-4)[0]
                else:
                    img_pred = interp_sf(img_pred[None], sf=-sf_lr)[0]
            else:
                pxs = pixel_size_xy
            res_da, curve_da = decorrelation_analysis(img_pred, pixel_size=pxs * 1000.0)
            metric_methods.append([psnr, msssim, zncc, rse, rsp, res_da])
        # calculate the resolution of the ground truth image
        res_da_gt, curve_da_gt = decorrelation_analysis(
            img_gt, pixel_size=pixel_size_xy * 1000.0
        )
        res_gt.append(res_da_gt)
        metric_values.append(metric_methods)
    pbar.close()

metric_values = np.array(metric_values)
metrics_name = ["PSNR", "MSSSIM", "ZNCC", "RSE", "RSP", "Resolution"]
# ------------------------------------------------------------------------------
# save source data
# ------------------------------------------------------------------------------
print("-" * 80)
print(f"[INFO] Save source data to: {path_source_data_npy}")

# save into npy
np.save(path_source_data_npy, metric_values)
np.save(path_source_data_npy_res_gt, res_gt)
# save each metric into a sheet in xlsx
writer = pandas.ExcelWriter(path_source_data_xlsx, engine="xlsxwriter")
for i_metric in range(len(metrics_name)):
    metric = metrics_name[i_metric]
    values = metric_values[:, :, i_metric]
    if metric == "Resolution":
        # append gt resolution
        values = np.concatenate([values, np.array(res_gt).reshape(-1, 1)], axis=1)
        names = methods_name
    else:
        names = methods_name[:-1]
    df = pandas.DataFrame(values, columns=names)
    df.to_excel(writer, sheet_name=metric, index=True)
writer.close()


# ------------------------------------------------------------------------------

# plot the metrics -------------------------------------------------------------
print("-" * 80)
# construct frame used for seaborn boxplot
df_metrics = pandas.DataFrame(
    columns=["Method", "Metric", "Value"],
)
for i_method in range(num_methods_show + 1):
    for i_metric in range(len(metrics_name)):
        metric = metrics_name[i_metric]
        value = metric_values[:, i_method, i_metric]
        df_metrics = pandas.concat(
            [
                df_metrics,
                pandas.DataFrame(
                    {
                        "Method": [methods_name[i_method]] * len(value),
                        "Metric": [metric] * len(value),
                        "Value": value,
                    }
                ),
            ],
            ignore_index=True,
        )
# append the resolution of the ground truth image into the dataframe
df_metrics = pandas.concat(
    [
        df_metrics,
        pandas.DataFrame(
            {
                "Method": [methods_name[-1]] * len(res_gt),
                "Metric": ["Resolution"] * len(res_gt),
                "Value": res_gt,
            }
        ),
    ],
    ignore_index=True,
)
print(df_metrics)
# ------------------------------------------------------------------------------
# nr, nc = 1, len(metrics_name)
nc, nr = 1, len(metrics_name)  # vertical
fig, axes = plt.subplots(nr, nc, figsize=(3 * nc, 3 * nr), **dict_fig)

for i_metric in range(len(metrics_name)):
    metric = metrics_name[i_metric]
    ax = axes[i_metric]

    # set y ticks
    ticks = metrics_ticks[i_metric]
    if metric == "PSNR":
        ticks = np.round(ticks, 1)
    elif metric in ["MSSSIM", "ZNCC", "RSE", "RSP"]:
        ticks = np.round(ticks, 2)
    elif metric == "Resolution":
        ticks = ticks.astype(int)

    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks, fontsize=10)

    df_metric = df_metrics[df_metrics["Metric"] == metric]
    colors_tmp = methods_colors if metric == "Resolution" else methods_colors[:-1]
    seaborn.boxplot(
        data=df_metric,
        x="Method",
        y="Value",
        ax=ax,
        hue="Method",
        palette=colors_tmp,
        showfliers=False,
        fill=False,
        legend="auto",
        width=0.5,
    )

    seaborn.stripplot(
        data=df_metric,
        x="Method",
        y="Value",
        hue="Method",
        ax=ax,
        jitter=True,
        size=4,
        palette=colors_tmp,
    )

    # del the xlabel
    ax.set_xlabel("")
    ax.set_ylabel(metric)
    # del upper and left axis
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_box_aspect(2)  # y:x
    # del x ticks and ticklabels
    ax.set_xticks([])
    ax.set_xticklabels([])

    # add legend ---------------------------------------------------------------
    names_tmp = methods_name if metric == "Resolution" else methods_name[:-1]
    if (i_metric == 0) or (metric == "Resolution"):
        legend = ax.legend(
            names_tmp,
            # loc="lower right",
            labelcolor=methods_colors[:-1],
            fontsize=8,
            frameon=False,
        )
        for i, handle in enumerate(legend.legend_handles):
            handle.set_color(methods_colors[i])

# save the figure
plt.savefig(os.path.join(path_save_fig, "metrics.svg"))
plt.savefig(os.path.join(path_save_fig, "metrics.png"))
