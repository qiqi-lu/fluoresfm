"""
Analysis the dianmeter and counts of CCP for all the sampels in a dataset.
- compared the raw, restored, and GT.
"""

import os, colorcet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from skimage import io
from skimage.measure import regionprops_table

from utils.data import win2linux, read_txt, normalization, interp_sf
from utils.analysis import pit_segmentation
from utils.plot import colorize, add_scale_bar
import logging

logging.getLogger().setLevel(logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

plt.rcParams["svg.fonttype"] = "none"

# ------------------------------------------------------------------------------
dataset_name = "biosr-cpp-sr-1"
id_sample_show = 0

# ------------------------------------------------------------------------------
methods = (
    ("Raw", "raw"),
    ("FluoResFM", "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16"),
    ("GT", "gt"),
)

path_prediction = os.path.join("results", "predictions", dataset_name)
path_metadata_excel = "dataset_test-v2.xlsx"
path_figure = os.path.join(
    "results", "figures", "analysis", "analysis_ccp", dataset_name
)

os.makedirs(path_figure, exist_ok=True)

# ------------------------------------------------------------------------------
# load metadata
# ------------------------------------------------------------------------------
df_metadata = pd.read_excel(path_metadata_excel)
info = df_metadata[df_metadata["id"] == dataset_name].iloc[0]
path_txt, path_lr, path_hr, pixel_size = (
    win2linux(info["path_index"]),
    win2linux(info["path_lr"]),
    win2linux(info["path_hr"]),
    float(info["target pixel size"].split("x")[0]) / 1000.0,
)

num_methods = len(methods)
methods_name = [method[0] for method in methods]

filenames = read_txt(path_txt)[:8]
num_sample = len(filenames)

print("-" * 80)
print(f"[INFO] Number of samples: {len(filenames)}")

# ------------------------------------------------------------------------------
# load images
# ------------------------------------------------------------------------------
dict_clip = {"a_min": 0.0, "a_max": 2.5}
normalizer = lambda image: normalization(image, p_low=0.03, p_high=0.995)

# ------------------------------------------------------------------------------
imgs_sample = []
for i_sample in range(num_sample):
    filename = filenames[i_sample]

    imgs_meth = []
    for i_meth in range(num_methods):
        method_name, method_path = methods[i_meth]
        if method_name == "GT":
            # load gt image
            img_gt = io.imread(os.path.join(path_hr, filename))
            img_gt = interp_sf(img_gt, sf=info["sf_hr"])[0]
            img_gt = normalizer(img_gt)
            img_gt = np.clip(img_gt, **dict_clip)
            imgs_meth.append(img_gt)
        elif method_name == "Raw":
            # load raw image
            img_raw = io.imread(os.path.join(path_lr, filename))
            img_raw = interp_sf(img_raw, sf=info["sf_lr"])[0]
            img_raw = normalizer(img_raw)
            img_raw = np.clip(img_raw, **dict_clip)
            imgs_meth.append(img_raw)
        else:
            # load results
            meth_id = methods[i_meth][1]
            img_meth = io.imread(os.path.join(path_prediction, meth_id, filename))[0]
            img_meth = normalizer(img_meth)
            img_meth = np.clip(img_meth, **dict_clip)
            imgs_meth.append(img_meth)

    imgs_sample.append(imgs_meth)

# ------------------------------------------------------------------------------
# calculate metrics
# ------------------------------------------------------------------------------
lables_sample = []
df_props_sample = []
for i_sample in range(num_sample):
    lables_meth = []
    df_props_meth = []
    for i_meth in range(num_methods):
        img = imgs_sample[i_sample][i_meth]
        img = img.squeeze().astype(np.float32)

        results = pit_segmentation(
            image=img,
            gaussian_sigma=1.0,
            norm_range=(0.03, 0.995),
            clip_range=(0, 2.0),
            min_area_px=3,
            hole_area_px=8,
            min_peak_distance_px=3,
            return_intermediate=False,
            otsu_thr_factor=1.0,
        )
        labels = results
        props = regionprops_table(
            labels,
            intensity_image=img,
            properties=["label", "area", "equivalent_diameter", "centroid"],
        )
        df = pd.DataFrame(props)
        df["equivalent_diameter_px"] = df["equivalent_diameter"]
        df["equivalent_diameter_um"] = df["equivalent_diameter_px"] * pixel_size
        lables_meth.append(labels)
        df_props_meth.append(df)
    lables_sample.append(lables_meth)
    df_props_sample.append(df_props_meth)

# ------------------------------------------------------------------------------
# show image
# ------------------------------------------------------------------------------
dict_fig = dict(dpi=600, constrained_layout=True)
dict_text_lt = dict(color="white", fontsize=14, ha="left", va="top")
dict_text_rt = dict(color="black", fontsize=14, ha="right", va="top")
dict_hist = dict(facecolor="none", edgecolor="black", linewidth=1)

cmap_glasbey = [(0, 0, 0)] + list(colorcet.cm.glasbey_dark.colors)
cmap_glasbey = ListedColormap(cmap_glasbey)
# ------------------------------------------------------------------------------
nr, nc = num_methods, 3
fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(nc * 3, nr * 3), **dict_fig)

imgs_show = imgs_sample[id_sample_show]
lables_show = lables_sample[id_sample_show]
df_props_show = df_props_sample[id_sample_show]

for i_meth in range(num_methods):
    img = imgs_show[i_meth]
    labels = lables_show[i_meth]
    df_props = df_props_show[i_meth]

    ax = axes[i_meth]
    img_color = colorize(img, vmin=0, vmax=0.9, color=(0, 255, 0))
    ax[0].imshow(img_color)
    ax[0].set_axis_off()
    ax[0].text(
        0.05, 0.95, methods_name[i_meth], transform=ax[0].transAxes, **dict_text_lt
    )

    ax[1].imshow(labels, cmap=cmap_glasbey)
    ax[1].set_axis_off()
    # add scale bar
    if i_meth == 0:
        img_shape = img.shape
        tp = 0.05
        dict_scale_bar = {
            "pixel_size": pixel_size,
            "bar_length": 5,  # um
            "bar_height": 0.01,
            "bar_color": "white",
            "pos": (int(img_shape[1] * tp), int(img_shape[0] * (1 - tp))),
        }
        add_scale_bar(ax[0], image=img, **dict_scale_bar)

    # show hist of diameters
    sns.histplot(
        data=df_props,
        x="equivalent_diameter_um",
        binwidth=0.05,
        kde=True,
        ax=ax[2],
        color="#C23637",
        **dict_hist,
    )
    if i_meth == num_methods - 1:
        ax[2].set_xlabel("Diameter ($\mu$m)", fontsize=12)
    else:
        ax[2].set_xlabel(None)
    ax[2].set_ylabel("Count", fontsize=12)
    ax[2].tick_params(axis="both", labelsize=12)
    ax[2].set_xlim([0, 1.5])
    # add line of median
    median = df_props["equivalent_diameter_um"].median()
    ax[2].axvline(median, color="#C23637", linestyle="--")
    print(f"[INFO] Median diameter ({methods_name[i_meth]}): {median:.4f}")

    # add text
    ax[2].text(
        0.95,
        0.95,
        f"Median: {median:.4f} $\mu$m",
        transform=ax[2].transAxes,
        **dict_text_rt,
    )


# save figure
fig.savefig(os.path.join(path_figure, f"sample_{id_sample_show}_seg_hist.png"))
fig.savefig(os.path.join(path_figure, f"sample_{id_sample_show}_seg_hist.svg"))

# save source data -------------------------------------------------------------
df_source_data = pd.DataFrame(columns=methods_name)
datas = []
for i_meth in range(num_methods):
    df = df_props_show[i_meth]
    data = list(df["equivalent_diameter_um"].values)
    datas.append(data)
length_max = max([len(data) for data in datas])
for i_meth in range(num_methods):
    data = datas[i_meth]
    if len(data) < length_max:
        data.extend([""] * (length_max - len(data)))
    df_source_data[methods_name[i_meth]] = data

df_source_data.to_excel(
    os.path.join(path_figure, f"sample_{id_sample_show}_hist_source_data.xlsx"),
    index=False,
)


# ------------------------------------------------------------------------------
# show all samples
# ------------------------------------------------------------------------------
# calculate the median diameter of each method and each sample
df_median_diameter = pd.DataFrame(columns=["methods", "sample", "median_diameter"])

for i_sample in range(num_sample):
    df_props = df_props_sample[i_sample]
    for i_meth in range(num_methods):
        props = df_props[i_meth]
        median_diameter = props["equivalent_diameter_um"].median()
        # add to the last line
        df_median_diameter.loc[len(df_median_diameter)] = (
            methods_name[i_meth],
            i_sample,
            median_diameter,
        )

colors_meth = ["#8E99AB", "#D95D5B", "#1F662A"]
nr, nc = 1, 1
fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(nc * 3, nr * 3), **dict_fig)
sns.barplot(
    data=df_median_diameter,
    x="methods",
    y="median_diameter",
    hue="methods",
    errorbar=("sd", 1),
    capsize=0.2,
    ax=axes,
    palette=colors_meth,
)

sns.stripplot(
    data=df_median_diameter,
    x="methods",
    y="median_diameter",
    hue="methods",
    ax=axes,
    jitter=True,
    size=4,
    palette=colors_meth,
    edgecolor="white",
    linewidth=0.5,
)

axes.set_xlabel(None)
axes.set_ylabel("Median diameter (um)", fontsize=12)
axes.tick_params(axis="both", labelsize=12)
# disbale x ticks and ticklabels
axes.set_xticks([])
axes.set_xticklabels([])
axes.set_box_aspect(1)
axes.spines[["top", "right"]].set_visible(False)

# save figure
fig.savefig(os.path.join(path_figure, "all_samples_median_diameter.png"))
fig.savefig(os.path.join(path_figure, "all_samples_median_diameter.svg"))

# save source data -------------------------------------------------------------
df_source_data = pd.DataFrame(columns=methods_name)
for i_meth in range(num_methods):
    data = df_median_diameter[df_median_diameter["methods"] == methods_name[i_meth]][
        "median_diameter"
    ].values
    df_source_data[methods_name[i_meth]] = data

df_source_data.to_excel(
    os.path.join(path_figure, "all_samples_median_diameter_source_data.xlsx"),
    index=False,
)
