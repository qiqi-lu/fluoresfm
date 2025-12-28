"""
Analysis the node degree of ER for all the sampels in a dataset.
- compared the raw, restored, and GT.
"""

import os, logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import io
from nellie.im_info.verifier import ImInfo, FileInfo
from nellie.utils.base_logger import logger

from utils.data import win2linux, read_txt, normalization, interp_sf
from utils.analysis import node_degree
from utils.plot import colorize, add_scale_bar

logging.getLogger().setLevel(logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

plt.rcParams["svg.fonttype"] = "none"
id_gpu = 0
os.environ["CUDA_VISIBLE_DEVICES"] = f"{id_gpu}"  # for nellie
logger.disable()

# ------------------------------------------------------------------------------
dataset_name = "biosr-er-dcv-2"
id_sample_show = 6

methods = (
    ("Raw", "raw"),
    ("FluoResFM", "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16"),
    ("GT", "gt"),
)

path_prediction = os.path.join("results", "predictions", dataset_name)
path_metadata_excel = "dataset_test-v2.xlsx"
path_figure = os.path.join(
    "results", "figures", "analysis", "analysis_er", dataset_name
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
methods_id = [method[1] for method in methods]

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
print("-" * 80)
print("[INFO] Calculate node degree...")
# # save the image and the get
# for i_meth in range(num_methods):
#     path_data_tmp = os.path.join(path_figure, "images", methods_id[i_meth])
#     os.makedirs(path_data_tmp, exist_ok=True)


nodes_sample = []
for i_sample in range(num_sample):
    nodes_meth = []
    for i_meth in range(num_methods):
        # img = imgs_sample[i_sample][i_meth]
        # path_img = os.path.join(
        #     path_figure, "images", methods_id[i_meth], filenames[i_sample]
        # )
        # io.imsave(path_img, img[None], check_contrast=False)
        path_img = os.path.join(
            path_prediction, methods_id[i_meth], filenames[i_sample]
        )

        res_xy = pixel_size
        file_info = FileInfo(path_img)
        file_info.find_metadata()
        file_info.load_metadata()
        file_info.change_axes("TYX")
        file_info.change_dim_res("T", 1)
        file_info.change_dim_res("Y", res_xy)
        file_info.change_dim_res("X", res_xy)
        im_info = ImInfo(file_info)
        node_info = node_degree(im_info, verbose=False)
        nodes_meth.append(node_info)
    nodes_sample.append(nodes_meth)


# ------------------------------------------------------------------------------
# show image
# ------------------------------------------------------------------------------
print("-" * 80)
print("[INFO] Show image...")
dict_fig = dict(dpi=600, constrained_layout=True)
dict_text_lt = dict(color="white", fontsize=14, ha="left", va="top", x=0.05, y=0.95)
dict_text_ltb = dict(color="black", fontsize=10, ha="left", va="top")
dict_text_rtb = dict(color="black", fontsize=12, ha="right", va="top", x=0.95, y=0.95)
dict_text_rb = dict(color="white", fontsize=12, ha="right", va="bottom", x=0.95, y=0.05)
dict_hist = dict(facecolor="none", edgecolor="black", linewidth=1)
dict_colors = {
    1: "#FADCC8",
    2: "#EC8860",
    3: "#2F67AC",
    4: "#B21F2B",
    5: "#1B3E22",
    6: "#57AA3E",
    7: "#D4E4BF",
}
# ------------------------------------------------------------------------------
nr, nc = num_methods, 3
fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(nc * 3, nr * 3), **dict_fig)

imgs_show = imgs_sample[id_sample_show]
nodes_show = nodes_sample[id_sample_show]

for i_meth in range(num_methods):
    img = imgs_show[i_meth]
    nodes = nodes_show[i_meth]

    # show images
    ax = axes[i_meth]
    img_color = colorize(img, vmin=0, vmax=0.9, color=(0, 255, 0))
    ax[0].imshow(img_color)
    ax[0].set_axis_off()
    ax[0].text(s=methods_name[i_meth], transform=ax[0].transAxes, **dict_text_lt)
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

    # show skel and nodes
    nodes_degree = nodes["degree"]
    nodes_coords = nodes["coords"]
    skel = nodes["pixel_class"].astype(bool)
    df_nodes = pd.DataFrame(columns=["x", "y", "degree"])
    df_nodes["x"] = nodes_coords[:, 1]
    df_nodes["y"] = nodes_coords[:, 0]
    df_nodes["degree"] = nodes_degree

    ax[1].imshow(skel, cmap="gray")
    sns.scatterplot(
        data=df_nodes,
        x="x",
        y="y",
        hue="degree",
        palette=dict_colors,
        s=3,
        edgecolor="none",
        linewidth=0.5,
        ax=ax[1],
        legend=True,
    )
    ax[1].set_axis_off()
    # add node count
    ax[1].text(
        s=f"num. of nodes: {nodes_degree.shape[0]}",
        transform=ax[1].transAxes,
        **dict_text_rb,
    )

    # show node degree histogram
    ax[2].set_yticks([0, 100, 200, 300, 400, 500, 600, 700])
    ax[2].set_yticklabels([0, 100, 200, 300, 400, 500, 600, 700])
    sns.histplot(
        data=df_nodes,
        x="degree",
        binwidth=1,
        ax=ax[2],
        kde=False,
        **dict_hist,
    )
    # add average node degree
    ax[2].text(
        s=f"{nodes_degree.mean():.2f} ({nodes_degree.std():.2f})",
        transform=ax[2].transAxes,
        **dict_text_rtb,
    )
    if i_meth == num_methods - 1:
        ax[2].set_xlabel("Node degree")
    else:
        ax[2].set_xlabel(None)
    ax[2].set_ylabel("count", fontsize=12)
    ax[2].tick_params(axis="both", which="major", labelsize=12)
    ax[2].set_xlim([1, 8])

    # add line of average node degree
    ax[2].axvline(nodes_degree.mean(), color="#C23637", linestyle="--", linewidth=1)

# save figure
fig.savefig(os.path.join(path_figure, f"sample_{id_sample_show}_node_degree.png"))
fig.savefig(os.path.join(path_figure, f"sample_{id_sample_show}_node_degree.svg"))

# save source data -------------------------------------------------------------
df_source_data = pd.DataFrame(columns=methods_name)
datas = []
for i_meth in range(num_methods):
    nodes = nodes_sample[id_sample_show][i_meth]
    data = list(nodes["degree"])
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
print("[INFO] Show all samples...")
# calculate the average node degree and node count
df_node_degree = pd.DataFrame(
    columns=["methods", "sample", "avg_node_degree", "node_count"]
)

for i_sample in range(num_sample):
    nodes_meth = nodes_sample[i_sample]
    for i_meth in range(num_methods):
        nodes = nodes_meth[i_meth]
        nodes_degree = nodes["degree"]
        avg_node_degree = nodes_degree.mean()
        node_count = nodes_degree.shape[0]
        df_node_degree.loc[len(df_node_degree)] = [
            methods_name[i_meth],
            i_sample,
            avg_node_degree,
            node_count,
        ]

# ------------------------------------------------------------------------------
colors_meth = ["#8E99AB", "#D95D5B", "#1F662A"]
nr, nc = 1, 2
fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(nc * 3, nr * 3), **dict_fig)

for ax, metric in zip(axes.ravel(), ["avg_node_degree", "node_count"]):
    sns.barplot(
        data=df_node_degree,
        x="methods",
        y=metric,
        hue="methods",
        errorbar=("sd", 1),
        capsize=0.2,
        ax=ax,
        palette=colors_meth,
        legend=True if metric == "avg_node_degree" else False,
    )

    sns.stripplot(
        data=df_node_degree,
        x="methods",
        y=metric,
        hue="methods",
        ax=ax,
        jitter=True,
        size=4,
        palette=colors_meth,
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_xlabel(None)
    ax.tick_params(axis="both", labelsize=12)
    # disbale x ticks and ticklabels
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_box_aspect(1)
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_ylabel("Average node degree", fontsize=12)
axes[1].set_ylabel("Number of nodes", fontsize=12)
axes[0].set_ylim([2.95, 3.15])

# save figure
fig.savefig(os.path.join(path_figure, "all_samples_avg_node_degree.png"))
fig.savefig(os.path.join(path_figure, "all_samples_avg_node_degree.svg"))

# save source data -------------------------------------------------------------
df_source_data_nd = pd.DataFrame(columns=methods_name)
for i_meth in range(num_methods):
    data = df_node_degree[df_node_degree["methods"] == methods_name[i_meth]][
        "avg_node_degree"
    ].values
    df_source_data_nd[methods_name[i_meth]] = data
df_source_data_nd.to_excel(
    os.path.join(path_figure, f"all_samples_avg_node_degree_source_data.xlsx"),
    index=False,
)
# ------------------------------------------------------------------------------
df_source_data_nc = pd.DataFrame(columns=methods_name)
for i_meth in range(num_methods):
    data = df_node_degree[df_node_degree["methods"] == methods_name[i_meth]][
        "node_count"
    ].values
    df_source_data_nc[methods_name[i_meth]] = data
df_source_data_nc.to_excel(
    os.path.join(path_figure, f"all_samples_node_count_source_data.xlsx"),
    index=False,
)
