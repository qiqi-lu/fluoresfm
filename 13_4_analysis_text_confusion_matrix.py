"""
Analysis of the influence of text condition on the results.
- confusion matrix
CCP/MT/ER
"""

import os, colorcet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from skimage import io
from skimage.measure import regionprops_table
from nellie.im_info.verifier import ImInfo, FileInfo
from nellie.utils.base_logger import logger

from utils.data import win2linux, read_txt, normalization, interp_sf
from utils.analysis import pit_segmentation, node_degree
from utils.plot import colorize, add_scale_bar, get_outlines
import logging
from scipy.io import loadmat

logging.getLogger().setLevel(logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

plt.rcParams["svg.fonttype"] = "none"
id_gpu = 0
os.environ["CUDA_VISIBLE_DEVICES"] = f"{id_gpu}"  # for nellie
logger.disable()

# ------------------------------------------------------------------------------
datasets = (
    # datasets name | true structure | input structure | sampel id (show) | method name
    ("biosr-cpp-sr-2", "CCP", "CCP", 1, "Raw"),
    ("biosr-cpp-sr-2", "CCP", "CCP", 1, "FluoResFM"),
    ("biosr-cpp-sr-2-in-mt", "CCP", "MT", 1, "FluoResFM"),
    ("biosr-cpp-sr-2-in-er", "CCP", "ER", 1, "FluoResFM"),
    ("biosr-cpp-sr-2", "CCP", "CCP", 1, "GT"),
    # # --------------------------------------------------------------------------
    ("biosr-mt-sr-2", "MT", "MT", 0, "Raw"),
    ("biosr-mt-sr-2-in-ccp", "MT", "CCP", 0, "FluoResFM"),
    ("biosr-mt-sr-2", "MT", "MT", 0, "FluoResFM"),
    ("biosr-mt-sr-2-in-er", "MT", "ER", 0, "FluoResFM"),
    ("biosr-mt-sr-2", "MT", "MT", 0, "GT"),
    # --------------------------------------------------------------------------
    # ("biosr-er-sr-2", "ER", "ER", 1, "Raw"),
    # ("biosr-er-sr-2-in-ccp", "ER", "CCP", 1, "FluoResFM"),
    # ("biosr-er-sr-2-in-mt", "ER", "MT", 1, "FluoResFM"),
    # ("biosr-er-sr-2", "ER", "ER", 1, "FluoResFM"),
    # ("biosr-er-sr-2", "ER", "ER", 1, "GT"),
    # --------------------------------------------------------------------------
    ("biosr-er-dcv-2", "ER", "ER", 6, "Raw"),
    ("biosr-er-dcv-2-in-ccp", "ER", "CCP", 6, "FluoResFM"),
    ("biosr-er-dcv-2-in-mt", "ER", "MT", 6, "FluoResFM"),
    ("biosr-er-dcv-2", "ER", "ER", 6, "FluoResFM"),
    ("biosr-er-dcv-2", "ER", "ER", 6, "GT"),
)

method_id_dict = {
    "Raw": "raw",
    "FluoResFM": "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16",
    "GT": "gt",
}

path_metadata_excel = "dataset_test-v2.xlsx"
path_prediction_root = os.path.join("results", "predictions")
path_figure = os.path.join(
    "results", "figures", "analysis", "analysis_text_confusion_matrix"
)
os.makedirs(path_figure, exist_ok=True)

df_metadata = pd.read_excel(path_metadata_excel)

print(f"-" * 80)
print(f"[INFO] Number of datasets = {len(datasets)}")
print(f'[INFO] Save figures to "{path_figure}"')

# ------------------------------------------------------------------------------
# load images
# ------------------------------------------------------------------------------
print(f"-" * 80)
print(f"[INFO] Load images ...")
dict_clip = {"a_min": 0.0, "a_max": 2.5}
normalizer = lambda image: normalization(image, p_low=0.03, p_high=0.995)

# ------------------------------------------------------------------------------
results = {}
num_datasets = len(datasets)
for i_ds in range(num_datasets):
    results_single_dataset = {}
    dataset_name, true_structure, input_structure, sample_id, method_name = datasets[
        i_ds
    ]
    method_id = method_id_dict[method_name]

    info = df_metadata[df_metadata["id"] == dataset_name].iloc[0]
    path_txt, path_lr, path_hr, pixel_size = (
        win2linux(info["path_index"]),
        win2linux(info["path_lr"]),
        win2linux(info["path_hr"]),
        float(info["target pixel size"].split("x")[0]) / 1000.0,
    )
    filenames = read_txt(path_txt)[:8]
    num_sample = len(filenames)

    imgs = []
    for i_sample in range(num_sample):
        filename = filenames[i_sample]
        if method_name == "GT":
            # load gt image
            img_gt = io.imread(os.path.join(path_hr, filename))
            img_gt = interp_sf(img_gt, sf=info["sf_hr"])[0]
            img_gt = normalizer(img_gt)
            img_gt = np.clip(img_gt, **dict_clip)
            imgs.append(img_gt)
        if method_name == "Raw":
            # load raw image
            img_raw = io.imread(os.path.join(path_lr, filename))
            img_raw = interp_sf(img_raw, sf=info["sf_lr"])[0]
            img_raw = normalizer(img_raw)
            img_raw = np.clip(img_raw, **dict_clip)
            imgs.append(img_raw)
        if method_name == "FluoResFM":
            # load results
            path_prediction = os.path.join(path_prediction_root, dataset_name)
            img_meth = io.imread(os.path.join(path_prediction, method_id, filename))[0]
            img_meth = normalizer(img_meth)
            img_meth = np.clip(img_meth, **dict_clip)
            imgs.append(img_meth)

    results_single_dataset["imgs"] = imgs
    results_single_dataset["pixel_size"] = pixel_size
    results_single_dataset["path_lr"] = path_lr
    results_single_dataset["path_hr"] = path_hr
    results_single_dataset["filenames"] = filenames
    results[f"{dataset_name}_{method_name}"] = results_single_dataset

# ------------------------------------------------------------------------------
# calculate metrics
# ------------------------------------------------------------------------------
print(f"-" * 80)
print(f"[INFO] Calculate metrics...")
for i_ds in range(num_datasets):
    print("-" * 80)
    print(f"[INFO] {datasets[i_ds]}")
    dataset_name, true_structure, input_structure, sample_id, method_name = datasets[
        i_ds
    ]
    result = results[f"{dataset_name}_{method_name}"]
    method_id = method_id_dict[method_name]

    imgs = result["imgs"]
    pixel_size = result["pixel_size"]
    path_lr = result["path_lr"]
    path_hr = result["path_hr"]
    filenames = result["filenames"]
    num_sample = len(imgs)
    # --------------------------------------------------------------------------
    # compute metrics
    if true_structure == "CCP":
        # calculate the diameter and count of pits
        labels_sample = []
        labels_sample_props = []
        median_diameter = []
        for i_sample in range(num_sample):
            img = imgs[i_sample]
            img = img.squeeze().astype(np.float32)

            labels = pit_segmentation(
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
            props = regionprops_table(
                labels,
                intensity_image=img,
                properties=["label", "area", "equivalent_diameter", "centroid"],
            )
            df = pd.DataFrame(props)
            df["equivalent_diameter_px"] = df["equivalent_diameter"]
            df["equivalent_diameter_um"] = df["equivalent_diameter_px"] * pixel_size
            labels_sample.append(labels)
            labels_sample_props.append(df)
            median_diameter.append(df["equivalent_diameter_um"].median())
        results[f"{dataset_name}_{method_name}"].update(
            {
                "labels": labels_sample,
                "labels_props": labels_sample_props,
                "median_diameter": median_diameter,
            }
        )
    elif true_structure == "MT":
        # calculate the length of microtubules
        # load labels
        if method_name == "GT":
            path_analysis = path_hr
        elif method_name == "Raw":
            path_analysis = path_lr + "_up2"
        else:
            path_analysis = os.path.join(path_prediction_root, dataset_name, method_id)

        labels_sample = []
        median_length = []
        for i_sample in range(num_sample):
            filename_analysis = filenames[i_sample].split(".")[0] + "_analysis"
            outputs = {}
            path_analysis_data = os.path.join(path_analysis, filename_analysis, "data")
            path_analysis_result = os.path.join(
                path_analysis, filename_analysis, "result"
            )

            assert os.path.exists(
                path_analysis_data
            ), f"[ERROR] {path_analysis_data} does not exist"
            assert os.path.exists(
                path_analysis_result
            ), f"[ERROR] {path_analysis_result} does not exist"

            # load filaments
            all_sorted_filament = loadmat(
                os.path.join(path_analysis_data, "all_sorted_filament.mat")
            )["all_sorted_filament"].astype(np.float32)

            R = loadmat(os.path.join(path_analysis_data, "R.mat"))["R"].astype(
                np.float32
            )

            # load junctions
            NewCrPts = loadmat(os.path.join(path_analysis_data, "NewCrPts.mat"))[
                "NewCrPts"
            ]

            # load filament length distribution
            analysis_info = loadmat(
                os.path.join(path_analysis_data, "AnalysisInfo.mat")
            )[
                "AnalysisInfo"
            ]  # ['Orientation','Total Length','End-to-End Distance','Centroid X','Centroid Y']
            analysis_info = analysis_info * np.array([1, pixel_size, 1, 1, 1])
            df_analysis_info = pd.DataFrame(
                analysis_info,
                columns=[
                    "Orientation",
                    "Total Length",
                    "End-to-End Distance",
                    "Centroid X",
                    "Centroid Y",
                ],
            )

            outputs["filaments"] = all_sorted_filament
            outputs["R"] = R
            outputs["junctions"] = NewCrPts
            outputs["analysis_info"] = analysis_info
            labels_sample.append(outputs)
            median_length.append(df_analysis_info["Total Length"].median())

        results[f"{dataset_name}_{method_name}"].update(
            {
                "labels": labels_sample,
                "median_length": median_length,
            }
        )
    elif true_structure == "ER":
        # calculate the node degree of ER
        labels_sample = []
        average_node_degree = []
        median_node_degree = []
        node_count = []
        for i_sample in range(num_sample):
            # if method_name == "GT":
            #     if "sr" in dataset_name:
            #         path_analysis = path_hr
            #     if "dcv" in dataset_name:
            #         path_analysis = path_hr + "_ave2"
            # elif method_name == "Raw":
            #     if "sr" in dataset_name:
            #         path_analysis = path_lr + "_up2"
            #     if "dcv" in dataset_name:
            #         path_analysis = path_lr
            # else:
            #     path_analysis = os.path.join(
            #         path_prediction_root, dataset_name, method_id
            #     )
            path_analysis = os.path.join(path_prediction_root, dataset_name, method_id)
            path_img = os.path.join(path_analysis, filenames[i_sample])
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

            avg_nd = np.mean(node_info["degree"])
            median_nd = np.median(node_info["degree"])
            count = len(list(node_info["degree"]))

            labels_sample.append(node_info)
            average_node_degree.append(avg_nd)
            median_node_degree.append(median_nd)
            node_count.append(count)

        results[f"{dataset_name}_{method_name}"].update(
            {
                "labels": labels_sample,
                "node_degree_average": average_node_degree,
                "node_degree_median": median_node_degree,
                "node_count": node_count,
            }
        )
    else:
        print(f'[WARRNING] Un-supported structure "{true_structure}"')

# ------------------------------------------------------------------------------
# shwo images
# ------------------------------------------------------------------------------
print(f"-" * 80)
print(f"[INFO] Show images...")
dict_fig = dict(dpi=600, constrained_layout=True)
dict_text_rt = dict(color="white", fontsize=14, ha="right", va="top", x=0.95, y=0.95)
dict_text_rb = dict(color="white", fontsize=14, ha="right", va="bottom", x=0.95, y=0.05)
dict_text_lt = dict(color="white", fontsize=14, ha="left", va="top", x=0.05, y=0.95)
color_list_glasbey = list(colorcet.cm.glasbey_dark.colors)
# dict_colors = {
#     1: "#FADCC8",
#     2: "#EC8860",
#     3: "#2F67AC",
#     4: "#B21F2B",
#     5: "#1B3E22",
#     6: "#57AA3E",
#     7: "#D4E4BF",
# }
dict_colors = {
    1: "#B21F2B",
    2: "#B21F2B",
    3: "#B21F2B",
    4: "#B21F2B",
    5: "#B21F2B",
    6: "#B21F2B",
    7: "#B21F2B",
}
# ------------------------------------------------------------------------------
nr, nc = 3, 5
fig, axes = plt.subplots(nr, nc, figsize=(nc * 3, nr * 3), **dict_fig)
axes = axes.ravel()
[ax.set_axis_off() for ax in axes]

for i_ds in range(num_datasets):
    dataset_name, true_structure, input_structure, sample_id, method_name = datasets[
        i_ds
    ]
    result = results[f"{dataset_name}_{method_name}"]
    imgs = result["imgs"]
    pixel_size = result["pixel_size"]

    # --------------------------------------------------------------------------
    # show image
    img = imgs[sample_id]
    ax = axes[i_ds]

    img_color = colorize(img, vmin=0, vmax=0.9, color=(0, 255, 0))
    ax.imshow(img_color)

    if true_structure == "CCP":
        # show pits outlines
        labels = result["labels"][sample_id]
        outlines = get_outlines(labels)
        for i in range(len(outlines)):
            ax.plot(
                outlines[i][:, 1], outlines[i][:, 0], color="magenta", linewidth=0.4
            )
        # add text to the right bottom corner
        median_diameter = result["median_diameter"][sample_id]
        ax.text(s=f"{median_diameter:.2f} µm", transform=ax.transAxes, **dict_text_rb)
    elif true_structure == "MT":
        # show filaments
        labels = result["labels"][sample_id]
        filaments = labels["filaments"]
        R = labels["R"][0, 0]
        num_filaments = filaments.shape[2]
        for i_filament in range(num_filaments):
            x = filaments[:, 0, i_filament]
            y = filaments[:, 1, i_filament]
            x = x[x != 0]
            y = y[y != 0]
            ax.plot(y - R, x - R, color="magenta", linewidth=0.4)
        # add text to the right bottom corner
        median_length = result["median_length"][sample_id]
        ax.text(s=f"{median_length:.2f} µm", transform=ax.transAxes, **dict_text_rb)
    elif true_structure == "ER":
        # show node degree
        labels = result["labels"][sample_id]
        nodes_coords = labels["coords"]
        nodes_degree = labels["degree"]
        df_nodes = pd.DataFrame(columns=["x", "y", "degree"])
        df_nodes["x"] = nodes_coords[:, 1]
        df_nodes["y"] = nodes_coords[:, 0]
        df_nodes["degree"] = nodes_degree
        # plot nodes
        sns.scatterplot(
            data=df_nodes,
            x="x",
            y="y",
            hue="degree",
            palette=dict_colors,
            s=4,
            edgecolor="white",
            linewidth=0.25,
            ax=ax,
            # legend=True,
            legend=False,
        )
        num_nodes = nodes_degree.shape[0]
        ax.text(s=f"{num_nodes} nodes", transform=ax.transAxes, **dict_text_rb)

    # add scale bar
    if method_name == "GT":
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

    # add text
    if method_name in ["Raw", "GT"]:
        ax.text(s=method_name, transform=ax.transAxes, **dict_text_rt)
    else:
        ax.text(s=f"({input_structure})", transform=ax.transAxes, **dict_text_rt)
    if method_name == "Raw":
        # add structure name at the top left
        ax.text(s=true_structure, transform=ax.transAxes, **dict_text_lt)

plt.savefig(os.path.join(path_figure, "confusion_matrix_image.png"))
plt.savefig(os.path.join(path_figure, "confusion_matrix_image.svg"))

# ------------------------------------------------------------------------------
# # show confusion matrix of metrics
# ------------------------------------------------------------------------------
# collect metrics used for plot
print(f"-" * 80)
print(f"[INFO] Collect metrics...")
df_metrics = pd.DataFrame(
    columns=[
        "dataset-method",
        "true_struture",
        "sample_id",
        "median_diameter",
        "median_length",
        "node_degree_average",
        "node_degree_median",
        "node_count",
    ]
)

for i_ds in range(num_datasets):
    dataset_name, true_structure, input_structure, sample_id, method_name = datasets[
        i_ds
    ]
    result = results[f"{dataset_name}_{method_name}"]

    for i_sample in range(len(result["filenames"])):
        head = [dataset_name + "_" + method_name, true_structure, i_sample]
        if true_structure == "CCP":
            median_diameter = result["median_diameter"]
            head.extend([median_diameter[i_sample], 0, 0, 0, 0])
        elif true_structure == "MT":
            median_length = result["median_length"]
            head.extend([0, median_length[i_sample], 0, 0, 0])
        elif true_structure == "ER":
            node_degree_average = result["node_degree_average"]
            node_degree_median = result["node_degree_median"]
            node_count = result["node_count"]
            head.extend(
                [
                    0,
                    0,
                    node_degree_average[i_sample],
                    node_degree_median[i_sample],
                    node_count[i_sample],
                ]
            )
        else:
            print(f'[WARRNING] Un-supported structure "{true_structure}"')
        df_metrics.loc[len(df_metrics)] = head
print(df_metrics)

# ------------------------------------------------------------------------------
print(f"-" * 80)
print(f"[INFO] Show confusion matrix (metrics)...")
colors_dict = {
    "CCP": ["#8E99AB", "#EA9A9D", "#D95D5B", "#C23637", "#1F662A"],
    "MT": ["#8E99AB", "#92C4E9", "#4D8FCB", "#0068A9", "#1F662A"],
    "ER": ["#8E99AB", "#CDA0CB", "#B271AB", "#9E4589", "#1F662A"],
}

metrics_show = (
    ("CCP", "median_diameter"),
    ("MT", "median_length"),
    ("ER", "node_count"),
    ("ER", "node_degree_average"),
)

# ------------------------------------------------------------------------------
nr, nc = 1, 4
fig, axes = plt.subplots(nr, nc, figsize=(nc * 3, nr * 3), **dict_fig)
axes = axes.ravel()

structures = df_metrics["true_struture"].unique()
print(f"[INFO] Structures = {structures}")

for i, (key, value) in enumerate(metrics_show):
    if key not in structures:
        print(f"[WARNING] Structure {key} not in metrics_dict")
        continue
    struc = key
    metric_name = value

    ax = axes[i]
    data = df_metrics[df_metrics["true_struture"] == struc]
    colors = colors_dict[struc]

    sns.barplot(
        data=data,
        x="dataset-method",
        y=metric_name,
        hue="dataset-method",
        errorbar=("sd", 1),
        capsize=0.2,
        ax=ax,
        palette=colors,
    )

    sns.stripplot(
        data=data,
        x="dataset-method",
        y=metric_name,
        hue="dataset-method",
        ax=ax,
        jitter=True,
        size=4,
        palette=colors,
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_xlabel(None)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_box_aspect(1)
    ax.spines[["top", "right"]].set_visible(False)

    if struc == "CCP":
        ax.set_ylabel("Median diameter (µm)", fontsize=14)
        ax.set_ylim(0.1, None)
    if struc == "MT":
        ax.set_ylabel("Median length (µm)", fontsize=14)
        ax.set_ylim(2.2, 3.8)
    if struc == "ER":
        if metric_name == "node_degree_average":
            ax.set_ylabel("Mean node degree", fontsize=14)
            ax.set_ylim(2.95, 3.15)
        if metric_name == "node_count":
            ax.set_ylabel("Node count", fontsize=14)
            # ax.set_ylim(100, 150)

fig.savefig(os.path.join(path_figure, "confusion_matrix_metrics.png"))
fig.savefig(os.path.join(path_figure, "confusion_matrix_metrics.svg"))

# save source data -------------------------------------------------------------
print(f"-" * 80)
print(f"[INFO] Save source data...")
df_metrics.to_excel(os.path.join(path_figure, "confusion_matrix_metrics.xlsx"))
print(f"[INFO] Done.")
