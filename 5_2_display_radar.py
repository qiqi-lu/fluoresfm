"""
Radar plot of the results of external datasets.
Only show the mean of each method.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas, os
from utils.plot import cal_radar_range
from dataset_analysis import dataset_names_radar

plt.rcParams["svg.fonttype"] = "none"

# ------------------------------------------------------------------------------
# dataset_group = "internal_dataset"
dataset_group = "external_dataset"

prefix = "compare_different_methods"
# prefix = "compare_different_text"


methods_info = (
    ("FluoResFM", "UNet-c:all-newnorm-ALL-v2-160-small-bs16", "#FF0000"),
    # ("FluoResFM", "UNet-c:all-newnorm-ALL-v2-160-small-bs8", "#FF0000"),
    # ("FluoResFM", "UNet-c:all-newnorm-ALL-v2-160-small-bs4", "#FF0000"),
    # ("FluoResFM-T", "UNet-c:all-newnorm-ALL-v2-small-bs16-T77"),
    # ("FluoResFM-TS", "UNet-c:all-newnorm-ALL-v2-small-bs16-TS77"),
    # ("FluoResFM-TSpixel", "UNet-c:all-newnorm-ALL-v2-small-bs16-TSpixel77"),
    # ("FluoResFM-TSmicro", "UNet-c:all-newnorm-ALL-v2-small-bs16-TSmicro77"),
    # ("FluoResFM-bs16", "UNet-c:all-newnorm-ALL-v2-160-small-bs16", "#4D8FCB"),
    # ("FluoResFM-bs8", "UNet-c:all-newnorm-ALL-v2-160-small-bs8", "#92C4E9"),
    # ("FluoResFM-bs4", "UNet-c:all-newnorm-ALL-v2-160-small-bs4", "#C1E4FA"),
    (
        "FluoResFM (w/o text)",
        "UNet-c:all-newnorm-ALL-v2-160-small-bs16-crossx",
        "#2962FF",
    ),
    ("UniFMIR", "UniFMIR:all-v2", "#00810A"),
    ("Raw", "raw", "#212C3E"),
)

colors_task = {"sr": "#D95D5B", "dcv": "#57AA3E", "dn": "#4D8FCB"}

# ------------------------------------------------------------------------------
# datasets
id_dataset_show = dataset_names_radar[dataset_group]
num_dataset_show = len(id_dataset_show)

# methods
titles = [meth[0] for meth in methods_info]
methods = [meth[1] + "-mean" for meth in methods_info]
colors_meth = [meth[2] for meth in methods_info]

# metrics
# metrics = ["PSNR", "SSIM", "ZNCC"]
metrics = ["PSNR", "MSSSIM", "ZNCC"]
metrics_precision = [0.1, 0.001, 0.001, 0.005, 0.005]
metrics_range = (0.2, 0.95)  # relative range of metric
num_metrics = len(metrics)

# file path
path_statistic = os.path.join("results", "statistic", dataset_group)
path_figure = os.path.join("results", "figures", "analysis", dataset_group)
path_xlsx = os.path.join(path_statistic, "all_mean_std_pvalue.xlsx")

print("-" * 80)
print("dataset_group:", dataset_group)
print("prefix:", prefix)
print("Methods:", methods)
print("Metrics:", metrics)
print("Number of dataset (show):", num_dataset_show)
print("-" * 80)

# ------------------------------------------------------------------------------
fig, axes = plt.subplots(
    nrows=1,
    ncols=num_metrics,
    figsize=(12 * num_metrics, 12),
    subplot_kw=dict(polar=True),
    dpi=300,
)

# Compute angle for each dataset
angles = np.linspace(0, 2 * np.pi, num_dataset_show, endpoint=False).tolist()
angles += angles[:1]

dict_ring = {"linewidth": 20, "linestyle": "solid"}
dict_line = {"linewidth": 1.5, "linestyle": "solid"}

for i_metric, metric in enumerate(metrics):
    ax = axes[i_metric]
    print("-" * 80)
    print(f"Metric: {metric}")

    df_metric = pandas.read_excel(path_xlsx, sheet_name=metric)[
        ["dataset-name", "task"] + methods
    ]
    df_metric = df_metric[df_metric["dataset-name"].isin(id_dataset_show)]
    df_metric = df_metric.set_index("dataset-name").loc[id_dataset_show].reset_index()
    metrics_value = np.array(df_metric[methods])

    tasks = list(df_metric["task"])
    num_sr, num_dcv, num_dn = tasks.count("sr"), tasks.count("dcv"), tasks.count("dn")

    data_range_h, data_range_l = cal_radar_range(
        metrics_value, percent=metrics_range, precision=metrics_precision[i_metric]
    )

    # --------------------------------------------------------------------------
    # Create the plot
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # add a circular ring around the plot
    ax.plot(angles[0:num_sr], [1] * num_sr, color=colors_task["sr"], **dict_ring)
    ax.plot(
        angles[num_sr : num_sr + num_dcv],
        [1] * num_dcv,
        color=colors_task["dcv"],
        **dict_ring,
    )
    ax.plot(
        angles[num_sr + num_dcv : -1],
        [1] * num_dn,
        color=colors_task["dn"],
        **dict_ring,
    )

    # Plot the data
    for i_meth, meth in enumerate(methods):
        values = np.array(list(df_metric[meth]))
        values = (values - data_range_l) / (data_range_h - data_range_l)
        values = list(values)
        values += values[:1]

        title = titles[i_meth]
        ax.plot(angles, values, label=title, color=colors_meth[i_meth], **dict_line)
        ax.fill(angles, values, colors_meth[i_meth], alpha=0.05)

    # print all the xlabels
    for i in range(num_dataset_show):
        print(
            f" [{i}] "
            + id_dataset_show[i]
            + f" ({data_range_l[i]:.4f},{data_range_h[i]:.4f})"
        )

    # Set the labels
    # ax.set_yticks(np.linspace(0, 1, 10)[1:-1])
    ax.set_yticks(np.linspace(0, 1, 5)[1:-1])
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    xlabels = [f"{i}" for i in range(num_dataset_show)]
    ax.set_xticklabels(xlabels, fontsize=10)
    ax.set_ylim((0.0, 1.0))
    # delete the outer circle
    ax.spines["polar"].set_visible(False)

ax.legend(loc="upper right", labelspacing=0.8, edgecolor="white")

plt.savefig(os.path.join(path_figure, f"{prefix}_radar.png"))
plt.savefig(os.path.join(path_figure, f"{prefix}_radar.svg"))

# save source data
writer = pandas.ExcelWriter(
    os.path.join(path_figure, f"{prefix}_radar.xlsx"), engine="xlsxwriter"
)
for i_metric, metric in enumerate(metrics):
    df_metric = pandas.read_excel(path_xlsx, sheet_name=metric)[
        ["dataset-name", "task"] + methods
    ]
    df_metric = df_metric[df_metric["dataset-name"].isin(id_dataset_show)]
    df_metric = df_metric.set_index("dataset-name").loc[id_dataset_show].reset_index()
    # rename the columns
    df_save = pandas.DataFrame()
    df_save["dataset-name"] = df_metric["dataset-name"]
    df_save["task"] = df_metric["task"]
    for i_meth, meth in enumerate(methods):
        df_save[titles[i_meth]] = df_metric[meth]
    df_save.to_excel(writer, sheet_name=metric, index=True)
writer.close()
