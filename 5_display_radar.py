"""
Radar plot of the results of external datasets.
Only show the mean of each method.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas, os
from utils.plot import cal_radar_range
from dataset_analysis import dataset_names_radar

# ------------------------------------------------------------------------------
excluded_datasets_id = None
dataset_group = "internal_dataset"
dataset_group = "external_dataset"

# prefix = "compare_different_text"
prefix = "compare_different_methods"


methods_title = (
    ("FluoResFM", "UNet-c:all-newnorm-ALL-v2-160-small-bs16", "#FF0000"),
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
print("prefix:", prefix)
titles = [meth[0] for meth in methods_title]
methods = [meth[1] + "-mean" for meth in methods_title]
colors_meth = [meth[2] for meth in methods_title]

metrics = ["PSNR", "SSIM", "ZNCC"]
metrics_precision = [0.1, 0.001, 0.001, 0.005, 0.005]
metrics_range = (0.2, 0.95)

path_statistic = os.path.join("results", "statistic", dataset_group)
path_figure = os.path.join("results", "figures", "analysis", dataset_group)

# ------------------------------------------------------------------------------
id_dataset_show_all = dataset_names_radar[dataset_group]
id_dataset_show = []

# exclude some datasets
if excluded_datasets_id is not None:
    for i in range(len(id_dataset_show_all)):
        if i not in excluded_datasets_id:
            id_dataset_show.append(id_dataset_show_all[i])
else:
    id_dataset_show = id_dataset_show_all


path_xlsx = os.path.join(path_statistic, "mean_std_pvalue.xlsx")
# ------------------------------------------------------------------------------
num_dataset_show = len(id_dataset_show)
print("Number of dataset (show):", num_dataset_show)

# Compute angle for each dataset
angles = np.linspace(0, 2 * np.pi, num_dataset_show, endpoint=False).tolist()
angles += angles[:1]

num_metrics = len(metrics)
print("Number of metrics:", num_metrics)

fig, axes = plt.subplots(
    nrows=1,
    ncols=num_metrics,
    figsize=(12 * num_metrics, 12),
    subplot_kw=dict(polar=True),
    dpi=300,
)

# loop over each metric
for i_metric, metric in enumerate(metrics):
    ax = axes[i_metric]
    print("-" * 50)
    print(f"Metric: {metric}")

    df_metric = pandas.read_excel(path_xlsx, sheet_name=metric)[
        ["dataset-name", "task"] + methods
    ]
    df_metric = df_metric[df_metric["dataset-name"].isin(id_dataset_show)]
    df_metric = df_metric.set_index("dataset-name").loc[id_dataset_show].reset_index()

    metrics_value = np.array(df_metric[methods])

    tasks = list(df_metric["task"])
    # count the number of 'sr', 'dcv', 'dn' in the tasks list
    num_sr = tasks.count("sr")
    num_dcv = tasks.count("dcv")
    num_dn = tasks.count("dn")

    data_range_h, data_range_l = cal_radar_range(
        metrics_value, percent=metrics_range, precision=metrics_precision[i_metric]
    )

    # --------------------------------------------------------------------------
    # Create the plot
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    dict_ring = {"linewidth": 20, "linestyle": "solid"}
    dict_line = {"linewidth": 1.5, "linestyle": "solid"}

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
        title = titles[i_meth]
        values = np.array(list(df_metric[meth]))
        values = (values - data_range_l) / (data_range_h - data_range_l)

        values = list(values)
        values += values[:1]
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
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    xlabels = [f"{i}" for i in range(num_dataset_show)]
    ax.set_xticklabels(xlabels, fontsize=10)
    ax.set_yticks(np.linspace(0, 1, 10)[1:-1])
    ax.set_ylim((0.0, 1.0))
    # delete the outer circle
    ax.spines["polar"].set_visible(False)

ax.legend(loc="upper right", labelspacing=0.8, edgecolor="white")

plt.savefig(os.path.join(path_figure, f"{prefix}_radar.png"))
plt.savefig(os.path.join(path_figure, f"{prefix}_radar.svg"))
plt.close()
