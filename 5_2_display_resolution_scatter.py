"""
Plot the average resolution of different methods and GT on internal and external
datasets.
Only show the mean of each method.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas, os
from utils.plot import cal_radar_range
from dataset_analysis import dataset_names_radar
import seaborn as sns
from scipy.stats import pearsonr

plt.rcParams["svg.fonttype"] = "none"

# ------------------------------------------------------------------------------
# dataset_group = "internal_dataset"
dataset_group = "external_dataset"

prefix = "compare_different_methods"
# prefix = "compare_different_text"

threshold_resolution = 700  # higher than threshold will be treated as outliers.
# ------------------------------------------------------------------------------

methods_info = (
    ("Raw", "raw", "#212C3E"),
    ("UniFMIR", "UniFMIR:all-v2", "#00810A"),
    (
        "FluoResFM (w/o text)",
        "UNet-c:all-newnorm-ALL-v2-160-small-bs16-crossx",
        "#2962FF",
    ),
    ("FluoResFM", "UNet-c:all-newnorm-ALL-v2-160-small-bs16", "#FF0000"),
    ("GT", "gt", "#000000"),
)

metric = "Resolution (DA)"

# ------------------------------------------------------------------------------
# datasets
id_dataset_show = dataset_names_radar[dataset_group]  # only show the dataset in radar
num_dataset_show = len(id_dataset_show)

# methods
titles = [meth[0] for meth in methods_info]
methods = [meth[1] + "-mean" for meth in methods_info]
colors_meth = [meth[2] for meth in methods_info]

# file path
path_statistic = os.path.join("results", "statistic", dataset_group)
path_figure = os.path.join("results", "figures", "analysis", dataset_group)
os.makedirs(path_figure, exist_ok=True)
path_xlsx = os.path.join(path_statistic, "all_mean_std_pvalue_res.xlsx")

# ------------------------------------------------------------------------------
print("-" * 80)
print("[INFO] dataset_group:", dataset_group)
print("[INFO] prefix:", prefix)
print("[INFO] Methods:", methods)
print("[INFO] Number of dataset (show):", num_dataset_show)
print("-" * 80)

# ------------------------------------------------------------------------------
# show the scatter plot
# ------------------------------------------------------------------------------
print("-" * 80)
print(f"[INFO] Metric: {metric}")

df_metric = pandas.read_excel(path_xlsx, sheet_name=metric)[
    ["dataset-name", "task"] + methods
]
df_metric = df_metric[df_metric["dataset-name"].isin(id_dataset_show)]
df_metric = df_metric.set_index("dataset-name").loc[id_dataset_show].reset_index()

if threshold_resolution is not None:
    print(
        f"[INFO] Number of dataset (show) before filtering:",
        len(df_metric),
    )
    id_drops = []
    # filter out the outliers
    # loop over each rows
    for i_row in range(len(df_metric)):
        # get the values
        value_gt = df_metric.iloc[i_row][methods[-1]]
        if value_gt > threshold_resolution:
            id_drops.append(i_row)
        value_raw = df_metric.iloc[i_row][methods[0]]
        if value_raw > 2000:
            id_drops.append(i_row)
    # drop the outliers
    df_metric = df_metric.drop(id_drops)
    print(f"[INFO] Number of dataset (show) after filtering:", len(df_metric))
    # update the dataset names to show
    id_dataset_show = df_metric["dataset-name"].values

# get max and min value
# max_gt = df_metric[methods[-1]].max()
# max_raw = np.percentile(df_metric[methods[0]], 95)
# # exclude the fliers
# max_global = np.maximum(max_gt, max_raw)
# xlim = (0, max_global * 1.05)
# ylim = (0, max_global * 1.05)
# xlim, ylim = (0, 750), (0, 750)

if dataset_group == "internal_dataset":
    xlim, ylim = (0, 700), (0, 700)
    xlim_zoomin, ylim_zoomin = (0, 700), (0, 700)
if dataset_group == "external_dataset":
    xlim, ylim = (0, 700), (0, 700)
    xlim_zoomin, ylim_zoomin = (0, 700), (0, 700)

data_x = df_metric[methods[-1]]
linex = np.linspace(xlim[0], xlim[1], 100)
dict_line = dict(linestyle="-", linewidth=1, alpha=0.5)
dict_line_1 = dict(linestyle="--", linewidth=1, alpha=0.5)
fontsize = 8
dict_text = dict(fontsize=fontsize, ha="center", va="center")
dict_fig = dict(dpi=600, constrained_layout=True)

# ------------------------------------------------------------------------------
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 3), **dict_fig)
ax = axes[0]
ax_zoomin = axes[1]

# ------------------------------------------------------------------------------
ax.plot(xlim, xlim, c="k", **dict_line_1)
# get the index of lower thatn 2000 in data_x
# index = np.where(data_x < 2000)[0]
for i_meth in range(0, len(methods) - 1):

    # plot the fitting line
    fit = np.polyfit(data_x, df_metric[methods[i_meth]], 1)
    fit_fn = np.poly1d(fit)
    ax.plot(linex, fit_fn(linex), color=colors_meth[i_meth], **dict_line)

    ax.scatter(
        x=data_x,
        y=df_metric[methods[i_meth]],
        c=colors_meth[i_meth],
        s=1,
        label=titles[i_meth]
        + f"\n(y={fit[0]:.2f}x+{fit[1]:.1f}) ($R^2$={(pearsonr(data_x, df_metric[methods[i_meth]])[0])**2:.2f})",
    )

ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_xlabel("Resolution (nm) - GT", **dict_text)
ax.set_ylabel("Resolution (nm) - Restored", **dict_text)
# add the legend
ax.legend(
    # loc="upper left",
    fontsize=5,
    frameon=False,
)
ax.set_box_aspect(1)
ax.tick_params(axis="both", which="major", labelsize=fontsize)
# ------------------------------------------------------------------------------
# zoom in the plot
# plot a box around the zoom in area
ax.add_patch(
    plt.Rectangle(
        (xlim_zoomin[0], ylim_zoomin[0]),
        xlim_zoomin[1] - xlim_zoomin[0],
        ylim_zoomin[1] - ylim_zoomin[0],
        fill=False,
        linestyle="--",
        edgecolor="k",
        linewidth=0.5,
    )
)

ax_zoomin.plot(xlim_zoomin, xlim_zoomin, c="k", **dict_line_1)
for i_meth in range(0, len(methods) - 1):
    fit = np.polyfit(data_x, df_metric[methods[i_meth]], 1)
    fit_fn = np.poly1d(fit)
    ax_zoomin.plot(linex, fit_fn(linex), color=colors_meth[i_meth], **dict_line)
    ax_zoomin.scatter(
        x=data_x, y=df_metric[methods[i_meth]], c=colors_meth[i_meth], s=1
    )

ax_zoomin.set_xlim(xlim_zoomin)
ax_zoomin.set_ylim(ylim_zoomin)
ax_zoomin.set_box_aspect(1)
ax_zoomin.tick_params(axis="both", which="major", labelsize=fontsize)
# ------------------------------------------------------------------------------
# save the figure
plt.savefig(os.path.join(path_figure, f"{prefix}_scatter.png"))
plt.savefig(os.path.join(path_figure, f"{prefix}_scatter.svg"))

# ------------------------------------------------------------------------------
# save source data
# ------------------------------------------------------------------------------
writer = pandas.ExcelWriter(
    os.path.join(path_figure, f"{prefix}_scatter.xlsx"), engine="xlsxwriter"
)
df_metric = pandas.read_excel(path_xlsx, sheet_name=metric)[
    ["dataset-name", "task"] + methods
]
df_metric = df_metric[df_metric["dataset-name"].isin(id_dataset_show)]
df_metric = df_metric.set_index("dataset-name").loc[id_dataset_show].reset_index()

df_save = pandas.DataFrame()
df_save["dataset-name"] = df_metric["dataset-name"]
df_save["task"] = df_metric["task"]
for i_meth, meth in enumerate(methods):
    df_save[titles[i_meth]] = df_metric[meth]
df_save.to_excel(writer, sheet_name=metric, index=True)
writer.close()
