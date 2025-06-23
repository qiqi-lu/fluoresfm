"""
Display the metrics of the fine-tuned model.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dataset_analysis import datasets_finetune
import os
import seaborn as sns
from utils.plot import add_significant_bars
from scipy.stats import wilcoxon

plt.rcParams["svg.fonttype"] = "none"

methods = [
    ("Raw", "raw", "#C1C7D5"),
    ("CARE", "CARE:v2-newnorm-ft-io-", "#8E99AB"),
    ("DFCAN", "DFCAN:v2-newnorm-ft-io-", "#647086"),
    ("UniFMIR", "UniFMIR:all-v2", "#4D8FCB"),
    ("UniFMIR (ft)", "UniFMIR:v2-newnorm-ft-io-", "#0068A9"),
    ("FluoResFM", "UNet-c:all-newnorm-ALL-v2-160-small-bs16", "#D95D5B"),
    ("FluoResFM (ft)", "UNet-c:all-newnorm-ALL-v2-160-small-bs16-ft-io-", "#C23637"),
]

metrics_name = ["PSNR", "MSSSIM", "ZNCC"]
path_predictions = os.path.join("results", "predictions")
path_figures = os.path.join("results", "figures", "analysis", "finetune")
os.makedirs(path_figures, exist_ok=True)

methods_title, methods_color = [], []
for method_name, method_id, color in methods:
    methods_title = methods_title + [method_name]
    methods_color = methods_color + [color]


num_methods = len(methods)
num_group = len(datasets_finetune)
num_metrics = len(metrics_name)
print("-" * 80)
print("Number fo datasets group:", num_group)
print("Number of methods:", num_methods)
print("Number of metrics:", num_metrics)
# get all the keys in the dicionary datasets_finetune
group_names = list(datasets_finetune.keys())
print("Group names:", group_names)
print()

dict_fig = {"dpi": 300, "constrained_layout": True}

# ------------------------------------------------------------------------------
# plot metrics of each datasets
# ------------------------------------------------------------------------------
nr, nc = num_metrics, num_group
fig, axes = plt.subplots(nr, nc, figsize=(3 * nc, 3 * nr), **dict_fig)

metrics_value_each_sample = {}
for metric in metrics_name:
    metrics_value_each_sample[metric] = []

for i_group, group_name in enumerate(group_names):
    print("-" * 80)
    print("Group:", group_name)
    datasets = datasets_finetune[group_name]
    num_datasets = len(datasets)
    print("Number of datasets:", num_datasets)

    # calculate mean and std
    metrics_mean = []  # (num_datasets, num_metrics, num_methods)
    metrics_std = []
    for i_dataset, datsaset_name in enumerate(datasets):
        methods_id = []
        for meth in methods:
            meth_id = meth[1]
            if "-ft-" in meth_id:
                meth_id += datsaset_name
            methods_id.append(meth_id)

        mean, std = [], []
        for metric in metrics_name:
            df = pd.read_excel(
                os.path.join(path_predictions, datsaset_name, "metrics-v2.xlsx"),
                sheet_name=metric,
            )
            df = df[methods_id]
            metrics_value_each_sample[metric].append(df.values)
            mean.append(df.mean(axis=0))
            std.append(df.std(axis=0))

        metrics_mean.append(mean)
        metrics_std.append(std)
    # convert to numpy array
    metrics_mean = np.array(metrics_mean)
    metrics_std = np.array(metrics_std)
    # plot
    dict_errorbar = {"fmt": "-o", "capsize": 3}
    for i_metric, metric in enumerate(metrics_name):
        ax = axes[i_metric, i_group]
        data_mean = metrics_mean[:, i_metric, :]  # (num_datasets, num_methods)
        data_std = metrics_std[:, i_metric, :]

        for i_meth in range(num_methods):
            ax.errorbar(
                np.arange(num_datasets),
                data_mean[:, i_meth],
                yerr=data_std[:, i_meth],
                color=methods_color[i_meth],
                **dict_errorbar,
            )

        ax.set_xticks(np.arange(num_datasets))
        ax.set_xticklabels(np.arange(num_datasets) + 1)
    axes[0, i_group].set_title(group_name)

plt.savefig(os.path.join(path_figures, "metrics_each_dataset.png"))

# ------------------------------------------------------------------------------
# plot overall statistic
# ------------------------------------------------------------------------------
nr, nc = 3, 1
fig, axes = plt.subplots(nr, nc, figsize=(3 * nc, 3 * nr), **dict_fig)
test_pairs = ((4, 6), (5, 6))


def plot_voilin(ax, data, metric, y_lim):
    ax_voilin = sns.violinplot(
        data=data,
        linewidth=1,
        # linecolor="white",
        linecolor="black",
        inner="box",
        palette=methods_color,
        ax=ax,
        cut=0,
        # inner_kws={"color": "black"},
    )
    ax_voilin.set_xticks([])
    ax_voilin.spines["right"].set_visible(False)
    ax_voilin.spines["top"].set_visible(False)

    # ----------------------------------------------------------------------
    # add significance test asterisks
    pvalues = []
    for pair in test_pairs:
        pvalue = wilcoxon(data[:, pair[1]], data[:, pair[0]], alternative="greater")[1]
        pvalues.append(pvalue)

    # ----------------------------------------------------------------------
    if metric == "PSNR":
        ax_voilin.set_yticks([10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
        ax_voilin.set_yticklabels([10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
        ax_voilin.set_ylim(y_lim[0])
        pos_y = y_lim[0][1] * 0.95
        for i, pvalue in enumerate(pvalues):
            add_significant_bars(
                ax_voilin, test_pairs[i][0], test_pairs[i][1], pos_y + i * 1, pvalue
            )
    if metric == "MSSSIM" or metric == "SSIM":
        ax_voilin.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax_voilin.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax_voilin.set_ylim(y_lim[1])
        pos_y = y_lim[1][1] * 0.95
        for i, pvalue in enumerate(pvalues):
            add_significant_bars(
                ax_voilin,
                test_pairs[i][0],
                test_pairs[i][1],
                pos_y + i * 0.02,
                pvalue,
            )
    if metric == "ZNCC":
        ax_voilin.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax_voilin.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax_voilin.set_ylim(y_lim[2])
        pos_y = y_lim[2][1] * 0.95
        for i, pvalue in enumerate(pvalues):
            add_significant_bars(
                ax_voilin,
                test_pairs[i][0],
                test_pairs[i][1],
                pos_y + i * 0.02,
                pvalue,
            )


y_lim = ((17, 39), (0.35, 1.07), (0.25, 1.05))

for i_metric, metric in enumerate(metrics_name):
    ax = axes[i_metric]
    # set ax to be square
    values = metrics_value_each_sample[metric]
    values = np.concatenate(values, axis=0)
    plot_voilin(ax, values, metric, y_lim)
    ax.set_box_aspect(1)
    ax.set_ylabel(metric)

plt.savefig(os.path.join(path_figures, "metrics_overall.png"))
plt.savefig(os.path.join(path_figures, "metrics_overall.svg"))
