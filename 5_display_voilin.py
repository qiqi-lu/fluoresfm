"""
Compare the metrics value between different methods for variaous task and metric.
"""

import pandas, os
import matplotlib.pyplot as plt
import seaborn as sns
from utils.plot import add_significant_bars
from scipy.stats import wilcoxon


suffix = "different_text_train_fusion"
# suffix = "different_text_train"

dataset_type = "internal_dataset"
# dataset_type = "external_dataset"

metrics_name = ["PSNR", "SSIM", "ZNCC"]
tasks = ["sr", "dcv", "dn"]
task_fusion = True
# task_fusion = False
path_statistic = os.path.join("results", "statistic", dataset_type)
path_figure = os.path.join("results", "figures", "analysis", dataset_type)

methods = [
    ("Raw", "raw", "#C1E4FA"),
    # ("UniFMIR", "UniFMIR:all-v2", "#92C4E9"),
    # (
    #     "FluoResFM\n(w/o text)",
    #     "UNet-c:all-newnorm-ALL-v2-160-small-bs16-crossx",
    #     "#96C36E",
    # ),
    # ("FluoResFM", "UNet-c:all-newnorm-ALL-v2-160-small-bs16", "#EA9A9D"),
    # ("FluoResFM-bs4", "UNet-c:all-newnorm-ALL-v2-160-small-bs4"),
    # ("FluoResFM-bs8", "UNet-c:all-newnorm-ALL-v2-160-small-bs8"),
    # ("FluoResFM-bs16", "UNet-c:all-newnorm-ALL-v2-160-small-bs16"),
    ("FluoResFM-T", "UNet-c:all-newnorm-ALL-v2-small-bs16-T77", "#92C4E9"),
    ("FluoResFM-TS", "UNet-c:all-newnorm-ALL-v2-small-bs16-TS77", "#4D8FCB"),
    ("FluoResFM-TSpixel", "UNet-c:all-newnorm-ALL-v2-small-bs16-TSpixel77", "#0068A9"),
    ("FluoResFM-TSmicro", "UNet-c:all-newnorm-ALL-v2-small-bs16-TSmicro77", "#004586"),
    ("FluoResFM", "UNet-c:all-newnorm-ALL-v2-160-small-bs16", "#002752"),
]

test_pairs = ((1, 3), (2, 3))
test_pairs = ((1, 5), (2, 5), (3, 5), (4, 5))

titles = [x[0] for x in methods]
methods_id = [x[1] for x in methods]
colors = [x[2] for x in methods]

plt.rcParams["svg.fonttype"] = "none"


def plot_voilin(ax, data, metric):
    ax_voilin = sns.violinplot(
        data=data,
        linewidth=1,
        linecolor="black",
        inner="box",
        palette=colors,
        ax=ax,
        cut=0,
    )
    ax_voilin.set_xticks([])
    ax_voilin.spines["right"].set_visible(False)
    ax_voilin.spines["top"].set_visible(False)

    # ----------------------------------------------------------------------
    # add significance test asterisks
    pvalues = []
    for pair in test_pairs:
        pvalue = wilcoxon(
            data[methods_id[pair[1]]],
            data[methods_id[pair[0]]],
            alternative="greater",
        )[1]
        pvalues.append(pvalue)

    # ----------------------------------------------------------------------
    if metric == "PSNR":
        ax_voilin.set_ylim([19, 55])
        pos_y = 52
        for i, pvalue in enumerate(pvalues):
            add_significant_bars(
                ax_voilin, test_pairs[i][0], test_pairs[i][1], pos_y + i * 1, pvalue
            )
    if metric == "SSIM":
        ax_voilin.set_ylim([0.18, 1.15])
        pos_y = 1.02
        for i, pvalue in enumerate(pvalues):
            add_significant_bars(
                ax_voilin,
                test_pairs[i][0],
                test_pairs[i][1],
                pos_y + i * 0.02,
                pvalue,
            )
    if metric == "ZNCC":
        ax_voilin.set_ylim([0.0, 1.15])
        pos_y = 1.02
        for i, pvalue in enumerate(pvalues):
            add_significant_bars(
                ax_voilin,
                test_pairs[i][0],
                test_pairs[i][1],
                pos_y + i * 0.02,
                pvalue,
            )


# ------------------------------------------------------------------------------
if task_fusion == False:
    nr, nc = len(metrics_name), len(tasks)
    fig, axs = plt.subplots(
        nr, nc, figsize=(nc * 3, nr * 3), dpi=300, constrained_layout=True
    )

    for i_metric, metric in enumerate(metrics_name):
        for i_task, task in enumerate(tasks):
            # read the data
            path = os.path.join(path_statistic, task + "_all_sample.xlsx")
            data_frame = pandas.read_excel(path, sheet_name=metric)

            # get data
            data = data_frame[methods_id]
            num_samples = data.shape[0]

            # ----------------------------------------------------------------------
            # plot the data
            plot_voilin(axs[i_metric, i_task], data, metric)

            if i_metric == 0:
                axs[i_metric, i_task].set_title(task.upper() + f" (N = {num_samples})")
            if i_task == 0:
                axs[i_metric, i_task].set_ylabel(metric)
            axs[i_metric, i_task].set_box_aspect(1)


if task_fusion == True:
    nr, nc = len(metrics_name), 1
    fig, axs = plt.subplots(
        nr, nc, figsize=(nc * 3, nr * 3), dpi=300, constrained_layout=True
    )
    for i_metric, metric in enumerate(metrics_name):
        # read the data
        data = []
        for task in tasks:
            path = os.path.join(path_statistic, task + "_all_sample.xlsx")
            data_frame = pandas.read_excel(path, sheet_name=metric)
            # get data
            data_single_task = data_frame[methods_id]
            data.append(data_single_task)
        # concatenate data from all tasks
        data = pandas.concat(data, axis=0)
        num_samples = data.shape[0]
        # ----------------------------------------------------------------------
        # plot the data
        plot_voilin(axs[i_metric], data, metric)
        axs[i_metric].set_ylabel(metric)
        if i_metric == 0:
            axs[i_metric].set_title(f"N = {num_samples}")
        axs[i_metric].set_box_aspect(1)


# save the figure
fig.savefig(os.path.join(path_figure, f"compare_{suffix}.png"))
fig.savefig(os.path.join(path_figure, f"compare_{suffix}.svg"))
