"""
Compare the metrics value between different methods for variaous task and metric.
"""

import pandas, os
import matplotlib.pyplot as plt
import seaborn as sns
from utils.plot import add_significant_bars
from scipy.stats import wilcoxon

suffix, inex, task_fusion = "different_methods", "internal_dataset", False
suffix, inex, task_fusion = "different_methods", "external_dataset", False
suffix, inex, task_fusion = "different_text_train_fusion", "internal_dataset", True
suffix, inex, task_fusion = "different_text_train_fusion", "external_dataset", True
suffix, inex, task_fusion = "different_text_test_fusion", "internal_dataset", True
suffix, inex, task_fusion = "different_text_test_fusion", "external_dataset", True

y_lim_dict = {
    "internal_dataset": {
        "sr": ((15, 47), (0.25, 1.07), (0.0, 1.1)),
        "dcv": ((15, 42), (0.37, 1.07), (0.0, 1.1)),
        "dn": ((15, 55), (0.35, 1.07), (0.0, 1.1)),
        "fusion": ((13, 55), (0.25, 1.07), (0.0, 1.1)),
    },
    "external_dataset": {
        "sr": ((18, 40), (0.46, 1.07), (0.42, 1.07)),
        "dcv": ((16, 41), (0.45, 1.07), (0.22, 1.07)),
        "dn": ((14, 46), (0.42, 1.07), (0.14, 1.1)),
        "fusion": ((14, 46), (0.42, 1.07), (0.12, 1.1)),
    },
}

metrics_name = ["PSNR", "MSSSIM", "ZNCC"]

tasks = ["sr", "dcv", "dn"]
path_statistic = os.path.join("results", "statistic", inex)
path_figure = os.path.join("results", "figures", "analysis", inex)

methods = [
    # --------------------------------------------------------------------------
    # Title | ID | Color
    # --------------------------------------------------------------------------
    # ------------------------ model comparison --------------------------------
    # ("Raw", "raw", "#C1C7D5"),
    # ("UniFMIR", "UniFMIR:all-v2", "#96C36E"),
    # (
    #     "FluoResFM\n(w/o text)",
    #     "UNet-c:all-newnorm-ALL-v2-160-small-bs16-crossx",
    #     "#92C4E9",
    # ),
    # ("FluoResFM", "UNet-c:all-newnorm-ALL-v2-160-small-bs16", "#EA9A9D"),
    # ------------------------ batch size --------------------------------------
    # ("Raw", "raw", "#C1C7D5"),
    # ("FluoResFM-bs4", "UNet-c:all-newnorm-ALL-v2-160-small-bs4"),
    # ("FluoResFM-bs8", "UNet-c:all-newnorm-ALL-v2-160-small-bs8"),
    # ("FluoResFM-bs16", "UNet-c:all-newnorm-ALL-v2-160-small-bs16"),
    # ------------------------ text used for training --------------------------
    # ("Raw", "raw", "#C1C7D5"),
    # ("FluoResFM-T", "UNet-c:all-newnorm-ALL-v2-small-bs16-T77", "#C1E4FA"),
    # ("FluoResFM-TS", "UNet-c:all-newnorm-ALL-v2-small-bs16-TS77", "#92C4E9"),
    # # ("FluoResFM-TSpixel", "UNet-c:all-newnorm-ALL-v2-small-bs16-TSpixel77", "#0068A9"),
    # # ("FluoResFM-TSmicro", "UNet-c:all-newnorm-ALL-v2-small-bs16-TSmicro77", "#004586"),
    # ("FluoResFM", "UNet-c:all-newnorm-ALL-v2-160-small-bs16", "#4D8FCB"),
    # ------------------------ Text used for testing ---------------------------
    ("Raw", "raw", "#C1C7D5"),
    ("FluoResFM-T", "UNet-c:all-newnorm-ALL-v2-160-small-bs16-in-T", "#C5E3EB"),
    ("FluoResFM-TS", "UNet-c:all-newnorm-ALL-v2-160-small-bs16-in-TS", "#8CCCCE"),
    # (
    #     "FluoResFM-TSmicro",
    #     "UNet-c:all-newnorm-ALL-v2-160-small-bs16-in-TSmicro",
    #     "#018F99",
    # ),
    # (
    #     "FluoResFM-TSpixel",
    #     "UNet-c:all-newnorm-ALL-v2-160-small-bs16-in-TSpixel",
    #     "#005D6E",
    # ),
    ("FluoResFM", "UNet-c:all-newnorm-ALL-v2-160-small-bs16", "#42B4B5"),
]

test_pairs = ((1, 3), (2, 3))
# test_pairs = ((1, 5), (2, 5), (3, 5), (4, 5))

titles = [x[0] for x in methods]
methods_id = [x[1] for x in methods]
colors = [x[2] for x in methods]

plt.rcParams["svg.fonttype"] = "none"

dict_fig = {"dpi": 300, "constrained_layout": True}


def plot_voilin(ax, data, metric, y_lim):
    ax_voilin = sns.violinplot(
        data=data,
        linewidth=1,
        # linecolor="white",
        linecolor="black",
        inner="box",
        palette=colors,
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
        pvalue = wilcoxon(
            data[methods_id[pair[1]]],
            data[methods_id[pair[0]]],
            alternative="greater",
        )[1]
        pvalues.append(pvalue)

    # ----------------------------------------------------------------------
    if metric == "PSNR":
        ax_voilin.set_yticks([10, 20, 30, 40, 50, 60])
        ax_voilin.set_yticklabels([10, 20, 30, 40, 50, 60])
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


# ------------------------------------------------------------------------------
if task_fusion == False:
    nr, nc = len(metrics_name), len(tasks)
    fig, axs = plt.subplots(nr, nc, figsize=(nc * 3, nr * 3), **dict_fig)

    for i_metric, metric in enumerate(metrics_name):
        for i_task, task in enumerate(tasks):
            # read the data
            path = os.path.join(path_statistic, task + "_value.xlsx")
            data_frame = pandas.read_excel(path, sheet_name=metric)

            # get data
            data = data_frame[methods_id]
            num_samples = data.shape[0]

            # ----------------------------------------------------------------------
            # plot the data
            y_lim = y_lim_dict[inex][task]
            plot_voilin(axs[i_metric, i_task], data, metric, y_lim)

            if i_metric == 0:
                axs[i_metric, i_task].set_title(task.upper() + f" (N = {num_samples})")
            if i_task == 0:
                axs[i_metric, i_task].set_ylabel(metric)
            axs[i_metric, i_task].set_box_aspect(1)


if task_fusion == True:
    y_lim = y_lim_dict[inex]["fusion"]
    nr, nc = len(metrics_name), 1
    fig, axs = plt.subplots(nr, nc, figsize=(nc * 3, nr * 3), **dict_fig)
    for i_metric, metric in enumerate(metrics_name):
        # read the data
        data = []
        for task in tasks:
            path = os.path.join(path_statistic, task + "_value.xlsx")
            data_frame = pandas.read_excel(path, sheet_name=metric)
            # get data
            data_single_task = data_frame[methods_id]
            data.append(data_single_task)
        # concatenate data from all tasks
        data = pandas.concat(data, axis=0)
        num_samples = data.shape[0]
        # ----------------------------------------------------------------------
        # plot the data
        plot_voilin(axs[i_metric], data, metric, y_lim)
        axs[i_metric].set_ylabel(metric)
        if i_metric == 0:
            axs[i_metric].set_title(f"N = {num_samples}")
        axs[i_metric].set_box_aspect(1)


# save the figure
fig.savefig(os.path.join(path_figure, f"compare_{suffix}.png"))
fig.savefig(os.path.join(path_figure, f"compare_{suffix}.svg"))
