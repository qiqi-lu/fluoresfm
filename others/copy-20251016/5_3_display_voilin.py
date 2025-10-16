"""
Compare the metrics value between different methods for variaous task and metric.
"""

import pandas, os
import matplotlib.pyplot as plt
import seaborn as sns
from utils.plot import add_significant_bars
from scipy.stats import wilcoxon

plt.rcParams["svg.fonttype"] = "none"
direction_fig = "vertical"
# direction_fig = "horizontal"  # arrangement of different metrics
# ------------------------------------------------------------------------------

# suffix, inex, task_fusion = "different_methods", "internal_dataset", False
suffix, inex, task_fusion = "different_methods", "external_dataset", False
# suffix, inex, task_fusion = "different_text_test_fusion", "internal_dataset", True
# suffix, inex, task_fusion = "different_text_test_fusion", "external_dataset", True

# suffix, inex, task_fusion = "different_text_train_fusion", "internal_dataset", True
# suffix, inex, task_fusion = "different_text_train_fusion", "external_dataset", True
# suffix, inex, task_fusion = "different_batch_size", "internal_dataset", True
# suffix, inex, task_fusion = "different_batch_size", "external_dataset", True

# ------------------------------------------------------------------------------
metrics_name = ["PSNR", "MSSSIM", "ZNCC"]
tasks = ["sr", "dcv", "dn"]

path_statistic = os.path.join("results", "statistic", inex)
path_figure = os.path.join("results", "figures", "analysis", inex)

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

# ------------------------------------------------------------------------------
#         Title | ID | Color
# ------------------------------------------------------------------------------
methods_dict = {
    "different_methods": [
        ("Raw", "raw", "#C1C7D5"),
        ("UniFMIR", "UniFMIR:all-v2", "#96C36E"),
        (
            "FluoResFM (w/o text)",
            "UNet-c:all-newnorm-ALL-v2-160-small-bs16-crossx",
            "#92C4E9",
        ),
        ("FluoResFM", "UNet-c:all-newnorm-ALL-v2-160-small-bs16", "#EA9A9D"),
    ],
    "different_text_test_fusion": [
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
    ],
    "different_text_train_fusion": [
        ("Raw", "raw", "#C1C7D5"),
        ("FluoResFM-T", "UNet-c:all-newnorm-ALL-v2-small-bs16-T77", "#C1E4FA"),
        ("FluoResFM-TS", "UNet-c:all-newnorm-ALL-v2-small-bs16-TS77", "#92C4E9"),
        # ("FluoResFM-TSpixel", "UNet-c:all-newnorm-ALL-v2-small-bs16-TSpixel77", "#0068A9"),
        # ("FluoResFM-TSmicro", "UNet-c:all-newnorm-ALL-v2-small-bs16-TSmicro77", "#004586"),
        ("FluoResFM", "UNet-c:all-newnorm-ALL-v2-160-small-bs16", "#4D8FCB"),
    ],
    "different_batch_size": [
        ("Raw", "raw", "#C1C7D5"),
        ("FluoResFM-bs4", "UNet-c:all-newnorm-ALL-v2-160-small-bs4", "#C1E4FA"),
        ("FluoResFM-bs8", "UNet-c:all-newnorm-ALL-v2-160-small-bs8", "#92C4E9"),
        ("FluoResFM-bs16", "UNet-c:all-newnorm-ALL-v2-160-small-bs16", "#4D8FCB"),
    ],
}

# ------------------------------------------------------------------------------
methods = methods_dict[suffix]
test_pairs = ((1, 3), (2, 3))
# test_pairs = ((1, 5), (2, 5), (3, 5), (4, 5))

titles = [x[0] for x in methods]
methods_id = [x[1] for x in methods]
colors = [x[2] for x in methods]

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
    )
    ax_voilin.set_xticks([])
    ax_voilin.spines["right"].set_visible(False)
    ax_voilin.spines["top"].set_visible(False)

    # add ref line
    # value_ref = data[data.columns[-1]].median()
    # ax_voilin.axhline(value_ref, color="gray", linestyle="--", linewidth=1)

    # ----------------------------------------------------------------------
    # add significance test asterisks
    pvalues = []
    for pair in test_pairs:
        test_result = wilcoxon(
            x=data[methods_id[pair[1]]],
            y=data[methods_id[pair[0]]],
            alternative="greater",
        )
        pvalues.append(test_result[1])

    # ----------------------------------------------------------------------
    if metric == "PSNR":
        ax_voilin.set_yticks([10, 20, 30, 40, 50, 60])
        ax_voilin.set_yticklabels([10, 20, 30, 40, 50, 60])
        ax_voilin.set_ylim(y_lim[0])
        pos_y = y_lim[0][1] * 0.95
        for i, pvalue in enumerate(pvalues):
            add_significant_bars(
                ax_voilin,
                test_pairs[i][0],
                test_pairs[i][1],
                pos_y + i * 1,
                pvalue,
            )
    if metric in ["MSSSIM", "SSIM", "ZNCC"]:
        ax_voilin.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax_voilin.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    if metric in ["MSSSIM", "SSIM"]:
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
        # ax_voilin.axhline(y=1.0, color="gray", linestyle="--", linewidth=1)

    if metric == "ZNCC":
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
        # ax_voilin.axhline(y=1.0, color="gray", linestyle="--", linewidth=1)


# save source data
writer = pandas.ExcelWriter(
    os.path.join(path_figure, f"compare_{suffix}.xlsx"), engine="xlsxwriter"
)

# ------------------------------------------------------------------------------
if task_fusion == False:
    if direction_fig == "vertical":
        nr, nc = len(metrics_name), len(tasks)
    else:
        nr, nc = len(tasks), len(metrics_name)
    fig, axs = plt.subplots(nr, nc, figsize=(nc * 3, nr * 3), **dict_fig)

    for i_metric, metric in enumerate(metrics_name):
        for i_task, task in enumerate(tasks):
            if direction_fig == "vertical":
                ax = axs[i_metric, i_task]
            else:
                ax = axs[i_task, i_metric]
            data_frame = pandas.read_excel(
                os.path.join(path_statistic, task + "_value.xlsx"),
                sheet_name=metric,
            )
            data = data_frame[methods_id]

            # ------------------------------------------------------------------
            # plot the data
            plot_voilin(ax, data, metric, y_lim_dict[inex][task])

            if i_metric == 0:
                ax.set_title(task.upper() + f" (N = {data.shape[0]})")
            if i_task == 0:
                ax.set_ylabel(metric)
            ax.set_box_aspect(1)

            # save source data
            data.columns = titles
            data.to_excel(writer, sheet_name=metric + "_" + task)

# ------------------------------------------------------------------------------
if task_fusion == True:
    if direction_fig == "vertical":
        nr, nc = len(metrics_name), 1
    else:
        nr, nc = 1, len(metrics_name)
    fig, axs = plt.subplots(nr, nc, figsize=(nc * 3, nr * 3), **dict_fig)
    for i_metric, metric in enumerate(metrics_name):
        ax = axs[i_metric]
        # read the data
        data = []
        for task in tasks:
            data_frame = pandas.read_excel(
                os.path.join(path_statistic, task + "_value.xlsx"),
                sheet_name=metric,
            )
            data_single_task = data_frame[methods_id]
            data.append(data_single_task)
        # concatenate data from all tasks
        data = pandas.concat(data, axis=0)
        # ----------------------------------------------------------------------
        # plot the data
        plot_voilin(ax, data, metric, y_lim_dict[inex]["fusion"])
        ax.set_ylabel(metric)
        if i_metric == 0:
            ax.set_title(f"N = {data.shape[0]}")
        ax.set_box_aspect(1)

        # save source data
        data.columns = titles
        # reindex
        data = data.reset_index(drop=True)
        data.to_excel(writer, sheet_name=metric, index=True)

# ------------------------------------------------------------------------------
writer.close()

# save the figure
fig.savefig(os.path.join(path_figure, f"compare_{suffix}.png"))
fig.savefig(os.path.join(path_figure, f"compare_{suffix}.svg"))
