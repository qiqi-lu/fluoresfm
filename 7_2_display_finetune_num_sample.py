"""
Display the change of metrics along the number of samples.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

plt.rcParams["svg.fonttype"] = "none"

# structure_display = "ccp"
structure_display = "lysosome"


# ------------------------------------------------------------------------------
methods = [
    ("CARE", "CARE:v2-newnorm-ft-io-", "#96C36E"),
    ("DFCAN", "DFCAN:v2-newnorm-ft-io-", "#92C4E9"),
]

method_ref = (
    "FluoResFM (ft)",
    "UNet-c:all-newnorm-ALL-v2-160-small-bs16-ft-io-",
    "#C23637",
)

metrics_name = ["PSNR", "MSSSIM", "ZNCC"]
dataset_name_dict = {
    "ccp": [
        "biotisr-ccp-sr-1",
        "biotisr-ccp-sr-2",
        "biotisr-ccp-sr-3",
    ],
    "lysosome": [
        "biotisr-lysosome-sr-1",
        "biotisr-lysosome-sr-2",
        "biotisr-lysosome-sr-3",
    ],
}
datasets_name = dataset_name_dict[structure_display]

# num_samples_train = [1, 2, 4, 8, 16, 64, 256]
num_samples_train = [1, 4, 16, 64, 256]

# ------------------------------------------------------------------------------
path_predictions = os.path.join("results", "predictions")
path_figures = os.path.join("results", "figures", "analysis", "finetune")
os.makedirs(path_figures, exist_ok=True)

dict_fig = {"dpi": 300, "constrained_layout": True}

num_metrics = len(metrics_name)
num_datasets = len(datasets_name)
num_methods = len(methods)
num_experiments = len(num_samples_train)

# ------------------------------------------------------------------------------
# collect the metrics values
metrics_values = {}
for metric in metrics_name:
    metrics_values[metric] = []

metrics_values_ref = {}
for metric in metrics_name:
    metrics_values_ref[metric] = []

for dataset_name in datasets_name:
    print("-" * 80)
    print("Dataset:", dataset_name)
    ids = []
    for method in methods:
        method_name, method_id, color = method
        for num_sample in num_samples_train:
            if num_sample == 1:
                ids.append(method_id + dataset_name)
            else:
                ids.append(method_id + dataset_name + "-" + str(num_sample))

    id_ref = method_ref[1] + dataset_name

    for metric in metrics_name:
        # load from excel
        df = pd.read_excel(
            os.path.join(path_predictions, dataset_name, "metrics-v2.xlsx"),
            sheet_name=metric,
        )

        num_samples = len(df)
        tmp = np.reshape(df[ids].values, (num_samples, num_methods, num_experiments))
        tmp = np.transpose(tmp, (1, 0, 2))
        metrics_values[metric].append(tmp)

        # ref
        tmp_ref = df[id_ref].values
        metrics_values_ref[metric].append(tmp_ref)

# metrics_values['PSNR']: [num_datasets, [num_methods, num_samples, num_experiments]]
print("-" * 80)
print("Metrics:", metrics_values.keys())
print("Number of datasets:", num_datasets)
print("Number of methods x sample:", metrics_values["PSNR"][0].shape)


# ------------------------------------------------------------------------------
# plot the metrics values

nr, nc = 3, 1
fig, axes = plt.subplots(nr, nc, figsize=(3 * nc, 3 * nr), **dict_fig)
fontsize = 8

axes[0].set_yticks(np.arange(10, 61, 4))
axes[0].set_yticklabels(np.arange(10, 61, 4), fontsize=fontsize)
axes[1].set_yticks(np.round(np.arange(0, 1.05, 0.1), 1))
axes[1].set_yticklabels(np.round(np.arange(0, 1.05, 0.1), 1), fontsize=fontsize)
axes[2].set_yticks(np.round(np.arange(0, 1.05, 0.1), 1))
axes[2].set_yticklabels(np.round(np.arange(0, 1.05, 0.1), 1), fontsize=fontsize)

dict_boxplot = dict(
    widths=0.15, flierprops=dict(marker="o", markerfacecolor="gray", markersize=2)
)
boxprops = dict(linestyle="-", linewidth=1)
dict_line = dict(linestyles="dashed", linewidth=1, color="gray")


# ------------------------------------------------------------------------------
# save source data
writer = pd.ExcelWriter(
    os.path.join(path_figures, f"metrics_num_sample_effect_{structure_display}.xlsx")
)

for i_metric, metric in enumerate(metrics_name):
    print("-" * 80)
    print("Metric:", metric)
    ax = axes[i_metric]
    for i_meth in range(num_methods):
        method_name, method_id, color = methods[i_meth]
        data = []
        for i_data in range(num_datasets):
            data.append(metrics_values[metric][i_data][i_meth])
        data = np.concatenate(data, axis=0)
        print("Number of samples:", data.shape)
        ax.boxplot(
            data,
            positions=np.arange(num_experiments) + i_meth * 0.2,
            boxprops={**boxprops, "color": color},
            **dict_boxplot,
        )

        # save data to excel
        df = pd.DataFrame(data, columns=num_samples_train)
        df.to_excel(writer, sheet_name=metric + "_" + method_name, index=False)

    # --------------------------------------------------------------------------
    # plot the reference line
    data_ref = []
    for i_data in range(num_datasets):
        data_ref.extend(metrics_values_ref[metric][i_data])
    data_ref = np.array(data_ref)
    print("Number of samples (ref):", data_ref.shape)

    ax.hlines(np.median(data_ref), -0.2, num_experiments - 0.8, **dict_line)
    ax.boxplot(
        data_ref[:, None],
        positions=[-0.2],
        boxprops={**boxprops, "color": method_ref[2]},
        **dict_boxplot,
    )
    # save data to excel
    df_ref = pd.DataFrame(data_ref, columns=["1"])
    df_ref.to_excel(writer, sheet_name=metric + "_" + method_ref[0], index=False)

    # --------------------------------------------------------------------------
    # set aspect as square
    ax.set_box_aspect(1)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xticks(np.arange(num_experiments))
    ax.set_xticklabels(num_samples_train, fontsize=fontsize)
    ax.set_ylabel(metric, fontsize=fontsize)
    if i_metric == num_metrics - 1:
        ax.set_xlabel("Number of training samples", fontsize=fontsize)
    if i_metric == 0:
        legend_elements = [
            Patch(facecolor="white", edgecolor=co, label=name, linewidth=1)
            for name, id, co in methods + [method_ref]
        ]
        ax.legend(handles=legend_elements, fontsize=fontsize, edgecolor="white")

writer.close()
plt.savefig(
    os.path.join(path_figures, f"metrics_num_sample_effect_{structure_display}.png")
)
plt.savefig(
    os.path.join(path_figures, f"metrics_num_sample_effect_{structure_display}.svg")
)
