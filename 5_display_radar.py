"""
Radar plot of the results of external datasets.
Only show the mean of each method.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas, os
from utils.plot import cal_radar_range
from dataset_analysis import dataset_names_radar

dataset_group = "internal_dataset"
dataset_group = "external_dataset"

methods = [
    "UNet-c:all-newnorm-mean",
    # "UNet-c:all-TSpixel-mean",
    # "UNet-c:all-newnorm_TSmicro-mean",
    "UNet-uc:all-newnorm-mean",
    # "UNet-c:all-mean",
    # "UNet-uc:all-mean",
    # "UniFMIR:all-mean",
    "raw-mean",
]


# ------------------------------------------------------------------------------
id_dataset_show = dataset_names_radar[dataset_group]

methods_colors = ["#D95D5B", "#4D8FCB", "#F48F3D", "#B78E72", "#B78E72"]
metrics = ["PSNR", "SSIM", "ZNCC"]
metrics_precision = [0.5, 0.005, 0.0005]

path_xlsx = os.path.join(
    "outputs",
    "unet_c",
    dataset_group,
    "0_evaluation_metrics",
    "all_mean_std_pvalue.xlsx",
)
path_figure = os.path.join("outputs", "figures", "analysis", dataset_group)

# load results
num_dataset_show = len(id_dataset_show)
print("Number of dataset (show):", num_dataset_show)

# Compute angle for each dataset
angles = np.linspace(0, 2 * np.pi, num_dataset_show, endpoint=False).tolist()
angles += angles[:1]

# loop over each metric
for i_metric, metric in enumerate(metrics):
    print("-" * 50)
    print(f"Metric: {metric}")
    print("-" * 50)

    df_metric = pandas.read_excel(path_xlsx, sheet_name=metric)[
        ["dataset-name"] + methods
    ]
    df_metric = df_metric[df_metric["dataset-name"].isin(id_dataset_show)]
    df_metric = df_metric.set_index("dataset-name").loc[id_dataset_show].reset_index()

    metrics_value = np.array(df_metric[methods])

    data_range_h, data_range_l = cal_radar_range(
        metrics_value, percent=(0.2, 0.95), precision=metrics_precision[i_metric]
    )

    # --------------------------------------------------------------------------
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True), dpi=600)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Plot the data
    for i_meth, meth in enumerate(methods):
        values = np.array(list(df_metric[meth]))
        values = (values - data_range_l) / (data_range_h - data_range_l)

        values = list(values)
        values += values[:1]
        ax.plot(
            angles,
            values,
            linewidth=1.5,
            linestyle="solid",
            # marker=".",
            label=meth,
            color=methods_colors[i_meth],
        )
        ax.fill(angles, values, methods_colors[i_meth], alpha=0.05)

    # Fill the area under the plot
    # ax.fill(angles, values, alpha=0.4)

    # Set the labels
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    xlabels = [
        # id_dataset_show[i] + f" ({data_range_l[i]},{data_range_h[i]})"
        # f" [{i}] ({data_range_l[i]},{data_range_h[i]})"
        f" [{i}]"
        for i in range(num_dataset_show)
    ]

    # print all the xlabels
    [
        print(
            f" [{i}] "
            + id_dataset_show[i]
            + f" ({data_range_l[i]:.4f},{data_range_h[i]:.4f})"
        )
        for i in range(num_dataset_show)
    ]

    ax.set_xticklabels(xlabels, fontsize=8)
    # 5 ticks between 0 and 1, delete the outer one
    ax.set_yticks(np.linspace(0, 1, 10)[1:-1])
    # ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.set_ylim((0.0, 1.0))
    # ax.set_axis_off()
    plt.legend()
    # ax.set_axis_off()
    # delete the outer circle
    ax.spines["polar"].set_visible(False)

    plt.savefig(os.path.join(path_figure, f"{metric}_radar.png"))
    plt.savefig(os.path.join(path_figure, f"{metric}_radar.svg"))
    plt.close()
