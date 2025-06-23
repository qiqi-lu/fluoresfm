"""
Dsipaly metrics of each dataset through box plot.
"""

import numpy as np
import pandas, os, tqdm
import matplotlib.pyplot as plt

dataset_group = "internal_dataset"
# dataset_group = "external_dataset"

methods = [
    ("Raw", "raw"),
    ("UniFMIR", "UniFMIR:all-v2"),
    ("FluoResFM-bs4", "UNet-c:all-newnorm-ALL-v2-160-small-bs4"),
    ("FluoResFM-bs8", "UNet-c:all-newnorm-ALL-v2-160-small-bs8"),
    ("FluoResFM-bs16", "UNet-c:all-newnorm-ALL-v2-160-small-bs16"),
    ("FluoResFM (w/o text)", "UNet-c:all-newnorm-ALL-v2-160-small-bs16-crossx"),
    ("FluoResFM-T", "UNet-c:all-newnorm-ALL-v2-small-bs16-T77"),
    ("FluoResFM-TS", "UNet-c:all-newnorm-ALL-v2-small-bs16-TS77"),
    ("FluoResFM-TSpixel", "UNet-c:all-newnorm-ALL-v2-small-bs16-TSpixel77"),
    ("FluoResFM-TSmicro", "UNet-c:all-newnorm-ALL-v2-small-bs16-TSmicro77"),
    ("FluoResFM-T (in)", "UNet-c:all-newnorm-ALL-v2-160-small-bs16-in-T"),
    ("FluoResFM-TS (in)", "UNet-c:all-newnorm-ALL-v2-160-small-bs16-in-TS"),
    ("FluoResFM-TSpixel (in)", "UNet-c:all-newnorm-ALL-v2-160-small-bs16-in-TSpixel"),
    ("FluoResFM-TSmicro (in)", "UNet-c:all-newnorm-ALL-v2-160-small-bs16-in-TSmicro"),
]

# ------------------------------------------------------------------------------
path_predictions = os.path.join("results", "predictions")
path_figures = os.path.join("results", "figures", "analysis", "metrics")


# ------------------------------------------------------------------------------
# get all the dataset names in the folder
dataset_names = os.listdir(path_predictions)
# only the folder
dataset_names = [
    x for x in dataset_names if os.path.isdir(os.path.join(path_predictions, x))
]
dataset_names.sort()

# metrics
metrics = ["PSNR", "SSIM", "ZNCC"]
methods_id = [x[1] for x in methods]
titles = [x[0] for x in methods]

# ------------------------------------------------------------------------------
# display the metrics of each dataset through box plot
pbar = tqdm.tqdm(desc="PLOT", total=len(dataset_names))
for dataset_name in dataset_names:
    path_xlsx = os.path.join(path_predictions, dataset_name, "metrics.xlsx")
    data_frames = []
    try:
        for metric in metrics:
            df_metric = pandas.read_excel(path_xlsx, sheet_name=metric)
            data_frames.append(df_metric)
    except Exception as e:
        print("Error:", e)
        continue

    num_sample = data_frames[0].shape[0]

    # plot the metrics
    fig, axes = plt.subplots(
        1, len(metrics), figsize=(len(metrics) * 3, 4), dpi=300, constrained_layout=True
    )
    for i_metric, metric_frame in enumerate(data_frames):
        metric_values = metric_frame[methods_id].values
        axes[i_metric].boxplot(metric_values)
        for i_method, method in enumerate(methods_id):
            axes[i_metric].scatter(
                np.ones(num_sample) * (i_method + 1),
                metric_values[:, i_method],
                color="blue",
                s=5,
            )
        axes[i_metric].set_title(metrics[i_metric])
        axes[i_metric].set_xticklabels(titles, rotation=90, fontsize=5)
        axes[i_metric].set_ylim(metric_values.min() * 0.99, metric_values.max() * 1.01)
    # save the figure
    plt.savefig(os.path.join(path_figures, dataset_name + ".png"))
    plt.close()
    del fig, axes
    pbar.update(1)
pbar.close()
