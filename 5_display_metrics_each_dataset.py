"""
Dsipaly metrics of each dataset.
"""

import numpy as np
import pandas, os, tqdm
import matplotlib.pyplot as plt

# path datasets
path_root = os.path.join("outputs", "unet_c", "external_dataset")
# path_root = os.path.join("outputs", "unet_c", "internal_dataset")

# ------------------------------------------------------------------------------
# get all the dataset names in the folder
dataset_names = os.listdir(path_root)
# only the folder
dataset_names = [x for x in dataset_names if os.path.isdir(os.path.join(path_root, x))]
# del the folder with 0 start
dataset_names = [x for x in dataset_names if not x.startswith("0")]
# sort the dataset names
dataset_names.sort()

# metrics
metrics = ["PSNR", "SSIM", "ZNCC"]

# display
pbar = tqdm.tqdm(desc="PLOT", total=len(dataset_names))
for dataset_name in dataset_names:
    # get the path of the dataset
    path_dataset = os.path.join(path_root, dataset_name)
    # load the calculated metrics in the xlsx file
    path_xlsx = os.path.join(path_dataset, "metrics.xlsx")
    metric_frames = []
    try:
        for metric in metrics:
            # load the calculated metrics in the xlsx file
            df_metric = pandas.read_excel(path_xlsx, sheet_name=metric)
            metric_frames.append(df_metric)
    except Exception as e:
        print("Error:", e)
        continue

    num_sample = metric_frames[0].shape[0]
    methods = metric_frames[0].columns[1:].tolist()

    # plot the metrics
    fig, axes = plt.subplots(
        1, len(metrics), figsize=(len(metrics) * 3, 4), dpi=300, constrained_layout=True
    )
    for i_metric, metric_frame in enumerate(metric_frames):
        # get the value in the frame
        metric_values = metric_frame.values[:, 1:]
        # boxplot and overlay the value on the boxplot
        axes[i_metric].boxplot(metric_values, tick_labels=methods)
        # overlay scatter plot on the boxplot
        for i_method, method in enumerate(methods):
            axes[i_metric].scatter(
                np.ones(num_sample) * (i_method + 1),
                metric_values[:, i_method],
                color="blue",
                s=5,
            )
        axes[i_metric].set_title(metrics[i_metric])
        axes[i_metric].set_xticklabels(methods, rotation=90, fontsize=5)
        # set the ylim
        axes[i_metric].set_ylim(metric_values.min() * 0.99, metric_values.max() * 1.01)
    # save the figure
    path_figure = os.path.join(path_root, "0_metrics", dataset_name + ".png")
    plt.savefig(path_figure)
    plt.close()
    pbar.update(1)
pbar.close()
