"""
Collect all the metrics of different datasets and calculate the mean and std of them.
Seperately run for each task. (as the methods used for each task may be different)
"""

import numpy as np
import pandas, os, tqdm
from dataset_analysis import dataset_names_all

dataset_group = "internal_dataset"
dataset_group = "external_dataset"

tasks = ["sr", "dn", "dcv"]
metrics_name = ["PSNR", "SSIM", "ZNCC"]

path_root = os.path.join("outputs", "unet_c", dataset_group)
path_save_to = os.path.join(path_root, "0_evaluation_metrics")
# ------------------------------------------------------------------------------
# path results
print("-" * 50)
print("Dataset Group:", dataset_group)

for task in tasks:
    print("-" * 50)
    print("Task:", task)
    dataset_names = dataset_names_all[dataset_group][task]
    num_dataset = len(dataset_names)
    print("Number of dataset:", num_dataset)

    writer = pandas.ExcelWriter(
        os.path.join(path_save_to, task + "_mean" + ".xlsx"),
        engine="xlsxwriter",
    )
    for _, metric_name in enumerate(metrics_name):
        mean_std_frame = pandas.DataFrame()

        pbar = tqdm.tqdm(total=num_dataset, desc=metric_name, ncols=80)
        for i in range(num_dataset):
            # read excel of result from current dataset
            data_frame = pandas.read_excel(
                os.path.join(path_root, dataset_names[i], "metrics.xlsx"),
                sheet_name=metric_name,
            )
            methods = list(data_frame.columns[1:])

            # insert inexistent methods
            methods_all = list(mean_std_frame.columns)
            # dataset name colume
            if "dataset-name" not in methods_all:
                mean_std_frame.insert(
                    mean_std_frame.shape[-1], "dataset-name", value=""
                )
            # methods columes
            for meth in methods:
                for name in [meth + "-mean", meth + "-std", meth + "-n"]:
                    if name not in methods_all:
                        mean_std_frame.insert(mean_std_frame.shape[-1], name, value="")

            # insert value
            idx = len(mean_std_frame)
            mean_std_frame.loc[idx, "dataset-name"] = dataset_names[i]  # dataset name

            n = data_frame[methods[0]].shape[0]  # number of sample
            for meth in methods:
                mean = data_frame[meth].mean().astype(np.float32)
                std = data_frame[meth].std().astype(np.float32)
                mean_std_frame.loc[idx, meth + "-mean"] = mean
                mean_std_frame.loc[idx, meth + "-std"] = std
                mean_std_frame.loc[idx, meth + "-n"] = n
            pbar.update(1)
        mean_std_frame.to_excel(writer, index=False, sheet_name=metric_name)
        pbar.close()
    writer.close()
