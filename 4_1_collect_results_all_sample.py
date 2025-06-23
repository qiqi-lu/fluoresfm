"""
Collect results from all samples.
Save ther results from each task to a separate excel file.
"""

import os, pandas
from dataset_analysis import dataset_names_all

# ------------------------------------------------------------------------------
# dataset_group = "internal_dataset"
dataset_group = "external_dataset"
methods = [
    "raw",
    "UniFMIR:all-v2",
    # "UNet-c:all-newnorm-ALL-v2-160-small-bs4",
    # "UNet-c:all-newnorm-ALL-v2-160-small-bs8",
    "UNet-c:all-newnorm-ALL-v2-160-small-bs16",
    "UNet-c:all-newnorm-ALL-v2-160-small-bs16-crossx",
    "UNet-c:all-newnorm-ALL-v2-small-bs16-T77",
    "UNet-c:all-newnorm-ALL-v2-small-bs16-TS77",
    # "UNet-c:all-newnorm-ALL-v2-small-bs16-TSpixel77",
    # "UNet-c:all-newnorm-ALL-v2-small-bs16-TSmicro77",
    "UNet-c:all-newnorm-ALL-v2-160-small-bs16-in-T",
    "UNet-c:all-newnorm-ALL-v2-160-small-bs16-in-TS",
    # "UNet-c:all-newnorm-ALL-v2-160-small-bs16-in-TSpixel",
    # "UNet-c:all-newnorm-ALL-v2-160-small-bs16-in-TSmicro",
]

tasks = ["sr", "dcv", "dn"]
metrics = ["PSNR", "MSSSIM", "ZNCC"]

path_prediciton = os.path.join("results", "predictions")
path_statistic = os.path.join("results", "statistic", dataset_group)

# ------------------------------------------------------------------------------
frame_title = ["dataset-name"] + methods

for task in tasks:
    path_save_to = os.path.join(path_statistic, task + "_value.xlsx")
    writer = pandas.ExcelWriter(path_save_to, engine="xlsxwriter")
    datasets = dataset_names_all[dataset_group][task]

    print("-" * 50)
    print("Task:", task)
    print("Number of dataset:", len(datasets))
    print("Save to:", path_save_to)
    print("-" * 50)

    frames = []
    for metric in metrics:
        metric_frame = pandas.DataFrame(columns=frame_title)
        for dataset in datasets:
            # read excel of result from current dataset
            data_frame = pandas.read_excel(
                os.path.join(path_prediciton, dataset, "metrics-v2.xlsx"),
                sheet_name=metric,
            )
            # get the number of samples
            n = data_frame[methods[0]].shape[0]
            assert n > 0, f"No samples found in the table of [{dataset}]."
            # get the data
            data_frame = data_frame[methods]
            # rename the columns
            data_frame.columns = methods
            # add the dataset name to the first column
            data_frame.insert(0, "dataset-name", [dataset] * n)
            # add the data to the metric frame
            metric_frame = pandas.concat([metric_frame, data_frame], ignore_index=True)
        frames.append(metric_frame)
    # save the metric frame to the excel file
    for i, metric in enumerate(metrics):
        frames[i].to_excel(writer, sheet_name=metric, index=False)
    # save the excel file
    writer.close()
