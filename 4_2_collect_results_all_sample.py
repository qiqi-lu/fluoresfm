"""
Collect results from all samples.
Save the results of each sample to a single excel file for each task.
Output:
    - results/statistic/{dataset_group}/{task}_value.xlsx
    -----------------------------------------------------
    dataset-name | method-1 | method-2 | ...
    -----------------------------------------------------
    dataset-1    | value-1  | value-2  |...
    dataset-1    | value-1  | value-2  |...
    dataset-1    | value-1  | value-2  |...
    dataset-2    | value-1  | value-2  |...
    ...          | ...      | ...      |...
    -----------------------------------------------------
"""

import os, pandas
from dataset_analysis import dataset_names_all

# ------------------------------------------------------------------------------
dataset_group = "internal_dataset"
# dataset_group = "external_dataset"
excel_file_name = "metrics-v3.xlsx"
# ------------------------------------------------------------------------------

methods = [
    "raw",
    "UniFMIR:all-v2",
    # "UNet-c:all-newnorm-ALL-v2-160-small-bs4",
    # "UNet-c:all-newnorm-ALL-v2-160-small-bs8",
    "UNet-c:all-newnorm-ALL-v2-160-small-bs16",
    "UNet-c:all-newnorm-ALL-v2-160-small-bs16-crossx",
    "UNet-c:all-newnorm-ALL-v2-small-bs16-T77",
    "UNet-c:all-newnorm-ALL-v2-small-bs16-TS77",
    "UNet-c:all-newnorm-ALL-v2-small-bs16-TSpixel77",
    "UNet-c:all-newnorm-ALL-v2-small-bs16-TSmicro77",
    "UNet-c:all-newnorm-ALL-v2-160-small-bs16-in-T",
    "UNet-c:all-newnorm-ALL-v2-160-small-bs16-in-TS",
    "UNet-c:all-newnorm-ALL-v2-160-small-bs16-in-TSpixel",
    "UNet-c:all-newnorm-ALL-v2-160-small-bs16-in-TSmicro",
    "UNet-c:all-newnorm-ALL-v2-160-small-bs16-structural-prompt",
    "UNet-c:all-newnorm-ALL-v2-160-small-bs16-wo-target-metadata",
    "UNet-c:all-newnorm-ALL-v2-160-small-bs16-in-wo-target-metadata",
]

tasks = ["sr", "dcv", "dn"]
metrics = ["PSNR", "MSSSIM", "ZNCC", "RSE", "RSP", "Resolution (DA)"]

# ------------------------------------------------------------------------------
path_prediciton = os.path.join("results", "predictions")
path_statistic = os.path.join("results", "statistic", dataset_group)
os.makedirs(path_statistic, exist_ok=True)

# ------------------------------------------------------------------------------
frame_title = ["dataset-name"] + methods

for task in tasks:
    path_save_to = os.path.join(path_statistic, task + "_value.xlsx")
    datasets = dataset_names_all[dataset_group][task]

    print("-" * 80)
    print("[INFO] Task:", task)
    print("[INFO] Number of dataset:", len(datasets))
    print("[INFO] Save to:", path_save_to)
    print("-" * 80)

    frames = []
    for metric in metrics:
        metric_frame = pandas.DataFrame(columns=frame_title)
        for dataset in datasets:
            try:
                # read excel of result from current dataset
                data_frame = pandas.read_excel(
                    os.path.join(path_prediciton, dataset, excel_file_name),
                    sheet_name=metric,
                )
            except:
                print(f"[WARNING] No result found in [{dataset}] for [{metric}]")
                continue

            # get the number of samples
            n = data_frame[methods[0]].shape[0]
            assert n > 0, f"[WARNNING] No samples found in the table of [{dataset}]."

            # check if all the methods in the data frame
            for method in methods:
                assert (
                    method in data_frame.columns
                ), f"[ERROR] Method [{method}] not found in the table of [{dataset}]."

            # get the data
            df = data_frame[methods]
            # rename the columns
            df.columns = methods
            # add the dataset name to the first column
            df.insert(0, "dataset-name", [dataset] * n)
            # add the data to the metric frame
            if metric_frame.empty:
                metric_frame = df
            else:
                metric_frame = pandas.concat([metric_frame, df], ignore_index=True)
        frames.append(metric_frame)

    # save the metric frame to the excel file
    writer = pandas.ExcelWriter(path_save_to, engine="xlsxwriter")
    for i, metric in enumerate(metrics):
        frames[i].to_excel(writer, sheet_name=metric, index=False)
    # save the excel file
    writer.close()
