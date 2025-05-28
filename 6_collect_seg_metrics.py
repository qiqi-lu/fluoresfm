"""
Collect segmentation evaluation metrics.
"""

import os, tqdm, pandas

id_datasets = [
    "cellpose3-2photon-dn-1",
    "cellpose3-2photon-dn-4",
    "cellpose3-2photon-dn-16",
    "cellpose3-2photon-dn-64",
    "colon-tissue-dn-high",
    "colon-tissue-dn-low",
    "hl60-high-noise-c00",
    "hl60-high-noise-c25",
    "hl60-high-noise-c50",
    "hl60-high-noise-c75",
    "hl60-low-noise-c00",
    "hl60-low-noise-c25",
    "hl60-low-noise-c50",
    "hl60-low-noise-c75",
    "scaffold-a549-dn",
    "granuseg-dn-high",
    "granuseg-dn-low",
    "colon-tissue-dcv-high",
    "colon-tissue-dcv-low",
    "hl60-high-noise-c00-dcv",
    "hl60-high-noise-c25-dcv",
    "hl60-high-noise-c50-dcv",
    "hl60-high-noise-c75-dcv",
    "hl60-low-noise-c00-dcv",
    "hl60-low-noise-c25-dcv",
    "hl60-low-noise-c50-dcv",
    "hl60-low-noise-c75-dcv",
    "granuseg-dcv-high",
    "granuseg-dcv-low",
    "deepbacs-seg-saureus-dcv",
    "deepbacs-seg-bsubtiles-dn",
]

methods = ["raw", "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16"]

metrics_name = ["AP", "IoU"]
path_results = os.path.join("results", "predictions")
path_statictic = os.path.join("results", "statistic", "seg_dataset")
os.makedirs(path_statictic, exist_ok=True)

num_datasets = len(id_datasets)
num_methods = len(methods)

# create xlsx file to same the statistics
writer = pandas.ExcelWriter(os.path.join(path_statictic, "mean_std.xlsx"))
# set the column names
columns = ["id"]
for i in range(num_methods):
    columns.append(f"{methods[i]}-mean")
    columns.append(f"{methods[i]}-std")
    columns.append(f"{methods[i]}-n")

df_mean_std_ap = pandas.DataFrame(columns=columns)
df_mean_std_iou = pandas.DataFrame(columns=columns)

for id_dataset in id_datasets:
    # load metrics
    df_ap = pandas.read_excel(
        os.path.join(path_results, id_dataset, "metrics_seg.xlsx"), sheet_name="AP"
    )
    df_iou = pandas.read_excel(
        os.path.join(path_results, id_dataset, "metrics_seg.xlsx"), sheet_name="UoI"
    )

    # add a new row to the dataframe
    row_ap = [id_dataset]
    row_iou = [id_dataset]
    for i in range(num_methods):
        row_ap.append(df_ap[methods[i]].mean())
        row_ap.append(df_ap[methods[i]].std())
        row_ap.append(len(df_ap[methods[i]]))
        row_iou.append(df_iou[methods[i]].mean())
        row_iou.append(df_iou[methods[i]].std())
        row_iou.append(len(df_iou[methods[i]]))
    # append the row to the dataframe
    df_mean_std_ap.loc[len(df_mean_std_ap)] = row_ap
    df_mean_std_iou.loc[len(df_mean_std_iou)] = row_iou

# save the dataframe to the xlsx file
df_mean_std_ap.to_excel(writer, sheet_name="AP", index=False)
df_mean_std_iou.to_excel(writer, sheet_name="IoU", index=False)
# save the xlsx file
writer.close()
