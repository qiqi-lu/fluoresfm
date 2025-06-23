"""
Collect segmentation evaluation metrics.
"""

import os, tqdm, pandas
from scipy.stats import wilcoxon
from dataset_analysis import datasets_seg_show

id_datasets = datasets_seg_show

methods = ["raw", "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16"]

metrics_name = ["AP", "IoU"]
path_results = os.path.join("results", "predictions")
path_statictic = os.path.join("results", "statistic", "seg_dataset")
os.makedirs(path_statictic, exist_ok=True)

num_datasets = len(id_datasets)
num_methods = len(methods)

# ------------------------------------------------------------------------------
# create xlsx file to same the statistics
writer = pandas.ExcelWriter(os.path.join(path_statictic, "mean_std.xlsx"))
# set the column names
columns_titles = ["id"]
for i in range(num_methods):
    columns_titles.extend(
        [
            f"{methods[i]}-mean",
            f"{methods[i]}-std",
            f"{methods[i]}-n",
        ]
    )

df_mean_std_ap = pandas.DataFrame(columns=columns_titles)
df_mean_std_iou = pandas.DataFrame(columns=columns_titles)

for id_dataset in id_datasets:
    # load metrics
    df_ap = pandas.read_excel(
        os.path.join(path_results, id_dataset, "metrics_seg.xlsx"), sheet_name="AP"
    )
    df_iou = pandas.read_excel(
        os.path.join(path_results, id_dataset, "metrics_seg.xlsx"), sheet_name="UoI"
    )

    # add a new row to the dataframe
    row_ap, row_iou = [id_dataset], [id_dataset]
    for i in range(num_methods):
        row_ap.extend(
            [df_ap[methods[i]].mean(), df_ap[methods[i]].std(), len(df_ap[methods[i]])]
        )
        row_iou.extend(
            [
                df_iou[methods[i]].mean(),
                df_iou[methods[i]].std(),
                len(df_iou[methods[i]]),
            ]
        )
    # append the row to the dataframe
    df_mean_std_ap.loc[len(df_mean_std_ap)] = row_ap
    df_mean_std_iou.loc[len(df_mean_std_iou)] = row_iou

# save the dataframe to the xlsx file
df_mean_std_ap.to_excel(writer, sheet_name="AP", index=False)
df_mean_std_iou.to_excel(writer, sheet_name="IoU", index=False)
# save the xlsx file
writer.close()
