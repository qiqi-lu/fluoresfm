"""
collect all the (mean, std, n, p value) into single xlsx file.
"""

import pandas, os

# dataset_group = "internal_dataset"
dataset_group = "external_dataset"
tasks = ["sr", "dcv", "dn"]
metrics = ["PSNR", "SSIM", "ZNCC"]
# methods = ["raw", "UniFMIR:all", "UNet-uc:all", "UNet-c:all"]
methods = [
    "raw",
    "UNet-uc:all",
    "UNet-c:all",
    "UNet-uc:all-newnorm",
    "UNet-c:all-TSpixel",
    "UNet-c:all-newnorm_TSmicro",
    "UNet-c:all-newnorm",
]

# ------------------------------------------------------------------------------
columns = ["dataset-name"]
for meth in methods:
    columns.extend([meth + "-mean", meth + "-std", meth + "-n", meth + "-pvalue"])
# delete the last pvalue column, which is the target method and does not have pvalue.
columns.pop(-1)

path_root = os.path.join("outputs", "unet_c", dataset_group, "0_evaluation_metrics")
all_writer = pandas.ExcelWriter(
    os.path.join(path_root, "all_mean_std_pvalue.xlsx"),
    engine="xlsxwriter",
)

for metric in metrics:
    frames = []
    for task in tasks:
        metric_frame = pandas.DataFrame()
        mean_std_frame = pandas.read_excel(
            os.path.join(path_root, f"{task}_mean.xlsx"), sheet_name=metric
        )
        pvalue_frame = pandas.read_excel(
            os.path.join(path_root, f"{task}_pvalue.xlsx"), sheet_name=metric
        )

        for col in columns:
            if "pvalue" in col:
                metric_frame[col] = pvalue_frame[col]
            else:
                metric_frame[col] = mean_std_frame[col]
        frames.append(metric_frame)
    all_frame = pandas.concat(frames, sort=False, ignore_index=True)
    all_frame.to_excel(all_writer, sheet_name=metric, index=False)
all_writer.close()
