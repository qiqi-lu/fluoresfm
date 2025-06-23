"""
collect all the (mean, std, n, p value) into single xlsx file.
"""

import pandas, os, tqdm, os, scipy
from dataset_analysis import dataset_names_all
from scipy.stats import wilcoxon


# dataset_group = "internal_dataset"
dataset_group = "external_dataset"

methods = [
    "raw",
    "UniFMIR:all-v2",
    # "UNet-c:all-newnorm-ALL-v2-160-small-bs4",
    # "UNet-c:all-newnorm-ALL-v2-160-small-bs8",
    "UNet-c:all-newnorm-ALL-v2-160-small-bs16-crossx",
    # "UNet-c:all-newnorm-ALL-v2-small-bs16-T77",
    # "UNet-c:all-newnorm-ALL-v2-small-bs16-TS77",
    # "UNet-c:all-newnorm-ALL-v2-small-bs16-TSpixel77",
    # "UNet-c:all-newnorm-ALL-v2-small-bs16-TSmicro77",
    # "UNet-c:all-newnorm-ALL-v2-160-small-bs16-in-T",
    # "UNet-c:all-newnorm-ALL-v2-160-small-bs16-in-TS",
    # "UNet-c:all-newnorm-ALL-v2-160-small-bs16-in-TSpixel",
    # "UNet-c:all-newnorm-ALL-v2-160-small-bs16-in-TSmicro",
    "UNet-c:all-newnorm-ALL-v2-160-small-bs16",  # must be the last one
]

methods_comp = methods[:-1]
methods_targ = methods[-1]

# metrics = ["PSNR", "SSIM", "ZNCC"]
metrics = ["PSNR", "MSSSIM", "ZNCC"]
# metrics = ["PSNR", "SSIM", "ZNCC", "NRMSE", "MSSSIM"]
tasks = ["sr", "dcv", "dn"]

# ------------------------------------------------------------------------------
path_statistic = os.path.join("results", "statistic", dataset_group)
path_predictions = os.path.join("results", "predictions")

columns = ["dataset-name", "task"]
for meth in methods:
    columns.extend(
        [
            meth + "-mean",
            meth + "-std",
            meth + "-n",
            meth + "-statistic",
            meth + "-pvalue",
        ]
    )
# delete the last pvalue column, which is the target method and does not have pvalue.
columns.pop(-1)
columns.pop(-1)

# ------------------------------------------------------------------------------
writer = pandas.ExcelWriter(
    os.path.join(path_statistic, "all_mean_std_pvalue.xlsx"), engine="xlsxwriter"
)

for metric in metrics:
    print("-" * 80)
    print("Metric:", metric)
    mean_std_pvalue_frame = pandas.DataFrame(columns=columns)

    for task in tasks:
        dataset_names = dataset_names_all[dataset_group][task]
        num_dataset = len(dataset_names)

        # ----------------------------------------------------------------------
        pbar = tqdm.tqdm(total=num_dataset, desc=task, ncols=80)
        for dataset_name in dataset_names:
            # read excel of result from current dataset
            data_frame = pandas.read_excel(
                os.path.join(path_predictions, dataset_name, "metrics-v2.xlsx"),
                sheet_name=metric,
            )
            idx = len(mean_std_pvalue_frame)
            mean_std_pvalue_frame.loc[idx, "dataset-name"] = dataset_name
            mean_std_pvalue_frame.loc[idx, "task"] = task
            # ------------------------------------------------------------------
            # calculate the mean, std, n, statistic, and p value.
            n = len(data_frame)
            assert n > 0, f"[ERROR] No samples found in the table of [{dataset_name}]."

            assert (
                methods_targ in data_frame.columns
            ), f"[ERROR] Target method [{methods_targ}] not found in the dataset [{dataset_name}]."

            data_targ = data_frame[methods_targ]

            # save to frame
            mean_std_pvalue_frame.loc[idx, methods_targ + "-mean"] = data_targ.mean()
            mean_std_pvalue_frame.loc[idx, methods_targ + "-std"] = data_targ.std()
            mean_std_pvalue_frame.loc[idx, methods_targ + "-n"] = n

            for meth in methods_comp:
                if meth not in data_frame.columns:
                    continue
                data_comp = data_frame[meth]
                res = wilcoxon(data_targ, data_comp, alternative="greater")

                # save to frame
                mean_std_pvalue_frame.loc[idx, meth + "-mean"] = data_comp.mean()
                mean_std_pvalue_frame.loc[idx, meth + "-std"] = data_comp.std()
                mean_std_pvalue_frame.loc[idx, meth + "-n"] = n
                mean_std_pvalue_frame.loc[idx, meth + "-statistic"] = res.statistic
                mean_std_pvalue_frame.loc[idx, meth + "-pvalue"] = res.pvalue
            pbar.update(1)
        pbar.close()
    mean_std_pvalue_frame.to_excel(writer, sheet_name=metric, index=False)
    del mean_std_pvalue_frame  # free memory
writer.close()
