"""
Pairwise comparison. (Test)
- Wilcoxon rank test.
Output statistics and pvalues.

"""

import scipy.stats
import pandas, tqdm, os, scipy
from dataset_analysis import dataset_names_all

dataset_group = "internal_dataset"
dataset_group = "external_dataset"

# ------------------------------------------------------------------------------
tasks = ["sr", "dn", "dcv"]
metrics_name = ["PSNR", "SSIM", "ZNCC"]

# target_method = "UNet-c:all"
target_method = "UNet-c:all-newnorm"
# compar_methods = ["UNet-uc:all", "UniFMIR:all", "raw"]
compar_methods = [
    "UNet-c:all",
    "UNet-uc:all",
    "UNet-uc:all-newnorm",
    "UNet-c:all-TSpixel",
    "UNet-c:all-newnorm_TSmicro",
    "raw",
]

path_root = os.path.join("outputs", "unet_c", dataset_group)
path_save_to = os.path.join(path_root, "0_evaluation_metrics")

# ------------------------------------------------------------------------------
print("-" * 50)
print("Dataset group:", dataset_group)

for task in tasks:
    print("-" * 50)
    print(f"Task: {task}")
    dataset_names = dataset_names_all[dataset_group][task]
    num_dataset = len(dataset_names)
    print("Number of dataset:", num_dataset)

    # create xlsx to write p values
    p_writer = pandas.ExcelWriter(
        os.path.join(path_save_to, task + "_pvalue.xlsx"),
        engine="xlsxwriter",
    )

    titles = []
    for meth in compar_methods:
        titles.extend([meth + "-statistic", meth + "-pvalue"])

    for metric_name in metrics_name:
        # calculate p value between the target method and the compared methods
        p_value_frame = pandas.DataFrame(columns=["dataset-name"] + titles)

        pbar = tqdm.tqdm(total=num_dataset, desc=metric_name, ncols=80)
        for id_dataset in dataset_names:
            # get the results of current dataset
            metrics_frame = pandas.read_excel(
                os.path.join(path_root, id_dataset, "metrics.xlsx"),
                sheet_name=metric_name,
            )

            idx = len(p_value_frame)  # index of the current row
            p_value_frame.loc[idx, "dataset-name"] = id_dataset
            targ = list(metrics_frame[target_method])

            for id_method in compar_methods:
                comp = list(metrics_frame[id_method])
                res = scipy.stats.wilcoxon(targ, comp, alternative="two-sided")
                statistic, pvalue = res.statistic, res.pvalue
                p_value_frame.loc[idx, id_method + "-statistic"] = statistic
                p_value_frame.loc[idx, id_method + "-pvalue"] = pvalue

            pbar.update(1)
        p_value_frame.to_excel(p_writer, sheet_name=metric_name, index=False)
        pbar.close()
    p_writer.close()
