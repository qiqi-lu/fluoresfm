"""
Dispaly the metrics value of masks.
"""

import os, pandas
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["svg.fonttype"] = "none"

datasets_info_whole = pandas.read_excel("dataset_test-v2.xlsx")
path_statistic = os.path.join("results", "statistic", "segmentation")
path_fig_save_to = os.path.join("results", "figures", "analysis", "segmentation")

# ------------------------------------------------------------------------------
#                             load data
# ------------------------------------------------------------------------------
os.makedirs(path_fig_save_to, exist_ok=True)
# load metrics data
df_ap = pandas.read_excel(
    os.path.join(path_statistic, "mean_std.xlsx"), sheet_name="AP"
)
df_iou = pandas.read_excel(
    os.path.join(path_statistic, "mean_std.xlsx"), sheet_name="IoU"
)

id_datasets = df_ap["id"].tolist()
datasets_info_current = datasets_info_whole[datasets_info_whole["id"].isin(id_datasets)]
# sort according to the order of id_datasets
datasets_info_current = (
    datasets_info_current.set_index("id").loc[id_datasets].reset_index()
)
seg_models = datasets_info_current["seg_model"].tolist()
structures = datasets_info_current["structure"].tolist()

# convert to colors
colors_ticklabels = []
for i in seg_models:
    if i == "cpsam":
        colors_ticklabels.append("#9E4589")
    elif i == "nellie":
        colors_ticklabels.append("#0068A9")

methods_info = [
    ("Raw", "raw", "#42B4B5"),
    ("After restoration", "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16", "#D95D5B"),
]

num_datasets = len(df_ap)
x = np.arange(num_datasets)

# ------------------------------------------------------------------------------
#                             plot the figure
# ------------------------------------------------------------------------------
dict_fig = dict(dpi=300, constrained_layout=True)

nr, nc = 2, 1
fig, axes = plt.subplots(nr, nc, figsize=(11 * nc, 5 * nr))
axes[0].set_box_aspect(0.25)
axes[1].set_box_aspect(0.25)

dict_errorbar = {"fmt": "o", "capsize": 3}

# AP
for meth in methods_info:
    axes[0].errorbar(
        x,
        df_ap[f"{meth[1]}-mean"],
        yerr=df_ap[f"{meth[1]}-std"],
        label=meth[0],
        color=meth[2],
        **dict_errorbar,
    )
axes[0].set_xticks(x)
axes[0].set_xticklabels([])
# axes[0].set_xticklabels(df_ap["id"], rotation=90)
axes[0].set_ylabel("AP")
axes[0].legend()

# IoU
for meth in methods_info:
    axes[1].errorbar(
        x,
        df_iou[f"{meth[1]}-mean"],
        yerr=df_iou[f"{meth[1]}-std"],
        label=meth[0],
        color=meth[2],
        **dict_errorbar,
    )
axes[1].set_xticks(x)
xt = [
    # f'{datasets_info_current["dataset-name"][i]} ({df_iou[f"{meth[1]}-n"][i]})'
    f'SEG-{i} ({structures[i]}) ({df_iou[f"{meth[1]}-n"][i]})'
    for i in range(num_datasets)
]
axes[1].set_xticklabels(xt, rotation=90)
xticklabels = axes[1].get_xticklabels()
for i in range(len(xticklabels)):
    xticklabels[i].set_color(colors_ticklabels[i])
axes[1].set_ylabel("IoU")
# axes[1].legend()

axes[0].set_xlim(-0.5, num_datasets - 0.5)
axes[1].set_xlim(-0.5, num_datasets - 0.5)
axes[0].set_ylim(-0.05, 1.05)
axes[1].set_ylim(-0.05, 1.05)

plt.savefig(os.path.join(path_fig_save_to, "mean_std.png"))
plt.savefig(os.path.join(path_fig_save_to, "mean_std.svg"))

# ------------------------------------------------------------------------------
#                             save source data
# ------------------------------------------------------------------------------
writer_surce = pandas.ExcelWriter(os.path.join(path_fig_save_to, "mean_std.xlsx"))
# save ap
table_title = ["dataset-name", "xticklabel"]
for meth in methods_info:
    table_title.extend([f"{meth[0]}-mean", f"{meth[0]}-std"])
df_ap_source = pandas.DataFrame(columns=table_title)
for i in range(num_datasets):
    row = [datasets_info_current["dataset-name"][i], xt[i]]
    for meth in methods_info:
        row.append(df_ap[f"{meth[1]}-mean"][i])
        row.append(df_ap[f"{meth[1]}-std"][i])
    df_ap_source.loc[len(df_ap_source)] = row
df_ap_source.to_excel(writer_surce, sheet_name="AP", index=False)

# save iou
df_iou_source = pandas.DataFrame(columns=table_title)
for i in range(num_datasets):
    row = [datasets_info_current["dataset-name"][i], xt[i]]
    for meth in methods_info:
        row.append(df_iou[f"{meth[1]}-mean"][i])
        row.append(df_iou[f"{meth[1]}-std"][i])
    df_iou_source.loc[len(df_iou_source)] = row
df_iou_source.to_excel(writer_surce, sheet_name="IoU", index=False)

writer_surce.close()
