"""
Display the results of the classification task.
"""

import os, pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["svg.fonttype"] = "none"

group = "internal"
# group = "external"

abbr_table = {
    "clathrin-coated pits": "CCP",
    "endoplasmic reticulum": "ER",
    "microtubule": "MT",
    "actin filament": "F-actin",
    "membrane": "membrane",
    "mitochondria": "Mito",
    "lysosome": "lysosome",
    "histone": "histone",
    "survivin": "survivin",
    "tubulin": "tubulin",
    "nuclear pore complex": "NPC",
    "Myosin-IIA": "Myosin-IIA",
    "nucleoid": "nucleoid",
    "MreB filament": "MreB",
    "nuclei": "nuclei",
    "neural crest cells": "NCC",
    "chromosome": "chromosome",
    "Golgi": "Golgi",
}

path_data_train = os.path.join("data", "dataset_classfication_train.npy")
path_results = os.path.join("results", "figures", "analysis", "classification", group)
path_excel_confusion_matrix = os.path.join(path_results, "confusion_matrix.xlsx")
path_excel_retrieval = os.path.join(path_results, "image_retrieval.xlsx")
path_excel_manual = os.path.join("data", "patches_in", "labels.xlsx")
# ------------------------------------------------------------------------------
# load the results
confusion_matrix = pandas.read_excel(path_excel_confusion_matrix, index_col=0).values
print("-" * 80)
print("[INFO] Confusion matrix:")
print(confusion_matrix)

retrieval = pandas.read_excel(path_excel_retrieval)
labels_gt = retrieval["label_gt"].values
labels_est = retrieval["label_est"].values


structure_types_train = np.load(path_data_train, allow_pickle=True).item()[
    "structure_types"
]
structure_types_train_abbr = [abbr_table[label] for label in structure_types_train]


# convert the labels to indices, use the types in training set as the labels
labels_gt = [structure_types_train.index(label) for label in labels_gt]
labels_est = [structure_types_train.index(label) for label in labels_est]

# ------------------------------------------------------------------------------
manual = pandas.read_excel(path_excel_manual)
manual_id_true = manual["id_true"].values
labels_manual = manual["label"].values

# ------------------------------------------------------------------------------
# only use the first 1800 samples
id_used = manual_id_true[:1800]
labels_gt_used = []
labels_est_used = []
for id in id_used:
    # convert str to int
    id = int(id)
    labels_gt_used.append(labels_gt[id])
    labels_est_used.append(labels_est[id])
# convert the labels in labels_manual to int
labels_manual = [int(label) - 1 for label in labels_manual[:1800]]
# ------------------------------------------------------------------------------
# calculate the accuracy between labels_gt_used and labels_est_used
accuracy_all = np.sum(np.array(labels_gt) == np.array(labels_est)) / len(labels_gt)
accuracy_1800_est = np.sum(np.array(labels_gt_used) == np.array(labels_est_used)) / len(
    labels_gt_used
)
# calculate the accuracy between labels_gt_used and labels_manual
accuracy_1800_manual = np.sum(
    np.array(labels_gt_used) == np.array(labels_manual)
) / len(labels_gt_used)

print(f"[INFO] Accuracy between labels_gt and labels_est: {accuracy_all * 100:.2f}%")
print(
    f"[INFO] Accuracy between labels_gt_used and labels_est_used: {accuracy_1800_est * 100:.2f}%"
)
print(
    f"[INFO] Accuracy between labels_gt_used and labels_manual: {accuracy_1800_manual * 100:.2f}%"
)

# ------------------------------------------------------------------------------
# plot the confusion matrix
# ------------------------------------------------------------------------------
font_size = 6

nr, nc = 1, 1
dict_fig = dict(dpi=300, constrained_layout=True)
fig, axes = plt.subplots(nr, nc, figsize=(nc * 3, nr * 3), **dict_fig)
# plot the confusion matrix
sns.heatmap(
    data=confusion_matrix,
    square=True,
    cmap="Blues",
    linewidths=0.5,
    xticklabels=structure_types_train_abbr,
    yticklabels=structure_types_train_abbr,
    ax=axes,
    annot=True,
    fmt="d",
    annot_kws={"size": 3, "ha": "center", "va": "center"},
    cbar=False,
    vmax=np.max(confusion_matrix),
    vmin=np.min(confusion_matrix),
)
# set the font size of the labels in x and y axis
axes.tick_params(axis="both", which="major", labelsize=5, length=0)

# set the tick label to red when it is in test
# get ticklabels
axes.set_xlabel("Predicted", fontsize=font_size)
axes.set_ylabel("True", fontsize=font_size)
axes.set_title(f"Accuracy: {accuracy_all * 100:.2f}%", fontsize=font_size)

fig.savefig(os.path.join(path_results, "confusion_matrix.png"))
fig.savefig(os.path.join(path_results, "confusion_matrix.svg"))

# ------------------------------------------------------------------------------
# plot comparsion to manual
# ------------------------------------------------------------------------------
nr, nc = 1, 1
dict_fig = dict(dpi=300, constrained_layout=True)
fig, axes = plt.subplots(nr, nc, figsize=(nc * 3, nr * 3), **dict_fig)
font_size = 8
axes.bar(
    x=[0, 1],
    height=[accuracy_1800_manual * 100, accuracy_1800_est * 100],
    color=["#004586", "#912322"],
    label=["Manual", "Retrieval"],
    width=0.75,
    edgecolor=None,
    linewidth=1,
)
# label value at the top of the bar
for i, v in enumerate([accuracy_1800_manual * 100, accuracy_1800_est * 100]):
    axes.text(
        i,
        v + 1,
        f"{v:.2f}%",
        ha="center",
        va="bottom",
        fontsize=font_size,
    )
axes.set_xticks([])
axes.set_xticklabels([])
axes.legend(fontsize=font_size, loc="best", frameon=False)
axes.set_ylabel("Accuracy (%)", fontsize=font_size)
axes.tick_params(axis="both", which="major", labelsize=font_size)
axes.set_ylim(50, 100)
axes.set_box_aspect(1.5)
# disable right and top axis
axes.spines["right"].set_visible(False)
axes.spines["top"].set_visible(False)

fig.savefig(os.path.join(path_results, "comparison_to_manual.png"))
fig.savefig(os.path.join(path_results, "comparison_to_manual.svg"))
