"""
Display the distriution of data used for classification.
"""

import os, pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap

plt.rcParams["svg.fonttype"] = "none"

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

# ------------------------------------------------------------------------------
path_data_train = os.path.join("data", "dataset_classfication_train.npy")
path_data_test_in = os.path.join("data", "dataset_classfication_test_in.npy")
path_data_test_ex = os.path.join("data", "dataset_classfication_test_ex.npy")

path_analysis = os.path.join("results", "figures", "analysis", "classification")
# ------------------------------------------------------------------------------
# load data
data_train = np.load(path_data_train, allow_pickle=True).item()
data_test_in = np.load(path_data_test_in, allow_pickle=True).item()
data_test_ex = np.load(path_data_test_ex, allow_pickle=True).item()

num_samples_train = len(data_train["images_embedding"])
num_types_train = len(data_train["structure_types"])

images_embedding_train = data_train["images_embedding"]
labels_train = data_train["labels"]
structure_types_train = data_train["structure_types"]
structure_types_train_abbr = [abbr_table[label] for label in structure_types_train]

images_embedding_test_in = data_test_in["images_embedding"]
labels_test_in = data_test_in["labels"]

images_embedding_test_ex_raw = data_test_ex["images_embedding"]
labels_test_ex_raw = data_test_ex["labels"]
# exclude the samples with label not in the training set
labels_test_ex = []
images_embedding_test_ex = []
for i in range(len(labels_test_ex_raw)):
    if labels_test_ex_raw[i] in structure_types_train:
        labels_test_ex.append(labels_test_ex_raw[i])
        images_embedding_test_ex.append(images_embedding_test_ex_raw[i])

# convert labels to indices
labels_train_id = [structure_types_train.index(label) for label in labels_train]
labels_test_in_id = [structure_types_train.index(label) for label in labels_test_in]
labels_test_ex_id = [structure_types_train.index(label) for label in labels_test_ex]

# ------------------------------------------------------------------------------
num_sample_test_in = len(images_embedding_test_in)
num_sample_test_ex = len(images_embedding_test_ex)

num_structure_types_train = len(structure_types_train)

print("-" * 80)
print("[INFO] Training set:")
print(f"[INFO] Number of samples: {num_samples_train}")
print(f'[INFO] Structure types: {data_train["structure_types"]}')
print(f"[INFO] Number of structure types: {num_types_train}")
print("-" * 80)
print("[INFO] Testing set (internal):")
print(f"[INFO] Number of samples: {num_sample_test_in}")
print(f'[INFO] Structure types: {data_test_in["structure_types"]}')
print("-" * 80)
print("[INFO] Testing set (external):")
print(f"[INFO] Number of samples: {num_sample_test_ex}")
print(f'[INFO] Structure types: {data_test_ex["structure_types"]}')
print("-" * 80)

# ------------------------------------------------------------------------------
# show the distribution of training set in low dimension space
# give each type a color
print("[INFO] use umap to transform the data ...")
colors = sns.color_palette("hls", num_types_train)
dict_class_color = dict(zip(structure_types_train, colors))

reducer = umap.UMAP(min_dist=0.5)
embedding_train = reducer.fit_transform(images_embedding_train)
dataframe = pandas.DataFrame(columns=["UMAP 1", "UMAP 2", "class"])
dataframe["UMAP 1"] = embedding_train[:, 0]
dataframe["UMAP 2"] = embedding_train[:, 1]
dataframe["class"] = labels_train

# use umap model to transform the test set
embedding_in = reducer.transform(images_embedding_test_in)
dataframe_in = pandas.DataFrame(columns=["UMAP 1", "UMAP 2", "class"])
dataframe_in["UMAP 1"] = embedding_in[:, 0]
dataframe_in["UMAP 2"] = embedding_in[:, 1]
dataframe_in["class"] = labels_test_in

embedding_ex = reducer.transform(images_embedding_test_ex)
dataframe_ex = pandas.DataFrame(columns=["UMAP 1", "UMAP 2", "class"])
dataframe_ex["UMAP 1"] = embedding_ex[:, 0]
dataframe_ex["UMAP 2"] = embedding_ex[:, 1]
dataframe_ex["class"] = labels_test_ex

font_size = 8

# ------------------------------------------------------------------------------
# plot the distribution of each class in the training set
print("-" * 80)
print("[INFO] Plotting the distribution of each class ...")
nr, nc = 1, 3
dict_fig = dict(dpi=1200, constrained_layout=True)
dict_scatter = dict(
    s=1, x="UMAP 1", y="UMAP 2", hue="class", palette=dict_class_color, legend=False
)
fig, axes = plt.subplots(nr, nc, figsize=(nc * 3, nr * 3), **dict_fig)

sns.scatterplot(data=dataframe, ax=axes[0], **dict_scatter)
sns.scatterplot(data=dataframe_in, ax=axes[1], **dict_scatter)
sns.scatterplot(data=dataframe_ex, ax=axes[2], **dict_scatter)

for ax in axes.ravel():
    # axes.legend(fontsize=font_size)
    ax.set_xlabel("UMAP 1", fontsize=font_size)
    ax.set_ylabel("UMAP 2", fontsize=font_size)
    ax.tick_params(axis="both", which="major", labelsize=font_size)


fig.savefig(os.path.join(path_analysis, "umap.png"))
fig.savefig(os.path.join(path_analysis, "umap.svg"))

# ------------------------------------------------------------------------------
# plot the distribution of each class
# ------------------------------------------------------------------------------
# count the number of samples in each class
counts_train = np.bincount(labels_train_id, minlength=num_structure_types_train)
counts_test_in = np.bincount(labels_test_in_id, minlength=num_structure_types_train)
counts_test_ex = np.bincount(labels_test_ex_id, minlength=num_structure_types_train)

# ------------------------------------------------------------------------------
dict_bar = dict(width=0.75, alpha=0.5)
nr, nc = 1, 4
fig, axes = plt.subplots(nr, nc, figsize=(nc * 3, nr * 3), **dict_fig)

xtick_pos = np.array(range(num_structure_types_train))

axes[0].set_yticks([0, 1000, 2000, 3000, 4000])
axes[0].set_yticklabels([0, 1000, 2000, 3000, 4000], fontsize=font_size)
axes[1].set_yticks([0, 200, 400, 600, 800])
axes[1].set_yticklabels([0, 200, 400, 600, 800], fontsize=font_size)
axes[2].set_yticks([0, 100, 200, 300, 400])
axes[2].set_yticklabels([0, 100, 200, 300, 400], fontsize=font_size)

axes[0].bar(xtick_pos, counts_train, color=colors)
axes[1].bar(xtick_pos, counts_test_in, color=colors)
axes[2].bar(xtick_pos, counts_test_ex, color=colors)

axes[0].set_title("Training set", fontsize=font_size)
axes[1].set_title("Testing set (in)", fontsize=font_size)
axes[2].set_title("Testing set (ex)", fontsize=font_size)

for ax in axes.ravel():
    ax.set_ylabel("Number of samples", fontsize=font_size)
    # ax.set_xticks(xtick_pos)
    # ax.set_xticklabels(structure_types_train, rotation=90)
    ax.tick_params(axis="both", which="major", labelsize=5)
    ax.set_xlim(-0.6, num_structure_types_train - 0.4)
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_box_aspect(0.5)

# put the lengend to axes[3]
axes[3].legend(
    handles=axes[0].containers[0],
    labels=structure_types_train_abbr,
    fontsize=font_size,
    loc="center",
    frameon=False,
)
axes[3].axis("off")


fig.savefig(os.path.join(path_analysis, "distribution_data.png"))
fig.savefig(os.path.join(path_analysis, "distribution_data.svg"))

# save source data (counts_train, counts_test_in) to excel ---------------------
dataframe = pandas.DataFrame(
    columns=["structure type", "counts_train", "counts_test_in", "counts_test_ex"]
)
dataframe["structure type"] = structure_types_train
dataframe["counts_train"] = counts_train
dataframe["counts_test_in"] = counts_test_in
dataframe["counts_test_ex"] = counts_test_ex
dataframe.to_excel(os.path.join(path_analysis, "distribution_data.xlsx"), index=False)
