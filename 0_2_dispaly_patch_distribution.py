"""
Display the distribution of pathces, according to the task, structure, and microscope.

Parameters:
- data_augmentation: The repeat number of data in the 64x64-aug sheet.
"""

import pandas, os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["svg.fonttype"] = "none"

data_augmentation = 3
# ------------------------------------------------------------------------------
# load dataset information from excel file
df_dataset = pandas.read_excel("dataset_train_transformer-v2.xlsx", sheet_name="64x64")
path_figures = os.path.join("results", "figures", "datasets")

# data augmentation
if data_augmentation > 0:
    print("Data Augmentation:", data_augmentation)
    # add the augmented datasets
    df_dataset_aug = pandas.read_excel(
        "dataset_train_transformer-v2.xlsx", sheet_name="64x64-aug"
    )
    df_dataset = pandas.concat([df_dataset] + [df_dataset_aug] * data_augmentation)

# get the task, structure, and microscope information
tasks = list(df_dataset["task"])
structures = list(df_dataset["structure"])
microscopes_input = list(df_dataset["input microscope-device"])
microscopes_target = list(df_dataset["target microscope-device"])

dict_fig = {"dpi": 300, "constrained_layout": True}
font_size = 8


# ------------------------------------------------------------------------------
# tasks
# ------------------------------------------------------------------------------
print("-" * 80)
tasks = list(set(tasks))
print("Number of tasks:", len(tasks), ":", tasks)

# number of patches for each tasks
num_patches, num_patches_clean = [0] * len(tasks), [0] * len(tasks)

for i in range(len(tasks)):
    # get all the rows with current task
    df_task = df_dataset[df_dataset["task"] == tasks[i]]
    # get the number of patches
    num_patches[i] = int(df_task["number of patches"].sum())
    num_patches_clean[i] = int(df_task["number of patches-clean"].sum())

df = pandas.DataFrame(
    {
        "task": tasks,
        "number of patches-clean": num_patches_clean,
        "number of patches": num_patches,
    }
)
# sorted by number of patches
df = df.sort_values(by=["number of patches-clean"], ascending=False)
print(df.to_string(index=False))

print("Sum:", np.sum(num_patches))
print("Sum (clean):", np.sum(num_patches_clean))
print("Ratio (clean):", np.sum(num_patches_clean) / np.sum(num_patches))

# ------------------------------------------------------------------------------
nr, nc = 1, 2
colors_task = ["#EA9A9D", "#96C36E", "#92C4E9"]
labels_task = ["SR", "DCV", "DN"]
dict_bar = dict(color=colors_task, height=0.5, y=[1.2, 0.6, 0])
fig, axes = plt.subplots(nr, nc, figsize=(3 * nc, 3 * nr), **dict_fig)
font_size = 8
# plot the bar chart
num_task_show = len(labels_task)

for i, tmp in enumerate(["number of patches", "number of patches-clean"]):
    ax = axes[i]
    data = [
        df[df["task"] == "sr"][tmp].values[0] + df[df["task"] == "iso"][tmp].values[0],
        df[df["task"] == "dcv"][tmp].values[0],
        df[df["task"] == "dn"][tmp].values[0],
    ]
    bars = ax.barh(width=data, **dict_bar)

    # add the number of patches on the right of the bar
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width - 0.05 * width,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.0f}",
            ha="right",
            va="center",
            color="white",
            fontsize=font_size,
        )

    ax.set_yticks([1.2, 0.6, 0])
    ax.set_yticklabels(labels_task, fontsize=font_size)
    ax.set_title(tmp, fontsize=font_size)
    ax.set_box_aspect(0.5)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

plt.savefig(os.path.join(path_figures, "patch_distribution_tasks.png"))
plt.savefig(os.path.join(path_figures, "patch_distribution_tasks.svg"))
# save source data
df_save = pandas.DataFrame(
    {
        "task": labels_task,
        "number of patches": data,
    }
)
df_save.to_excel(
    os.path.join(path_figures, "patch_distribution_tasks.xlsx"), index=False
)


# ------------------------------------------------------------------------------
# structures
# ------------------------------------------------------------------------------
print("-" * 80)
structures = list(set(structures))
num_patches, num_patches_clean = [0] * len(structures), [0] * len(structures)

for i in range(len(structures)):
    # get all the rows with current structure
    df_structure = df_dataset[df_dataset["structure"] == structures[i]]
    # get the number of patches
    num_patches[i] = int(df_structure["number of patches"].sum())
    num_patches_clean[i] = int(df_structure["number of patches-clean"].sum())

df = pandas.DataFrame(
    {
        "structure": structures,
        "number of patches-clean": num_patches_clean,
        "number of patches": num_patches,
    }
)
# sorted by number of patches
df = df.sort_values(by=["number of patches-clean"], ascending=False)
print(df.to_string(index=False))

# ------------------------------------------------------------------------------
nr, nc = 2, 1
fig, axes = plt.subplots(nr, nc, figsize=(7.5 * nc, 3 * nr), **dict_fig)

cmap = plt.cm.Blues_r
colors_structure = [cmap(i) for i in np.linspace(0.2, 0.8, len(structures))]
dict_bar = dict(width=0.5, color=colors_structure)

xtick_pos = np.arange(len(df["structure"])) * 0.6

for i, tmp in enumerate(["number of patches", "number of patches-clean"]):
    ax = axes[i]
    ax.bar(xtick_pos, df[tmp], **dict_bar)

    # add the number of patches on the top of the bar
    for pos, value in zip(xtick_pos, df[tmp]):
        ax.text(
            pos,
            value,
            f"{value:.0f}",
            ha="center",
            va="bottom",
            fontsize=font_size * 0.8,
            rotation=90,
        )
    ax.set_box_aspect(0.3)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(df["structure"], fontsize=font_size, rotation=90)
    ax.set_xlim(-0.5, xtick_pos.max() + 0.5)
    ax.set_ylim(0, df[tmp].max() * 1.3)
    ax.set_title(tmp, fontsize=font_size)

plt.savefig(os.path.join(path_figures, "patch_distribution_structure.png"))
plt.savefig(os.path.join(path_figures, "patch_distribution_structure.svg"))
# save source data
df_save = pandas.DataFrame(
    {
        "structure": df["structure"],
        "number of patches": df["number of patches-clean"],
    }
)
df_save.to_excel(
    os.path.join(path_figures, "patch_distribution_structure.xlsx"), index=False
)


# ------------------------------------------------------------------------------
# input microscope
# ------------------------------------------------------------------------------
print("-" * 80)
microscopes_input = list(set(microscopes_input))
# number of patches for eahc microscopes
num_patches = [0] * len(microscopes_input)
num_patches_clean = [0] * len(microscopes_input)

for i in range(len(microscopes_input)):
    # get all the rows with current microscope
    df_microscope = df_dataset[
        df_dataset["input microscope-device"] == microscopes_input[i]
    ]
    # get the number of patches
    num_patches[i] = int(df_microscope["number of patches"].sum())
    num_patches_clean[i] = int(df_microscope["number of patches-clean"].sum())

df = pandas.DataFrame(
    {
        "microscope": microscopes_input,
        "number of patches-clean": num_patches_clean,
        "number of patches": num_patches,
    }
)
# sorted by number of patches
df = df.sort_values(by=["number of patches-clean"], ascending=False)
print(df.to_string(index=False))

# ------------------------------------------------------------------------------
# target microscope
# ------------------------------------------------------------------------------
print("-" * 80)
microscopes_target = list(set(microscopes_target))
# number of patches for eahc microscopes
num_patches = [0] * len(microscopes_target)
num_patches_clean = [0] * len(microscopes_target)

for i in range(len(microscopes_target)):
    # get all the rows with current microscope
    df_microscope = df_dataset[
        df_dataset["target microscope-device"] == microscopes_target[i]
    ]
    # get the number of patches
    num_patches[i] = int(df_microscope["number of patches"].sum())
    num_patches_clean[i] = int(df_microscope["number of patches-clean"].sum())

df = pandas.DataFrame(
    {
        "microscope": microscopes_target,
        "number of patches-clean": num_patches_clean,
        "number of patches": num_patches,
    }
)
# sorted by number of patches
df = df.sort_values(by=["number of patches-clean"], ascending=False)
print(df.to_string(index=False))
