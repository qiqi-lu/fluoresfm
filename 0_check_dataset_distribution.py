"""
Check the distribution of dataset.
- task
- structure
- microscope
"""

import pandas, os, tqdm
import numpy as np

data_augmentation = 3
data_augmentation = 0

if data_augmentation:
    print("Data Augmentation:", data_augmentation)

# load dataset information from excel file
# df_dataset = pandas.read_excel("dataset_train_transformer.xlsx", sheet_name="64x64")
df_dataset = pandas.read_excel("dataset_train_transformer-v2.xlsx", sheet_name="64x64")

if data_augmentation > 0:
    # load dataset information from excel file
    df_dataset_aug = pandas.read_excel(
        "dataset_train_transformer-v2.xlsx", sheet_name="64x64-aug"
    )
    df_dataset = pandas.concat([df_dataset] + [df_dataset_aug] * data_augmentation)

# get the task, structure, and microscope information
tasks = list(df_dataset["task"])
structures = list(df_dataset["structure"])
microscopes_input = list(df_dataset["input microscope-device"])
microscopes_target = list(df_dataset["target microscope-device"])

# ------------------------------------------------------------------------------
# tasks
print("-" * 50)
tasks = list(set(tasks))
# number of patches for eahc tasks
num_patches = [0] * len(tasks)
num_patches_clean = [0] * len(tasks)

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


# ------------------------------------------------------------------------------
# structures
print("-" * 50)
structures = list(set(structures))
# number of patches for eahc structures
num_patches = [0] * len(structures)
num_patches_clean = [0] * len(structures)
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
# input microscope
print("-" * 50)
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
print("-" * 50)
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
