"""
Read all the training patches.
Exclude the patches with only background noise based on the value in the high-quality patches
Save the filename of the patch into the training txt file with a suffix `_clean`.

Parameters:
- condition_type:   Use `max` or `mean` as the filtering condition.
- threshold:        The threshold value for the filtering condition.
- specific_dataset: The specific dataset to be processed.
                    If not specified, all the datasets will be processed.
"""

import pandas, tqdm, os
import skimage.io as io
from utils.data import read_txt, win2linux

condition_type, threshold = "max", 0.02  # all
# condition_type, threshold = "mean", 0.06  # golgi

specific_dataset = [
    # "vmsim5-mito-dcv",
    # "vmsim5-mito-sr",
    # "rcan3d-c2s-mt-dcv",
    # "rcan3d-c2s-mt-sr",
    # "rcan3d-c2s-npc-dcv",
    # "rcan3d-c2s-npc-sr",
    # "rcan3d-dn-golgi-dn",
    # "biotisr-ccp-sr-1",
    # "biotisr-ccp-sr-1-2",
]

# ------------------------------------------------------------------------------
data_frame = pandas.read_excel("dataset_train_transformer-v2.xlsx", sheet_name="64x64")

print("Condition Type:", condition_type)
print("Threshold:", threshold)

# load training dataset information
if specific_dataset:
    data_frame = data_frame[data_frame["id"].isin(specific_dataset)]

path_dataset_hr = list(data_frame["path_hr"])
path_index_file = list(data_frame["path_index"])

num_dataset = len(list(data_frame["index"]))
print("Number of Dataset:", num_dataset)

# ------------------------------------------------------------------------------
path_txt_collect = []  # collect all the datasets that have been processed

pbar_dataset = tqdm.tqdm(desc="DATA CLEANING", total=num_dataset, ncols=80, leave=True)
for i_dataset in range(num_dataset):
    path_txt = path_index_file[i_dataset]
    path_hr = path_dataset_hr[i_dataset]

    if path_txt in path_txt_collect:
        pbar_dataset.update(1)
        continue
    else:
        path_txt_collect.append(path_txt)

    path_txt = win2linux(path_txt)
    path_txt_clean = path_txt.split(".")[0] + "_clean.txt"
    path_hr = win2linux(path_hr)

    # load all the sample names
    file_names = read_txt(path_txt)
    num_patches = len(file_names)

    pbar = tqdm.tqdm(desc=i_dataset, total=num_patches, ncols=80, leave=False)
    if os.path.exists(path_txt_clean):
        file = open(path_txt_clean, "w")
    else:
        file = open(path_txt_clean, "x")

    for i_patch in range(num_patches):
        filename = file_names[i_patch]
        patch_hr = io.imread(os.path.join(path_hr, filename))
        if condition_type == "max":
            if patch_hr.max() >= threshold:
                file.write(filename + "\n")
        if condition_type == "mean":
            if patch_hr.mean() >= threshold:
                file.write(filename + "\n")
        pbar.update(1)
    pbar.close()
    file.close()
    pbar_dataset.update(1)
pbar_dataset.close()
