"""
read all the training patches.
del the pathces not satisfied the custom conditions.
regenerate the training .txt file with a suffix `_clean`.
"""

import skimage.io as io
import numpy as np
import pandas, tqdm, os
from utils.data import read_txt, win2linux

# threshold = 0.02
threshold = 0.06  # golgi
print("Threshold:", threshold)

specific_dataset = [
    # "vmsim5-mito-dcv",
    # "vmsim5-mito-sr",
    # "rcan3d-c2s-mt-dcv",
    # "rcan3d-c2s-mt-sr",
    # "rcan3d-c2s-npc-dcv",
    # "rcan3d-c2s-npc-sr",
    "rcan3d-dn-golgi-dn",
]
# specific_dataset = []

# load training dataset information
data_frame = pandas.read_excel("dataset_train_transformer-v2.xlsx", sheet_name="64x64")

# ------------------------------------------------------------------------------
if specific_dataset:
    data_frame = data_frame[data_frame["id"].isin(specific_dataset)]

path_dataset_hr = list(data_frame["path_hr"])
path_index_file = list(data_frame["path_index"])

num_dataset = len(list(data_frame["index"]))
print("Number of Dataset:", num_dataset)

path_txt_collect = []  # append after processing

pbar_dataset = tqdm.tqdm(desc="DATA_CLEAN", total=num_dataset, ncols=80, leave=True)
for i_dataset in range(num_dataset):
    path_txt = path_index_file[i_dataset]
    path_hr = path_dataset_hr[i_dataset]

    if path_txt in path_txt_collect:
        pbar_dataset.update(1)
        continue
    else:
        path_txt_collect.append(path_txt)

    if os.name == "posix":
        path_txt = win2linux(path_txt)
        path_hr = win2linux(path_hr)

    # load all the sample names
    file_names = read_txt(path_txt)
    num_patch = len(file_names)
    path_txt_clean = path_txt.split(".")[0] + "_clean.txt"

    pbar = tqdm.tqdm(desc="CLEAN", total=num_patch, ncols=80, leave=False)
    if os.path.exists(path_txt_clean):
        file = open(path_txt_clean, "w")
    else:
        file = open(path_txt_clean, "x")

    for i_patch in range(num_patch):
        fielname = file_names[i_patch]
        # read patch
        patch_hr = io.imread(os.path.join(path_hr, fielname))
        # if patch_hr.max() >= threshold:
        if patch_hr.mean() >= threshold:
            file.write(fielname + "\n")
        pbar.update(1)
    pbar.close()
    file.close()
    pbar_dataset.update(1)
pbar_dataset.close()
