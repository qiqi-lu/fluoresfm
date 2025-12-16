"""
Count the number of pathces in each dataset.
- before clean
- after clean
"""

import pandas, os, tqdm
import numpy as np
from utils.data import read_txt, win2linux

# load dataset information from excel file
df_dataset = pandas.read_excel("dataset_train_transformer-v2.xlsx", sheet_name="64x64")
path_save_to = os.path.join("results", "statistic")
assert os.path.exists(path_save_to), "Path does not exist: {}".format(path_save_to)

# number of datasets
num_dataset = len(list(df_dataset["id"]))

print("-" * 80)
print("Number of Dataset:", num_dataset)

# ------------------------------------------------------------------------------
# get the path of index file
path_index_file = list(df_dataset["path_index"])
# get the number of patches
num_patches_before, num_patches_after = [0] * num_dataset, [0] * num_dataset

pbar = tqdm.tqdm(desc="Patch Counting", total=num_dataset, ncols=80)
for i_dataset in range(num_dataset):
    # before clean
    path_txt = win2linux(path_index_file[i_dataset])
    # load all the sample names
    file_names = read_txt(path_txt)

    # if the last is \n, remove it
    if file_names[-1] == "\n":
        file_names.pop(-1)
    num_patches_before[i_dataset] = len(file_names)

    # after clean
    path_txt_clean = win2linux(path_txt.split(".")[0] + "_clean.txt")
    # load all the sample names
    file_names = read_txt(path_txt_clean)

    # if the last is \n, remove it
    if file_names[-1] == "\n":
        file_names.pop(-1)
    num_patches_after[i_dataset] = len(file_names)
    pbar.update(1)
pbar.close()

# print the number of patches
print("Number of patches:", np.sum(num_patches_before))
print("Number of patches after clean:", np.sum(num_patches_after))

# dataframe
df = pandas.DataFrame(
    {
        "index": list(df_dataset["index"]),
        "id": list(df_dataset["id"]),
        "task": list(df_dataset["task"]),
        "structure": list(df_dataset["structure"]),
        "number of patches": num_patches_before,
        "number of patches-clean": num_patches_after,
    }
)
print("-" * 80)

# save to excel
df.to_excel(os.path.join(path_save_to, "patch_count.xlsx"), index=False)
