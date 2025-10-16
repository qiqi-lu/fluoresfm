"""
File check.
- Check the existence of all the path in the excel file.
- Check the existence of all images in the training datasets.
"""

import pandas, os, tqdm
from utils.data import win2linux

path_dataset_excel = "dataset_train_transformer-v2.xlsx"
enable_image_check = False

# ------------------------------------------------------------------------------
# load the dataset information from excel file
datasets_frame = pandas.read_excel(path_dataset_excel, sheet_name="64x64")
num_patches = list(datasets_frame["number of patches"])
num_patches_clean = list(datasets_frame["number of patches-clean"])
num_dataset = len(num_patches)
print("-" * 80)
print("Number of datasets:", num_dataset)

path_lr = list(datasets_frame["path_lr"])
path_hr = list(datasets_frame["path_hr"])
path_index = list(datasets_frame["path_index"])

# convert windows path to linux path
for i in range(num_dataset):
    path_lr[i] = win2linux(path_lr[i])
    path_hr[i] = win2linux(path_hr[i])
    path_index[i] = win2linux(path_index[i])

# ------------------------------------------------------------------------------
#                              check path exist
# ------------------------------------------------------------------------------
print("-" * 80)
print("Check path existence")
pbar = tqdm.tqdm(total=num_dataset, desc="Check path existence", ncols=80)
for i in range(num_dataset):
    if not os.path.exists(path_lr[i]):
        print("[WARNNING] Path inexist: ", i, path_lr[i])
    if not os.path.exists(path_hr[i]):
        print("[WARNNING] Path inexist: ", i, path_hr[i])
    if not os.path.exists(path_index[i]):
        print("[WARNNING] Path inexist: ", i, path_index[i])
    pbar.update(1)
pbar.close()

# ------------------------------------------------------------------------------
#                         check dataset size consistent
# ------------------------------------------------------------------------------
# the number labeled in the excel file should be the same as
# the number of files in the index file
print("-" * 80)
print("Check dataset size")
pbar = tqdm.tqdm(total=num_dataset, desc="Check dataset size", ncols=80)
for i in range(num_dataset):
    # check the number of patches before cleaning
    num_p = num_patches[i]
    path_txt = path_index[i]
    with open(path_txt) as f:
        files = f.read().splitlines()
        #  del the \n in the end of the file
        if files[-1] == "":
            files = files[:-1]
    if num_p != len(files):
        print("[WARNNING] Number inconsistent", num_p, path_txt)

    # check the number of patches after cleaning
    num_p = num_patches_clean[i]
    path_txt = path_index[i].replace(".txt", "_clean.txt")
    with open(path_txt) as f:
        files = f.read().splitlines()
        #  del the \n in the end of the file
        if files[-1] == "":
            files = files[:-1]
    if num_p != len(files):
        print("[WARNNING] Number inconsistent", num_p, path_txt)

    pbar.update(1)
pbar.close()

# ------------------------------------------------------------------------------
#                               check image exist
# ------------------------------------------------------------------------------
if enable_image_check:
    print("-" * 80)
    print("Check image exist")
    pbar = tqdm.tqdm(total=num_dataset, desc="Check all image", ncols=80)
    for i in range(num_dataset):
        with open(path_index[i]) as f:
            files = f.read().splitlines()
        files_in_folder_lr = os.listdir(path_lr[i])
        files_in_folder_hr = os.listdir(path_hr[i])
        for file in files:
            # if not os.path.exists(os.path.join(path_lr[i], file)):
            if file not in files_in_folder_lr:
                print(os.path.join(path_lr[i], file))
            # if not os.path.exists(os.path.join(path_hr[i], file)):
            if file not in files_in_folder_hr:
                print(os.path.join(path_hr[i], file))
        pbar.update(1)
    pbar.close()
