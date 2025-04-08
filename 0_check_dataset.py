"""
Check dataset: check dataset exist, check dataset size, check all images exist.
check the modified data of images.
"""

import pandas, os, tqdm, time
from utils.data import win2linux

path_dataset_xlx = "dataset_train_transformer.xlsx"

# ------------------------------------------------------------------------------
# load the dataset information from excel file
datasets_frame = pandas.read_excel(path_dataset_xlx, sheet_name="64x64")
num_patches = list(datasets_frame["number of patches"])
num_datset = len(num_patches)
print(num_datset)

path_lr = list(datasets_frame["path_lr"])
path_hr = list(datasets_frame["path_hr"])
path_index = list(datasets_frame["path_index"])

# convert windows path to linux path
for i in range(num_datset):
    path_lr[i] = win2linux(path_lr[i])
    path_hr[i] = win2linux(path_hr[i])
    path_index[i] = win2linux(path_index[i])

# ------------------------------------------------------------------------------
# check path exist
pbar = tqdm.tqdm(total=num_datset, desc="check exist", ncols=100)
for i in range(num_datset):
    if not os.path.exists(path_lr[i]):
        print(i, path_lr[i])
    if not os.path.exists(path_hr[i]):
        print(i, path_hr[i])
    if not os.path.exists(path_index[i]):
        print(i, path_index[i])
    pbar.update(1)
pbar.close()

# ------------------------------------------------------------------------------
# check dataset size
pbar = tqdm.tqdm(total=num_datset, desc="check dataset size", ncols=100)
for i in range(num_datset):
    num_p = num_patches[i]
    with open(path_index[i]) as f:
        files = f.read().splitlines()
    if num_p != len(files):
        print(num_p, files[-1], path_index[i])
    pbar.update(1)
pbar.close()

# ------------------------------------------------------------------------------
# check the modified date of image
for i in range(num_datset):
    with open(path_index[i]) as f:
        files = f.read().splitlines()

    # low resolution image
    # print the modified time of first image in a year/month/day format.
    modification_time = os.path.getmtime(win2linux(os.path.join(path_lr[i], files[0])))
    # Convert the modification time to a struct_time object
    time_struct = time.localtime(modification_time)
    # Format the time as year/month/day
    formatted_time_lr = time.strftime("%Y/%m/%d", time_struct)

    # high resolution image, same as low-resolution
    modification_time = os.path.getmtime(win2linux(os.path.join(path_hr[i], files[0])))
    time_struct = time.localtime(modification_time)
    formatted_time_hr = time.strftime("%Y/%m/%d", time_struct)

    if formatted_time_lr not in ["2025/03/28", "2025/03/29"]:
        print(path_lr[i], formatted_time_lr)
    if formatted_time_hr not in ["2025/03/28", "2025/03/29"]:
        print(path_hr[i], formatted_time_hr)


# ------------------------------------------------------------------------------
# check exist of all images
# pbar = tqdm.tqdm(total=num_datset, desc="check all image", ncols=100)
# for i in range(num_datset):
#     with open(path_index[i]) as f:
#         files = f.read().splitlines()
#     files_in_folder_lr = os.listdir(path_lr[i])
#     files_in_folder_hr = os.listdir(path_hr[i])
#     for file in files:
#         # if not os.path.exists(os.path.join(path_lr[i], file)):
#         if file not in files_in_folder_lr:
#             print(os.path.join(path_lr[i], file))
#         # if not os.path.exists(os.path.join(path_hr[i], file)):
#         if file not in files_in_folder_hr:
#             print(os.path.join(path_hr[i], file))
#     pbar.update(1)
# pbar.close()
