import pandas, os, tqdm

path_dataset_xlx = "dataset_train_transformer.xlsx"

datasets_frame = pandas.read_excel(path_dataset_xlx, sheet_name="64x64")
num_patches = list(datasets_frame["number of patches"])
num_datset = len(num_patches)
print(num_datset)

path_lr = list(datasets_frame["path_lr"])
path_hr = list(datasets_frame["path_hr"])
path_index = list(datasets_frame["path_index"])

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

# check datset size
pbar = tqdm.tqdm(total=num_datset, desc="check dataset size", ncols=100)
for i in range(num_datset):
    num_p = num_patches[i]
    with open(path_index[i]) as f:
        files = f.read().splitlines()
    if num_p != len(files):
        print(num_p, files[-1], path_index[i])
    pbar.update(1)
pbar.close()


# check exist of all images
pbar = tqdm.tqdm(total=num_datset, desc="check all image", ncols=100)
for i in range(num_datset):
    with open(path_index[i]) as f:
        files = f.read().splitlines()
    for file in files:
        if not os.path.exists(os.path.join(path_lr[i], file)):
            print(os.path.join(path_lr[i], file))
        if not os.path.exists(os.path.join(path_hr[i], file)):
            print(os.path.join(path_hr[i], file))
    pbar.update(1)
pbar.close()
