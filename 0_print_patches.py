import matplotlib.pyplot as plt
import os, pandas, tqdm, torch
import skimage.io as io
import numpy as np


def interp_sf(x, sf):
    x = torch.tensor(x)
    x = torch.unsqueeze(x, dim=0)
    if sf > 0:
        x_inter = torch.nn.functional.interpolate(x, scale_factor=sf, mode="nearest")
    if sf < 0:
        x_inter = torch.nn.functional.avg_pool2d(x, kernel_size=-sf, stride=-sf)
    return x_inter[0].numpy()


path_dataset_xlx = "dataset_train_transformer.xlsx"
datasets_frame = pandas.read_excel(path_dataset_xlx, sheet_name="64x64")

path_dataset_lr = list(datasets_frame["path_lr"])
path_dataset_hr = list(datasets_frame["path_hr"])
path_dataset_index = list(datasets_frame["path_index"])
sf_lr = list(datasets_frame["sf_lr"])
sf_hr = list(datasets_frame["sf_hr"])
tasks = list(datasets_frame["task"])

num_datsets = len(path_dataset_lr)
print("num of datasets:", num_datsets)

save_path = "outputs\\figures\\datasets"
pbar = tqdm.tqdm(desc="show patches", total=num_datsets, ncols=100)

# show_ids = range(num_datsets)
show_ids = range(2250, num_datsets)

fig, axes = plt.subplots(
    nrows=1, ncols=2, figsize=(3, 1.5), dpi=300, constrained_layout=True
)

for i in show_ids:
    with open(path_dataset_index[i]) as f:
        files = f.read().splitlines()
    for file in files:
        img_hr = io.imread(os.path.join(path_dataset_hr[i], file))
        if np.mean(img_hr) > 0.05:
            img_lr = io.imread(os.path.join(path_dataset_lr[i], file))

            img_hr = interp_sf(img_hr, sf_hr[i])
            img_lr = interp_sf(img_lr, sf_lr[i])

            axes[0].cla()
            axes[1].cla()
            axes[0].set_axis_off()
            axes[1].set_axis_off()
            axes[0].imshow(img_lr[0], cmap="hot")
            axes[1].imshow(img_hr[0], cmap="hot")
            plt.savefig(os.path.join(save_path, str(i) + "_" + tasks[i] + ".png"))
            break
    pbar.update(1)
pbar.close()
