"""
For test datasets.
Check the nromalized raw input to the models.
To confirm that whether the normalized raw input is resonable.
Plot the raw input, and saved into the results folder.
"""

import os, pandas, tqdm, math
import matplotlib.pyplot as plt
import utils.data as utils_data
from dataset_analysis import dataset_names_all

# dataset_group = "internal_dataset"
dataset_group = "external_dataset"

# ------------------------------------------------------------------------------
dataset_names = []
keys = dataset_names_all[dataset_group]
for key in keys:
    dataset_names.extend(dataset_names_all[dataset_group][key])

path_dataset_test = "dataset_test-v2.xlsx"
num_sample_max = 8
path_save_to = os.path.join(
    "results", "figures", "datasets", "test_input", dataset_group
)
os.makedirs(path_save_to, exist_ok=True)

normalizer = utils_data.NormalizePercentile(p_low=0.03, p_high=0.995)

# ------------------------------------------------------------------------------
data_frame = pandas.read_excel(path_dataset_test)
num_dataset = len(dataset_names)
print("-" * 80)
print("Number of datasets:", num_dataset)

dict_fig = {"dpi": 300, "constrained_layout": True}

pbar = tqdm.tqdm(total=num_dataset, desc="Display input (test)", ncols=80)
for id_dataset in dataset_names:
    # get the information of current dataset
    ds = data_frame[data_frame["id"] == id_dataset].iloc[0]

    # read filenames of test images
    sample_filenames = utils_data.read_txt(ds["path_index"])

    # set the number of samples used for test
    if num_sample_max is not None:
        if num_sample_max > len(sample_filenames):
            num_sample_analysis = len(sample_filenames)
        else:
            num_sample_analysis = num_sample_max
    else:
        num_sample_analysis = len(sample_filenames)

    # --------------------------------------------------------------------------
    nr, nc = int(math.ceil(num_sample_analysis / 4)), 4
    fig, axes = plt.subplots(ncols=nc, nrows=nr, figsize=(nc * 3, nr * 3), **dict_fig)
    axes = axes.flatten()
    for ax in axes:
        ax.set_axis_off()

    for i_sample in range(num_sample_analysis):
        sample_name = sample_filenames[i_sample]
        # get raw image and apply normalization
        img_raw = utils_data.read_image(os.path.join(ds["path_lr"], sample_name))
        img_raw = utils_data.utils_data.interp_sf(img_raw, sf=ds["sf_lr"])[0]
        img_raw = normalizer(img_raw)

        axes[i_sample].imshow(img_raw, cmap="hot", vmin=0.0, vmax=1.5)
        axes[i_sample].set_title(sample_name)

    plt.savefig(os.path.join(path_save_to, ds["id"] + ".png"))
    plt.close()
    del fig, axes
    pbar.update(1)
pbar.close()
