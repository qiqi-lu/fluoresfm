import numpy as np
import pandas, os
import matplotlib.pyplot as plt
import utils.data as utils_data
import utils.evaluation as eva

# set which to display
save_file_name = "sr-w2s"
samples = (
    # ("biosr-cpp-sr-1", 1),
    # ("biosr-cpp-sr-9", 2),
    # ("biosr-mt-sr-1", 1),
    # ("biosr-mt-sr-9", 2),
    # ("biosr-er-sr-2", 1),
    # ("biosr-er-sr-6", 2),
    # ("biosr-actin-sr-1", 1),
    # ("biosr-actin-sr-12", 2),
    # --------------------------------------------------------------------------
    # ("biosr-cpp-dcv-1", 0),
    # ("biosr-cpp-dcv-9", 3),
    # ("biosr-mt-dcv-1", 0),
    # ("biosr-mt-dcv-9", 3),
    # ("biosr-er-dcv-2", 0),
    # ("biosr-er-dcv-6", 3),
    # ("biosr-actin-dcv-1", 0),
    # ("biosr-actin-dcv-12", 3),
    # ("biosr-actinnl-dcv-1", 0),
    # ("biosr-actinnl-dcv-9", 3),
    # --------------------------------------------------------------------------
    # ("biosr-cpp-dn-1", 4),
    # ("biosr-cpp-dn-4", 5),
    # ("biosr-mt-dn-1", 4),
    # ("biosr-mt-dn-4", 5),
    # ("biosr-er-dn-2", 4),
    # ("biosr-er-dn-4", 5),
    # ("biosr-actin-dn-1", 4),
    # ("biosr-actin-dn-6", 5),
    # ("biosr-actinnl-dn-1", 4),
    # ("biosr-actinnl-dn-4", 5),
    # --------------------------------------------------------------------------
    # ("biosr+-mt-dn-1", 0),
    # ("biosr+-mt-dn-4", 1),
    # ("biosr+-er-dn-2", 0),
    # ("biosr+-er-dn-4", 1),
    # ("biosr+-actin-dn-1", 0),
    # ("biosr+-actin-dn-6", 1),
    # ("biosr+-myosin-dn-1", 0),
    # ("biosr+-myosin-dn-4", 1),
    # ("biosr+-ccp-dn-1", 0),
    # ("biosr+-ccp-dn-4", 1),
    # --------------------------------------------------------------------------
    # ("w2s-c0-dcv-1", 0),
    # ("w2s-c0-dcv-7", 1),
    # ("w2s-c1-dcv-1", 0),
    # ("w2s-c1-dcv-7", 1),
    # ("w2s-c2-dcv-1", 0),
    # ("w2s-c2-dcv-7", 1),
    # --------------------------------------------------------------------------
    # ("w2s-c0-dn-1", 2),
    # ("w2s-c0-dn-6", 3),
    # ("w2s-c1-dn-1", 2),
    # ("w2s-c1-dn-6", 3),
    # ("w2s-c2-dn-1", 2),
    # ("w2s-c2-dn-6", 3),
    # --------------------------------------------------------------------------
    ("w2s-c0-sr-1", 4),
    ("w2s-c0-sr-7", 5),
    ("w2s-c1-sr-1", 4),
    ("w2s-c1-sr-7", 5),
    ("w2s-c2-sr-1", 4),
    ("w2s-c2-sr-7", 5),
)
# ------------------------------------------------------------------------------
methods = ("unet_sd_c_all",)
path_figures = os.path.join("outputs", "figures", "imgtext")
path_results = os.path.join("outputs", "unet_c")
p_low = 0.0
p_high = 0.9999
with_gt = True
num_samples = len(samples)
num_methods = len(methods)
path_datsets_test = "dataset_test.xlsx"
normalizer = utils_data.NormalizePercentile(p_low=p_low, p_high=p_high)

data_frame = pandas.read_excel(path_datsets_test)

# ------------------------------------------------------------------------------
nr, nc = num_methods + 2, num_samples
if not with_gt:
    nr -= 1

fig, axes = plt.subplots(
    nrows=nr, ncols=nc, figsize=(2 * nc, 2 * nr), dpi=300, constrained_layout=True
)
[ax.set_axis_off() for ax in axes.ravel()]

for i_sample, sample in enumerate(samples):
    ds = data_frame[data_frame["id"] == sample[0]]
    filenames = utils_data.read_txt(ds["path_index"].iloc[0])
    filename = filenames[sample[1]]

    dict_show = {"cmap": "hot"}
    # load data
    img_raw = utils_data.read_image(os.path.join(ds["path_lr"].iloc[0], filename))
    img_raw = utils_data.utils_data.interp_sf(img_raw, sf=ds["sf_lr"].iloc[0])
    img_raw = normalizer(img_raw)[0]

    if with_gt:
        img_gt = utils_data.read_image(os.path.join(ds["path_hr"].iloc[0], filename))
        img_gt = utils_data.interp_sf(img_gt, sf=ds["sf_hr"].iloc[0])
        img_gt = normalizer(img_gt)[0]
        img_raw = eva.linear_transform(img_true=img_gt, img_test=img_raw)

    for i_meth, meth in enumerate(methods):
        img_meth = utils_data.read_image(
            os.path.join(path_results, sample[0], meth, filename)
        )
        if with_gt:
            img_meth = eva.linear_transform(img_true=img_gt, img_test=img_meth)[0]
        axes[1 + i_meth, i_sample].imshow(img_meth, **dict_show)

    axes[0, i_sample].imshow(img_raw, **dict_show)
    axes[0, i_sample].text(15, 65, sample[0], color="white")
    if with_gt:
        axes[-1, i_sample].imshow(img_gt, **dict_show)

plt.savefig(os.path.join(path_figures, f"comparision-{save_file_name}.png"))
