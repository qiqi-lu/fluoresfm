import numpy as np
import os, pandas
import skimage.io as io
import matplotlib.pyplot as plt
import utils.data import win2linux, read_txt, normalization

id_datasets = [
    "biosr-cpp-sr-1",
    "biosr-er-sr-1",
    "biosr-mt-sr-1",
    "biosr-actin-sr-1",
    # "biosr-cpp-dcv-1",
    # "biosr-er-dcv-1",
    # "biosr-mt-dcv-1",
    # "biosr-actin-dcv-1",
    # "biosr-cpp-dn-1",
    # "biosr-er-dn-1",
    # "biosr-mt-dn-1",
    # "biosr-actin-dn-1",
    # "care-liver-iso",
]

id_sample = [0, 0, 0, 0]
data_frame = pandas.read_excel("dataset_test.xlsx")
num_datasets = len(id_datasets)

methods = [
    # ("CARE:sr", "care_sr"),
    # ("DFCAN:sr-2", "dfcan_sr_2"),
    ("UniFMIR:all", "unifmir_all"),
    ("UNet-uc:all", "unet_sd_c_all_cross"),
    ("UNet-c:all", "unet_sd_c_all"),
]

nr, nc = num_datasets, len(methods) + 1
fig, axes = plt.subplots(
    nrows=nr, ncols=nc, dpi=300, figsize=(nc * 3, nr * 3), constrained_layout=True
)
[ax.set_axis_off() for ax in axes.ravel()]

for i_dataset in range(num_datasets):
    ds = data_frame[data_frame["id"] == id_datasets[i_dataset]].loc[0]
    path_txt, path_lr, path_hr = (
        win2linux(ds["path_index"]),
        win2linux(ds["path_lr"]),
        win2linux(ds["path_hr"]),
    )
    path_results = os.path.join("outputs", "unet_c", id_datasets[i_dataset])
    imgs = []
    for meth in methods:
        img_raw = io.imread(
            os.path.join(
                path_lr,
            )
        )
