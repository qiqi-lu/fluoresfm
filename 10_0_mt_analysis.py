"""
MT structure analysis for a single image.
- filament segmentation
- filament length distribution
- junctions detection

Output a image:
--------------------------------------------------------------------------------
        | image   | segmentation   | junctions   | filament length distribution
--------------------------------------------------------------------------------
LR      |
HR      |
Pred    |
--------------------------------------------------------------------------------
"""

from utils.data import win2linux, read_txt, normalization
from utils.plot import plot_and_save_2d_image
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import os, json, pandas
from scipy.io import loadmat

# set font in svg
plt.rcParams["svg.fonttype"] = "none"

method_id = "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16"
method_id = "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16"
dataset_name = "biosr-mt-sr-9"
id_img = 6

normalizer = lambda image: normalization(image, p_low=0.03, p_high=0.995)
dict_clip = {"a_min": 0.0, "a_max": 2.5}

# ------------------------------------------------------------------------------
# load data info
df_info = pandas.read_excel("dataset_test-v2.xlsx")
info = df_info[df_info["id"] == dataset_name].iloc[0]

path_lr = win2linux(info["path_lr"]) + "_up2"
path_hr = win2linux(info["path_hr"])
path_txt = win2linux(info["path_index"])
pixel_size = float(info["target pixel size"].split("x")[0]) / 1000  # um
filenames = read_txt(path_txt)

path_predict = os.path.join("results", "predictions", dataset_name, method_id)

path_figure = os.path.join("results", "figures", "images", dataset_name)
os.makedirs(path_figure, exist_ok=True)


print(f"[INFO] Load image from {path_lr}")
print(f"[INFO] Load image from {path_hr}")
print(f"[INFO] Pixel size: {pixel_size} um")

# ------------------------------------------------------------------------------
# load images
filename = filenames[id_img]
filename_analysis = os.path.splitext(filename)[0] + "_analysis"

paths = [path_lr, path_hr, path_predict]

results = []
for path in paths:
    print(f"[INFO] Load image from {path}")
    results_single_meth = []
    # load image
    img = io.imread(os.path.join(path, filename))[0]
    img = np.clip(normalizer(img), **dict_clip)
    results_single_meth.append(img)

    # load analysis results ----------------------------------------------------
    path_analysis_data = os.path.join(path, filename_analysis, "data")
    path_analysis_result = os.path.join(path, filename_analysis, "result")

    assert os.path.exists(
        path_analysis_data
    ), f"[ERROR] {path_analysis_data} does not exist"
    assert os.path.exists(
        path_analysis_result
    ), f"[ERROR] {path_analysis_result} does not exist"

    # load filaments
    all_sorted_filament = loadmat(
        os.path.join(path_analysis_data, "all_sorted_filament.mat")
    )["all_sorted_filament"]
    results_single_meth.append(all_sorted_filament)

    # load junctions
    NewCrPts = loadmat(os.path.join(path_analysis_data, "NewCrPts.mat"))["NewCrPts"]
    results_single_meth.append(NewCrPts)

    # load filament length distribution
    filament_info = 1

    results.append(results_single_meth)


# ------------------------------------------------------------------------------
# show images
nr, nc = 3, 4
dict_fig = dict(dpi=300, constrained_layout=True)
fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(nc * 3, nr * 3), **dict_fig)

for i_meth in range(nr):
    res = results[i_meth]
    ax = axes[i_meth]
    # show image
    ax[0].imshow(res[0], cmap="gray")

    # show filaments


fig.savefig(os.path.join(path_figure, f"{filename_analysis}.png"))
fig.savefig(os.path.join(path_figure, f"{filename_analysis}.svg"))
