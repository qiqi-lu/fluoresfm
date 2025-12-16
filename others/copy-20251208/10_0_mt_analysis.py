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
from utils.plot import colorize, add_scale_bar
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import os, pandas
from scipy.io import loadmat

# set font in svg
plt.rcParams["svg.fonttype"] = "none"
# ------------------------------------------------------------------------------
method_id = "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16"
dataset_name = "biosr-mt-sr-9"
#
img_info = (6, (387, 480, 150), (810, 206, 150))
id_img, box_1, box_2 = img_info

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
filename = filenames[id_img]

path_predict = os.path.join("results", "predictions", dataset_name, method_id)
path_figure = os.path.join(
    "results", "figures", "images", dataset_name, filename.split(".")[0]
)
os.makedirs(path_figure, exist_ok=True)


print(f"[INFO] Load image from {path_lr}")
print(f"[INFO] Load image from {path_hr}")
print(f"[INFO] Pixel size: {pixel_size} um")

# ------------------------------------------------------------------------------
# load images
filename_analysis = os.path.splitext(filename)[0] + "_analysis"

paths = [path_lr, path_hr, path_predict]
methods_name = ["Raw", "GT", "Restored"]

results = []
for path in paths:
    print(f"[INFO] Load image from {path}")
    results_single_meth = []
    # load image ---------------------------------------------------------------
    img = io.imread(os.path.join(path, filename))[0]
    img = np.clip(normalizer(img), **dict_clip)
    results_single_meth.append(img)

    # load results -------------------------------------------------------------
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
    )["all_sorted_filament"].astype(np.float32)
    results_single_meth.append(all_sorted_filament)

    R = loadmat(os.path.join(path_analysis_data, "R.mat"))["R"].astype(np.float32)
    results_single_meth.append(R)

    # load junctions
    NewCrPts = loadmat(os.path.join(path_analysis_data, "NewCrPts.mat"))["NewCrPts"]
    results_single_meth.append(NewCrPts)

    # load filament length distribution
    analysis_info = loadmat(os.path.join(path_analysis_data, "AnalysisInfo.mat"))[
        "AnalysisInfo"
    ]  # ['Orientation','Total Length','End-to-End Distance','Centroid X','Centroid Y']
    results_single_meth.append(analysis_info)
    results.append(results_single_meth)

# calculate the maximum length of the filaments
max_length = 0
min_length = 0
for res in results:
    analysis_info = res[4]
    max_length = max(max_length, analysis_info[:, 1].max())
    min_length = min(min_length, analysis_info[:, 1].min())


# ------------------------------------------------------------------------------
# show images
nr, nc = 3, 3
dict_fig = dict(dpi=600, constrained_layout=True)
num_colors = 32
ColorList = np.random.rand(num_colors, 3)
dict_filament = dict(linewidth=0.75)
dict_overlap = dict(linewidth=0.75, color="black")
dict_junction = dict(
    linestyle="",
    marker=".",
    markersize=1.0,
    markeredgecolor="#A6FF00",
    markerfacecolor="#A6FF00",
)
dict_hist = dict(facecolor="none", edgecolor="black", linewidth=1)
dict_fit_line = dict(color="#C23637", linewidth=1)
dict_rect = dict(facecolor="none", edgecolor="white", linewidth=1, linestyle="-")


fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(nc * 3, nr * 3), **dict_fig)
fig_len, axes_len = plt.subplots(nrows=nr, ncols=1, figsize=(3, nr * 3), **dict_fig)

for i_meth in range(nr):
    res = results[i_meth]
    ax = axes[i_meth]
    ax_len = axes_len[i_meth]

    # show image ---------------------------------------------------------------
    img = res[0]
    img_color = colorize(img, vmin=0, vmax=0.9, color=(0, 255, 0))
    ax[0].imshow(img_color, cmap="gray")
    ax[0].set_axis_off()
    ax[0].text(
        0.05,
        0.95,
        methods_name[i_meth],
        ha="left",
        va="top",
        transform=ax[0].transAxes,
        color="white",
        fontsize=14,
    )

    # show scale bar -----------------------------------------------------------
    if i_meth == 0:
        img_shape = img.shape
        tp = 0.05
        dict_scale_bar = {
            "pixel_size": pixel_size,
            "bar_length": 5,  # um
            "bar_height": 0.01,
            "bar_color": "white",
            "pos": (int(img_shape[1] * tp), int(img_shape[0] * (1 - tp))),
        }
        add_scale_bar(ax[0], image=img, **dict_scale_bar)

    # show filaments -----------------------------------------------------------
    all_sorted_filament = res[1]
    R = res[2][0, 0].astype(np.int32)

    num_filaments = all_sorted_filament.shape[2]
    # plot each filament
    for i_filament in range(num_filaments):
        x = all_sorted_filament[:, 0, i_filament]
        y = all_sorted_filament[:, 1, i_filament]
        x = x[x != 0]
        y = y[y != 0]
        colr = tuple(ColorList[i_filament % num_colors])
        ax[1].plot(y - R, x - R, color=colr, **dict_filament)
        ax[2].plot(y - R, x - R, **dict_overlap)
    ax[1].invert_yaxis()
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_xlim([0, img.shape[1]])
    ax[1].set_ylim([img.shape[0], 0])
    ax[1].set_facecolor("black")
    ax[1].set_box_aspect(1)

    # show junctions -----------------------------------------------------------
    NewCrPts = res[3]
    # get all the cooridinates of points == 1 in overlap_map

    ax[2].plot(NewCrPts[:, 1] - R, NewCrPts[:, 0] - R, **dict_junction)
    ax[2].invert_yaxis()
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    ax[2].set_xlim([0, img.shape[1]])
    ax[2].set_ylim([img.shape[0], 0])
    ax[2].set_facecolor("#C23637")
    ax[2].set_box_aspect(1)

    # add titles ---------------------------------------------------------------
    if i_meth == 0:
        for i_txt, txt in enumerate(["Image", "Filaments", "Junctions"]):
            ax[i_txt].text(
                0.95,
                0.95,
                txt,
                ha="right",
                va="top",
                transform=ax[i_txt].transAxes,
                color="white",
                fontsize=14,
            )

    # show filament length distribution ----------------------------------------
    analysis_info = res[4]
    # get the total length of each filament
    total_length = analysis_info[:, 1]

    # plot
    xlim_h = (max_length // 100 + 1) * 100
    freq, bins, _ = ax_len.hist(
        total_length, bins=25, range=(0, xlim_h), cumulative=True, **dict_hist
    )
    if i_meth == nr - 1:
        ax_len.set_xlabel("Filament length (nm)")
    # ax_len.set_ylabel("Frequency")
    ax_len.set_xlim([0, xlim_h])
    ax_len.set_yticks([0, 100, 200, 300, 400])
    ax_len.set_ylim([0, 400])
    ax_len.set_box_aspect(1)

    # add a fitted polynomial curve of the histogram
    x = bins[:-1] + np.diff(bins) / 2
    y = freq
    fit = np.polyfit(x, y, 6)
    fitted_curve = np.polyval(fit, x)
    ax_len.plot(x, fitted_curve, **dict_fit_line)

# add boxes --------------------------------------------------------------------
for ax in axes.ravel():
    for box in [box_1, box_2]:
        ax.add_patch(
            plt.Rectangle((box[0], box[1]), box[2], box[2], fill=False, **dict_rect)
        )

# save figures -----------------------------------------------------------------
fig.savefig(os.path.join(path_figure, f"{filename_analysis}.png"))
fig.savefig(os.path.join(path_figure, f"{filename_analysis}.svg"))
fig_len.savefig(os.path.join(path_figure, f"{filename_analysis}_len.png"))
fig_len.savefig(os.path.join(path_figure, f"{filename_analysis}_len.svg"))
