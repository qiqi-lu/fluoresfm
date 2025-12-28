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

from utils.data import win2linux, read_txt, normalization, interp_sf
from utils.plot import colorize, add_scale_bar
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import os, pandas
from scipy.io import loadmat
import seaborn as sns

# set font in svg
plt.rcParams["svg.fonttype"] = "none"

# ------------------------------------------------------------------------------
dataset_name = "biosr-mt-sr-2"
id_sample_show = 5

# ------------------------------------------------------------------------------
methods = (
    ("Raw", "raw"),
    ("FluoResFM", "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16"),
    ("GT", "gt"),
)

path_metadata_excel = "dataset_test-v2.xlsx"
path_prediction = os.path.join("results", "predictions", dataset_name)
path_figure = os.path.join(
    "results", "figures", "analysis", "analysis_mt", dataset_name
)
os.makedirs(path_figure, exist_ok=True)

# ------------------------------------------------------------------------------
# load metadata
# ------------------------------------------------------------------------------
df_info = pandas.read_excel(path_metadata_excel)
info = df_info[df_info["id"] == dataset_name].iloc[0]
path_lr = win2linux(info["path_lr"])
path_hr = win2linux(info["path_hr"])
path_txt = win2linux(info["path_index"])
pixel_size = float(info["target pixel size"].split("x")[0]) / 1000  # um

num_methods = len(methods)
methods_name = [method[0] for method in methods]
methods_id = [method[1] for method in methods]

filenames = read_txt(path_txt)[:8]
num_sample = len(filenames)
print("-" * 80)
print(f"[INFO] Number of samples: {len(filenames)}")
print(f"[INFO] Load image from {path_lr}")
print(f"[INFO] Load image from {path_hr}")
print(f"[INFO] Pixel size: {pixel_size} um")

# ------------------------------------------------------------------------------
# load images
# ------------------------------------------------------------------------------
normalizer = lambda image: normalization(image, p_low=0.03, p_high=0.995)
dict_clip = {"a_min": 0.0, "a_max": 2.5}

# ------------------------------------------------------------------------------
filename = filenames[id_sample_show]

imgs_meth = []
for i_meth in range(num_methods):
    method_name, method_path = methods[i_meth]
    if method_name == "GT":
        # load gt image
        img_gt = io.imread(os.path.join(path_hr, filename))
        img_gt = interp_sf(img_gt, sf=info["sf_hr"])[0]
        img_gt = normalizer(img_gt)
        img_gt = np.clip(img_gt, **dict_clip)
        imgs_meth.append(img_gt)
    elif method_name == "Raw":
        # load raw image
        img_raw = io.imread(os.path.join(path_lr, filename))
        img_raw = interp_sf(img_raw, sf=info["sf_lr"])[0]
        img_raw = normalizer(img_raw)
        img_raw = np.clip(img_raw, **dict_clip)
        imgs_meth.append(img_raw)
    else:
        # load results
        meth_id = methods[i_meth][1]
        img_meth = io.imread(os.path.join(path_prediction, meth_id, filename))[0]
        img_meth = normalizer(img_meth)
        img_meth = np.clip(img_meth, **dict_clip)
        imgs_meth.append(img_meth)


# ------------------------------------------------------------------------------
# load labels
# ------------------------------------------------------------------------------
filename_analysis = filenames[id_sample_show].split(".")[0] + "_analysis"
results_meth = []
for i_meth in range(num_methods):
    if methods_name[i_meth] == "Raw":
        path = path_lr + "_up2"
    elif methods_name[i_meth] == "GT":
        path = path_hr
    else:
        path = os.path.join(path_prediction, methods_id[i_meth])

    # load results
    results_single_meth = {}
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

    R = loadmat(os.path.join(path_analysis_data, "R.mat"))["R"].astype(np.float32)

    # load junctions
    NewCrPts = loadmat(os.path.join(path_analysis_data, "NewCrPts.mat"))["NewCrPts"]

    # load filament length distribution
    analysis_info = loadmat(os.path.join(path_analysis_data, "AnalysisInfo.mat"))[
        "AnalysisInfo"
    ]  # ['Orientation','Total Length','End-to-End Distance','Centroid X','Centroid Y']

    results_single_meth["filaments"] = all_sorted_filament
    results_single_meth["R"] = R
    results_single_meth["junctions"] = NewCrPts
    results_single_meth["analysis_info"] = analysis_info * np.array(
        [1, pixel_size, 1, 1, 1]
    )
    results_meth.append(results_single_meth)


# ------------------------------------------------------------------------------
# show images
# ------------------------------------------------------------------------------
print("-" * 80)
print(f"[INFO] Show images ...")
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
dict_rect = dict(facecolor="none", edgecolor="white", linewidth=1, linestyle="-")
dict_text_rt = dict(color="black", fontsize=14, ha="right", va="top")

# ------------------------------------------------------------------------------
nr, nc = 3, 4
fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(nc * 3, nr * 3), **dict_fig)

imgs_show = imgs_meth
results_show = results_meth

for i_meth in range(nr):
    img = imgs_show[i_meth]
    res = results_show[i_meth]
    ax = axes[i_meth]

    # show image ---------------------------------------------------------------
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
    # (num_pixel x coordinate x num_filaments)
    all_sorted_filament = res["filaments"]  # shape = (N, 2, num_filaments)
    R = res["R"][0, 0].astype(np.int32)

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
    # (num_junctions x coordinate)
    NewCrPts = res["junctions"]  # shape (N, 2)
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
    # (num_filaments x 5)
    # ['Orientation','Total Length','End-to-End Distance','Centroid X','Centroid Y']
    analysis_info = res["analysis_info"]
    df_analysis_info = pandas.DataFrame(
        analysis_info,
        columns=[
            "Orientation",
            "Total Length",
            "End-to-End Distance",
            "Centroid X",
            "Centroid Y",
        ],
    )
    sns.histplot(
        data=df_analysis_info,
        x="Total Length",
        binwidth=1,
        kde=True,
        ax=ax[3],
        color="#C23637",
        **dict_hist,
    )
    if i_meth == nr - 1:
        ax[3].set_xlabel("Filament length ($\mu$m)")
    # ax_len.set_ylabel("Frequency")
    ax[3].set_xlim([0, 16])
    ax[3].set_yticks([0, 100, 200, 300, 400])
    # ax[3].set_ylim([0, 150])
    ax[3].set_box_aspect(1)
    ax[3].set_xlabel(None)

    median_length = df_analysis_info["Total Length"].median()
    print(
        f"[INFO] median filament length ({methods_name[i_meth]}): {median_length:.2f} um"
    )
    # add text at the top right of the histogram
    ax[3].text(
        0.95, 0.95, f"{median_length:.2f} um", transform=ax[3].transAxes, **dict_text_rt
    )

# save figures -----------------------------------------------------------------
fig.savefig(
    os.path.join(path_figure, f"{filenames[id_sample_show].split('.')[0]}_image.png")
)
fig.savefig(
    os.path.join(path_figure, f"{filenames[id_sample_show].split('.')[0]}_image.svg")
)
