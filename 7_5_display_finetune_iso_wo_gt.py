"""
Display the results of FluoResFM on isotropic reconstruction data without ground truth.
"""

import pandas, os, tqdm, seaborn
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

from utils.data import normalization, win2linux, read_txt, interp_sf, iso_xy
from utils.plot import colorize, add_scale_bar

# GLOBAL SETTINGS --------------------------------------------------------------
plt.rcParams["svg.fonttype"] = "none"
GREEN, BlUE, RED, YELLOW = (0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)
fig_direction = "vertical"  # [methods x 1]
# fig_direction ='horizontal' # [1 x methods]

# datsets and methods to show --------------------------------------------------
#               dataset name, id sample, color
dataset_show = ("care-drosophila-iso", 3, GREEN)

methods_show = (
    # (
    #     "FluoResFM (dn)",
    #     "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-ft-inout-care-denoising-flywing-1",
    #     "#B271AB",
    # ),
    (
        "FluoResFM",
        "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-ft-inout-care-iso-drosophila-3d",
        "#6E2769",
    ),
)

dataset_id, id_sample, dataset_color = dataset_show
methods_colors = ["#8E99AB"] + [m[2] for m in methods_show]
methods_name = ["Raw"] + [m[0] for m in methods_show]
num_methods_show = len(methods_show)
num_sample_show = 8

# dataset infomation -----------------------------------------------------------
# load dataset info
path_info_excel = "dataset_test-v2.xlsx"
path_save_fig = os.path.join("results", "figures", "images", dataset_id)
os.makedirs(path_save_fig, exist_ok=True)
path_prediction = os.path.join("results", "predictions", dataset_id)

df_info = pandas.read_excel(path_info_excel)
info = df_info.loc[df_info["id"] == dataset_id].iloc[0]

path_lr = win2linux(info["path_lr"])
path_hr = win2linux(info["path_hr"])
path_index = win2linux(info["path_index"])

pixel_size_xy = float(info["target pixel size"].split("x")[0]) / 1000.0

sf_hr = int(info["sf_hr"])
sf_lr = int(info["sf_lr"])

structure_type = info["structure"]

# print dataset info
print("-" * 80)
print(f"[INFO] Dataset:   {dataset_id}")
print(f"[INFO] Path text: {path_index}")
print(f"[INFO] Path LR:   {path_lr}")
print(f"[INFO] Path HR:   {path_hr}")
print(f"[INFO] Pixel size (xy): {pixel_size_xy} x {pixel_size_xy} um")

# preprocessing settings -------------------------------------------------------
normalizer = lambda image: normalization(image, p_low=0.03, p_high=0.995)
dict_clip = {"a_min": 0.0, "a_max": 2.5}

# ------------------------------------------------------------------------------
# load image
# ------------------------------------------------------------------------------
# load all the image in the datasets
filenames = read_txt(path_index)
print("-" * 80)
print(f"[INFO] Number of samples: {len(filenames)}")
num_samples = min(len(filenames), num_sample_show)

imgs = []
for i_sample in range(num_samples):
    filename = filenames[i_sample]

    imgs_one = []
    # load the raw image
    img_raw = io.imread(os.path.join(path_lr, filename)).astype(np.float32)
    img_raw = interp_sf(img_raw, sf=sf_lr)
    img_raw = normalizer(img_raw)
    img_raw = np.clip(img_raw, **dict_clip)
    img_raw = np.transpose(img_raw, (0, 2, 1))
    imgs_one.append(img_raw[0])

    # load the prediction image
    for i_method in range(num_methods_show):
        method_title, method_id, method_color = methods_show[i_method]
        img_pred = io.imread(os.path.join(path_prediction, method_id, filename))
        img_pred = normalizer(img_pred)
        img_pred = np.clip(img_pred, **dict_clip)
        img_pred = np.transpose(img_pred, (0, 2, 1))
        imgs_one.append(img_pred[0])

    imgs.append(imgs_one)

# ------------------------------------------------------------------------------
# show image
# ------------------------------------------------------------------------------
dict_fig = {"dpi": 300, "constrained_layout": True}
dict_colorize = {"vmin": 0.0, "vmax": 0.9, "color": dataset_color}
dict_text_lt = {"fontsize": 14, "color": "white", "ha": "left", "va": "top"}
dict_text_rt = {"fontsize": 14, "color": "white", "ha": "right", "va": "top"}
dict_text_lb = {"fontsize": 14, "color": "white", "ha": "left", "va": "bottom"}
dict_text_rb = {"fontsize": 14, "color": "white", "ha": "right", "va": "bottom"}
dict_line = {"linewidth": 1, "color": "magenta", "linestyle": "--"}

# ------------------------------------------------------------------------------
if fig_direction == "vertical":
    nr, nc = num_methods_show + 1, 1
elif fig_direction == "horizontal":
    nr, nc = 1, num_methods_show + 1
else:
    raise ValueError(
        f"fig_direction must be 'vertical' or 'horizontal', but got {fig_direction}"
    )

fig, axes = plt.subplots(nr, nc, figsize=(nc * 3, nr * 3), **dict_fig)
[ax.set_axis_off() for ax in axes.ravel()]

imgs_one = imgs[id_sample]
for i_method in range(num_methods_show + 1):
    ax = axes[i_method]
    img = imgs_one[i_method]

    img_color = colorize(img, **dict_colorize)

    ax.imshow(img_color)

    img_shape = img.shape
    # add text -----------------------------------------------------------------
    # method name
    pos_text = (int(img_shape[1] * 0.96), int(img_shape[0] * 0.04))
    ax.text(pos_text[0], pos_text[1], methods_name[i_method], **dict_text_rt)

    # add scale bar ------------------------------------------------------------
    if i_method == num_methods_show:
        tp = 0.05
        dict_scale_bar = {
            "pixel_size": pixel_size_xy,
            "bar_length": 50,  # um
            "bar_height": 0.01,
            "bar_color": "white",
            "pos": (int(img_shape[1] * tp), int(img_shape[0] * (1 - tp))),
        }
        add_scale_bar(ax, image=img, **dict_scale_bar)

# save the figure
plt.savefig(os.path.join(path_save_fig, f"sample_{id_sample}.svg"))
plt.savefig(os.path.join(path_save_fig, f"sample_{id_sample}.png"))
