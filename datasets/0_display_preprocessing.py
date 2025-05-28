"""
check the result of preprocessing.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io

from utils_data import normalization, linear_trasnform, interp_sf

path_gt = "BioSR\\transformed\MTs\\train\channel_0\SIM"
# path_gt = "BioSR\\transformed\MTs\\train\channel_0\WF_noise_level_6"
path_raw = "BioSR\\transformed\MTs\\train\channel_0\WF_noise_level_1"
id_check = 1  # id of the image to check

# ------------------------------------------------------------------------------
norm_l, norm_h = 0.03, 0.995
# norm_l, norm_h = 0.0, 0.995
# ------------------------------------------------------------------------------
list_gt = os.listdir(path_gt)
list_gt = [x for x in list_gt if x.endswith(".tif")]  # filter out the non-tif files

# load the gt image
gt = io.imread(os.path.join(path_gt, list_gt[id_check]))[0]
raw = io.imread(os.path.join(path_raw, list_gt[id_check]))[0]
raw = interp_sf(raw, 2)

num_pixels = gt.shape[0] * gt.shape[1]

# normalize the gt and raw image to [0, 1]
gt_norm, vmin_gt, vmax_gt = normalization(gt, norm_l, norm_h)
raw_norm, vmin_raw, vmax_raw = normalization(raw, norm_l, norm_h)

# gt_norm, vmin_gt, vmax_gt = gt, gt.min(), gt.max()
# raw_norm, vmin_raw, vmax_raw = raw, raw.min(), raw.max()

# raw_norm = np.clip(raw_norm, a_min=0.0, a_max=None)  # clip negative values to 0
# gt_norm = np.clip(gt_norm, a_min=0.0, a_max=None)  # clip negative values to 0
# raw_norm = raw_norm / raw_norm.sum() * num_pixels * 0.2
# gt_norm = gt_norm / gt_norm.sum() * num_pixels * 0.2

# raw_norm = linear_trasnform(raw_norm, gt_norm)

# raw_norm = (raw - vmin_gt) / (vmax_gt - vmin_gt)
# vmin_raw, vmax_raw = vmin_gt, vmax_gt

# ------------------------------------------------------------------------------
# plot the gt and raw images
fig, axes = plt.subplots(nrows=3, ncols=2, dpi=300, figsize=(6, 9))
vmax = 2.0
vmin = 0.0

# raw image
axes[0, 0].imshow(raw_norm, vmin=vmin, vmax=vmax, cmap="magma")
# use min and max value as title
axes[0, 0].set_title(
    "raw image, min={:.3f}, max={:.3f}".format(np.min(raw_norm), np.max(raw_norm)),
    fontsize=6,
)
raw_mean = np.mean(raw)
raw_norm_mean = np.mean(raw_norm)
gt_norm_mean = np.mean(gt_norm)
print("raw_norm mean: {:.3f}, gt_norm mean: {:.3f}".format(raw_norm_mean, gt_norm_mean))

# plot the histogram of raw image
axes[1, 0].hist(raw.flatten(), bins=100, log=False)
# plot a line of vin_raw, and vmax_raw in the histogram
axes[1, 0].axvline(x=vmin_raw, color="r")
axes[1, 0].axvline(x=vmax_raw, color="r")
# plot a line of raw_mean in the histogram
axes[1, 0].axvline(x=raw_mean, color="g")
# use the vim_raw and vmax_raw as title, and add the value of vmin_raw and vmax_raw to the title
axes[1, 0].set_title(
    "raw image, vmin={:.3f}, vmax={:.3f}".format(vmin_raw, vmax_raw), fontsize=6
)

# gt image
axes[0, 1].imshow(gt_norm, vmin=vmin, vmax=vmax, cmap="magma")
# use min and max value as title
axes[0, 1].set_title(
    "gt image, min={:.3f}, max={:.3f}".format(np.min(gt_norm), np.max(gt_norm)),
    fontsize=6,
)
# same as raw
axes[1, 1].hist(gt.flatten(), bins=100, log=False)
axes[1, 1].axvline(x=vmin_gt, color="r")
axes[1, 1].axvline(x=vmax_gt, color="r")
axes[1, 1].set_title(
    "gt image, vmin={:.3f}, vmax={:.3f}".format(vmin_gt, vmax_gt), fontsize=6
)

# plot the profile line of raw and gt image
pos = raw_norm.shape[0] // 2
axes[2, 0].plot(raw_norm[pos], label="raw", linewidth=0.5)
axes[2, 0].plot(gt_norm[pos], label="gt", linewidth=0.5)
axes[2, 0].legend()
# zero line
axes[2, 0].axhline(y=0, color="r", linewidth=0.5)

# another line
pos = raw_norm.shape[1] // 2
axes[2, 1].plot(raw_norm[:, pos], label="raw", linewidth=0.5)
axes[2, 1].plot(gt_norm[:, pos], label="gt", linewidth=0.5)
axes[2, 1].legend()
# zero line
axes[2, 1].axhline(y=0, color="r", linewidth=0.5)

# save image to check_preprocessing.png
plt.savefig("check_preprocessing.png")
