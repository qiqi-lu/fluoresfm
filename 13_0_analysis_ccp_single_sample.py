"""
Analysis CCP diameter.
"""

import os, colorcet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage import io
from skimage.color import label2rgb
from skimage.exposure import rescale_intensity
from skimage.measure import regionprops_table
from utils.analysis import pit_segmentation
from utils.data import win2linux
from utils.plot import get_outlines
import logging

logging.getLogger().setLevel(logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# ------------------------------------------------------------------------------
# tif_path = "results\predictions\\biosr-cpp-dcv-1\\unet_sd_c_all_newnorm-ALL-v2-160-s123-bs16\\40.tif"
# tif_path = "results\predictions\\biosr-cpp-dcv-1\gt\\40.tif"
# tif_path = "results\predictions\\biosr-cpp-dcv-1\\raw\\40.tif"
tif_path = "results\predictions\\biotisr-ccp-dcv-1\\unet_sd_c_all_newnorm-ALL-v2-160-small-bs16\Cell_044_0.tif"
tif_path = "results\predictions\\biotisr-ccp-dcv-1\\gt\Cell_044_0.tif"
# tif_path = "results\predictions\\biotisr-ccp-dcv-1\\raw\Cell_044_0.tif"
tif_path = "results\predictions\\biotisr-lysosome-dcv-3\\unet_sd_c_all_newnorm-ALL-v2-160-small-bs16\Cell_044_0.tif"
tif_path = "results\predictions\\biotisr-lysosome-dcv-3\\gt\Cell_044_0.tif"
tif_path = "results\predictions\\biotisr-lysosome-dcv-3\\raw\Cell_044_0.tif"
path_save = os.path.join("results", "figures", "analysis", "analysis_ccp")

os.makedirs(path_save, exist_ok=True)
tif_path = win2linux(tif_path)

pixel_size = 62.6  # nm/pixel

print(f"-" * 80)
# ------------------------------------------------------------------------------
# load image
img = io.imread(tif_path)
img2d = img.squeeze().astype(np.float32)
assert img2d.ndim == 2, "[ERROR] Image must be 2D."

# ------------------------------------------------------------------------------
results = pit_segmentation(
    image=img2d,
    gaussian_sigma=1.0,
    norm_range=(0.03, 0.995),
    clip_range=(0, 2.0),
    min_area_px=3,
    hole_area_px=8,
    min_peak_distance_px=3,
    return_intermediate=True,
    otsu_thr_factor=1.0,
)
pixel_size_um = pixel_size / 1000.0  # um/px

labels = results["labels_ws_clean"]
props = regionprops_table(
    labels,
    intensity_image=img2d,
    properties=["label", "area", "equivalent_diameter", "centroid"],
)
df = pd.DataFrame(props)
df["equivalent_diameter_px"] = df["equivalent_diameter"]
df["equivalent_diameter_um"] = df["equivalent_diameter_px"] * pixel_size_um

# save results
df.to_excel(os.path.join(path_save, "ccp_props.xlsx"), index=False)

n = len(df)
d_mean = df["equivalent_diameter_um"].mean()
d_std = df["equivalent_diameter_um"].std()
d_median = df["equivalent_diameter_um"].median()
print(f"Pits after watershed  = {n}")
print(f"Median equiv diameter = {d_median:.4f} µm")
print(f"Mean equiv diameter   = {d_mean:.4f} µm")

# ------------------------------------------------------------------------------
# Display results
# ------------------------------------------------------------------------------
dict_fig = dict(dpi=600, constrained_layout=True)
cmap_glasbey = [(0, 0, 0)] + list(colorcet.cm.glasbey_dark.colors)
cmap_glasbey = ListedColormap(cmap_glasbey)

# # display masks
nr, nc = 2, 4
fig, axes = plt.subplots(2, 4, figsize=(nc * 3, nr * 3), **dict_fig)
axes = axes.ravel()
[ax.set_axis_off() for ax in axes]

axes[0].imshow(img2d, cmap="hot")
axes[0].set_title("Original")

axes[1].imshow(results["normalized"], cmap="hot")
axes[1].set_title("Normalized")

axes[2].imshow(results["mask_ostu"], cmap="gray")
axes[2].set_title(f"mask (Otsu)")

axes[3].imshow(results["mask_init"], cmap="gray")
axes[3].set_title("mask (pre-watershed)")

axes[4].imshow(results["dist_map"], cmap="hot")
axes[4].set_title("Distance to backrgound")

axes[5].imshow(results["markers"] > 0, cmap="gray")
axes[5].set_title(f"Seed")

axes[6].imshow(results["labels_ws"], cmap=cmap_glasbey)
axes[6].set_title("Watershed labels (raw)")

axes[7].imshow(results["labels_ws_clean"], cmap=cmap_glasbey)
axes[7].set_title("Watershed labels (- small objects)")

plt.savefig(os.path.join(path_save, "ccp_masks.png"))

# # ------------------------------------------------------------------------------
# # display overlay
overlay = label2rgb(
    labels,
    image=rescale_intensity(img2d, out_range=(0, 1)),
    colors=colorcet.cm.glasbey_dark.colors,
    alpha=0.35,
    bg_label=0,
)

nr, nc = 1, 3
fig, axes = plt.subplots(nr, nc, figsize=(nc * 3, nr * 3), **dict_fig)

# show overlay
axes[0].imshow(overlay)
axes[0].set_axis_off()

# show histogram
hist_range = (0, 1.5)
axes[1].hist(df["equivalent_diameter_um"], bins=30, range=hist_range)
axes[1].set_xlim(hist_range)
axes[1].set_xlabel("Diameter (µm)")
axes[1].set_ylabel("Count")
axes[1].set_box_aspect(1)

# show outline
axes[2].imshow(img2d, cmap=colorcet.cm.fire)
outlines = get_outlines(labels)
for i in range(n):
    axes[2].plot(outlines[i][:, 1], outlines[i][:, 0], color="green", linewidth=0.5)
axes[2].set_axis_off()

# add text about avergae diameter, and number of pits
axes[2].text(
    0.95,
    0.05,
    f"{d_mean:.2f} ({d_std:.2f}) µm \n {n} pits",
    transform=axes[2].transAxes,
    fontsize=8,
    color="white",
    ha="right",
    va="bottom",
)
plt.savefig(os.path.join(path_save, "ccp_overlay_diameter_hist.png"))
