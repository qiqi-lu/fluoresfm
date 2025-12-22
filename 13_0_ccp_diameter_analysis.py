"""
Analysis CCP diameter.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from skimage import io
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import remove_small_objects, remove_small_holes, disk, opening
from skimage.exposure import rescale_intensity
from skimage.segmentation import clear_border, watershed
from skimage.measure import regionprops_table
from skimage.feature import peak_local_max
from skimage.color import label2rgb
from scipy.ndimage import distance_transform_edt

from utils.data import win2linux

# ------------------------------------------------------------------------------
tif_path = "results\predictions\\biosr-cpp-dn-8\\unet_sd_c_all_newnorm-ALL-v2-160-s123-bs16\\40.tif"
path_save_to = os.path.join("results", "figures", "analysis", "ccp_analysis")
os.makedirs(path_save_to, exist_ok=True)

# ------------------------------------------------------------------------------
pixel_size_um = 62.6 / 1000.0  # um/px
gaussian_sigma = 1.0
opening_radius = 1
min_area_px = 6
hole_area_px = 6
remove_edge_objects = False
norm_range = (3, 99.5)

# Watershed splitting controls (MOST IMPORTANT TUNING KNOBS)
min_peak_distance_px = 3  # increase to reduce over-splitting; decrease to split more
min_seed_intensity = (
    None  # optional: set e.g. 0.05-0.2 to ignore weak peaks; None = auto
)
exclude_border_for_peaks = False  # set True to avoid seeds on edge

# Optional post-filters to remove implausible objects
min_area_after_split = 4  # after splitting, remove tiny shards
max_area_after_split = None  # optionally remove huge plaques

# ------------------------------------------------------------------------------
# LOAD IMAGE
# ------------------------------------------------------------------------------
tif_path = win2linux(tif_path)
img = io.imread(tif_path)
img2d = img.mean(axis=0).astype(np.float32)

# ------------------------------------------------------------------------------
# BUILD BASE CCP MASK
# ------------------------------------------------------------------------------
# Gaussian smoothing
# smoothed = gaussian(img2d, sigma=gaussian_sigma)
smoothed = img2d

# Normalize to 0-1
p1, p99 = np.percentile(smoothed, norm_range)
normalized = rescale_intensity(smoothed, in_range=(p1, p99), out_range=(0, 1))

# Thresholding
thr = threshold_otsu(normalized)
binary_raw = normalized > thr

# Morphological operations
opened = opening(binary_raw, footprint=disk(opening_radius))
small_removed = remove_small_objects(opened, min_size=min_area_px)
final_mask = remove_small_holes(small_removed, area_threshold=hole_area_px)

if remove_edge_objects:
    final_mask = clear_border(final_mask)

# -----------------------------
# WATERSHED SPLITTING (FOR PIT COUNT)
# -----------------------------
# Distance transform: peaks correspond to pit centers
dist = distance_transform_edt(final_mask)

# Seed points: local maxima in distance map
# peak_local_max returns coordinates; we convert to marker image
peak_coords = peak_local_max(
    dist,
    labels=final_mask,
    min_distance=min_peak_distance_px,
    threshold_abs=min_seed_intensity,
    exclude_border=exclude_border_for_peaks,
)

markers = np.zeros(dist.shape, dtype=np.int32)
for i, (r, c) in enumerate(peak_coords, start=1):
    markers[r, c] = i

# Watershed on negative distance splits blobs at saddle points
labels_ws = watershed(-dist, markers, mask=final_mask)

# Post-clean: remove tiny fragments created by splitting
labels_ws_clean = labels_ws.copy()
# Convert to boolean per label by relabeling after removing small objects:
# easiest: make boolean mask of "keep" pixels from objects >= min_area_after_split
# then re-run watershed labels to contiguous labels (via relabeling)
# We'll do a simple relabel by zeroing small labels:
if min_area_after_split is not None and min_area_after_split > 0:
    # compute areas quickly
    areas = np.bincount(labels_ws_clean.ravel())
    # areas[0] is background
    small_labs = np.where(areas < min_area_after_split)[0]
    if len(small_labs) > 0:
        small_labs = small_labs[small_labs != 0]
        mask_small = np.isin(labels_ws_clean, small_labs)
        labels_ws_clean[mask_small] = 0

# Optional remove huge plaques
if max_area_after_split is not None:
    areas = np.bincount(labels_ws_clean.ravel())
    big_labs = np.where(areas > max_area_after_split)[0]
    big_labs = big_labs[big_labs != 0]
    if len(big_labs) > 0:
        labels_ws_clean[np.isin(labels_ws_clean, big_labs)] = 0

# Relabel to make labels contiguous 1..N
# (simple way: use skimage.measure.label on nonzero with connectivity, but we want instance labels)
# We'll map existing labels to 1..N
uniq = np.unique(labels_ws_clean)
uniq = uniq[uniq != 0]
new_labels = np.zeros_like(labels_ws_clean, dtype=np.int32)
for new_id, old_id in enumerate(uniq, start=1):
    new_labels[labels_ws_clean == old_id] = new_id

labels_ws_clean = new_labels

# -----------------------------
# MEASURE CCPs (ON WATERSHED LABELS)
# -----------------------------
props = regionprops_table(
    labels_ws_clean,
    intensity_image=img2d,
    properties=[
        "label",
        "area",
        "equivalent_diameter",
        "centroid",
        "mean_intensity",
        "max_intensity",
    ],
)
df = pd.DataFrame(props)
df["equivalent_diameter_px"] = df["equivalent_diameter"]
df["equivalent_diameter_um"] = df["equivalent_diameter_px"] * pixel_size_um

n = len(df)
print(f"Otsu threshold               = {thr:.4f}")
print(f"Seeds found (candidate pits) = {len(peak_coords)}")
print(f"Pits after watershed         = {n}")
print(f"Median equiv diameter        = {df['equivalent_diameter_um'].median():.4f} µm")
print(f"Mean equiv diameter          = {df['equivalent_diameter_um'].mean():.4f} µm")

# -----------------------------
# SAVE RESULTS
# -----------------------------
out_csv = os.path.join(path_save_to, "ccp_diameters_watershed.csv")
df.to_csv(out_csv, index=False)
print(f"Saved: {out_csv}")

# -----------------------------
# SAVE MASKS FIGURE (ALL STEPS)
# -----------------------------
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.ravel()

axes[0].imshow(img2d, cmap="gray")
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(normalized, cmap="gray")
axes[1].set_title("Normalized (p1–p99)")
axes[1].axis("off")

axes[2].imshow(binary_raw, cmap="gray")
axes[2].set_title(f"Raw binary (Otsu={thr:.3f})")
axes[2].axis("off")

axes[3].imshow(final_mask, cmap="gray")
axes[3].set_title("Final mask (pre-watershed)")
axes[3].axis("off")

axes[4].imshow(dist, cmap="gray")
axes[4].set_title("Distance transform")
axes[4].axis("off")

axes[5].imshow(markers > 0, cmap="gray")
axes[5].set_title(f"Seeds (min_distance={min_peak_distance_px})")
axes[5].axis("off")

axes[6].imshow(labels_ws, cmap="nipy_spectral")
axes[6].set_title("Watershed labels (raw)")
axes[6].axis("off")

axes[7].imshow(labels_ws_clean, cmap="nipy_spectral")
axes[7].set_title("Watershed labels (clean)")
axes[7].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(path_save_to, "ccp_masks_watershed.png"), dpi=200)
plt.close(fig)

# -----------------------------
# SAVE OVERLAY FOR QC
# -----------------------------
overlay = label2rgb(
    labels_ws_clean,
    image=rescale_intensity(img2d, out_range=(0, 1)),
    alpha=0.35,
    bg_label=0,
)
plt.figure(figsize=(6, 6))
plt.imshow(overlay)
plt.title("Watershed labels overlay (QC)")
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(path_save_to, "ccp_overlay_watershed.png"), dpi=200)
plt.close()

# -----------------------------
# SAVE HISTOGRAM
# -----------------------------
plt.figure(figsize=(6, 4))
plt.hist(df["equivalent_diameter_um"], bins=30)
plt.title("CCP equivalent diameter distribution (watershed)")
plt.xlabel("Diameter (µm)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(path_save_to, "ccp_diameter_hist_watershed.png"), dpi=200)
plt.close()
