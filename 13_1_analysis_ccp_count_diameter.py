"""
Use Nellie to segment the CCPs in the image and analysis the counts of CCPs and
the diameter of CCPs.
Some CCPs are touching or partially merged, thus they may be segmented as one CCP.
So, the number of the CCPs can be underestimated, and the diameter of the CCPs
can be overestimated.
"""

import os
import numpy as np
from skimage import io
from scipy import ndimage as ndi
from skimage import measure, morphology
from skimage.feature import peak_local_max
from skimage.segmentation import watershed



# ------------------------------------------------------------------------------
path_image = "results\predictions\\biosr-cpp-dn-8\\unet_sd_c_all_newnorm-ALL-v2-160-s123-bs16\\40.tif"
path_mask = 
path_save_to = os.path.join("results", "figures", "analysis", "ccp_analysis")
os.makedirs(path_save_to, exist_ok=True)

# ------------------------------------------------------------------------------

# --- Inputs ---
mask = your_binary_mask.astype(bool)  # CCP mask
pixel_size_nm = 100  # example: 100 nm per pixel

# --- Distance transform ---
distance = ndi.distance_transform_edt(mask)

# --- Find CCP centers ---
local_max = peak_local_max(
    distance,
    labels=mask,
    min_distance=2,   # adjust based on CCP size
    exclude_border=False
)

# Create marker image
markers = np.zeros_like(mask, dtype=int)
for i, (r, c) in enumerate(local_max, start=1):
    markers[r, c] = i

# --- Watershed segmentation ---
labels = watershed(-distance, markers, mask=mask)

# --- Count CCPs ---
num_ccps = labels.max()
print("Number of CCPs:", num_ccps)

# --- Measure CCP properties ---
props = measure.regionprops(labels)

# Equivalent diameter (circle with same area)
diameters_pixels = [p.equivalent_diameter for p in props]
diameters_nm = np.array(diameters_pixels) * pixel_size_nm

print("Average CCP diameter (nm):", diameters_nm.mean())
print("Std CCP diameter (nm):", diameters_nm.std())