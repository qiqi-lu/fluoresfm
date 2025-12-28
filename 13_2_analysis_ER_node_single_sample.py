"""
Analysis the node degree of ER network using Nellie.
"""

import os, seaborn, pandas
import numpy as np
from skimage import io
from utils.data import win2linux
from nellie.im_info.verifier import ImInfo, FileInfo
from nellie.utils.base_logger import logger

from utils.analysis import node_degree
import matplotlib.pyplot as plt
from utils.data import normalization

# ------------------------------------------------------------------------------
id_gpu = 0
# path_image = "results\predictions\\biosr-er-dn-2\gt\\58.tif"
path_image = (
    "results\\figures\\analysis\\analysis_er\\biosr-er-sr-2\\images\\gt\\51.tif"
)
path_save = os.path.join("results", "figures", "analysis", "analysis_er")
os.makedirs(path_save, exist_ok=True)

path_image = win2linux(path_image)

# ------------------------------------------------------------------------------
print("-" * 80)
os.environ["CUDA_VISIBLE_DEVICES"] = f"{id_gpu}"  # for nellie
logger.disable()
normalizer = lambda x: normalization(x, p_low=0.03, p_high=0.995)

# --------------------------------------------------------------------------
print("-" * 80)
print("[INFO] Load image ...")
res_xy = 62.6 / 1000.0  # um/pixel
file_info = FileInfo(path_image)
file_info.find_metadata()
file_info.load_metadata()
file_info.change_axes("TYX")
file_info.change_dim_res("T", 1)
file_info.change_dim_res("Y", res_xy)
file_info.change_dim_res("X", res_xy)
im_info = ImInfo(file_info)

# --------------------------------------------------------------------------
print("-" * 80)
print("[INFO] ER network analysis...")
node_info = node_degree(im_info, verbose=False)
node_coords = node_info["coords"]
node_degree = node_info["degree"]
# np.savez(os.path.join(path_save, "node_yx_degree.npy"), node_info, allow_pickle=True)

print("[INFO] Number of node centroids: ", node_coords.shape[0])
print("[INFO] Node degree: ", node_degree.shape)

print(
    f"[INFO] Average node degree (std):  {node_degree.mean():.4f} ({node_degree.std():.4f})"
)
print(f"[INFO] Node degree range: {np.unique(node_degree)}")

# --------------------------------------------------------------------------
print("[INFO] Display nodes ...")
dict_colors = {
    1: "#FADCC8",
    2: "#EC8860",
    3: "#2F67AC",
    4: "#B21F2B",
    5: "#1B3E22",
}
dict_fig = dict(dpi=600, constrained_layout=True)
nr, nc = 1, 1
fig, axes = plt.subplots(nr, nc, figsize=(nc * 3, nr * 3), **dict_fig)

img = io.imread(path_image)[0]
img = normalizer(img)
img = np.clip(img, 0, 0.9)
axes.imshow(img, cmap="gray")

# add node markers -------------------------------------------------------------
df_node = pandas.DataFrame(columns=["x", "y", "degree"])
df_node["x"] = node_coords[:, 1]
df_node["y"] = node_coords[:, 0]
df_node["degree"] = node_degree

seaborn.scatterplot(
    data=df_node,
    x="x",
    y="y",
    hue="degree",
    palette=dict_colors,
    s=3,
    edgecolor="none",
    ax=axes,
    legend=False,
)

# add text ---------------------------------------------------------------------
dict_text = dict(fontsize=8, color="white", ha="right", va="bottom")
# add average node degree
axes.text(
    0.95,
    0.05,
    f"avg. degree: {node_degree.mean():.2f} ({node_degree.std():.2f})",
    transform=axes.transAxes,
    **dict_text,
)

axes.axis("off")
# save
path_fig = os.path.join(path_save, "node_yx_degree.png")
plt.savefig(path_fig)
