"""
MT structure analysis for a single image.
"""

from utils.data import win2linux
from utils.plot import plot_and_save_2d_image
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import os, json

from methods.SIFNE.data_preprocessing import load_image, preview_filter, LFT_OFT

dataset_name = "SIFNE"

path_images = win2linux("results/figures/images")
path_save = os.path.join(path_images, dataset_name)
path_save_data = os.path.join(path_save, "data")
os.makedirs(path_save_data, exist_ok=True)

params = dict(
    imgpath=win2linux("E:\qiqilu\datasets\SIFNE\\unzip\SIFNE\MT.tiff"),
    maskpath=win2linux("E:\qiqilu\datasets\SIFNE\\unzip\SIFNE\MT_cell_mask.tiff"),
    radius_of_filter=10,  # pixels
    num_of_filter_orientations=20,  # in the range of Pi
)

# save params to json file
with open(os.path.join(path_save_data, "params.json"), "w") as f:
    json.dump(params, f, indent=4)


# ------------------------------------------------------------------------------
# load image
imgpath = params["imgpath"]
OriginImg = load_image(imgpath)
# save image
io.imsave(os.path.join(path_save_data, "OriginImg.tif"), OriginImg)
plot_and_save_2d_image(OriginImg, os.path.join(path_save_data, "OriginImg.png"))

# ------------------------------------------------------------------------------
# LFT_OFT

preview_filter(
    OriginImg,
    path_save_to=path_save_data,
    R=params["radius_of_filter"],
    NofOrientations_FT=params["num_of_filter_orientations"],
)

LFT_OFT(
    OriginImg,
    params["maskpath"],
    path_save_to=path_save_data,
    R=params["radius_of_filter"],
    NofOrientations_FT=params["num_of_filter_orientations"],
)
