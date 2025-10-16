import numpy as np
import os
import skimage.io as io
from methods.deconvolution import Deconvolution
import utils.data as utils_data

# ------------------------------------------------------------------------------
# parameters
# ------------------------------------------------------------------------------
params = {
    "dataset_name": "SimuMix",
    "path_dataset_gt": "E:\qiqilu\datasets\SimuMix\gt",
    "path_dataset_raw": "E:\qiqilu\datasets\SimuMix\\raw\data_128_128_128_gauss_0_poiss_1_ratio_1_457",
    "num_iter_trad": 100,
}

# ------------------------------------------------------------------------------
path_results = os.path.join("outputs", params["dataset_name"], "results")
# file_names = utils_data.read_txt(os.path.join(params["path_dataset_raw"], "test.txt"))
# file_names = utils_data.read_txt(os.path.join(params["path_dataset_raw"], "train.txt"))
file_names = utils_data.read_txt(
    os.path.join(params["path_dataset_raw"], "validation.txt")
)

# ------------------------------------------------------------------------------
# deconvolution
# ------------------------------------------------------------------------------
print("- Number of files:", len(file_names))

idxs = [0, 1, 2]
psf = io.imread(fname=os.path.join(params["path_dataset_raw"], "PSF.tif"))

for i in idxs:
    img_raw = io.imread(
        fname=os.path.join(params["path_dataset_raw"], "images", file_names[i])
    )
    img_gt = io.imread(
        fname=os.path.join(params["path_dataset_gt"], "images", file_names[i])
    )

    print("-" * 80)
    print(f"- Filename: {file_names[i]}")
    print(
        "- RAW: {}\n- GT: {}\n- PSF: {}".format(img_raw.shape, img_gt.shape, psf.shape)
    )

    DCV = Deconvolution(PSF=psf, bp_type="traditional")
    img_deconv = DCV.deconv(
        stack=img_raw, num_iter=params["num_iter_trad"], domain="fft"
    )

    print(f"DCV: {img_deconv.shape}")
    utils_data.make_path(os.path.join(path_results, "traditional"))
    io.imsave(
        fname=os.path.join(path_results, "traditional", file_names[i]),
        arr=img_deconv,
        check_contrast=False,
    )
