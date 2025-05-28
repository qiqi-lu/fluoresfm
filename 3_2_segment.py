"""
Use segmentation model to segment the raw and the predicted images, and generate the masks.
"""

import os, tqdm, pandas, torch
import numpy as np
import skimage.io as skio
from utils.data import win2linux, read_txt, interp_sf, NormalizePercentile
from cellpose import models, denoise, io
from nellie.im_info.verifier import FileInfo, ImInfo
from nellie.segmentation.filtering import Filter
from nellie.segmentation.labelling import Label
from nellie.segmentation.networking import Network
from nellie.utils.base_logger import logger

id_gpu = 0
model_name = "cpsam"
# model_name = "nellie"

path_test_xlsx = "dataset_test-v2.xlsx"
id_datasets = [
    # "cellpose3-2photon-dn-1",
    # "cellpose3-2photon-dn-4",
    # "cellpose3-2photon-dn-16",
    # "cellpose3-2photon-dn-64",
    # "colon-tissue-dn-high",
    # "colon-tissue-dn-low",
    # "hl60-high-noise-c00",
    # "hl60-high-noise-c25",
    # "hl60-high-noise-c50",
    # "hl60-high-noise-c75",
    # "hl60-low-noise-c00",
    # "hl60-low-noise-c25",
    # "hl60-low-noise-c50",
    # "hl60-low-noise-c75",
    # "scaffold-a549-dn",
    # "granuseg-dn-high",
    # "granuseg-dn-low",
    # "colon-tissue-dcv-high",
    # "colon-tissue-dcv-low",
    # "hl60-high-noise-c00-dcv",
    # "hl60-high-noise-c25-dcv",
    # "hl60-high-noise-c50-dcv",
    # "hl60-high-noise-c75-dcv",
    # "hl60-low-noise-c00-dcv",
    # "hl60-low-noise-c25-dcv",
    # "hl60-low-noise-c50-dcv",
    # "hl60-low-noise-c75-dcv",
    # "granuseg-dcv-high",
    # "granuseg-dcv-low",
    # "deepbacs-seg-saureus-dcv",
    # "deepbacs-seg-bsubtiles-dn",
    # --------------------------------------------------------------------------
    "biotisr-mito-dn-1",
    # "biotisr-mito-dn-2",
    # "rcan3d-dn-er-dn",
    # "rcan3d-dn-golgi-dn",
    # "rcan3d-dn-tomm20mito-dn",
]

methods = [
    "gt",
    "raw",
    "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16",
]

path_results = os.path.join("results", "predictions")
num_samples = 8
input_normallizer = NormalizePercentile(0.03, 0.995)

# ------------------------------------------------------------------------------
print("-" * 80)
print("Load segmentation model:", model_name)

if model_name == "cpsam":
    io.logger_setup()  # cellpose logger
    model_seg = models.CellposeModel(gpu=True, device=torch.device(f"cuda:{id_gpu}"))
elif model_name == "nellie":
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{id_gpu}"  # for nellie
    logger.disable()  # nellie logger
else:
    raise ValueError("Unknown model name.")

# ------------------------------------------------------------------------------
print("\n")
print("Number of dataset:", len(id_datasets))
print("Number of methods:", len(methods))

# load datasets info
df_test = pandas.read_excel(path_test_xlsx)

# loop over each dataset and each method
for id_dataset in id_datasets:
    print("-" * 80)
    print("Dataset:", id_dataset)
    df_dataset = df_test[df_test["id"] == id_dataset].iloc[0]

    path_lr = win2linux(df_dataset["path_lr"])
    path_hr = win2linux(df_dataset["path_hr"])
    path_mask_gt = win2linux(df_dataset["path_mask"])
    path_index = win2linux(df_dataset["path_index"])

    # get the resolution of image
    res_xy = float(df_dataset["target pixel size"].split("x")[0]) / 1000.0  # um

    # get all the filenames used for test
    filenames = read_txt(path_index)[:num_samples]

    # get the path of lr
    for meth in methods:
        print("\nMethod:", meth)
        if meth == "raw":
            path_image = path_lr
            path_mask = os.path.join(path_results, id_dataset, "raw_mask")
            path_tmp = os.path.join(path_results, id_dataset, "raw")
            scale_factor = df_dataset["sf_lr"]
        elif meth == "gt":
            if path_mask_gt == "Unknown":
                print(f">> Dataset {id_dataset} has no ground truth mask.")
                print(">> Use the high quality image to make ground truth mask.")
                path_image = path_hr
                path_mask = os.path.join(path_results, id_dataset, "gt_mask")
                path_tmp = os.path.join(path_results, id_dataset, "gt")
                scale_factor = df_dataset["sf_hr"]
            else:
                print(f">> Dataset {id_dataset} has ground truth mask.")
                continue
        else:
            path_image = os.path.join(path_results, id_dataset, meth)
            path_mask = path_image + "_mask"

        path_mask = path_mask + "_" + model_name
        os.makedirs(path_mask, exist_ok=True)

        if meth in ["raw", "gt"]:
            os.makedirs(path_tmp, exist_ok=True)

        # loop over each image
        pbar = tqdm.tqdm(total=len(filenames), desc=f"SEGMENTATION", ncols=80)
        for filename in filenames:
            path_file = os.path.join(path_image, filename)
            img = skio.imread(path_file)
            img = np.clip(img, 0, None)
            img = input_normallizer(img)

            if meth in ["raw", "gt"]:
                img = interp_sf(img, scale_factor)  # resample

            if meth in ["raw", "gt"]:
                # save the preprocessed image into tmp folder
                path_file = os.path.join(path_tmp, filename)
                skio.imsave(path_file, img, check_contrast=False)

            # segemntation
            if model_name == "cpsam":
                mask, flows, styles = model_seg.eval(img, niter=1000)

            elif model_name == "nellie":
                file_info = FileInfo(path_file)
                file_info.find_metadata()
                file_info.load_metadata()
                file_info.change_axes("TYX")
                file_info.change_dim_res("T", 1)
                file_info.change_dim_res("Y", res_xy)
                file_info.change_dim_res("X", res_xy)
                im_info = ImInfo(file_info)
                preprocessing = Filter(im_info, remove_edges=False)
                preprocessing.run()
                segmentation = Label(
                    im_info, otsu_thresh_intensity=False, threshold=None
                )
                segmentation.run()

                # move the segmentation to the mask folder
                path_saved = im_info.pipeline_paths["im_instance_label"]
                mask = io.imread(path_saved)
                mask = mask.astype("uint16")
            skio.imsave(
                os.path.join(path_mask, filename), mask[None], check_contrast=False
            )
            pbar.update(1)
        pbar.close()
