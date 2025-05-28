"""
Evaluate the segmentation results.
"""

import os, tqdm, pandas
import numpy as np
from utils.data import win2linux, read_txt
import skimage.io as skio
import skimage.measure as skm
from utils.evaluation import average_precision, IoU
import scipy


path_test_xlsx = "dataset_test-v2.xlsx"
id_datasets = [
    "cellpose3-2photon-dn-1",
    "cellpose3-2photon-dn-4",
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
]
num_samples = 8

methods = [
    (
        "UNet-c:all-newnorm-ALL-v2-160-small-bs16",
        "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16",
    ),
]
model_name_seg = "cpsam"

# ------------------------------------------------------------------------------
# load test dataset info
print("-" * 50)
df_test = pandas.read_excel(path_test_xlsx)
path_results_root = os.path.join("results", "predictions")
titles = [i[0] for i in methods]  # table title
methods = [i[1] for i in methods]
num_methods = len(methods)

for id_dataset in id_datasets:
    print("-" * 50)
    print(f"Dataset: {id_dataset}")
    # get info of current dataset
    info = df_test[df_test["id"] == id_dataset]
    path_mask_gt = win2linux(info["path_mask"].values[0])
    path_index = win2linux(info["path_index"].values[0])
    path_result = os.path.join(path_results_root, id_dataset)
    filenames = read_txt(path_index)

    if path_mask_gt == "Unknown":
        print(f"[WARNNING] Dataset {id_dataset} has no ground truth.")
        print("[INFO] check the mask made from high quality images.")
        path_mask_gt_from_hr = os.path.join(path_result, "gt_mask_" + model_name_seg)
        if os.path.exists(path_mask_gt_from_hr):
            path_mask_gt = path_mask_gt_from_hr
            print(f"[INFO] Use the mask made from high quality images.")
        else:
            print(f"[WARNNING] No mask made from high quality images.")
            continue

    if num_samples is not None:
        filenames = filenames[:num_samples]
    num_samples = len(filenames)
    print(f"Number of samples: {num_samples}")

    # create/get the excel to write results
    path_metrics_file = os.path.join(path_result, "metrics_seg.xlsx")
    try:
        df_iou = pandas.read_excel(path_metrics_file, sheet_name="UoI")
    except:
        df_iou = pandas.DataFrame()
    try:
        df_ap = pandas.read_excel(path_metrics_file, sheet_name="AP")
    except:
        df_ap = pandas.DataFrame()

    # --------------------------------------------------------------------------
    # load results
    writer = pandas.ExcelWriter(path_metrics_file, engine="xlsxwriter")
    for meth in ["raw"] + methods:
        print(f"Method: {meth}")
        masks_gt, masks_est, metrics = [], [], []
        for filename in filenames:
            # read ground truth
            mask_gt = skio.imread(os.path.join(path_mask_gt, filename))
            mask_gt = mask_gt.astype(np.uint16)
            # convert binary masks into masks with labels
            mask_gt = skm.label(mask_gt).astype(np.uint16)
            masks_gt.append(mask_gt)

            # read prediction
            mask_est = skio.imread(
                os.path.join(path_result, f"{meth}_mask_{model_name_seg}", filename)
            )
            mask_est = mask_est.astype(np.uint16)
            masks_est.append(mask_est)

        # calculate metrics
        ious = IoU(masks_gt, masks_est, threshold=0.5)
        aps = average_precision(masks_gt, masks_est, threshold=0.5)
        aps = aps[0].squeeze()

        # write to excel
        df_iou[meth] = ious
        df_ap[meth] = aps
        # print mena and std
        print(f"Mean IoU: {np.mean(ious):.4f}, Std IoU: {np.std(ious):.4f}")
        print(f"Mean AP: {np.mean(aps):.4f}, Std AP: {np.std(aps):.4f}")

    df_iou.to_excel(writer, sheet_name="UoI", index=False)
    df_ap.to_excel(writer, sheet_name="AP", index=False)
    writer.close()
