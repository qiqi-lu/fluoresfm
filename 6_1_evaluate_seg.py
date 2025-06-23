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
datasets_info = [
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
    # "omnipose-bact-fluor-a22",
    # "omnipose-bact-fluor-bthai-cyto",
    # "omnipose-bact-fluor-bthai-membrane",
    # "omnipose-bact-fluor-cex",
    # "omnipose-bact-fluor-vibrio",
    # "omnipose-bact-fluor-wiggins",
    # "omnisegger-cyto-lysC",
    # "omnisegger-mem-ygaW",
    # "omnisegger-mem-over",
    # "omnisegger-mem-under",
    # "stardist",
    # "cellpose3-ccdb6843-dn",
    # "care-planaria-dn-1",
    # "care-planaria-dn-2",
    # "care-planaria-dn-3",
    # "care-tribolium-dn-1",
    # "care-tribolium-dn-2",
    # "care-tribolium-dn-3",
    # "fmd-confocal-bpae-b-avg2",
    # "fmd-confocal-bpae-b-avg4",
    # "fmd-confocal-bpae-b-avg8",
    # "fmd-confocal-bpae-b-avg16",
    # "fmd-twophoton-bpae-b-avg2",
    # "fmd-twophoton-bpae-b-avg4",
    # "fmd-twophoton-bpae-b-avg8",
    # "fmd-twophoton-bpae-b-avg16",
    # "fmd-wf-bpae-b-avg2",
    # "fmd-wf-bpae-b-avg4",
    # "fmd-wf-bpae-b-avg8",
    # "fmd-wf-bpae-b-avg16",
    # "deepbacs-ecoli-dn",
    # "deepbacs-ecoli2-dn",
    # "deepbacs-sim-ecoli-dcv",
    # "deepbacs-sim-saureus-dcv",
    # "care-liver-iso",
    # "srcaco2-h2b-sr-2",
    # "srcaco2-h2b-dn-8",
    # "srcaco2-h2b-dn-4",
    # "srcaco2-h2b-dn-2",
    # --------------------------------------------------------------------------
    # "biotisr-ccp-dcv-1",
    # "biotisr-ccp-dcv-2",
    # "biotisr-ccp-dcv-3",
    # "biotisr-factin-dcv-1",
    # "biotisr-factin-dcv-2",
    # "biotisr-factin-dcv-3",
    # "biotisr-factin-nonlinear-dcv-1",
    # "biotisr-factin-nonlinear-dcv-2",
    # "biotisr-factin-nonlinear-dcv-3",
    # "biotisr-lysosome-dcv-1",
    # "biotisr-lysosome-dcv-2",
    # "biotisr-lysosome-dcv-3",
    # "biotisr-mt-dcv-1",
    # "biotisr-mt-dcv-2",
    # "biotisr-mt-dcv-3",
    # "biotisr-mito-dcv-1",
    # "biotisr-mito-dcv-2",
    # "biotisr-mito-dcv-3",
    # "biotisr-ccp-dn-1",
    # "biotisr-ccp-dn-2",
    # "biotisr-factin-dn-1",
    # "biotisr-factin-dn-2",
    # "biotisr-factin-nonlinear-dn-1",
    # "biotisr-factin-nonlinear-dn-2",
    # "biotisr-lysosome-dn-1",
    # "biotisr-lysosome-dn-2",
    # "biotisr-mt-dn-1",
    # "biotisr-mt-dn-2",
    # "biotisr-mito-dn-1",
    # "biotisr-mito-dn-2",
    # "rcan3d-dn-er-dn",
    # "rcan3d-dn-golgi-dn",
    # "rcan3d-dn-tomm20mito-dn",
    # "rcan3d-dn-actin-dn",
    # "rcan3d-dn-lysosome-dn",
    # "rcan3d-dn-mixtrixmito-dn",
    # "rcan3d-dn-mt-dn",
    "biosr-er-sr-1",
    "biosr-er-sr-2",
    "biosr-er-sr-3",
    "biosr-er-sr-4",
    "biosr-er-sr-5",
    "biosr-er-sr-6",
    "biosr-er-dcv-1",
    "biosr-er-dcv-2",
    "biosr-er-dcv-3",
    "biosr-er-dcv-4",
    "biosr-er-dcv-5",
    "biosr-er-dcv-6",
    "biosr-er-dn-1",
    "biosr-er-dn-2",
    "biosr-er-dn-3",
    "biosr-er-dn-4",
    "biosr-er-dn-5",
    # "stardist-25",
    # "stardist-50",
    # "stardist-100",
]

num_samples_eva = 8

methods_info = [
    (
        "UNet-c:all-newnorm-ALL-v2-160-small-bs16",
        "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16",
    ),
]

# ------------------------------------------------------------------------------
# load test dataset info
print("-" * 50)
df_test = pandas.read_excel(path_test_xlsx)
path_results_root = os.path.join("results", "predictions")
# methods_titles = [i[0] for i in methods_info]  # table title
methods_id = [i[1] for i in methods_info]
num_methods = len(methods_id)

for id_dataset in datasets_info:
    print("-" * 50)
    print(f"Dataset: {id_dataset}")
    # get info of current dataset
    info = df_test[df_test["id"] == id_dataset].iloc[0]
    model_name_seg = info["seg_model"]
    path_mask_gt = win2linux(info["path_mask"])
    filenames = read_txt(win2linux(info["path_index"]))

    path_result = os.path.join(path_results_root, id_dataset)
    path_metrics_file = os.path.join(path_result, "metrics_seg.xlsx")

    if path_mask_gt == "Unknown":
        print(f"[WARNNING] Dataset {id_dataset} has no ground truth.")
        path_mask_gt_from_hr = os.path.join(path_result, "gt_mask_" + model_name_seg)
        if os.path.exists(path_mask_gt_from_hr):
            path_mask_gt = path_mask_gt_from_hr
            print(f"[INFO] Use the mask made from high quality images.")
        else:
            print(f"[WARNNING] No mask made from high quality images.")
            continue

    filenames = filenames[:num_samples_eva]
    num_samples = len(filenames)
    print(f"Number of samples: {num_samples}")

    # create/get the excel to write results
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
    try:
        for meth in ["raw"] + methods_id:
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
            if len(df_iou) < num_samples:
                df_iou = df_iou.reindex(range(num_samples))
            if len(df_ap) < num_samples:
                df_ap = df_ap.reindex(range(num_samples))
            df_iou[meth] = ious
            df_ap[meth] = aps
            # print mena and std
            print(f"Mean IoU: {np.mean(ious):.4f}, Std IoU: {np.std(ious):.4f}")
            print(f"Mean AP: {np.mean(aps):.4f}, Std AP: {np.std(aps):.4f}")

        df_iou.to_excel(writer, sheet_name="UoI", index=False)
        df_ap.to_excel(writer, sheet_name="AP", index=False)
        writer.close()
        del df_iou, df_ap
    except Exception as e:
        print(f"[ERROR] {e}")
        df_iou.to_excel(writer, sheet_name="UoI", index=False)
        df_ap.to_excel(writer, sheet_name="AP", index=False)
        writer.close()
        del df_iou, df_ap
