"""
Calculate the metrics of each image in each dataset.
Saved into a excel file.
Each row is a dataset, each column is a methods, each sheet is a metric.
"""

import os, pandas, traceback
import numpy as np
import utils.data as utils_data
import utils.evaluation as eva
from utils.data import rolling_ball_approximation, interp_sf
from dataset_analysis import datasets_need_bkg_sub
from finetune_evaluation_methods import dataset_method

# ------------------------------------------------------------------------------
# parameters
# ------------------------------------------------------------------------------
params = {
    "dataset_names": [],
    "num_sample": 8,
    # "num_sample": None,
    "path_dataset_test": "dataset_test-v2.xlsx",
    "path_results": "results\\predictions",
    "methods": None,
    "percentiles": (0.03, 0.995),
}


normalizer = utils_data.NormalizePercentile(
    p_low=params["percentiles"][0], p_high=params["percentiles"][1]
)

params["dataset_names"] = dataset_method[0][0]
params["methods"] = dataset_method[0][1]

# ------------------------------------------------------------------------------
params["path_results"] = utils_data.win2linux(params["path_results"])
titles = [i[0] for i in params["methods"]]  # table title
methods = [i[1] for i in params["methods"]]

# ------------------------------------------------------------------------------
data_frame = pandas.read_excel(params["path_dataset_test"])
datasets_not_processed = []  # the datasets that are not processed by the script.
dict_clip = {"a_min": 0.0, "a_max": 2.5}
data_range = dict_clip["a_max"] - dict_clip["a_min"]


def bkg_subtraction(image):
    radius, sf = 25, 16
    image_rb, bg = rolling_ball_approximation(image, radius=radius, sf=sf)
    image_rb = np.clip(image_rb, 0, None)
    return image_rb


for id_dataset in params["dataset_names"]:
    print("-" * 80)
    print("Datasset:", id_dataset)

    if id_dataset in datasets_need_bkg_sub:
        sub_bkg = True
        print("Background subtraction is applied.")
    else:
        sub_bkg = False

    # get the information of current dataset
    ds = data_frame[data_frame["id"] == id_dataset].iloc[0]
    path_results = os.path.join(params["path_results"], ds["id"])
    path_metrics_file = os.path.join(path_results, f"metrics-v2.xlsx")

    # --------------------------------------------------------------------------
    # collect all the datasets that are not processed successfully.
    if not os.path.exists(path_results):
        datasets_not_processed.append(id_dataset)
        continue

    # check if the dataset has ground truth
    if ds["path_hr"] == "Unknown":
        print("The ground truth is inexistent.")
        continue

    # read filenames of test images
    sample_filenames = utils_data.read_txt(ds["path_index"])

    # set the number of samples used for test
    if params["num_sample"] is not None:
        if params["num_sample"] > len(sample_filenames):
            num_sample_analysis = len(sample_filenames)
        else:
            num_sample_analysis = params["num_sample"]
    else:
        num_sample_analysis = len(sample_filenames)
    print(f"Number of test samples: {num_sample_analysis}/{len(sample_filenames)}")

    # --------------------------------------------------------------------------
    # load results
    # --------------------------------------------------------------------------
    try:
        df_psnr = pandas.read_excel(path_metrics_file, sheet_name="PSNR", index_col=0)
    except:
        df_psnr = pandas.DataFrame()

    try:
        df_ssim = pandas.read_excel(path_metrics_file, sheet_name="SSIM", index_col=0)
    except:
        df_ssim = pandas.DataFrame()

    try:
        df_zncc = pandas.read_excel(path_metrics_file, sheet_name="ZNCC", index_col=0)
    except:
        df_zncc = pandas.DataFrame()

    try:
        df_nrmse = pandas.read_excel(path_metrics_file, sheet_name="NRMSE", index_col=0)
    except:
        df_nrmse = pandas.DataFrame()

    try:
        df_msssim = pandas.read_excel(
            path_metrics_file, sheet_name="MSSSIM", index_col=0
        )
    except:
        df_msssim = pandas.DataFrame()

    # --------------------------------------------------------------------------
    writer = pandas.ExcelWriter(path_metrics_file, engine="xlsxwriter")
    try:
        metrics_dataset = []
        for i_sample in range(num_sample_analysis):
            sample_name = sample_filenames[i_sample]
            img_est_multi_meth = []  # collect estimated images

            print(f"load results of {sample_name} ...")
            img_gt = utils_data.read_image(os.path.join(ds["path_hr"], sample_name))
            img_gt = utils_data.interp_sf(img_gt, sf=ds["sf_hr"])[0]
            img_gt = normalizer(img_gt)
            img_gt = np.clip(img_gt, **dict_clip)

            # ------------------------------------------------------------------
            # get raw image and apply normalization
            img_raw = utils_data.read_image(os.path.join(ds["path_lr"], sample_name))
            img_raw = utils_data.utils_data.interp_sf(img_raw, sf=ds["sf_lr"])[0]
            # img_raw = eva.linear_transform(img_true=img_gt, img_test=img_raw)
            img_raw = normalizer(img_raw)
            img_raw = np.clip(img_raw, **dict_clip)
            img_est_multi_meth.append(img_raw)

            # ------------------------------------------------------------------
            # get estimated images form different methods
            for meth in methods:
                img_tmp = utils_data.read_image(
                    os.path.join(path_results, meth, sample_name)
                )

                if sub_bkg:
                    img_tmp = bkg_subtraction(img_tmp)

                # img_tmp = eva.linear_transform(img_true=img_gt, img_test=img_tmp)[0]
                img_tmp = normalizer(img_tmp[0])
                img_tmp = np.clip(img_tmp, **dict_clip)
                img_est_multi_meth.append(img_tmp)

            # ------------------------------------------------------------------
            # evaluate all the images
            metrics_method = []
            for img in img_est_multi_meth:
                metrics_metric = []
                dict_tmp = {"img_true": img_gt, "img_test": img}
                psnr = eva.PSNR(data_range=data_range, **dict_tmp)
                ssim = eva.SSIM(data_range=data_range, **dict_tmp)
                zncc = eva.ZNCC(**dict_tmp)
                nrmse = eva.NRMSE(**dict_tmp)
                try:
                    msssim = eva.MSSSIM(data_range=data_range, **dict_tmp)
                except:
                    try:
                        msssim = eva.MSSSIM(
                            img_true=interp_sf(x=img_gt[None], sf=2)[0],
                            img_test=interp_sf(x=img[None], sf=2)[0],
                            data_range=data_range,
                        )
                    except Exception as e:
                        traceback.print_exc()
                        msssim = 0
                metrics_metric.extend([psnr, ssim, zncc, nrmse, msssim])
                metrics_method.append(metrics_metric)  # (num_meth, num_metric)
            metrics_dataset.append(metrics_method)  # (num_sample, num_meth, num_metric)

        # (num_sample, num_meth, num_metric)
        metrics_dataset = np.array(metrics_dataset)

        # --------------------------------------------------------------------------
        all_titles = ["raw"] + titles
        for i_df, df in enumerate([df_psnr, df_ssim, df_zncc, df_nrmse, df_msssim]):
            for i_title, title in enumerate(all_titles):
                # if current method is not in the dataframe, add it at the last column
                if title not in list(df.columns):
                    df.insert(df.shape[-1], title, value="")
                # insert the metrics of current method into the dataframe
                df[title] = metrics_dataset[:, i_title, i_df]

        # --------------------------------------------------------------------------
        # save all the metrics value into a excel file
        df_psnr.to_excel(writer, sheet_name="PSNR", index_label="id")
        df_ssim.to_excel(writer, sheet_name="SSIM", index_label="id")
        df_zncc.to_excel(writer, sheet_name="ZNCC", index_label="id")
        df_nrmse.to_excel(writer, sheet_name="NRMSE", index_label="id")
        df_msssim.to_excel(writer, sheet_name="MSSSIM", index_label="id")
        writer.close()
    except Exception as e:
        df_psnr.to_excel(writer, sheet_name="PSNR", index_label="id")
        df_ssim.to_excel(writer, sheet_name="SSIM", index_label="id")
        df_zncc.to_excel(writer, sheet_name="ZNCC", index_label="id")
        df_nrmse.to_excel(writer, sheet_name="NRMSE", index_label="id")
        df_msssim.to_excel(writer, sheet_name="MSSSIM", index_label="id")
        writer.close()
        traceback.print_exc()

print("-" * 80)
if datasets_not_processed:
    print("The following datasets are not processed:")
    for dataset in datasets_not_processed:
        print(dataset)
