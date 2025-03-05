import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import os
import utils.data as utils_data
import utils.evaluation as eva
from skimage.measure import profile_line
import pandas

# ------------------------------------------------------------------------------
# parameters
# ------------------------------------------------------------------------------
params = {
    "dataset_names": [
        # "biosr-cpp-sr-1",
        # "biosr-cpp-sr-2",
        # "biosr-cpp-sr-3",
        # "biosr-cpp-sr-4",
        # "biosr-cpp-sr-5",
        # "biosr-cpp-sr-6",
        # "biosr-cpp-sr-7",
        # "biosr-cpp-sr-8",
        # "biosr-cpp-sr-9",
        # "biosr-er-sr-1",
        # "biosr-er-sr-2",
        # "biosr-er-sr-3",
        # "biosr-er-sr-4",
        # "biosr-er-sr-5",
        # "biosr-er-sr-6",
        # "biosr-mt-sr-1",
        # "biosr-mt-sr-2",
        # "biosr-mt-sr-3",
        # "biosr-mt-sr-4",
        # "biosr-mt-sr-5",
        # "biosr-mt-sr-6",
        # "biosr-mt-sr-7",
        # "biosr-mt-sr-8",
        # "biosr-mt-sr-9",
        # "biosr-actin-sr-1",
        # "biosr-actin-sr-2",
        # "biosr-actin-sr-3",
        # "biosr-actin-sr-4",
        # "biosr-actin-sr-5",
        # "biosr-actin-sr-6",
        # "biosr-actin-sr-7",
        # "biosr-actin-sr-8",
        # "biosr-actin-sr-9",
        # "biosr-actin-sr-10",
        # "biosr-actin-sr-11",
        # "biosr-actin-sr-12",
        # "deepbacs-sim-ecoli-sr",
        # "deepbacs-sim-saureus-sr",
        # "w2s-c0-sr-1",
        # "w2s-c0-sr-2",
        # "w2s-c0-sr-3",
        # "w2s-c0-sr-4",
        # "w2s-c0-sr-5",
        # "w2s-c0-sr-6",
        # "w2s-c0-sr-7",
        # "w2s-c1-sr-1",
        # "w2s-c1-sr-2",
        # "w2s-c1-sr-3",
        # "w2s-c1-sr-4",
        # "w2s-c1-sr-5",
        # "w2s-c1-sr-6",
        # "w2s-c1-sr-7",
        # "w2s-c2-sr-1",
        # "w2s-c2-sr-2",
        # "w2s-c2-sr-3",
        # "w2s-c2-sr-4",
        # "w2s-c2-sr-5",
        # "w2s-c2-sr-6",
        # "w2s-c2-sr-7",
        # "srcaco2-h2b-sr-8",
        # "srcaco2-h2b-sr-4",
        # "srcaco2-h2b-sr-2",
        # "srcaco2-survivin-sr-8",
        # "srcaco2-survivin-sr-4",
        # "srcaco2-survivin-sr-2",
        # "srcaco2-tubulin-sr-8",
        # "srcaco2-tubulin-sr-4",
        # "srcaco2-tubulin-sr-2",
        # ----------------------------------------------------------------------
        # "biosr-cpp-dn-1",
        # "biosr-cpp-dn-2",
        # "biosr-cpp-dn-3",
        # "biosr-cpp-dn-4",
        # "biosr-cpp-dn-5",
        # "biosr-cpp-dn-6",
        # "biosr-cpp-dn-7",
        # "biosr-cpp-dn-8",
        # "biosr-er-dn-1",
        # "biosr-er-dn-2",
        # "biosr-er-dn-3",
        # "biosr-er-dn-4",
        # "biosr-er-dn-5",
        # "biosr-mt-dn-1",
        # "biosr-mt-dn-2",
        # "biosr-mt-dn-3",
        # "biosr-mt-dn-4",
        # "biosr-mt-dn-5",
        # "biosr-mt-dn-6",
        # "biosr-mt-dn-7",
        # "biosr-mt-dn-8",
        # "biosr-actin-dn-1",
        # "biosr-actin-dn-2",
        # "biosr-actin-dn-3",
        # "biosr-actin-dn-4",
        # "biosr-actin-dn-5",
        # "biosr-actin-dn-6",
        # "biosr-actin-dn-7",
        # "biosr-actin-dn-8",
        # "biosr-actin-dn-9",
        # "biosr-actin-dn-10",
        # "biosr-actin-dn-11",
        # "biosr-actinnl-dn-1",
        # "biosr-actinnl-dn-2",
        # "biosr-actinnl-dn-3",
        # "biosr-actinnl-dn-4",
        # "biosr-actinnl-dn-5",
        # "biosr-actinnl-dn-6",
        # "biosr-actinnl-dn-7",
        # "biosr-actinnl-dn-8",
        # "biosr+-ccp-dn-1",
        # "biosr+-ccp-dn-2",
        # "biosr+-ccp-dn-3",
        # "biosr+-ccp-dn-4",
        # "biosr+-ccp-dn-5",
        # "biosr+-ccp-dn-6",
        # "biosr+-ccp-dn-7",
        # "biosr+-ccp-dn-8",
        # "biosr+-er-dn-1",
        # "biosr+-er-dn-2",
        # "biosr+-er-dn-3",
        # "biosr+-er-dn-4",
        # "biosr+-er-dn-5",
        # "biosr+-er-dn-6",
        # "biosr+-actin-dn-1",
        # "biosr+-actin-dn-2",
        # "biosr+-actin-dn-3",
        # "biosr+-actin-dn-4",
        # "biosr+-actin-dn-5",
        # "biosr+-actin-dn-6",
        # "biosr+-actin-dn-7",
        # "biosr+-actin-dn-8",
        # "biosr+-actin-dn-9",
        # "biosr+-actin-dn-10",
        # "biosr+-actin-dn-11",
        # "biosr+-mt-dn-1",
        # "biosr+-mt-dn-2",
        # "biosr+-mt-dn-3",
        # "biosr+-mt-dn-4",
        # "biosr+-mt-dn-5",
        # "biosr+-mt-dn-6",
        # "biosr+-mt-dn-7",
        # "biosr+-mt-dn-8",
        # "biosr+-myosin-dn-1",
        # "biosr+-myosin-dn-2",
        # "biosr+-myosin-dn-3",
        # "biosr+-myosin-dn-4",
        # "biosr+-myosin-dn-5",
        # "biosr+-myosin-dn-6",
        # "biosr+-myosin-dn-7",
        # "biosr+-myosin-dn-8",
        # "care-planaria-dn-1",
        # "care-planaria-dn-2",
        # "care-planaria-dn-3",
        # "care-tribolium-dn-1",
        # "care-tribolium-dn-2",
        # "care-tribolium-dn-3",
        # "deepbacs-ecoli-dn",
        # "deepbacs-ecoli2-dn",
        # "w2s-c0-dn-1",
        # "w2s-c0-dn-2",
        # "w2s-c0-dn-3",
        # "w2s-c0-dn-4",
        # "w2s-c0-dn-5",
        # "w2s-c0-dn-6",
        # "w2s-c1-dn-1",
        # "w2s-c1-dn-2",
        # "w2s-c1-dn-3",
        # "w2s-c1-dn-4",
        # "w2s-c1-dn-5",
        # "w2s-c1-dn-6",
        # "w2s-c2-dn-1",
        # "w2s-c2-dn-2",
        # "w2s-c2-dn-3",
        # "w2s-c2-dn-4",
        # "w2s-c2-dn-5",
        # "w2s-c2-dn-6",
        # "srcaco2-h2b-dn-8",
        # "srcaco2-h2b-dn-4",
        # "srcaco2-h2b-dn-2",
        # "srcaco2-survivin-dn-8",
        # "srcaco2-survivin-dn-4",
        # "srcaco2-survivin-dn-2",
        # "srcaco2-tubulin-dn-8",
        # "srcaco2-tubulin-dn-4",
        # "srcaco2-tubulin-dn-2",
        # "fmd-confocal-bpae-b-avg2",
        # "fmd-confocal-bpae-b-avg4",
        # "fmd-confocal-bpae-b-avg8",
        # "fmd-confocal-bpae-b-avg16",
        # "fmd-confocal-bpae-g-avg2",
        # "fmd-confocal-bpae-g-avg4",
        # "fmd-confocal-bpae-g-avg8",
        # "fmd-confocal-bpae-g-avg16",
        # "fmd-confocal-bpae-r-avg2",
        # "fmd-confocal-bpae-r-avg4",
        # "fmd-confocal-bpae-r-avg8",
        # "fmd-confocal-bpae-r-avg16",
        # "fmd-confocal-fish-avg2",
        # "fmd-confocal-fish-avg4",
        # "fmd-confocal-fish-avg8",
        # "fmd-confocal-fish-avg16",
        # "fmd-confocal-mice-avg2",
        # "fmd-confocal-mice-avg4",
        # "fmd-confocal-mice-avg8",
        # "fmd-confocal-mice-avg16",
        # "fmd-twophoton-mice-avg2",
        # "fmd-twophoton-mice-avg4",
        # "fmd-twophoton-mice-avg8",
        # "fmd-twophoton-mice-avg16",
        # "fmd-twophoton-bpae-b-avg2",
        # "fmd-twophoton-bpae-b-avg4",
        # "fmd-twophoton-bpae-b-avg8",
        # "fmd-twophoton-bpae-b-avg16",
        # "fmd-twophoton-bpae-g-avg2",
        # "fmd-twophoton-bpae-g-avg4",
        # "fmd-twophoton-bpae-g-avg8",
        # "fmd-twophoton-bpae-g-avg16",
        # "fmd-twophoton-bpae-r-avg2",
        # "fmd-twophoton-bpae-r-avg4",
        # "fmd-twophoton-bpae-r-avg8",
        # "fmd-twophoton-bpae-r-avg16",
        # "fmd-wf-bpae-b-avg2",
        # "fmd-wf-bpae-b-avg4",
        # "fmd-wf-bpae-b-avg8",
        # "fmd-wf-bpae-b-avg16",
        # "fmd-wf-bpae-g-avg2",
        # "fmd-wf-bpae-g-avg4",
        # "fmd-wf-bpae-g-avg8",
        # "fmd-wf-bpae-g-avg16",
        # "fmd-wf-bpae-r-avg2",
        # "fmd-wf-bpae-r-avg4",
        # "fmd-wf-bpae-r-avg8",
        # "fmd-wf-bpae-r-avg16",
        # # ----------------------------------------------------------------------
        # "biosr-cpp-dcv-1",
        # "biosr-cpp-dcv-2",
        # "biosr-cpp-dcv-3",
        # "biosr-cpp-dcv-4",
        # "biosr-cpp-dcv-5",
        # "biosr-cpp-dcv-6",
        # "biosr-cpp-dcv-7",
        # "biosr-cpp-dcv-8",
        # "biosr-cpp-dcv-9",
        # "biosr-er-dcv-1",
        # "biosr-er-dcv-2",
        # "biosr-er-dcv-3",
        # "biosr-er-dcv-4",
        # "biosr-er-dcv-5",
        # "biosr-er-dcv-6",
        # "biosr-mt-dcv-1",
        # "biosr-mt-dcv-2",
        # "biosr-mt-dcv-3",
        # "biosr-mt-dcv-4",
        # "biosr-mt-dcv-5",
        # "biosr-mt-dcv-6",
        # "biosr-mt-dcv-7",
        # "biosr-mt-dcv-8",
        # "biosr-mt-dcv-9",
        # "biosr-actin-dcv-1",
        # "biosr-actin-dcv-2",
        # "biosr-actin-dcv-3",
        # "biosr-actin-dcv-4",
        # "biosr-actin-dcv-5",
        # "biosr-actin-dcv-6",
        # "biosr-actin-dcv-7",
        # "biosr-actin-dcv-8",
        # "biosr-actin-dcv-9",
        # "biosr-actin-dcv-10",
        # "biosr-actin-dcv-11",
        # "biosr-actin-dcv-12",
        # "biosr-actinnl-dcv-1",
        # "biosr-actinnl-dcv-2",
        # "biosr-actinnl-dcv-3",
        # "biosr-actinnl-dcv-4",
        # "biosr-actinnl-dcv-5",
        # "biosr-actinnl-dcv-6",
        # "biosr-actinnl-dcv-7",
        # "biosr-actinnl-dcv-8",
        # "biosr-actinnl-dcv-9",
        # "care-synthe-granules-dcv",
        # "care-synthe-tubulin-dcv",
        # "care-synthe-tubulin-gfp-dcv",
        # "deepbacs-sim-ecoli-dcv",
        # "deepbacs-sim-saureus-dcv",
        # "w2s-c0-dcv-1",
        # "w2s-c0-dcv-2",
        # "w2s-c0-dcv-3",
        # "w2s-c0-dcv-4",
        # "w2s-c0-dcv-5",
        # "w2s-c0-dcv-6",
        # "w2s-c0-dcv-7",
        # "w2s-c1-dcv-1",
        # "w2s-c1-dcv-2",
        # "w2s-c1-dcv-3",
        # "w2s-c1-dcv-4",
        # "w2s-c1-dcv-5",
        # "w2s-c1-dcv-6",
        # "w2s-c1-dcv-7",
        # "w2s-c2-dcv-1",
        # "w2s-c2-dcv-2",
        # "w2s-c2-dcv-3",
        # "w2s-c2-dcv-4",
        # "w2s-c2-dcv-5",
        # "w2s-c2-dcv-6",
        # "w2s-c2-dcv-7",
        # ----------------------------------------------------------------------
        # "care-drosophila-iso",
        # "care-retina0-iso",
        # "care-retina1-iso",
        # "care-liver-iso",
    ],
    "num_sample": 8,
    "index_show": 2,
    "path_dataset_test": "dataset_test.xlsx",
    "path_results": "outputs\\unet_c",
    "path_figure": "outputs\\figures\\imgtext",
    "methods": (
        # ("CARE:biosr-sr-cpp", "care_biosr_sr_cpp"),
        # ("CARE:biosr-sr-actin", "care_biosr_sr_actin"),
        # ("CARE:biosr-sr", "care_biosr_sr"),
        # ("CARE:sr", "care_sr"),
        # ("CARE:biosr-dcv", "care_biosr_dcv"),
        # ("DFCAN:biosr-sr-2", "dfcan_biosr_sr_2"),
        # ("DFCAN:sr-2", "dfcan_sr_2"),
        # ("UNet-uc:sr", "unet_sd_c_sr_crossx"),
        # ("UNet-c:sr", "unet_sd_c_sr"),
        # ----------------------------------------------------------------------
        # ("CARE:dcv", "care_dcv"),
        # ("DFCAN:dcv", "dfcan_dcv"),
        # ("UNet-uc:dcv", "unet_sd_c_dcv_crossx"),
        # ("UNet-c:dcv", "unet_sd_c_dcv"),
        # ----------------------------------------------------------------------
        # ("CARE:dn", "care_dn"),
        # ("DFCAN:dn", "dfcan_dn"),
        # ("UNet-uc:dn", "unet_sd_c_dn_crossx"),
        # ("UNet-c:dn", "unet_sd_c_dn"),
        # ----------------------------------------------------------------------
        # ("CARE:sr", "care_sr"),
        # ("DFCAN:sr-2", "dfcan_sr_2"),
        # ("UNet-uc:sr", "unet_sd_c_sr_crossx"),
        # ("UNet-c:sr", "unet_sd_c_sr"),
        # ----------------------------------------------------------------------
        ("CARE:iso", "care_iso"),
        ("DFCAN:iso", "dfcan_iso"),
        ("UNet-uc:iso", "unet_sd_c_iso_crossx"),
        ("UNet-c:iso", "unet_sd_c_iso"),
        # ----------------------------------------------------------------------
        ("UNet-c:all", "unet_sd_c_all"),
    ),
    "p_low": 0.0,
    "p_high": 0.9999,
}

# ------------------------------------------------------------------------------
if os.name == "posix":
    params["path_results"] = utils_data.win2linux(params["path_results"])
    params["path_figure"] = utils_data.win2linux(params["path_figure"])

titles = [i[0] for i in params["methods"]]
methods = [i[1] for i in params["methods"]]

# ------------------------------------------------------------------------------
data_frame = pandas.read_excel(params["path_dataset_test"])
normalizer = utils_data.NormalizePercentile(
    p_low=params["p_low"], p_high=params["p_high"]
)

# ------------------------------------------------------------------------------
for id_dataset in params["dataset_names"]:
    print("-" * 80)
    print("- Datasset:", id_dataset)

    ds = data_frame[data_frame["id"] == id_dataset]
    if ds["path_hr"].iloc[0] == "Unknown":
        print("The ground truth is inexistent.")
        continue
    # read file names of test images
    sample_filenames = utils_data.read_txt(ds["path_index"].iloc[0])
    path_results = os.path.join(params["path_results"], ds["id"].iloc[0])

    # set the number of samples used for test
    if params["num_sample"] > len(sample_filenames):
        params["num_sample"] = len(sample_filenames)
    num_sample_eva = params["num_sample"]
    print(f"- Number of test samples: {num_sample_eva}/{len(sample_filenames)}")

    # check the existence of metrics file
    path_metrics_file = os.path.join(path_results, f"metrics.xlsx")
    if os.path.exists(path_metrics_file):
        df_psnr = pandas.read_excel(path_metrics_file, sheet_name="PSNR", index_col=0)
        df_ssim = pandas.read_excel(path_metrics_file, sheet_name="SSIM", index_col=0)
        df_zncc = pandas.read_excel(path_metrics_file, sheet_name="ZNCC", index_col=0)
    else:
        df_psnr = pandas.DataFrame()
        df_ssim = pandas.DataFrame()
        df_zncc = pandas.DataFrame()

    # --------------------------------------------------------------------------
    # load results
    # --------------------------------------------------------------------------
    with pandas.ExcelWriter(path_metrics_file, engine="xlsxwriter") as writer:
        imgs_gt, imgs_est, metrics = [], [], []
        for i_sample in range(params["num_sample"]):
            print(f"load results of {sample_filenames[i_sample]} ...")

            # ----------------------------------------------------------------------
            # load ground truth image and apply normalization
            path_gt = os.path.join(ds["path_hr"].iloc[0], sample_filenames[i_sample])
            img_gt = utils_data.read_image(path_gt)
            img_gt = utils_data.interp_sf(img_gt, sf=ds["sf_hr"].iloc[0])
            img_gt = normalizer(img_gt)[0]
            imgs_gt.append(img_gt)

            # ----------------------------------------------------------------------
            img_est_multi_meth = []  # collect estimated images
            # ----------------------------------------------------------------------
            # load raw image and apply normalization
            img_raw = utils_data.read_image(
                os.path.join(ds["path_lr"].iloc[0], sample_filenames[i_sample])
            )
            img_raw = utils_data.utils_data.interp_sf(img_raw, sf=ds["sf_lr"].iloc[0])
            img_raw = normalizer(img_raw)[0]
            # linear transform
            img_raw = eva.linear_transform(img_true=img_gt, img_test=img_raw)

            img_est_multi_meth.append(img_raw)

            # ----------------------------------------------------------------------
            # images form different methods
            for meth in methods:
                tmp = utils_data.read_image(
                    os.path.join(path_results, meth, sample_filenames[i_sample])
                )
                tmp = eva.linear_transform(img_true=img_gt, img_test=tmp)[0]
                img_est_multi_meth.append(tmp)

            imgs_est.append(img_est_multi_meth)

            # ----------------------------------------------------------------------
            # evaluate all results
            metrics_multi_meth = []
            for img in img_est_multi_meth:
                m = []
                psnr = eva.PSNR(img_true=img_gt, img_test=img, data_range=None)
                ssim = eva.SSIM(
                    img_true=img_gt, img_test=img, data_range=None, version_wang=False
                )
                zncc = eva.ZNCC(img_true=img_gt, img_test=img)
                m.extend([psnr, ssim, zncc])
                metrics_multi_meth.append(m)
            metrics.append(metrics_multi_meth)

        metrics = np.array(metrics)  # (N_sample, N_method, N_metrics)
        # --------------------------------------------------------------------------
        all_titles = ["raw"] + titles
        for i_df, df in enumerate([df_psnr, df_ssim, df_zncc]):
            for i_title, title in enumerate(all_titles):
                if title not in list(df.columns):
                    df.insert(df.shape[-1], title, value="")
                df[title] = metrics[:, i_title, i_df]

        # --------------------------------------------------------------------------
        # save all the metrics value into a excel file
        df_psnr.to_excel(writer, sheet_name="PSNR", index_label="id")
        df_ssim.to_excel(writer, sheet_name="SSIM", index_label="id")
        df_zncc.to_excel(writer, sheet_name="ZNCC", index_label="id")
