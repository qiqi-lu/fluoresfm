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
    "id_dataset": [
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
        # "biosr-cpp-sr-9",
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
        "biosr-actin-sr-12",
        # "biosr-actin-dcv-1",
        # "biosr-actin-dcv-12",
        # "biosr-actin-dn-1",
        # "w2s-c0-sr-7",
        # "deepbacs-sim-ecoli-sr",
        # "deepbacs-sim-saureus-sr",
        # "fmd-wf-bpae-r-avg2",
        # "fmd-wf-bpae-r-avg4",
        # "fmd-wf-bpae-r-avg8",
        # "fmd-wf-bpae-r-avg16",
    ],
    "num_sample": 8,
    "id_sample_show": 2,
    "path_dataset_test": "dataset_test.xlsx",
    "path_results": "outputs\\unet_c",
    "path_figure": "outputs\\figures\\imgtext",
    "methods": (
        ("CARE:biosr-sr-cpp", "care_biosr_sr_cpp"),
        ("CARE:biosr-sr-actin", "care_biosr_sr_actin"),
        ("CARE:biosr-sr", "care_biosr_sr"),
        ("CARE:sr", "care_sr"),
        # ("CARE:dcv", "care_dcv"),
        # ("CARE:biosr-dcv", "care_biosr_dcv"),
        ("DFCAN:biosr-sr-2", "dfcan_biosr_sr_2"),
        ("DFCAN:sr-2", "dfcan_sr_2"),
        # ("DFCAN:dcv", "dfcan_dcv"),
        ("UNet-uc:sr", "unet_sd_c_sr_crossx"),
        ("UNet-c:sr", "unet_sd_c_sr"),
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

for id_dataset in params["id_dataset"]:
    print("- Dataset:", id_dataset)
    ds = data_frame[data_frame["id"] == id_dataset]

    sample_filenames = utils_data.read_txt(ds["path_index"].iloc[0])
    path_results = os.path.join(params["path_results"], ds["id"].iloc[0])
    path_figures = os.path.join(params["path_figure"], id_dataset)
    utils_data.make_path(path_figures)

    # --------------------------------------------------------------------------
    # load results
    # --------------------------------------------------------------------------
    num_sample_show = params["num_sample"]
    print(f"- Number of test samples: {num_sample_show}/{len(sample_filenames)}")

    if params["num_sample"] > len(sample_filenames):
        params["num_sample"] = len(sample_filenames)

        print(f"load results of {sample_filenames[params['id_sample_show']]} ...")

    # --------------------------------------------------------------------------
    # ground truth
    img_gt = utils_data.read_image(
        os.path.join(ds["path_hr"].iloc[0], sample_filenames[params["id_sample_show"]])
    )
    img_gt = utils_data.interp_sf(img_gt, sf=ds["sf_hr"].iloc[0])
    img_gt = normalizer(img_gt)[0]

    imgs_est = []  # collect estimated images
    # --------------------------------------------------------------------------
    # raw image
    img_raw = utils_data.read_image(
        os.path.join(ds["path_lr"].iloc[0], sample_filenames[params["id_sample_show"]])
    )
    img_raw = utils_data.utils_data.interp_sf(img_raw, sf=ds["sf_lr"].iloc[0])
    img_raw = normalizer(img_raw)[0]
    img_raw = eva.linear_transform(img_true=img_gt, img_test=img_raw)
    imgs_est.append(img_raw)

    # --------------------------------------------------------------------------
    # images form different methods
    for meth in methods:
        tmp = utils_data.read_image(
            os.path.join(path_results, meth, sample_filenames[params["id_sample_show"]])
        )
        tmp = eva.linear_transform(img_true=img_gt, img_test=tmp)[0]
        imgs_est.append(tmp)

    # ------------------------------------------------------------------------------
    # display
    # ------------------------------------------------------------------------------
    def show_slice(
        img,
        img_gt,
        ax,
        i_slice=None,
        title=None,
        vmin=None,
        vmax=None,
        cmap=None,
        evaluate=True,
        patch_size=(256, 256),
    ):
        patch_pos = (img.shape[-2] // 3, img.shape[-1] // 3)
        ax[0].set_title(title, fontdict={"fontsize": 10})

        if len(img.shape) == 3:
            ax[0].imshow(img[i_slice], vmin=vmin, vmax=vmax, cmap=cmap)
        if len(img.shape) == 2:
            ax[0].imshow(img, vmin=vmin, vmax=vmax, cmap=cmap)
            ax[1].imshow(
                img[
                    patch_pos[0] : patch_pos[0] + patch_size[0],
                    patch_pos[1] : patch_pos[1] + patch_size[1],
                ],
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
            )

        if evaluate:
            if len(img.shape) == 3:
                ax[2].imshow(
                    (img - img_gt)[i_slice], cmap="seismic", vmin=-vmax, vmax=vmax
                )
            if len(img.shape) == 2:
                ax[2].imshow((img - img_gt), cmap="seismic", vmin=-vmax, vmax=vmax)

            ssim = eva.SSIM(img_true=img_gt, img_test=img)
            psnr = eva.PSNR(img_true=img_gt, img_test=img)
            zncc = eva.ZNCC(img_true=img_gt, img_test=img)
            mae = eva.MAE(img_true=img_gt, img_test=img)
            mse = eva.MSE(img_true=img_gt, img_test=img)

            ax[2].set_title(
                "{:>.4f} | {:>.4f} | {:>.4f}\n{:>.8f} | {:>.8f}".format(
                    psnr, ssim, zncc, mae, mse
                ),
                fontdict={"fontsize": 10},
            )

    # ------------------------------------------------------------------------------
    # show specific sample
    nr, nc = 3, len(methods) + 2

    fig, axes = plt.subplots(
        nrows=nr, ncols=nc, figsize=(nc * 3, nr * 3), dpi=300, constrained_layout=True
    )
    [ax.set_axis_off() for ax in axes.ravel()]
    dict_img = {"vmin": 0, "vmax": 1.0, "cmap": "hot"}

    show_slice(img_gt, img_gt, ax=axes[:, -1], title="GT", evaluate=False, **dict_img)

    show_slice(imgs_est[0], img_gt, ax=axes[:, 0], title="RAW", **dict_img)

    for i in range(len(methods)):
        show_slice(
            imgs_est[i + 1], img_gt, ax=axes[:, 1 + i], title=titles[i], **dict_img
        )

    axes[2, -1].set_title(ds["id"].iloc[0])

    plt.savefig(os.path.join(path_figures, "comparision.png"))

    # ------------------------------------------------------------------------------
    # show statics
    metrics = []
    metrics_title = ["PSNR", "SSIM", "ZNCC"]
    num_metris = len(metrics_title)

    for m in metrics_title:
        metrics.append(
            pandas.read_excel(os.path.join(path_results, "metrics.xlsx"), sheet_name=m)
        )
    x_lables = metrics[0].columns[1:]

    nr, nc = 1, num_metris
    fig, axes = plt.subplots(
        nrows=nr, ncols=nc, figsize=(nc * 3, nr * 3), dpi=300, constrained_layout=True
    )
    colors = plt.cm.rainbow(np.linspace(0, 1, len(x_lables)))

    for i in range(num_metris):
        mean = np.mean(metrics[i].to_numpy()[:, 1:], axis=0)
        std = np.std(metrics[i].to_numpy()[:, 1:], axis=0)
        axes[i].bar(
            x=list(range(len(x_lables))),
            height=mean,
            yerr=std,
            label=x_lables,
            color=colors,
        )
        print(metrics_title[i], "| mean : ", mean, "\n  std : ", std)
        axes[i].set_title(metrics_title[i])
        axes[i].set_xticks([])
        axes[i].set_ylim([np.round(mean.min() - std.max() * 1.5, decimals=2), None])

    axes[0].legend(fontsize=4, loc="upper left")
    axes[0].set_ylabel(ds["id"].iloc[0])
    plt.savefig(os.path.join(path_figures, "comparision_metrics.png"))
