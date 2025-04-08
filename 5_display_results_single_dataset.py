"""
Visualization of the results for one sample in one dataset.
- single sample
- single dataset
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os, pandas
import utils.data as utils_data
import utils.evaluation as eva
import numpy as np
import utils.plot as utils_plot

# ------------------------------------------------------------------------------
# parameters
# ------------------------------------------------------------------------------
params = {
    "id_datasets": [
        # ("biosr-cpp-sr-2", 1, (300, 300), 256),
        # ("biosr-er-sr-3", 1, (300, 300), 256),
        # ("biosr-mt-sr-2", 1, (400, 300), 256),
        # ("biosr-actin-sr-2", 1, (400, 300), 256),
        # ("deepbacs-sim-ecoli-sr", 1, (550, 700), 256),
        # ("deepbacs-sim-saureus-sr", 1, (280, 350), 128),
        # ("w2s-c0-sr-4", 2, (300, 300), 256),
        # ("w2s-c0-sr-5", 2, (300, 300), 256),
        # ("w2s-c0-sr-6", 2, (300, 300), 256),
        # ("w2s-c2-sr-2", 2, (100, 100), 256),
        # ("w2s-c2-sr-4", 2, (100, 100), 256),
        # ("w2s-c2-sr-5", 2, (100, 100), 256),
        # ("srcaco2-h2b-sr-2", 2, (400, 300), 256),
        # ("srcaco2-survivin-sr-2", 2, (400, 300), 256),
        # ("srcaco2-tubulin-sr-2", 2, (400, 300), 256),
        # ("vmsim-mito-sr", 1, (400, 300), 300),
        # ("vmsim-er-sr", 1, (400, 300), 256),
        # ("vmsim-mito-dcv", 1, (150, 150), 200),
        # ("vmsim-mito-dcv", 2, (150, 150), 200),
        # ("vmsim-er-dcv", 1, (150, 150), 256),
        # ("vmsim-er-dcv", 2, (150, 150), 256),
        # ("sim-microtubule-2d-patch-dcv", 1, (150, 150), 256),
        # ("vmsim488-bead-patch-dcv", 1, (150, 150), 256),
        # ("vmsim3-mito-dcv", 3, (150, 150), 256),
        # ("vmsim5-mito-dcv", 0, (150, 150), 256),
        # ("biotisr-mito-dcv-1", 1, (150, 150), 256),
        # ("biotisr-mito-dcv-2", 1, (150, 150), 256),
        # ("biotisr-mito-dcv-3", 1, (150, 150), 256),
        ("rcan3d-dn-er-dn", 0, (400, 400), 256),
        ("rcan3d-dn-golgi-dn", 0, (400, 400), 256),
    ],
    "path_dataset_test": "dataset_test.xlsx",
    # "path_results": "outputs\\unet_c\\internal_dataset",
    "path_results": "outputs\\unet_c\\external_dataset",
    "path_figure": "outputs\\figures\\images",
    "methods": (
        # ("CARE:biosr-sr-cpp", "care_biosr_sr_cpp"),
        # ("CARE:biosr-sr-actin", "care_biosr_sr_actin"),
        # ("CARE:biosr-sr", "care_biosr_sr"),
        # ("CARE:sr", "care_sr"),
        # ("CARE:dcv", "care_dcv"),
        # ("CARE:biosr-dcv", "care_biosr_dcv"),
        # ("DFCAN:biosr-sr-2", "dfcan_biosr_sr_2"),
        # ("DFCAN:sr-2", "dfcan_sr_2"),
        # ("DFCAN:dcv", "dfcan_dcv"),
        # ("UNet-uc:sr", "unet_sd_c_sr_crossx"),
        # ("UNet-c:sr", "unet_sd_c_sr"),
        # ("UniFMIR:all", "unifmir_all"),
        ("UNet-uc:all", "unet_sd_c_all_cross"),
        ("UNet-c:all", "unet_sd_c_all"),
        ("UNet-uc:all-newnorm", "unet_sd_c_all_crossx_newnorm"),
        ("UNet-c:all-newnorm", "unet_sd_c_all_newnorm"),
    ),
    "p_low": 0.001,
    "p_high": 0.999,
}

# ------------------------------------------------------------------------------
# convert to linux path
if os.name == "posix":
    params["path_results"] = utils_data.win2linux(params["path_results"])
    params["path_figure"] = utils_data.win2linux(params["path_figure"])

titles = [i[0] for i in params["methods"]]
methods = [i[1] for i in params["methods"]]

normalizer = lambda x: utils_data.normalization(
    x, p_low=params["p_low"], p_high=params["p_high"], clip=False
)

data_info_frame = pandas.read_excel(params["path_dataset_test"])
num_methods = len(methods)
print("Number of methods shown:", num_methods)


# ------------------------------------------------------------------------------
def show_slice(
    ax,
    img_gt,
    img,
    title=None,
    vmin=None,
    vmax=None,
    cmap=None,
    patch_pos=(100, 100),
    patch_size=256,
    evaluate=True,
    boader_size=0,
):
    img_gt = utils_plot.crop_edge(img_gt, boader_size)
    img = utils_plot.crop_edge(img, boader_size)

    img_shape = img.shape
    patch = img[
        patch_pos[0] : patch_pos[0] + patch_size,
        patch_pos[1] : patch_pos[1] + patch_size,
    ]
    patch_gt = img_gt[
        patch_pos[0] : patch_pos[0] + patch_size,
        patch_pos[1] : patch_pos[1] + patch_size,
    ]

    ax[0].text(30, 60, title, fontsize=10, color="white")
    # full size image
    ax[0].imshow(img, vmin=vmin, vmax=vmax, cmap=cmap)
    # rectangle window
    rect = patches.Rectangle(
        (patch_pos[1], patch_pos[0]),
        patch_size,
        patch_size,
        linewidth=1,
        edgecolor="white",
        facecolor="none",
    )
    ax[0].add_patch(rect)
    # patch
    ax[1].imshow(patch, vmin=vmin, vmax=vmax, cmap=cmap)

    if evaluate:
        # error image
        ax[2].imshow((patch - patch_gt), cmap="seismic", vmin=-vmax, vmax=vmax)
        # metrics
        psnr = eva.PSNR(img_true=img_gt, img_test=img)
        ssim = eva.SSIM(img_true=img_gt, img_test=img)
        zncc = eva.ZNCC(img_true=img_gt, img_test=img)
        ax[0].text(
            30,
            img_shape[0] - 30,
            "{:>.4f} | {:>.4f} | {:>.4f}".format(psnr, ssim, zncc),
            fontsize=10,
            color="white",
        )


# ------------------------------------------------------------------------------
for idx, dataset_setting in enumerate(params["id_datasets"]):
    # parse settings
    id_dataset, id_sample, patch_pos, patch_size = dataset_setting

    # --------------------------------------------------------------------------
    print("-" * 80)
    print("Dataset:", id_dataset)
    # save to
    path_figures = os.path.join(params["path_figure"], id_dataset)
    utils_data.make_path(path_figures)

    # load dataset information
    ds = data_info_frame[data_info_frame["id"] == id_dataset].iloc[0]
    sample_filenames = utils_data.read_txt(ds["path_index"])
    path_results = os.path.join(params["path_results"], ds["id"])

    sample_name = sample_filenames[id_sample]
    print(f"load results of {sample_name}")

    # --------------------------------------------------------------------------
    # ground truth image
    img_gt = utils_data.read_image(os.path.join(ds["path_hr"], sample_name))
    img_gt = utils_data.interp_sf(img_gt, sf=ds["sf_hr"])[0]
    img_gt = normalizer(img_gt)

    # --------------------------------------------------------------------------
    imgs_est = []  # collect estimated images

    # raw image
    img_raw = utils_data.read_image(os.path.join(ds["path_lr"], sample_name))
    img_raw = utils_data.utils_data.interp_sf(img_raw, sf=ds["sf_lr"])[0]
    img_raw = eva.linear_transform(img_true=img_gt, img_test=img_raw)
    imgs_est.append(img_raw)

    # images form different methods
    for meth in methods:
        tmp = utils_data.read_image(os.path.join(path_results, meth, sample_name))[0]
        tmp = eva.linear_transform(img_true=img_gt, img_test=tmp)
        imgs_est.append(tmp)

    # ------------------------------------------------------------------------------
    # show
    nr, nc = 3, num_methods + 2
    fig, axes = plt.subplots(
        nrows=nr, ncols=nc, figsize=(nc * 3, nr * 3), dpi=300, constrained_layout=True
    )
    [ax.set_axis_off() for ax in axes.ravel()]
    dict_img = {
        "vmin": 0,
        "vmax": 1.0,
        "cmap": "hot",
        "patch_pos": patch_pos,
        "patch_size": patch_size,
    }
    show_slice(axes[:, -1], img_gt, img_gt, title="GT", evaluate=False, **dict_img)
    show_slice(axes[:, 0], img_gt, imgs_est[0], title="RAW", **dict_img)

    for i in range(num_methods):
        show_slice(axes[:, 1 + i], img_gt, imgs_est[i + 1], title=titles[i], **dict_img)
    axes[0, 0].text(30, 120, id_dataset, color="white")
    # axes[2, -1].set_title(ds["id"])
    plt.savefig(
        os.path.join(path_figures, f"comparision_{sample_name.split('.')[0]}.png")
    )
