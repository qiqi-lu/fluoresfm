import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import os
import utils.data as utils_data
import utils.evaluation as eva
from skimage.measure import profile_line

# ------------------------------------------------------------------------------
# parameters
# ------------------------------------------------------------------------------
params = {
    "dataset_name": "SimuMix_457",
    "path_dataset_raw": "E:\qiqilu\datasets\SimuMix\\raw\data_128_128_128_gauss_0_poiss_1_ratio_1_457",
    # "dataset_name": "SimuMix_528",
    # "path_dataset_raw": "E:\qiqilu\datasets\SimuMix\\raw\data_128_128_128_gauss_0_poiss_1_ratio_1_528",
    "path_dataset_gt": "E:\qiqilu\datasets\SimuMix\gt",
}

num_data = 3
index_show = 0

# ------------------------------------------------------------------------------
path_results = os.path.join("outputs", params["dataset_name"])
path_figures = os.path.join("outputs", "figures")
utils_data.make_path(path_figures)

filenames = utils_data.read_txt(os.path.join(params["path_dataset_raw"], "test.txt"))

methods = [
    # "traditional",
    # "ddn3d",
    # "unet3d",
    # "unet3d_sim",
    "unet3d_sim_138_large_mix3",
    "rcan3d_138_large_mix3_55_f16",
    "teeresnet_138_large_mix3_55_f16_sep",
    # "teenet_sq_zncc+mse_ss_groups_large_alter_fcn_half",
    # "teenet_sq_zncc+mse_ss_groups_large_alter_fcn_bw",
    # "psf_estimator+unet3d_sim_zncc_ss_large_half",
    # "psf_estimator+unet3d_sim_zncc_ss_large_bw",
]
titles = [
    # "traditional",
    # "ddn3d",
    # "unet3d",
    # "UNet3D_sim",
    "UNet3D",
    "RCAN3D",
    "TeeResNet",
    # "teenet_sq_half",
    # "teenet_sq_bw",
]
norm = utils_data.NormalizePercentile(p_low=0.0, p_high=0.9)

# ------------------------------------------------------------------------------
# load results
# ------------------------------------------------------------------------------
print(f"- Number of test samples: {len(filenames)}")

img_gt_multi_sample = []
img_est_multi_sample = []
metrics_multi_sample = []

for idx in range(num_data):
    print(f"load results of sampel [{idx}] ...")
    img_gt = utils_data.read_image(
        os.path.join(params["path_dataset_gt"], "images", filenames[idx])
    )
    img_gt = norm(img_gt)
    img_gt_multi_sample.append(img_gt)

    # collect estimated images
    img_est_multi_meth = []
    img_raw = utils_data.read_image(
        os.path.join(params["path_dataset_raw"], "images", filenames[idx])
    )
    img_raw = norm(img_raw)
    img_raw = eva.linear_transform(img_true=img_gt, img_test=img_raw)
    img_est_multi_meth.append(img_raw)

    for meth in methods:
        tmp = utils_data.read_image(os.path.join(path_results, meth, filenames[idx]))
        tmp = eva.linear_transform(img_true=img_gt, img_test=tmp)
        img_est_multi_meth.append(tmp)

    img_est_multi_sample.append(img_est_multi_meth)

    # evaluate all results
    metrics_multi_meth = []
    for img in img_est_multi_meth:
        metrics = []
        psnr = eva.PSNR(img_true=img_gt, img_test=img, data_range=None)
        ssim = eva.SSIM(
            img_true=img_gt, img_test=img, data_range=None, version_wang=False
        )
        zncc = eva.ZNCC(img_true=img_gt, img_test=img)
        metrics.extend([psnr, ssim, zncc])
        metrics_multi_meth.append(metrics)
    metrics_multi_sample.append(metrics_multi_meth)

metrics_multi_sample = np.array(metrics_multi_sample)


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
):
    if i_slice is None:
        i_slice = img.shape[0] // 2

    assert i_slice >= 0 and i_slice <= img.shape[0], "ERROR: Out of slice range."

    ax[0].imshow(img[i_slice], vmin=vmin, vmax=vmax, cmap=cmap)
    ax[0].set_title(title, fontdict={"fontsize": 10})
    if evaluate:
        ax[1].imshow((img - img_gt)[i_slice], cmap="seismic", vmin=-vmax, vmax=vmax)
        ssim = eva.SSIM(img_true=img_gt, img_test=img)
        psnr = eva.PSNR(img_true=img_gt, img_test=img)
        zncc = eva.ZNCC(img_true=img_gt, img_test=img)
        ax[1].set_title(
            "{:>.4f} | {:>.4f} | {:>.4f}".format(ssim, psnr, zncc),
            fontdict={"fontsize": 10},
        )


# ------------------------------------------------------------------------------
# show specific sample
nr, nc = 2, len(methods) + 2
fig, axes = plt.subplots(
    nrows=nr, ncols=nc, figsize=(nc * 3, nr * 3), dpi=300, constrained_layout=True
)
[ax.set_axis_off() for ax in axes.ravel()]
dict_img = {"vmin": 0, "vmax": 2.5, "cmap": "hot"}

show_slice(
    img_gt_multi_sample[index_show],
    img_gt_multi_sample[index_show],
    ax=axes[:, 0],
    title="GT",
    evaluate=False,
    **dict_img,
)
show_slice(
    img_est_multi_sample[index_show][0],
    img_gt_multi_sample[index_show],
    ax=axes[:, 1],
    title="RAW",
    **dict_img,
)

for i in range(len(methods)):
    show_slice(
        img_est_multi_sample[index_show][i + 1],
        img_gt_multi_sample[index_show],
        ax=axes[:, 2 + i],
        title=titles[i],
        **dict_img,
    )

plt.savefig(os.path.join(path_figures, "comparision.png"))

# ------------------------------------------------------------------------------
# show statics
x_lables = ["raw"]
x_lables = x_lables + methods
num_metris = metrics_multi_sample.shape[-1]
nr, nc = 1, num_metris
fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(nc * 3, nr * 3), dpi=300)

colors = plt.cm.rainbow(np.linspace(0, 1, len(x_lables)))
for i in range(num_metris):
    mean = np.mean(metrics_multi_sample[..., i], axis=0)
    std = np.std(metrics_multi_sample[..., i], axis=0)
    axes[i].bar(
        x=list(range(len(x_lables))),
        height=mean,
        yerr=std,
        label=x_lables,
        color=colors,
    )
    print("Mean : ", mean, "std : ", std)
axes[0].legend(fontsize=4, loc="lower right")
axes[0].set_ylim([24, 32.0])
axes[1].set_ylim([0.7, 0.95])
axes[2].set_ylim([0.9, 1.0])

plt.savefig(os.path.join(path_figures, "comparision_metrics.png"))
