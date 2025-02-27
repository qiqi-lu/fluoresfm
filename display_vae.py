import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import os, pandas, torch
import utils.data as utils_data
import utils.evaluation as eva
from skimage.measure import profile_line

# ------------------------------------------------------------------------------
# parameters
# ------------------------------------------------------------------------------
params = {
    # "dataset_name": "CCPs_noise_level_9",
    # "path_dataset": "E:\qiqilu\datasets\BioSR\\transformed\CCPs\\test\channel_0\WF_noise_level_9",
    # "dataset_name": "CCPs_SIM",
    # "path_dataset": "E:\qiqilu\datasets\BioSR\\transformed\CCPs\\test\channel_0\SIM",
    # "path_index_file": "E:\qiqilu\datasets\BioSR\\transformed\CCPs\\test.txt",
    # "dataset_name": "ER_noise_level_6",
    # "path_dataset": "E:\qiqilu\datasets\BioSR\\transformed\ER\\test\channel_0\WF_noise_level_6",
    # "dataset_name": "ER_SIM",
    # "path_dataset": "E:\qiqilu\datasets\BioSR\\transformed\ER\\test\channel_0\SIM",
    # "path_index_file": "E:\qiqilu\datasets\BioSR\\transformed\ER\\test.txt",
    # "dataset_name": "MTs_noise_level_9",
    # "path_dataset": "E:\qiqilu\datasets\BioSR\\transformed\MTs\\test\channel_0\WF_noise_level_9",
    # "dataset_name": "MTs_SIM",
    # "path_dataset": "E:\qiqilu\datasets\BioSR\\transformed\MTs\\test\channel_0\SIM",
    # "path_index_file": "E:\qiqilu\datasets\BioSR\\transformed\MTs\\test.txt",
    # "dataset_name": "F_actin_noise_level_12",
    # "path_dataset": "E:\qiqilu\datasets\BioSR\\transformed\F_actin\\test\channel_0\WF_noise_level_12",
    "dataset_name": "F_actin_SIM",
    "path_dataset": "E:\qiqilu\datasets\BioSR\\transformed\F_actin\\test\channel_0\SIM",
    "path_index_file": "E:\qiqilu\datasets\BioSR\\transformed\F_actin\\test.txt",
    "scale_factor": 1,
}

print(params)
num_data = 10
index_show = 2

# ------------------------------------------------------------------------------
path_results = os.path.join("outputs", "vae", params["dataset_name"])
path_figures = os.path.join("outputs", "figures\imgtext\\vae")
utils_data.make_path(path_figures)

filenames = utils_data.read_txt(params["path_index_file"])

methods = ["vae_biosrall_sf_c4", "vae_biosrall_sf_c16"]
titles = ["VAE-c4", "VAE-c16"]
normalizer = utils_data.NormalizePercentile(p_low=0.0, p_high=0.9999)

# ------------------------------------------------------------------------------
# load results
# ------------------------------------------------------------------------------
print(f"- Number of test samples: {len(filenames)}")
imgs_gt, imgs_est, metrics = [], [], []

for idx in range(num_data):
    print(f"load results of sampel [{idx}] ...")
    # --------------------------------------------------------------------------
    # ground truth
    img_gt = utils_data.read_image(
        os.path.join(params["path_dataset"], filenames[idx])
    )[0]

    # interpolat low-resolution image
    if params["scale_factor"] > 1:
        img_gt = torch.nn.functional.interpolate(
            torch.tensor(img_gt[None, None]),
            scale_factor=(params["scale_factor"], params["scale_factor"]),
            mode="nearest",
        )[0, 0].numpy()

    img_gt = normalizer(img_gt)
    imgs_gt.append(np.squeeze(img_gt))

    # collect estimated images
    img_est_multi_meth = []

    # --------------------------------------------------------------------------
    # images form different methods
    for meth in methods:
        tmp = utils_data.read_image(os.path.join(path_results, meth, filenames[idx]))
        tmp = eva.linear_transform(img_true=img_gt, img_test=tmp)
        img_est_multi_meth.append(np.squeeze(tmp))

    imgs_est.append(img_est_multi_meth)

    # --------------------------------------------------------------------------
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
metrics = np.array(metrics)

# save all the metrics value into a excel file

with pandas.ExcelWriter(
    os.path.join(path_results, "metrics.xlsx"), engine="xlsxwriter"
) as writer:
    for i, metric_name in enumerate(["PSNR", "SSIM", "ZNCC"]):
        df = pandas.DataFrame(metrics[..., i], columns=methods)
        df.to_excel(writer, sheet_name=metric_name)


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
    patch_pos=(200, 400),
    patch_size=(100, 100),
):
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
            ax[2].imshow((img - img_gt)[i_slice], cmap="seismic", vmin=-vmax, vmax=vmax)
        if len(img.shape) == 2:
            ax[2].imshow((img - img_gt), cmap="seismic", vmin=-vmax, vmax=vmax)

        ssim = eva.SSIM(img_true=img_gt, img_test=img)
        psnr = eva.PSNR(img_true=img_gt, img_test=img)
        zncc = eva.ZNCC(img_true=img_gt, img_test=img)

        ax[2].set_title(
            "{:>.4f} | {:>.4f} | {:>.4f}".format(psnr, ssim, zncc),
            fontdict={"fontsize": 10},
        )


# ------------------------------------------------------------------------------
# show specific sample
nr, nc = 3, len(methods) + 1

fig, axes = plt.subplots(
    nrows=nr, ncols=nc, figsize=(nc * 3, nr * 3), dpi=300, constrained_layout=True
)
[ax.set_axis_off() for ax in axes.ravel()]
dict_img = {"vmin": 0, "vmax": 1.0, "cmap": "hot"}

show_slice(
    imgs_gt[index_show],
    imgs_gt[index_show],
    ax=axes[:, 0],
    title="GT",
    evaluate=False,
    **dict_img,
)

for i in range(len(methods)):
    show_slice(
        imgs_est[index_show][i],
        imgs_gt[index_show],
        ax=axes[:, 1 + i],
        title=titles[i],
        **dict_img,
    )

axes[2, 0].set_title(params["dataset_name"])

plt.savefig(os.path.join(path_figures, "comparision.png"))

# ------------------------------------------------------------------------------
# show statics
x_lables = titles
num_metris = metrics.shape[-1]

nr, nc = 1, num_metris
fig, axes = plt.subplots(
    nrows=nr, ncols=nc, figsize=(nc * 3, nr * 3), dpi=300, constrained_layout=True
)
colors = plt.cm.rainbow(np.linspace(0, 1, len(x_lables)))

metrics_title = ["PSNR", "SSIM", "ZNCC"]
for i in range(num_metris):
    mean = np.mean(metrics[..., i], axis=0)
    std = np.std(metrics[..., i], axis=0)
    axes[i].bar(
        x=list(range(len(x_lables))),
        height=mean,
        yerr=std,
        label=x_lables,
        color=colors,
    )
    print("mean : ", mean)
    print("std : ", std)
    axes[i].set_title(metrics_title[i])
    axes[i].set_xticks([])
    axes[i].set_ylim([np.round(mean.min() - std.max() * 1.5, decimals=2), None])

axes[0].legend(fontsize=4, loc="upper left")
axes[0].set_ylabel(params["dataset_name"])
# axes[0].set_ylim([15, None])
# axes[1].set_ylim([0.3, None])
# axes[2].set_ylim([mean.min() * 0.2, None])

plt.savefig(os.path.join(path_figures, "comparision_metrics.png"))

print(metrics)
