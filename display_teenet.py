import numpy as np
import torch
import skimage.io as io
import matplotlib.pyplot as plt
import os
import utils.data as utils_data
import utils.evaluation as utils_eva
from skimage.measure import profile_line
from methods.convolution import convolution as conv_fft
import models.PSFmodels as PSFModel

import utils.loss_functions as loss_func

# ------------------------------------------------------------------------------
# parameters
# ------------------------------------------------------------------------------
params = {
    "method": "teeresnet_138_large_mix3_55_f16_sep",
    # "method": "teenet_sq_zncc+mse_ss_groups_large_alter_fcn_half",
    # "method": "teenet_sq_zncc+mse_ss_groups_large_alter_fcn_bw_mix",
    # "method": "teenet_sq_zncc+mse_ss_groups_large_alter_fcn_half_10",
    # "method": "teenet_sq_mse_ss_groups_large_alter_fcn_half_ft",
    # "method": "teenet_sq_zncc+mse_ss_groups_large_alter_fcn_bw",
    # "method": "teenet_sq_zncc+mse_ss_groups_large_alter_fcn_gm",
    # "method": "psf_estimator+unet3d_sim_zncc_large_fcn_half",
    # "method": "psf_estimator+unet3d_sim_zncc_large_fcn_bw",
    # "method": "psf_estimator+unet3d_sim_zncc_large_fcn_gm",
    # "dataset_name": "SimuMix_457",
    # "path_dataset_lr": "E:\qiqilu\datasets\SimuMix\\raw\data_128_128_128_gauss_0_poiss_1_ratio_1_457",
    # "dataset_name": "SimuMix_528",
    # "path_dataset_lr": "E:\qiqilu\datasets\SimuMix\\raw\data_128_128_128_gauss_0_poiss_1_ratio_1_528",
    # "dataset_name": "SimuMix_404",
    # "path_dataset_lr": "E:\qiqilu\datasets\SimuMix\\raw\data_128_128_128_gauss_0_poiss_1_ratio_1_404",
    "dataset_name": "SimuMix_751",
    "path_dataset_lr": "E:\qiqilu\datasets\SimuMix\\raw\data_128_128_128_gauss_0_poiss_1_ratio_1_751",
    "path_dataset_hr": "E:\qiqilu\datasets\SimuMix\gt",
    # "dataset_name": "SimuMix_large",
    # "path_dataset_lr": "E:\qiqilu\datasets\SimuMix_large\\raw\data_128_128_128_gauss_0_poiss_1_ratio_1",
    # "path_dataset_hr": "E:\qiqilu\datasets\SimuMix_large\gt",
    "p_low": 0.0,
    "p_high": 0.9,
}

idx = 2

# ------------------------------------------------------------------------------
utils_data.print_dict(params)

path_results = os.path.join("outputs", params["dataset_name"])
path_figures = os.path.join("outputs", "figures")
utils_data.make_path(path_figures)
norm = utils_data.NormalizePercentile(p_low=params["p_low"], p_high=params["p_high"])
filenames = utils_data.read_txt(os.path.join(params["path_dataset_lr"], "test.txt"))

# ------------------------------------------------------------------------------
# load results
# ------------------------------------------------------------------------------
print(f"- Number of test samples: {len(filenames)}")

img_hr = io.imread(os.path.join(params["path_dataset_hr"], "images", filenames[idx]))
img_lr = io.imread(os.path.join(params["path_dataset_lr"], "images", filenames[idx]))

if os.path.exists(os.path.join(params["path_dataset_lr"], "psf.tif")):
    psf_lr = io.imread(os.path.join(params["path_dataset_lr"], "psf.tif"))
else:
    psf_lr = io.imread(os.path.join(params["path_dataset_lr"], "PSF", filenames[idx]))
    print("check")

img_hr_reblur_lr = conv_fft(img_hr, psf_lr, padding_mode="constant")

img_lr, img_hr = norm(img_lr), norm(img_hr)

img_est = io.imread(os.path.join(path_results, params["method"], filenames[idx]))
img_est = utils_eva.linear_transform(img_true=img_hr, img_test=img_est)
psf_est = io.imread(os.path.join(path_results, params["method"], "PSF", filenames[idx]))

img_hr_reblur_est = conv_fft(img_hr, psf_est, padding_mode="constant")

img_hr_reblur_lr = utils_eva.linear_transform(
    img_true=img_hr, img_test=img_hr_reblur_lr
)
img_lr = utils_eva.linear_transform(img_true=img_hr, img_test=img_lr)
img_hr_reblur_est = utils_eva.linear_transform(
    img_true=img_hr, img_test=img_hr_reblur_est
)

img_est_reblur = conv_fft(img_est, psf_est, padding_mode="constant")
img_est_reblur = utils_eva.linear_transform(img_true=img_hr, img_test=img_est_reblur)

# ------------------------------------------------------------------------------
# init
# psf_init = PSFModel.GaussianModel(
#     kernel_size=(127, 127, 127), kernel_norm=True, num_params=2
# )(torch.tensor([0.75, 2.5]).reshape(shape=(1, 1, 2)))[0, 0]

psf_init = PSFModel.BWModel(
    kernel_size=(127, 127, 127),
    kernel_norm=True,
    num_integral=100,
    over_sampling=1,
    # )(torch.tensor([2.6, 0.9]).reshape(shape=(1, 1, 2)))[0, 0]
)(torch.tensor([528 / 100 / 3 * 2, 0.933]).reshape(shape=(1, 1, 2)))[0, 0]

psf_init = psf_init.clone().detach().numpy()
img_reblur_init = conv_fft(img_hr, psf_init, padding_mode="constant")
img_reblur_init = utils_eva.linear_transform(img_true=img_hr, img_test=img_reblur_init)

print("LR", img_lr.shape, "HR", img_hr.shape)
print(f"IMG: {img_est.shape}, PSF: {psf_est.shape}")

# ------------------------------------------------------------------------------
# display
# ------------------------------------------------------------------------------
# PSF
nr, nc = 2, 4
fig, axes = plt.subplots(
    nrows=nr, ncols=nc, figsize=(nc * 3, nr * 3), dpi=300, constrained_layout=True
)
for ax in [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]:
    ax.set_axis_off()

dict_psf = {"vmin": 0, "vmax": 1.0, "cmap": "hot"}
dict_psf_profile = {
    "linewidth": 1.0,
}

psf_est_crop = utils_data.center_crop(psf_est, size=[31, 31, 31], verbose=True)
psf_lr_crop = utils_data.center_crop(psf_lr, size=[31, 31, 31], verbose=True)
psf_init_crop = utils_data.center_crop(psf_init, size=[31, 31, 31], verbose=True)

i_slice = psf_lr_crop.shape[0] // 2
i_row = psf_lr_crop.shape[1] // 2
i_col = psf_lr_crop.shape[2] // 2


def max_norm(x):
    return x / x.max()


axes[0, 0].imshow(max_norm(psf_lr_crop[i_slice]), **dict_psf)
axes[1, 0].imshow(max_norm(psf_lr_crop[:, i_row]), **dict_psf)

axes[0, 1].imshow(max_norm(psf_est_crop[i_slice]), **dict_psf)
axes[1, 1].imshow(max_norm(psf_est_crop[:, i_row]), **dict_psf)

axes[0, 2].imshow(max_norm(psf_init_crop[i_slice]), **dict_psf)
axes[1, 2].imshow(max_norm(psf_init_crop[:, i_row]), **dict_psf)

axes[0, 3].plot(
    max_norm(psf_lr_crop[i_slice, i_row]), label="GT", color="red", **dict_psf_profile
)
axes[0, 3].plot(max_norm(psf_est_crop[i_slice, i_row]), label="EST", **dict_psf_profile)
axes[0, 3].set_ylim([0.0, 1.1])
axes[0, 3].set_title("PSF profiel (x)")

axes[1, 3].plot(
    max_norm(psf_lr_crop[:, i_row, i_col]), label="GT", color="red", **dict_psf_profile
)
axes[1, 3].plot(
    max_norm(psf_est_crop[:, i_row, i_col]), label="EST", **dict_psf_profile
)
axes[0, 3].plot(
    max_norm(psf_init_crop[i_slice, i_row]),
    label="Init",
    color="black",
    **dict_psf_profile,
)
axes[1, 3].plot(
    max_norm(psf_init_crop[:, i_row, i_col]),
    label="Init",
    color="black",
    **dict_psf_profile,
)
axes[1, 3].set_ylim([0.0, 1.1])
axes[1, 3].set_title("PSF profiel (z)")
axes[0, 3].legend()
axes[1, 3].legend()

plt.savefig(os.path.join(path_figures, "compare_psf.png"))

# ------------------------------------------------------------------------------
# Re-blurred image
nr, nc = 4, 5
fig, axes = plt.subplots(
    nrows=nr, ncols=nc, figsize=(nc * 3, nr * 3), dpi=300, constrained_layout=True
)
[ax.set_axis_off() for ax in axes.ravel()]
i_slice = 63
dict_img = {"vmin": 0, "vmax": 2.5, "cmap": "hot"}

axes[0, 0].imshow(img_hr[i_slice], **dict_img)
axes[0, 0].set_title("GT (x,y)")
axes[1, 0].imshow(img_hr[:, i_slice], **dict_img)
axes[1, 0].set_title("GT (z,x)")

axes[2, 0].imshow(img_est[i_slice], **dict_img)
axes[2, 0].set_title("EST (x,y)")
axes[3, 0].imshow(img_est[:, i_slice], **dict_img)
axes[3, 0].set_title("EST (z,x)")

axes[2, 1].imshow(img_est_reblur[i_slice], **dict_img)
axes[2, 1].set_title("EST_reblur")
axes[3, 1].imshow(img_est_reblur[:, i_slice], **dict_img)
axes[3, 1].set_title(
    loss_func.ZNCC(
        torch.tensor(img_lr[None, None]), torch.tensor(img_est_reblur[None, None])
    ).numpy()
)

axes[0, 1].imshow(img_hr_reblur_lr[i_slice], **dict_img)
axes[0, 1].set_title("GT_reblur_lr")
axes[1, 1].imshow(img_hr_reblur_lr[:, i_slice], **dict_img)
axes[1, 1].set_title(
    loss_func.ZNCC_ss(
        torch.tensor(img_lr[None, None]),
        torch.tensor(img_hr[None, None]),
        torch.tensor(psf_lr[None, None]),
    ).numpy()
)

axes[0, 2].imshow(img_lr[i_slice], **dict_img)
axes[0, 2].set_title("LR")
axes[1, 2].imshow(img_lr[:, i_slice], **dict_img)


axes[0, 3].imshow(img_hr_reblur_est[i_slice], **dict_img)
axes[0, 3].set_title("GT_reblur_est")
axes[1, 3].imshow(img_hr_reblur_est[:, i_slice], **dict_img)
axes[1, 3].set_title(
    loss_func.ZNCC_ss(
        torch.tensor(img_lr[None, None]),
        torch.tensor(img_hr[None, None]),
        torch.tensor(psf_est[None, None]),
    ).numpy()
)

axes[0, 4].imshow(img_reblur_init[i_slice], **dict_img)
axes[0, 4].set_title("GT_reblur_init")
axes[1, 4].imshow(img_reblur_init[:, i_slice], **dict_img)
axes[1, 4].set_title(
    loss_func.ZNCC_ss(
        torch.tensor(img_lr[None, None]),
        torch.tensor(img_hr[None, None]),
        torch.tensor(psf_init[None, None]),
    ).numpy()
)

plt.savefig(os.path.join(path_figures, "re-blurred.png"))


print(np.mean((max_norm(psf_lr) - max_norm(psf_est)) ** 2))
print(np.mean((max_norm(psf_lr) - max_norm(psf_init)) ** 2))

# ------------------------------------------------------------------------------
