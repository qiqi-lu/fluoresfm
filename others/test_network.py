import numpy as np
import torch, os, json
import skimage.io as io
from torchvision.transforms import v2

from models.rcan3d import RCAN
from models.unet3d import UNet3D
from models.ddn3d import DenseDeconNet
from models.unet3d_sim import UNet3D_SIM
from models.teenet3d import TeeNet3D, TeeNet3D_Att, TeeNet3D_sq
from models.psfestimator import PSFEstimator
from models.teeresnet import TeeResNet

import utils.data as utils_data
import utils.evaluation as utils_eva


# ------------------------------------------------------------------------------
# parameters
# ------------------------------------------------------------------------------
params = {
    # "path_dataset_hr": "E:\qiqilu\datasets\SimuMix\gt",
    # "dataset_name": "SimuMix_457",
    # "path_dataset_lr": "E:\qiqilu\datasets\SimuMix\\raw\data_128_128_128_gauss_0_poiss_1_ratio_1_457",
    # "dataset_name": "SimuMix_528",
    # "path_dataset_lr": "E:\qiqilu\datasets\SimuMix\\raw\data_128_128_128_gauss_0_poiss_1_ratio_1_528",
    # "dataset_name": "SimuMix_404",
    # "path_dataset_lr": "E:\qiqilu\datasets\SimuMix\\raw\data_128_128_128_gauss_0_poiss_1_ratio_1_404",
    # "dataset_name": "SimuMix_751",
    # "path_dataset_lr": "E:\qiqilu\datasets\SimuMix\\raw\data_128_128_128_gauss_0_poiss_1_ratio_1_751",
    # "path_dataset_hr": "E:\qiqilu\datasets\BioSR\\transformed\MTs\\test\channel_0\sim",
    # "dataset_name": "Real_MTs",
    # "path_dataset_lr": "E:\qiqilu\datasets\BioSR\\transformed\MTs\\test\channel_0\wf_noise_level_9",
    # "path_index_file": "E:\qiqilu\datasets\BioSR\\transformed\MTs\\test.txt",
    "path_dataset_hr": "E:\qiqilu\datasets\RCAN3D\\transformed\C2S_MT\\test\channel_0\STED",
    "dataset_name": "Real_C2S_MT",
    "path_dataset_lr": "E:\qiqilu\datasets\RCAN3D\\transformed\C2S_MT\\test\channel_0\confocal",
    "path_index_file": "E:\qiqilu\datasets\RCAN3D\\transformed\C2S_MT\\test.txt",
    # "model_name": "ddn3d",
    # "path_model": "checkpoints\SimuMix\ddn3d_zncc_bs_1_lr_0.01\epoch_100_iter_10000.pt",
    # "model_name": "unet3d",
    # "path_model": "checkpoints\SimuMix\\unet3d_zncc_bs_1_lr_0.01\epoch_100_iter_10000.pt",
    # "model_name": "unet3d_sim",
    # "use_bn": False,
    # "path_model": "checkpoints\SimuMix\\unet3d_sim_zncc_bs_1_lr_0.001\epoch_100_iter_10000.pt",
    # "path_model": "checkpoints\SimuMix\\unet3d_sim_zncc_bs_1_lr_0.001_138_large\epoch_0_iter_14000.pt",
    # "path_model": "checkpoints\SimuMix\\unet3d_sim_zncc_bs_1_lr_0.001_mix_large_3_bnx\epoch_0_iter_14000.pt",
    # "suffix": "_138_large_mix3_bnx",
    "model_name": "rcan3d",
    # "path_model": "checkpoints\SimuMix\\rcan3d_zncc_bs_1_lr_0.001_mix_large_3_posx_55_f16\epoch_1_iter_16500.pt",
    "path_model": "checkpoints\Real\\rcan3d_mse_bs_1_lr_0.0001_real_posx_55_zpad0\epoch_0_iter_44500.pt",
    "suffix": "_138_large_mix3_55_f16",
    # "model_name": "teenet_sq",
    # "path_model": "checkpoints\SimuMix\\teenet_sq_zncc+mse_ss_bs_1_lr_0.001_mix_groups_resx_large_alter_bw\epoch_0_iter_6500.pt",
    # "path_model": "checkpoints\SimuMix\\teenet_sq_mse_ss_bs_1_lr_1e-05_138_groups_resx_large_alter_half_ft\epoch_10_iter_1000.pt",
    # "model_name": "teeresnet",
    # "path_model": "checkpoints\SimuMix\\teeresnet_mse_bs_1_lr_0.001_mix_large_3_pos_55_f16\epoch_1_iter_16500.pt",
    # "path_model": "checkpoints\SimuMix\\teeresnet_mse+mse_ss_norm_bs_4_lr_0.01_mix_large_3_pos_55_f16_trainpsf\epoch_1_iter_4125.pt",
    # "suffix": "_138_large_mix3_55_f16_sep",
    # "model_name": "psf_estimator+unet3d_sim",
    # "path_model": [
    #     "checkpoints\SimuMix\\psf_estimator_zncc_ss_bs_1_lr_0.001_138_large_gm\epoch_0_iter_7000.pt",
    #     "checkpoints\SimuMix\\unet3d_sim_zncc_bs_1_lr_0.001_138_large\epoch_0_iter_16000.pt",
    # ],
    # "suffix": "_zncc+mse_ss_groups_large_alter_fcn_bw_mix",
    # "suffix": "_mse_ss_groups_large_alter_fcn_half_ft",
    # "suffix": "_zncc_large_fcn_half",
    "num_features": 32,
    "num_groups": 5,
    "num_blocks": 5,
    "pos_out": False,
    "enable_standardize": True,
    "psf_model": "bw",
    "over_sampling": 2,
    "psf_size": (127, 127, 127),
    "kernel_norm": True,
    "enable_groups": True,
    "residual": False,
    "enable_constraints": True,
    "num_gaussian_model": 2,
    "center_one": False,
    "num_integral": 100,
    "pixel_size_z": 1,
    "in_channels": 1,
    "out_channels": 1,
    "p_low": 0.0,
    # "p_high": 0.9,
    "p_high": 0.999,
    "device": "cuda:0",
}

# idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
idxs = [0, 1, 2]

utils_data.print_dict(params)

# ------------------------------------------------------------------------------
device = torch.device(params["device"])
path_results = os.path.join(
    "outputs", params["dataset_name"], params["model_name"] + params["suffix"]
)
utils_data.make_path(path_results)


# ------------------------------------------------------------------------------
# datasets
# ------------------------------------------------------------------------------
transform = v2.Compose(
    [
        utils_data.NormalizePercentile(p_low=params["p_low"], p_high=params["p_high"]),
    ]
)


# dataset_test = utils_data.MicroDataset(
#     path_dataset_lr=params["path_dataset_lr"],
#     path_dataset_hr=params["path_dataset_hr"],
#     mode="test",
#     transform=transform,
# )


dataset_test = utils_data.RealFMDataset(
    path_index_file=params["path_index_file"],
    path_dataset_lr=params["path_dataset_lr"],
    path_dataset_hr=params["path_dataset_hr"],
    transform=transform,
    z_padding=0,
)

print("- Test dataset size:", dataset_test.__len__())

# ------------------------------------------------------------------------------
# model
# ------------------------------------------------------------------------------
if params["model_name"] == "ddn3d":
    model = DenseDeconNet(
        in_channels=params["in_channels"],
        out_channels=params["out_channels"],
        bias=False,
        scale_factor=1,
    ).to(device=device)

if params["model_name"] == "unet3d":
    model = UNet3D(
        in_channels=params["in_channels"],
        out_channels=params["out_channels"],
        use_bn=True,
        bias=True,
    ).to(device=device)

if params["model_name"] == "unet3d_sim":
    model = UNet3D_SIM(
        in_channels=params["in_channels"],
        out_channels=params["out_channels"],
        use_bn=params["use_bn"],
        bias=True,
    ).to(device=device)

if params["model_name"] == "rcan3d":
    print("[RCAN]")
    model = RCAN(
        in_channels=params["in_channels"],
        out_channels=params["out_channels"],
        num_residual_blocks=params["num_blocks"],
        num_residual_groups=params["num_groups"],
        enable_standardize=params["enable_standardize"],
        pos_out=params["pos_out"],
    ).to(device=device)

if params["model_name"] == "teenet":
    model = TeeNet3D(
        in_channels=params["in_channels"],
        out_channels=params["out_channels"],
        use_bn=True,
        bias=True,
        encoder_type="doubleconv",
        decoder_type="doubleconv",
        psf_size=params["psf_size"],
    ).to(device=device)

if params["model_name"] == "teenet_att":
    model = TeeNet3D_Att(
        in_channels=params["in_channels"],
        out_channels=params["out_channels"],
        use_bn=True,
        bias=True,
        encoder_type="doubleconv",
        decoder_type="doubleconv",
        psf_size=params["psf_size"],
        use_att=params["use_att"],
    ).to(device=device)

if params["model_name"] == "teenet_sq":
    print(f'- construct model ({params["model_name"]}) ... ')
    model = TeeNet3D_sq(
        in_channels=params["in_channels"],
        out_channels=params["out_channels"],
        use_bn=True,
        bias=True,
        encoder_type="doubleconv",
        decoder_type="doubleconv",
        psf_size=params["psf_size"],
        psf_model=params["psf_model"],
        kernel_norm=params["kernel_norm"],
        groups=params["enable_groups"],
        residual=params["residual"],
        over_sampling=params["over_sampling"],
        num_gauss_model=params["num_gaussian_model"],
        enable_constaints=params["enable_constraints"],
        num_integral=params["num_integral"],
        pixel_size_z=params["pixel_size_z"],
        center_one=params["center_one"],
    ).to(device=device)

if params["model_name"] == "teeresnet":
    model = TeeResNet(
        in_channels=params["in_channels"],
        out_channels=params["out_channels"],
        num_dims=3,
        num_features=params["num_features"],
        num_blocks=params["num_blocks"],
        num_groups=params["num_groups"],
        block_type="RCAB",
        channel_reduction=8,
        residual_scaling=1,
        enable_standardize=params["enable_standardize"],
        pos_out=params["pos_out"],
        psf_size=params["psf_size"],
        psf_model=params["psf_model"],
        kernel_norm=params["kernel_norm"],
        enable_groups=params["enable_groups"],
        over_sampling=params["over_sampling"],
        center_one=params["center_one"],
    ).to(device=device)

model_names = params["model_name"].split("+")
if len(model_names) == 2:
    model_psf = PSFEstimator(
        in_channels=params["in_channels"],
        psf_model=params["psf_model"],
        kernel_size=params["psf_size"],
        kernel_norm=params["kernel_norm"],
        num_gauss_model=params["num_gaussian_model"],
        enable_constraint=params["enable_constraints"],
        over_sampling=params["over_sampling"],
        center_one=params["center_one"],
        use_bn=True,
        bias=True,
    ).to(device=device)

    model_img = UNet3D_SIM(
        in_channels=params["in_channels"],
        out_channels=params["out_channels"],
        use_bn=True,
        bias=True,
    ).to(device=device)

# ------------------------------------------------------------------------------
# load model parameters
# ------------------------------------------------------------------------------

if len(model_names) == 2:
    if model_names[0] == "psf_estimator":
        model_psf.load_state_dict(
            torch.load(params["path_model"][0], map_location=device, weights_only=True)[
                "model_state_dict"
            ]
        )
        model_img.load_state_dict(
            torch.load(params["path_model"][1], map_location=device, weights_only=True)[
                "model_state_dict"
            ]
        )
    else:
        model_psf.load_state_dict(
            torch.load(params["path_model"], map_location=device, weights_only=True)[
                "model_state_dict_psf"
            ]
        )
        model_img.load_state_dict(
            torch.load(params["path_model"], map_location=device, weights_only=True)[
                "model_state_dict_img"
            ]
        )
else:
    model.load_state_dict(
        torch.load(params["path_model"], map_location=device, weights_only=True)[
            "model_state_dict"
        ]
    )
# model.eval()

# ------------------------------------------------------------------------------
# predict
# ------------------------------------------------------------------------------
for i in idxs:
    img_file_name = dataset_test[i]["file_name"]

    img_lr = dataset_test[i]["lr"][None].to(device)
    img_hr = dataset_test[i]["hr"][None].to(device)

    with torch.no_grad():
        if params["model_name"] in ["teenet", "teenet_att", "teenet_sq", "teeresnet"]:
            img_est, psf_est = model(img_lr)
        elif len(model_names) == 2:
            img_est = model_img(img_lr)
            psf_est = model_psf(img_lr)
        else:
            img_est = model(img_lr)

    print(f"- File Name: {img_file_name}")

    # --------------------------------------------------------------------------
    imgs_est = utils_eva.linear_transform(
        img_true=img_hr, img_test=img_est, axis=(2, 3, 4)
    )

    ssim_val = utils_eva.SSIM_tb(
        img_true=img_hr,
        img_test=imgs_est,
        data_range=None,
        version_wang=False,
    )
    psnr_val = utils_eva.PSNR_tb(img_true=img_hr, img_test=imgs_est, data_range=None)

    print(ssim_val, psnr_val)
    # --------------------------------------------------------------------------

    img_est = img_est.cpu().detach().numpy()

    io.imsave(
        os.path.join(path_results, img_file_name),
        arr=img_est[0, 0],
        check_contrast=False,
    )

    if (
        params["model_name"] in ["teenet", "teenet_att", "teenet_sq", "teeresnet"]
        or len(model_names) == 2
    ):
        psf_est = psf_est.cpu().detach().numpy()[0, 0]
        utils_data.make_path(os.path.join(path_results, "PSF"))
        io.imsave(
            os.path.join(path_results, "PSF", img_file_name),
            arr=psf_est,
            check_contrast=False,
        )
