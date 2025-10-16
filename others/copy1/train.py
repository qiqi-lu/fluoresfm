import torch, os, time, tqdm, json
import numpy as np
from torchinfo import summary
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2

from models.unet3d import UNet3D
from models.ddn3d import DenseDeconNet
from models.unet3d_sim import UNet3D_SIM
from models.teenet3d import TeeNet3D, TeeNet3D_Att, TeeNet3D_sq
from models.psfestimator import PSFEstimator
from models.rcan3d import RCAN
from models.teeresnet import TeeResNet

import utils.data as utils_data
import utils.evaluation as utils_eva
import utils.optim as utils_optim
import utils.loss_functions as utils_loss

# ------------------------------------------------------------------------------
# parameters
# ------------------------------------------------------------------------------
# wavelength = [
#     404,
#     419,
#     420,
#     421,
#     424,
#     434,
#     435,
#     439,
#     441,
#     442,
#     444,
#     447,
#     448,
#     450,
#     455,
#     457,
#     458,
#     461,
#     463,
#     464,
#     470,
#     474,
#     479,
#     480,
#     482,
#     484,
#     486,
#     495,
#     496,
#     501,
#     502,
#     503,
#     504,
#     505,
#     506,
#     507,
#     509,
#     511,
#     512,
#     513,
#     514,
#     515,
#     516,
#     517,
#     518,
#     519,
#     520,
#     521,
#     522,
#     523,
#     524,
#     525,
#     526,
#     527,
#     529,
#     530,
#     531,
#     532,
#     533,
#     534,
#     537,
#     538,
#     540,
#     541,
#     542,
#     546,
#     547,
#     548,
#     549,
#     553,
#     554,
#     559,
#     562,
#     565,
#     567,
#     568,
#     569,
#     570,
#     571,
#     573,
#     574,
#     575,
#     576,
#     577,
#     578,
#     579,
#     580,
#     581,
#     582,
#     584,
#     585,
#     589,
#     590,
#     591,
#     593,
#     595,
#     596,
#     599,
#     602,
#     603,
#     604,
#     605,
#     608,
#     610,
#     613,
#     614,
#     616,
#     617,
#     618,
#     619,
#     621,
#     627,
#     629,
#     630,
#     631,
#     636,
#     637,
#     646,
#     649,
#     657,
#     659,
#     660,
#     661,
#     664,
#     666,
#     668,
#     669,
#     670,
#     673,
#     674,
#     691,
#     692,
#     702,
#     703,
#     706,
#     719,
#     720,
#     751,
# ]
# path_dataset_lr = []
# path_dataset_hr = []
# for wl in wavelength:
#     path_dataset_lr.append(
#         "E:\qiqilu\datasets\SimuMix\\raw\data_128_128_128_gauss_0_poiss_1_ratio_1_"
#         + str(wl)
#     )
#     path_dataset_hr.append("E:\qiqilu\datasets\SimuMix\gt")

# path_dataset_lr_val = [
#     "E:\qiqilu\datasets\SimuMix\\raw\data_128_128_128_gauss_0_poiss_1_ratio_1_457"
# ]
# path_dataset_hr_val = ["E:\qiqilu\datasets\SimuMix\gt"]

# ------------------------------------------------------------------------------
# train
# path_dataset_lr = [
#     "E:\qiqilu\datasets\SimuMix_large\\raw\data_128_128_128_gauss_0_poiss_1_ratio_1_mix3"
# ]
# path_dataset_hr = ["E:\qiqilu\datasets\SimuMix_large\gt"]

# path_dataset_lr_val = [
#     "E:\qiqilu\datasets\SimuMix_large\\raw\data_128_128_128_gauss_0_poiss_1_ratio_1_mix3"
# ]
# path_dataset_hr_val = ["E:\qiqilu\datasets\SimuMix_large\gt"]

# ------------------------------------------------------------------------------
# train (real)
path_dataset_lr = [
    "E:\qiqilu\datasets\BioSR\\transformed\CCPs\\train\channel_0\WF_noise_level_1",
    "E:\qiqilu\datasets\BioSR\\transformed\CCPs\\train\channel_0\WF_noise_level_2",
    "E:\qiqilu\datasets\BioSR\\transformed\CCPs\\train\channel_0\WF_noise_level_3",
    "E:\qiqilu\datasets\BioSR\\transformed\CCPs\\train\channel_0\WF_noise_level_4",
    "E:\qiqilu\datasets\BioSR\\transformed\CCPs\\train\channel_0\WF_noise_level_5",
    "E:\qiqilu\datasets\BioSR\\transformed\CCPs\\train\channel_0\WF_noise_level_6",
    "E:\qiqilu\datasets\BioSR\\transformed\CCPs\\train\channel_0\WF_noise_level_7",
    "E:\qiqilu\datasets\BioSR\\transformed\CCPs\\train\channel_0\WF_noise_level_8",
    "E:\qiqilu\datasets\BioSR\\transformed\CCPs\\train\channel_0\WF_noise_level_9",
    "E:\qiqilu\datasets\BioSR\\transformed\ER\\train\channel_0\WF_noise_level_1",
    "E:\qiqilu\datasets\BioSR\\transformed\ER\\train\channel_0\WF_noise_level_2",
    "E:\qiqilu\datasets\BioSR\\transformed\ER\\train\channel_0\WF_noise_level_3",
    "E:\qiqilu\datasets\BioSR\\transformed\ER\\train\channel_0\WF_noise_level_4",
    "E:\qiqilu\datasets\BioSR\\transformed\ER\\train\channel_0\WF_noise_level_5",
    "E:\qiqilu\datasets\BioSR\\transformed\ER\\train\channel_0\WF_noise_level_6",
    "E:\qiqilu\datasets\BioSR\\transformed\F_actin\\train\channel_0\WF_noise_level_1",
    "E:\qiqilu\datasets\BioSR\\transformed\F_actin\\train\channel_0\WF_noise_level_2",
    "E:\qiqilu\datasets\BioSR\\transformed\F_actin\\train\channel_0\WF_noise_level_3",
    "E:\qiqilu\datasets\BioSR\\transformed\F_actin\\train\channel_0\WF_noise_level_4",
    "E:\qiqilu\datasets\BioSR\\transformed\F_actin\\train\channel_0\WF_noise_level_5",
    "E:\qiqilu\datasets\BioSR\\transformed\F_actin\\train\channel_0\WF_noise_level_6",
    "E:\qiqilu\datasets\BioSR\\transformed\F_actin\\train\channel_0\WF_noise_level_7",
    "E:\qiqilu\datasets\BioSR\\transformed\F_actin\\train\channel_0\WF_noise_level_8",
    "E:\qiqilu\datasets\BioSR\\transformed\F_actin\\train\channel_0\WF_noise_level_9",
    "E:\qiqilu\datasets\BioSR\\transformed\F_actin\\train\channel_0\WF_noise_level_10",
    "E:\qiqilu\datasets\BioSR\\transformed\F_actin\\train\channel_0\WF_noise_level_11",
    "E:\qiqilu\datasets\BioSR\\transformed\F_actin\\train\channel_0\WF_noise_level_12",
    "E:\qiqilu\datasets\BioSR\\transformed\F_actin_nonlinear\\train\channel_0\WF_noise_level_1",
    "E:\qiqilu\datasets\BioSR\\transformed\F_actin_nonlinear\\train\channel_0\WF_noise_level_2",
    "E:\qiqilu\datasets\BioSR\\transformed\F_actin_nonlinear\\train\channel_0\WF_noise_level_3",
    "E:\qiqilu\datasets\BioSR\\transformed\F_actin_nonlinear\\train\channel_0\WF_noise_level_4",
    "E:\qiqilu\datasets\BioSR\\transformed\F_actin_nonlinear\\train\channel_0\WF_noise_level_5",
    "E:\qiqilu\datasets\BioSR\\transformed\F_actin_nonlinear\\train\channel_0\WF_noise_level_6",
    "E:\qiqilu\datasets\BioSR\\transformed\F_actin_nonlinear\\train\channel_0\WF_noise_level_7",
    "E:\qiqilu\datasets\BioSR\\transformed\F_actin_nonlinear\\train\channel_0\WF_noise_level_8",
    "E:\qiqilu\datasets\BioSR\\transformed\F_actin_nonlinear\\train\channel_0\WF_noise_level_9",
    "E:\qiqilu\datasets\BioSR\\transformed\MTs\\train\channel_0\WF_noise_level_1",
    "E:\qiqilu\datasets\BioSR\\transformed\MTs\\train\channel_0\WF_noise_level_2",
    "E:\qiqilu\datasets\BioSR\\transformed\MTs\\train\channel_0\WF_noise_level_3",
    "E:\qiqilu\datasets\BioSR\\transformed\MTs\\train\channel_0\WF_noise_level_4",
    "E:\qiqilu\datasets\BioSR\\transformed\MTs\\train\channel_0\WF_noise_level_5",
    "E:\qiqilu\datasets\BioSR\\transformed\MTs\\train\channel_0\WF_noise_level_6",
    "E:\qiqilu\datasets\BioSR\\transformed\MTs\\train\channel_0\WF_noise_level_7",
    "E:\qiqilu\datasets\BioSR\\transformed\MTs\\train\channel_0\WF_noise_level_8",
    "E:\qiqilu\datasets\BioSR\\transformed\MTs\\train\channel_0\WF_noise_level_9",
    "E:\qiqilu\datasets\W2S\\transformed\\train\channel_0\wf_ave_400",
    "E:\qiqilu\datasets\W2S\\transformed\\train\channel_0\sim_ave",
    "E:\qiqilu\datasets\W2S\\transformed\\train\channel_1\wf_ave_400",
    "E:\qiqilu\datasets\W2S\\transformed\\train\channel_1\sim_ave",
    "E:\qiqilu\datasets\W2S\\transformed\\train\channel_2\wf_ave_400",
    "E:\qiqilu\datasets\W2S\\transformed\\train\channel_2\sim_ave",
    "E:\qiqilu\datasets\RCAN3D\\transformed\C2S_MT\\train\channel_0\confocal",
    "E:\qiqilu\datasets\RCAN3D\\transformed\C2S_NPC\\train\channel_0\confocal",
    "E:\qiqilu\datasets\RCAN3D\\transformed\C2S_SirDNA\\train\channel_0\condition_0",
    "E:\qiqilu\datasets\RCAN3D\\transformed\Phantom_spheres\\train\channel_0\condition_2xblur",
    "E:\qiqilu\datasets\RCAN3D\\transformed\Phantom_spheres\\train\channel_0\condition_3xblur",
    "E:\qiqilu\datasets\RCAN3D\\transformed\Phantom_spheres\\train\channel_0\condition_4xblur",
]
path_dataset_hr = list(
    ("E:\qiqilu\datasets\BioSR\\transformed\CCPs\\train\channel_0\SIM",) * 9
    + ("E:\qiqilu\datasets\BioSR\\transformed\ER\\train\channel_0\SIM",) * 6
    + ("E:\qiqilu\datasets\BioSR\\transformed\F_actin\\train\channel_0\SIM",) * 12
    + (
        "E:\qiqilu\datasets\BioSR\\transformed\F_actin_nonlinear\\train\channel_0\SIM_nonlinear",
    )
    * 9
    + ("E:\qiqilu\datasets\BioSR\\transformed\MTs\\train\channel_0\SIM",) * 9
    + ("E:\qiqilu\datasets\W2S\\transformed\\train\channel_0\sim",) * 2
    + ("E:\qiqilu\datasets\W2S\\transformed\\train\channel_1\sim",) * 2
    + ("E:\qiqilu\datasets\W2S\\transformed\\train\channel_2\sim",) * 2
    + ("E:\qiqilu\datasets\RCAN3D\\transformed\C2S_MT\\train\channel_0\STED",) * 1
    + ("E:\qiqilu\datasets\RCAN3D\\transformed\C2S_NPC\\train\channel_0\STED",) * 1
    + ("E:\qiqilu\datasets\RCAN3D\\transformed\C2S_SirDNA\\train\channel_0\gt",) * 1
    + ("E:\qiqilu\datasets\RCAN3D\\transformed\Phantom_spheres\\train\channel_0\gt",)
    * 3
)

path_index_file = list(
    ("E:\qiqilu\datasets\BioSR\\transformed\CCPs\\train",) * 9
    + ("E:\qiqilu\datasets\BioSR\\transformed\ER\\train",) * 6
    + ("E:\qiqilu\datasets\BioSR\\transformed\F_actin\\train",) * 12
    + ("E:\qiqilu\datasets\BioSR\\transformed\F_actin_nonlinear\\train",) * 9
    + ("E:\qiqilu\datasets\BioSR\\transformed\MTs\\train",) * 9
    + ("E:\qiqilu\datasets\W2S\\transformed\\train",) * 6
    + ("E:\qiqilu\datasets\RCAN3D\\transformed\C2S_MT\\train",) * 1
    + ("E:\qiqilu\datasets\RCAN3D\\transformed\C2S_NPC\\train",) * 1
    + ("E:\qiqilu\datasets\RCAN3D\\transformed\C2S_SirDNA\\train",) * 1
    + ("E:\qiqilu\datasets\RCAN3D\\transformed\Phantom_spheres\\train",) * 3
)

path_dataset_lr_val = None
path_dataset_hr_val = None

for i, path in enumerate(path_dataset_lr):
    path_dataset_lr[i] = path + "_patch"

for i, path in enumerate(path_dataset_hr):
    path_dataset_hr[i] = path + "_patch"

for i, path in enumerate(path_index_file):
    path_index_file[i] = path + "_patch.txt"


# ------------------------------------------------------------------------------
# finetune
# path_dataset_lr = [
#     "E:\qiqilu\datasets\SimuMix\\raw\data_128_128_128_gauss_0_poiss_1_ratio_1_528"
# ]
# path_dataset_hr = ["E:\qiqilu\datasets\SimuMix\gt"]

# path_dataset_lr_val = [
#     "E:\qiqilu\datasets\SimuMix\\raw\data_128_128_128_gauss_0_poiss_1_ratio_1_528"
# ]
# path_dataset_hr_val = ["E:\qiqilu\datasets\SimuMix\gt"]

# ------------------------------------------------------------------------------
params = {
    "device": "cuda:0",
    "num_workers": 0,
    "random_seed": 7,
    "data_shuffle": True,
    "enable_amp": False,
    "enable_gradscaler": False,
    "mean_one": False,
    # "model_name": "ddn3d",
    # "model_name": "unet3d",
    "model_name": "rcan3d",
    # "model_name": "unet3d_sim",
    # "use_bn": True,
    # "model_name": "teenet_sq",
    # "model_name": "teeresnet",
    "num_features": 32,
    "num_blocks": 5,
    "num_groups": 5,
    "enable_standardize": True,
    "pos_out": False,
    "psf_size": (127, 127, 127),
    "psf_model": "bw",
    "num_gaussian_model": 1,
    "enable_constraints": True,
    "over_sampling": 2,
    "kernel_norm": True,
    "residual": False,
    "center_one": False,
    "pixel_size_z": 1,
    "enable_groups": True,
    # "loss": "mse_ss",
    # "loss": "zncc+mse_ss",
    # "loss": "mse",
    "loss": "mae",
    # "loss": "mse+mse_ss_norm",
    # "loss": "mse+mae_ss_norm",
    # "loss": "zncc",
    # "loss": "zncc+zncc_psf",
    # "loss": "mse+ae_psf",
    # "loss": "zncc+mae_ss",
    # "loss": "zncc+poisson",
    # "loss": "zncc+zncc_ss",
    # "loss": "zncc_ss",
    # "loss": "zncc_x+mse_ss+mse_ss_2+zncc_ss_12+zncc_ss_21",
    # "loss": "zncc_x+zncc_ss+zncc_ss_2+zncc_ss_12+zncc_ss_21",
    "loss_weight": [1.0, 1.0],
    "alternate_opt": False,
    # "suffix": "_138_groups_resx_large_alter_half",
    # "suffix": "_mix_groups_resx_large_alter_bw",
    # "suffix": "_mix_large_3_pos_55_f16_trainpsf",
    "suffix": "_real_posx_55_zpad0",
    "fine_tune": False,
    "psf_part_freeze": False,
    "img_part_freeze": True,
    "head_tail_freeze": True,
    "path_model_pretrain": "checkpoints\SimuMix\\teeresnet_mse_bs_1_lr_0.001_mix_large_3_pos_55_f16\epoch_1_iter_16500.pt",
    "lr": 0.0001,
    "batch_size": 1,
    "num_epochs": 10,
    "warm_up": 0,
    "lr_decay_every_iter": 5000,
    "lr_decay_rate": 0.5,
    # "dataset_name": "SimuMix",
    "dataset_name": "Real",
    "z_padding": 0,
    "path_dataset_lr": path_dataset_lr,
    "path_dataset_hr": path_dataset_hr,
    "path_index_file": path_index_file,
    "path_dataset_lr_val": path_dataset_lr_val,
    "path_dataset_hr_val": path_dataset_hr_val,
    "in_channels": 1,
    "out_channels": 1,
    "path_checkpoints": "checkpoints",
    "save_every_iter": 500,
    "plot_every_iter": 20,
    "enable_validation": False,
    "validate_every_iter": 200,
    "p_low": 0.0,
    "p_high": 0.999,
}

# ------------------------------------------------------------------------------
if params["fine_tune"] == False:
    params["psf_part_freeze"] = False
    params["img_part_freeze"] = False
    params["head_tail_freeze"] = False

loss_names = params["loss"].split("+")

if len(loss_names) > 1:
    params["alternate_opt"] = False

device = torch.device(params["device"])
torch.manual_seed(params["random_seed"])

path_save_model = os.path.join(
    params["path_checkpoints"],
    params["dataset_name"],
    "{}_{}_bs_{}_lr_{}{}".format(
        params["model_name"],
        params["loss"],
        params["batch_size"],
        params["lr"],
        params["suffix"],
    ),
)
print(f"save model to {path_save_model}")
utils_data.make_path(path_save_model)

# save parameters
with open(os.path.join(path_save_model, "parameters.json"), "w") as f:
    f.write(json.dumps(params, indent=1))

utils_data.print_dict(params)

# ------------------------------------------------------------------------------
# dataset
# ------------------------------------------------------------------------------
transform = v2.Compose(
    [utils_data.NormalizePercentile(p_low=params["p_low"], p_high=params["p_high"])]
)

# training dataset
if params["dataset_name"] == "SimuMix":
    dataset_train = utils_data.MicroDataset(
        path_dataset_lr=params["path_dataset_lr"],
        path_dataset_hr=params["path_dataset_hr"],
        mode="train",
        transform=transform,
    )
elif params["dataset_name"] == "Real":
    dataset_train = utils_data.RealFMDataset(
        path_index_file=params["path_index_file"],
        path_dataset_lr=params["path_dataset_lr"],
        path_dataset_hr=params["path_dataset_hr"],
        transform=transform,
        z_padding=params["z_padding"],
    )
else:
    print("Un-supported dataset.")
    os._exit()

dataloader_train = DataLoader(
    dataset=dataset_train,
    batch_size=params["batch_size"],
    shuffle=params["data_shuffle"],
    num_workers=params["num_workers"],
)
num_batches_train = len(dataloader_train)

# ------------------------------------------------------------------------------
# validation dataset
if params["enable_validation"]:
    dataset_validation = utils_data.MicroDataset(
        path_dataset_lr=params["path_dataset_lr_val"],
        path_dataset_hr=params["path_dataset_hr_val"],
        mode="validation",
        transform=transform,
    )
    dataloader_validation = DataLoader(
        dataset=dataset_validation,
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
    )
    num_batches_validation = len(dataloader_validation)
else:
    num_batches_validation = 0
# ------------------------------------------------------------------------------

img_lr_shape = dataset_train[0]["lr"].shape
img_hr_shape = dataset_train[0]["hr"].shape

print(
    "- Num of Batches (train| validation): {}|{}\n- Input shape: {}\n- GT shape: {}".format(
        num_batches_train, num_batches_validation, img_lr_shape, img_hr_shape
    )
)


# ------------------------------------------------------------------------------
# model
# ------------------------------------------------------------------------------
if params["model_name"] == "ddn3d":
    model = DenseDeconNet(
        in_channels=params["in_channels"],
        out_channels=params["out_channels"],
        bias=False,
        scale_factor=1,
    )

if params["model_name"] == "unet3d":
    model = UNet3D(
        in_channels=params["in_channels"],
        out_channels=params["out_channels"],
        use_bn=True,
        bias=True,
    )

if params["model_name"] == "rcan3d":
    model = RCAN(
        in_channels=params["in_channels"],
        out_channels=params["out_channels"],
        fea_channels=params["num_features"],
        num_residual_blocks=params["num_blocks"],
        num_residual_groups=params["num_groups"],
        enable_standardize=params["enable_standardize"],
        pos_out=params["pos_out"],
    )

if params["model_name"] == "unet3d_sim":
    model = UNet3D_SIM(
        in_channels=params["in_channels"],
        out_channels=params["out_channels"],
        use_bn=params["use_bn"],
        bias=True,
        pos_out=True,
    )

if params["model_name"] == "teenet_sq":
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
        residual=params["residual"],
        groups=params["enable_groups"],
        over_sampling=params["over_sampling"],
        num_gauss_model=params["num_gaussian_model"],
        enable_constaints=params["enable_constraints"],
        center_one=params["center_one"],
        num_integral=100,
        pixel_size_z=params["pixel_size_z"],
        psf_part_freeze=params["psf_part_freeze"],
        img_part_freeze=params["img_part_freeze"],
        head_tail_freeze=params["head_tail_freeze"],
    )

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
        psf_part_freeze=params["psf_part_freeze"],
        img_part_freeze=params["img_part_freeze"],
        head_tail_freeze=params["head_tail_freeze"],
    )

model.to(device=device)

summary(model=model, input_size=(1,) + img_lr_shape)
# ------------------------------------------------------------------------------
# load pretrained model
# ------------------------------------------------------------------------------
if params["fine_tune"] == True:
    print("fine tuning ...")
    model.load_state_dict(
        torch.load(
            params["path_model_pretrain"], map_location=device, weights_only=True
        )["model_state_dict"]
    )

# ------------------------------------------------------------------------------
# optimization
# ------------------------------------------------------------------------------
if params["fine_tune"] == False:
    optimizer = torch.optim.Adam(params=model.parameters(), lr=params["lr"])
else:
    # optimizer = torch.optim.Adam(params=model.get_img_part_params(), lr=params["lr"])
    optimizer = torch.optim.Adam(params=model.get_psf_part_params(), lr=params["lr"])
    # optimizer = torch.optim.Adam(params=model.get_head_tail_params(), lr=params["lr"])

log_writer = SummaryWriter(os.path.join(path_save_model, "log"))

# ------------------------------------------------------------------------------
# train
# ------------------------------------------------------------------------------
print(
    "Batch size: {} | Num of Batches: {}".format(
        params["batch_size"], num_batches_train
    )
)

scaler = torch.GradScaler("cuda", enabled=params["enable_gradscaler"])

for i_epoch in range(params["num_epochs"]):
    pbar = tqdm.tqdm(
        total=num_batches_train,
        desc=f"Epoch {i_epoch + 1}|{params['num_epochs']}",
        leave=True,
        ncols=120,
    )

    # --------------------------------------------------------------------------
    for i_batch, data in enumerate(dataloader_train):
        i_iter = i_batch + i_epoch * num_batches_train

        imgs_est, psfs_est = None, None

        imgs_lr, imgs_hr, psfs_lr = (
            data["lr"].to(device),
            data["hr"].to(device),
            data["psf_lr"].to(device),
        )

        if psfs_lr != 0:
            # crop psf
            psfs_lr = utils_data.center_crop(
                psfs_lr, size=params["psf_size"], verbose=False
            )

        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=params["enable_amp"]
        ):
            if params["model_name"] in [
                "teenet",
                "teenet_att",
                "teenet_sq",
                "teeresnet",
            ]:
                imgs_est, psfs_est = model(imgs_lr)
            else:
                imgs_est = model(imgs_lr)

            if params["z_padding"] > 0:
                imgs_est = imgs_est[:, params["z_padding"] : -params["z_padding"]]

            # ------------------------------------------------------------------
            # loss
            loss_log = []

            if params["loss"] == "mse":
                loss = torch.nn.MSELoss()(imgs_est, imgs_hr)
                loss_log.append(loss)

            if params["loss"] == "mae":
                loss = torch.nn.L1Loss()(imgs_est, imgs_hr)
                loss_log.append(loss)

            if params["loss"] == "zncc":
                loss = utils_loss.ZNCC(imgs_est, imgs_hr, params["mean_one"])
                loss_log.append(loss)

            if params["loss"] == "mse_ss":
                loss = utils_loss.MSE_ss(
                    img_in=imgs_lr, img_est=imgs_est, psf_est=psfs_est
                )
                loss_log.append(loss)

            if params["loss"] == "zncc_ss":
                loss = utils_loss.ZNCC_ss(
                    img_in=imgs_lr, img_est=imgs_est, psf_est=psfs_est
                )
                loss_log.append(loss)

            if params["loss"] in [
                "zncc+ae_psf",
                "zncc+zncc_psf",
                "mse+mse_ss",
                "mse+mse_ss_norm",
                "mse+mae_ss_norm",
                "zncc+mse_ss",
                "zncc+mae_ss",
                "zncc+poisson",
                "zncc+zncc_ss",
            ]:
                if loss_names[0] == "zncc":
                    loss1 = utils_loss.ZNCC(
                        x=imgs_est, y=imgs_hr, mean_one=params["mean_one"]
                    )
                if loss_names[0] == "mse":
                    loss1 = torch.nn.MSELoss()(imgs_est, imgs_hr)

                loss1 = loss1 * params["loss_weight"][0]

                if params["alternate_opt"] == True:
                    optimizer.zero_grad()
                    scaler.scale(loss1).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    imgs_est, psfs_est = model(imgs_lr)

                if loss_names[1] == "mse_ss":
                    loss2 = utils_loss.MSE_ss(
                        img_in=imgs_lr, img_est=imgs_est, psf_est=psfs_est
                    )
                if loss_names[1] == "mse_ss_norm":
                    loss2 = utils_loss.MSE_ss_norm(
                        img_in=imgs_lr, img_est=imgs_est, psf_est=psfs_est
                    )
                if loss_names[1] == "mae_ss":
                    loss2 = utils_loss.MAE_ss(
                        img_in=imgs_lr, img_est=imgs_est, psf_est=psfs_est
                    )

                if loss_names[1] == "mae_ss_norm":
                    loss2 = utils_loss.MAE_ss_norm(
                        img_in=imgs_lr, img_est=imgs_est, psf_est=psfs_est
                    )

                if loss_names[1] == "poisson":
                    loss2 = utils_loss.Poisson(
                        img_in=imgs_lr, img_est=imgs_est, psf_est=psfs_est
                    )
                if loss_names[1] == "zncc_ss":
                    loss2 = utils_loss.ZNCC_ss(
                        img_in=imgs_lr, img_est=imgs_est, psf_est=psfs_est
                    )
                if loss_names[1] == "zncc_psf":
                    loss2 = utils_loss.ZNCC(psfs_est, psfs_lr)
                if loss_names[1] == "ae_psf":
                    loss2 = utils_loss.AE(psfs_est, psfs_lr)

                loss2 = loss2 * params["loss_weight"][1]

                if params["alternate_opt"] == True:
                    optimizer.zero_grad()
                    scaler.scale(loss2).backward()
                    scaler.step(optimizer)
                    scaler.update()

                loss = loss1 + loss2
                loss_log.extend([loss1, loss2])

        # ----------------------------------------------------------------------
        if torch.isnan(loss):
            print("NaN")

        if params["alternate_opt"] == False:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # loss.backward()
            # optimizer.step()

        # ----------------------------------------------------------------------
        # evaluation
        imgs_est = utils_eva.linear_transform(
            img_true=imgs_hr, img_test=imgs_est, axis=(2, 3, 4)
        )

        if (psfs_est is not None) and (psfs_lr != 0):
            mse_psf = torch.nn.MSELoss()(
                utils_data.max_norm(psfs_lr), utils_data.max_norm(psfs_est)
            )
            ZNCC_ss_psf = utils_loss.ZNCC_ss(
                img_in=imgs_lr, img_est=imgs_hr, psf_est=psfs_est
            )

        ssim = utils_eva.SSIM_tb(
            img_true=imgs_hr, img_test=imgs_est, data_range=None, version_wang=False
        )
        psnr = utils_eva.PSNR_tb(img_true=imgs_hr, img_test=imgs_est, data_range=None)

        pbar.set_postfix(
            Metrics="Loss: {:>.6f}, PSNR: {:>.6f}, SSIM: {:>.6f}".format(
                loss.cpu().detach().numpy(), psnr, ssim
            )
        )
        pbar.update(1)

        # ----------------------------------------------------------------------
        # update learning rate
        utils_optim.StepLR_iter(
            i_iter=i_iter,
            lr_start=params["lr"],
            optimizer=optimizer,
            decay_every_iter=params["lr_decay_every_iter"],
            lr_min=0.0,
            warm_up=params["warm_up"],
            decay_rate=params["lr_decay_rate"],
        )

        # ----------------------------------------------------------------------
        # log
        if i_iter % params["plot_every_iter"] == 0:
            if log_writer is not None:
                log_writer.add_scalar(
                    "Learning rate", optimizer.param_groups[-1]["lr"], i_iter
                )
                for i, ls in enumerate(loss_log):
                    log_writer.add_scalar(f"{loss_names[i]}", ls, i_iter)
                if (psfs_est is not None) and (psfs_lr != 0):
                    log_writer.add_scalar("MSE_PSF", mse_psf, i_iter)
                    log_writer.add_scalar("ZNCC_ss_PSF", ZNCC_ss_psf, i_iter)
                log_writer.add_scalar("PSNR", psnr, i_iter)
                log_writer.add_scalar("SSIM", ssim, i_iter)

        if i_iter % params["save_every_iter"] == 0:
            print("\nsave model (epoch: {}, iter: {})".format(i_epoch, i_iter))
            model_dict = {"model_state_dict": model.state_dict()}
            torch.save(
                model_dict,
                os.path.join(path_save_model, f"epoch_{i_epoch}_iter_{i_iter}.pt"),
            )

        # ----------------------------------------------------------------------
        # validation
        if (i_iter % params["validate_every_iter"] == 0) and (
            params["enable_validation"] == True
        ):
            print("validating ...")
            running_val_ssim, running_val_psnr = 0, 0
            for i_batch_val, sample_val in enumerate(dataloader_validation):
                imgs_lr_val, imgs_hr_val, psfs_lr_val = (
                    sample_val["lr"].to(device),
                    sample_val["hr"].to(device),
                    sample_val["psf_lr"].to(device),
                )

                with torch.no_grad():
                    if params["model_name"] in [
                        "teenet",
                        "teenet_att",
                        "teenet_sq",
                        "teeresnet",
                    ]:
                        imgs_est_val, psfs_est_val = model(imgs_lr_val)
                    else:
                        imgs_est_val = model(imgs_lr_val)

                # evaluation
                imgs_est_val = utils_eva.linear_transform(
                    img_true=imgs_hr_val, img_test=imgs_est_val, axis=(2, 3, 4)
                )

                ssim_val = utils_eva.SSIM_tb(
                    img_true=imgs_hr_val,
                    img_test=imgs_est_val,
                    data_range=None,
                    version_wang=False,
                )
                psnr_val = utils_eva.PSNR_tb(
                    img_true=imgs_hr_val, img_test=imgs_est_val, data_range=None
                )

                running_val_psnr += psnr_val
                running_val_ssim += ssim_val

            if log_writer is not None:
                log_writer.add_scalar(
                    "psnr_val", running_val_psnr / num_batches_validation, i_iter
                )
                log_writer.add_scalar(
                    "ssim_val", running_val_ssim / num_batches_validation, i_iter
                )
        # ----------------------------------------------------------------------
    pbar.close()

# ------------------------------------------------------------------------------
# save and finish
# ------------------------------------------------------------------------------
print("save model (epoch: {}, iter: {})".format(i_epoch, i_iter))

# saving general checkpoint
model_dict = {"model_state_dict": model.state_dict()}
torch.save(
    model_dict,
    os.path.join(path_save_model, f"epoch_{i_epoch+1}_iter_{i_iter+1}.pt"),
)

log_writer.flush()
log_writer.close()
print("Training done.")
