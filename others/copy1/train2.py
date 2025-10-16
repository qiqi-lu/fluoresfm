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

import utils.data as utils_data
import utils.evaluation as utils_eva
import utils.optim as utils_optim
import utils.loss_functions as utils_loss

# ------------------------------------------------------------------------------
# parameters
# ------------------------------------------------------------------------------
# dataset
path_dataset_lr = [
    "E:\qiqilu\datasets\SimuMix_large\\raw\data_128_128_128_gauss_0_poiss_1_ratio_1"
]
path_dataset_hr = ["E:\qiqilu\datasets\SimuMix_large\gt"]

path_dataset_lr_val = [
    "E:\qiqilu\datasets\SimuMix_large\\raw\data_128_128_128_gauss_0_poiss_1_ratio_1"
]
path_dataset_hr_val = ["E:\qiqilu\datasets\SimuMix_large\gt"]

# ------------------------------------------------------------------------------
params = {
    "device": "cuda:1",
    "num_workers": 0,
    "random_seed": 7,
    "data_shuffle": True,
    "model_name": "half+unet3d_sim",
    "psf_model": "half",
    # "psf_size": (127, 127, 127),
    "psf_size": (31, 31, 31),
    "over_sampling": 2,
    "center_one": True,
    "kernel_norm": True,
    "num_gaussian_model": 2,
    "enable_constraints": True,
    "model_weights_path": [
        None,
        "E:\qiqilu\Project\\2024 Foundation model\code\checkpoints\SimuMix\\unet3d_sim_zncc_bs_1_lr_0.001_138_large\epoch_0_iter_16000.pt",
    ],
    "enable_train": [True, False],
    # "loss": "zncc+zncc_psf",
    # "loss": "mse+ae_psf",
    # "loss": "zncc+mse_ss",
    # "loss": "zncc+mae_ss",
    "loss": "zncc+poisson",
    # "loss": "zncc+zncc_ss",
    # "suffix": "_138_large_alter10_gmm2_cons_31",
    "suffix": "_138_large_alter10_os2_cons_31",
    "lr": [0.001, 0.001],
    "batch_size": 1,
    "num_epochs": 2,
    "warm_up": 0,
    "lr_decay_every_iter": 2000,
    "lr_decay_rate": 0.5,
    "dataset_name": "SimuMix",
    "path_dataset_lr": path_dataset_lr,
    "path_dataset_hr": path_dataset_hr,
    "path_dataset_lr_val": path_dataset_lr_val,
    "path_dataset_hr_val": path_dataset_hr_val,
    "in_channels": 1,
    "out_channels": 1,
    "path_checkpoints": "checkpoints",
    "save_every_iter": 500,
    "plot_every_iter": 1,
    "enable_validation": False,
    "validate_every_iter": 100,
    "p_low": 0.0,
    "p_high": 0.9,
}

# ------------------------------------------------------------------------------
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

utils_data.make_path(path_save_model)

# save parameters
with open(os.path.join(path_save_model, "parameters.json"), "w") as f:
    f.write(json.dumps(params, indent=1))

utils_data.print_dict(params)
print(f"save model to {path_save_model}")
log_writer = SummaryWriter(os.path.join(path_save_model, "log"))

# ------------------------------------------------------------------------------
# dataset
# ------------------------------------------------------------------------------
transform = v2.Compose(
    [utils_data.NormalizePercentile(p_low=params["p_low"], p_high=params["p_high"])]
)

# training dataset
dataset_train = utils_data.MicroDataset(
    path_dataset_lr=params["path_dataset_lr"],
    path_dataset_hr=params["path_dataset_hr"],
    mode="train",
    transform=transform,
)

dataloader_train = DataLoader(
    dataset=dataset_train,
    batch_size=params["batch_size"],
    shuffle=params["data_shuffle"],
    num_workers=params["num_workers"],
)


# validation dataset
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

num_batches_train = len(dataloader_train)
num_batches_validation = len(dataloader_validation)

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
model_names = params["model_name"].split("+")

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

summary(model=model_psf, input_size=(1,) + img_lr_shape)
summary(model=model_img, input_size=(1,) + img_lr_shape)

# ------------------------------------------------------------------------------
# load saved model weights
# ------------------------------------------------------------------------------
if len(params["model_weights_path"]) == 2:
    if params["model_weights_path"][0] is not None:
        model_psf.load_state_dict(
            torch.load(
                params["model_weights_path"][0], map_location=device, weights_only=True
            )["model_state_dict"]
        )

    if params["model_weights_path"][1] is not None:
        model_img.load_state_dict(
            torch.load(
                params["model_weights_path"][1], map_location=device, weights_only=True
            )["model_state_dict"]
        )

if len(params["model_weights_path"]) == 1:
    model_psf.load_state_dict(
        torch.load(
            params["model_weights_path"], map_location=device, weights_only=True
        )["model_state_dict_psf"]
    )

    model_img.load_state_dict(
        torch.load(
            params["model_weights_path"][1], map_location=device, weights_only=True
        )["model_state_dict_img"]
    )

# ------------------------------------------------------------------------------
# optimization
# ------------------------------------------------------------------------------
if params["enable_train"][0] == True:
    optimizer_psf = torch.optim.Adam(params=model_psf.parameters(), lr=params["lr"][0])

if params["enable_train"][1] == True:
    optimizer_img = torch.optim.Adam(params=model_img.parameters(), lr=params["lr"][1])

# ------------------------------------------------------------------------------
# train
# ------------------------------------------------------------------------------
print(
    "Batch size: {} | Num of Batches: {}".format(
        params["batch_size"], num_batches_train
    )
)
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

        imgs_lr, imgs_hr, psfs_lr = (
            data["lr"].to(device),
            data["hr"].to(device),
            data["psf_lr"].to(device),
        )

        # crop psf
        psfs_lr = utils_data.center_crop(
            psfs_lr, size=params["psf_size"], verbose=False
        )

        # ----------------------------------------------------------------------
        # loss
        loss_log = []
        loss_names = params["loss"].split("+")

        # training img part
        if params["enable_train"][1] == False:
            with torch.no_grad():
                imgs_est = model_img(imgs_lr)
        else:
            imgs_est = model_img(imgs_lr)

        if loss_names[0] == "zncc":
            loss1 = utils_loss.ZNCC(x=imgs_est, y=imgs_hr)
        if loss_names[0] == "mse":
            loss1 = torch.nn.MSELoss()(x=imgs_est, y=imgs_hr)

        if params["enable_train"][1] == True:
            optimizer_img.zero_grad()
            loss1.backward()
            optimizer_img.step()

        # training psf part
        if params["enable_train"][0] == True:
            psfs_est = model_psf(imgs_lr)
            imgs_est = model_img(imgs_lr)
        else:
            with torch.no_grad():
                psfs_est = model_psf(imgs_lr)
                imgs_est = model_img(imgs_lr)

        if loss_names[1] == "mse_ss":
            loss2 = utils_loss.MSE_ss(
                img_in=imgs_lr, img_est=imgs_est, psf_est=psfs_est
            )
        if loss_names[1] == "mae_ss":
            loss2 = utils_loss.MAE_ss(
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

        if params["enable_train"][0] == True:
            optimizer_psf.zero_grad()
            loss2.backward()
            optimizer_psf.step()

        loss = loss1 + loss2
        loss_log.extend([loss1, loss2])

        # ----------------------------------------------------------------------
        # evaluation
        imgs_est = utils_eva.linear_transform(
            img_true=imgs_hr, img_test=imgs_est, axis=(2, 3, 4)
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
        if params["enable_train"][0]:
            utils_optim.StepLR_iter(
                i_iter=i_iter,
                lr_start=params["lr"][0],
                optimizer=optimizer_psf,
                decay_every_iter=params["lr_decay_every_iter"],
                lr_min=0.0,
                warm_up=params["warm_up"],
                decay_rate=params["lr_decay_rate"],
            )
        if params["enable_train"][1]:
            utils_optim.StepLR_iter(
                i_iter=i_iter,
                lr_start=params["lr"][1],
                optimizer=optimizer_img,
                decay_every_iter=params["lr_decay_every_iter"],
                lr_min=0.0,
                warm_up=params["warm_up"],
                decay_rate=params["lr_decay_rate"],
            )

        # ----------------------------------------------------------------------
        # log
        if i_iter % params["plot_every_iter"] == 0:
            if log_writer is not None:
                for i, ls in enumerate(loss_log):
                    log_writer.add_scalar(f"{loss_names[i]}", ls, i_iter)
                log_writer.add_scalar("PSNR", psnr, i_iter)
                log_writer.add_scalar("SSIM", ssim, i_iter)

        if i_iter % params["save_every_iter"] == 0:
            print("\nsave model (epoch: {}, iter: {})".format(i_epoch, i_iter))
            model_dict = {
                "model_state_dict_psf": model_psf.state_dict(),
                "model_state_dict_img": model_img.state_dict(),
            }
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
                    psfs_est_val = model_psf(imgs_lr_val)
                    imgs_est_val = model_img(imgs_lr_val)

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
model_dict = {
    "model_state_dict_psf": model_psf.state_dict(),
    "model_state_dict_img": model_img.state_dict(),
}
torch.save(
    model_dict,
    os.path.join(path_save_model, f"epoch_{i_epoch+1}_iter_{i_iter+1}.pt"),
)

log_writer.flush()
log_writer.close()
print("Training done.")
