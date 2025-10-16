import torch, os, time, tqdm, json
import numpy as np
from torchinfo import summary
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2
from models.psfestimator import PSFEstimator

import utils.data as utils_data
import utils.evaluation as utils_eva
import utils.optim as utils_optim
import utils.loss_functions as utils_loss

# ------------------------------------------------------------------------------
# parameters
# ------------------------------------------------------------------------------
path_dataset_lr = [
    "E:\qiqilu\datasets\SimuMix_large\\raw\data_128_128_128_gauss_0_poiss_1_ratio_1"
]
path_dataset_hr = ["E:\qiqilu\datasets\SimuMix_large\gt"]

path_dataset_lr_val = [
    "E:\qiqilu\datasets\SimuMix_large\\raw\data_128_128_128_gauss_0_poiss_1_ratio_1"
]
path_dataset_hr_val = ["E:\qiqilu\datasets\SimuMix_large\gt"]

params = {
    "device": "cuda:0",
    "num_workers": 0,
    "random_seed": 7,
    "data_shuffle": True,
    "model_name": "psf_estimator",
    "psf_model": "gm",
    "psf_size": (127, 127, 127),
    # "psf_size": (31, 31, 31),
    "over_sampling": 2,
    "kernel_norm": True,
    # "loss": "mae_ss",
    # "loss": "mse_ss",
    "loss": "zncc_ss",
    "num_gaussian_model": 2,
    "enable_constraints": True,
    "center_one": False,
    "pixel_size_z": 1,
    "lr": 0.0001,
    "suffix": "_138_large_gm",
    "batch_size": 1,
    "num_epochs": 1,
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
    "validate_every_iter": 200,
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


def max_norm(x):
    x_max = torch.amax(x, dim=(2, 3, 4), keepdim=True)
    return x / x_max


# ------------------------------------------------------------------------------
# model
# ------------------------------------------------------------------------------
if params["model_name"] == "psf_estimator":
    model = PSFEstimator(
        in_channels=params["in_channels"],
        psf_model=params["psf_model"],
        kernel_norm=params["kernel_norm"],
        kernel_size=params["psf_size"],
        num_gauss_model=params["kernel_norm"],
        enable_constraint=params["enable_constraints"],
        over_sampling=params["over_sampling"],
        center_one=params["center_one"],
        use_bn=True,
        bias=True,
        pixel_size_z=params["pixel_size_z"],
    ).to(device)

summary(model=model, input_size=(1,) + img_lr_shape)

# ------------------------------------------------------------------------------
# optimization
# ------------------------------------------------------------------------------
optimizer = torch.optim.Adam(params=model.parameters(), lr=params["lr"])

log_writer = SummaryWriter(os.path.join(path_save_model, "log"))

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

        psfs_est = model(imgs_lr)
        if params["loss"] == "mse_ss":
            loss = utils_loss.MSE_ss(img_in=imgs_lr, img_est=imgs_hr, psf_est=psfs_est)
            loss_log.append(loss)

        if params["loss"] == "zncc_ss":
            loss = utils_loss.ZNCC_ss(img_in=imgs_lr, img_est=imgs_hr, psf_est=psfs_est)
            loss_log.append(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ----------------------------------------------------------------------
        # evaluation
        mse_psf = torch.nn.MSELoss()(max_norm(psfs_lr), max_norm(psfs_est))
        pbar.set_postfix(
            Metrics="Loss: {:>.6f}, MSE_PSF: {:>.6f}".format(
                loss.cpu().detach().numpy(), mse_psf
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
                log_writer.add_scalar("MSE_PSF", mse_psf, i_iter)

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
            running_val_mse_psf = 0

            for i_batch_val, sample_val in enumerate(dataloader_validation):
                imgs_lr_val, imgs_hr_val, psfs_lr_val = (
                    sample_val["lr"].to(device),
                    sample_val["hr"].to(device),
                    sample_val["psf_lr"].to(device),
                )

                with torch.no_grad():
                    psfs_est_val = model(imgs_lr_val)

                # evaluation
                mse_psf_val = torch.nn.MSELoss()(
                    max_norm(psfs_lr_val), max_norm(psfs_est_val)
                )

                running_val_mse_psf += mse_psf_val

            if log_writer is not None:
                log_writer.add_scalar(
                    "mse_psf_val", running_val_mse_psf / num_batches_validation, i_iter
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
