import torch, os, tqdm, json, pandas, datetime
from torchinfo import summary
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2
import numpy as np

from models.unet import UNet
from models.care import CARE
from models.dfcan import DFCAN

import utils.data as utils_data
import utils.evaluation as utils_eva
import utils.optim as utils_optim

# ------------------------------------------------------------------------------
# parameters
# ------------------------------------------------------------------------------
params = {
    # device
    "device": "cuda:0",
    "num_workers": 5,
    "random_seed": 3,
    "data_shuffle": True,
    # mixed-precision ----------------------------------------------------------
    "enable_amp": False,
    "enable_gradscaler": False,
    # model parameters ---------------------------------------------------------
    "dim": 2,
    # "model_name": "care",
    "model_name": "dfcan",
    # loss function ------------------------------------------------------------
    # "loss": "mse",
    "loss": "mae",
    # learning rate ------------------------------------------------------------
    "lr": 0.0001,
    "batch_size": 4,
    "num_epochs": 20,
    "warm_up": 0,
    "lr_decay_every_iter": 10000 * 10,
    "lr_decay_rate": 0.5,
    "lr_min": 0.0000001,
    # validation ---------------------------------------------------------------
    "enable_validation": True,
    "frac_val": 0.01,
    "validate_every_iter": 5000,
    # dataset ------------------------------------------------------------------
    "path_dataset_excel": "dataset_train_transformer.xlsx",
    "sheet_name": "64x64",
    "datasets_id": [],
    # "datasets_id": [
    #     "biosr-cpp",
    #     "biosr-er",
    #     "biosr-actin",
    #     "biosr-mt",
    #     "biosr-actinnl",
    # ],
    # "datasets_id": [
    #     "biosr-cpp",
    #     "biosr-er",
    #     "biosr-actin",
    #     "biosr-mt",
    #     "w2s-c0",
    #     "w2s-c1",
    #     "w2s-c2",
    #     "srcaco2-h2b-2",
    #     "srcaco2-survivin-2",
    #     "srcaco2-tubulin-2",
    # ],
    # "datasets_id": ["biosr-cpp"],
    # "datasets_id": ["biosr-er"],
    # "datasets_id": ["biosr-actin"],
    # "datasets_id": ["biosr-mt"],
    # "task": ["sr"],
    "scale_factor": 1,
    "task": ["dcv"],
    # "task": ["dn"],
    # "task": ["iso"],
    # checkpoints --------------------------------------------------------------
    "suffix": "_dcv",
    "path_checkpoints": "checkpoints\conditional",
    "save_every_iter": 5000,
    "plot_every_iter": 100,
    # saved model --------------------------------------------------------
    "saved_checkpoint": None,
}

# ------------------------------------------------------------------------------
device = torch.device(params["device"])
torch.manual_seed(params["random_seed"])

if os.name == "posix":
    params["path_checkpoints"] = utils_data.win2linux(params["path_checkpoints"])
    if params["saved_checkpoint"] is not None:
        params["saved_checkpoint"] = utils_data.win2linux(params["saved_checkpoint"])


# checkpoints save path
path_save_model = os.path.join(
    params["path_checkpoints"],
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
with open(
    os.path.join(path_save_model, f"parameters-{datetime.date.today()}.json"), "w"
) as f:
    f.write(json.dumps(params, indent=1))

utils_data.print_dict(params)

# ------------------------------------------------------------------------------
# dataset
# ------------------------------------------------------------------------------
data_frame = pandas.read_excel(
    params["path_dataset_excel"], sheet_name=params["sheet_name"]
)

if params["task"]:
    data_frame = data_frame[data_frame["task"].isin(params["task"])]
if params["datasets_id"]:
    data_frame = data_frame[data_frame["id"].isin(params["datasets_id"])]

print("Number of datasets:", data_frame.shape[0])

path_dataset_lr = list(data_frame["path_lr"])
path_dataset_hr = list(data_frame["path_hr"])
path_index_file = list(data_frame["path_index"])
dataset_index = list(data_frame["index"])
dataset_scale_factor_lr = list(data_frame["sf_lr"])
dataset_scale_factor_hr = list(data_frame["sf_hr"])

if params["scale_factor"] != 1:
    dataset_scale_factor_lr = [1] * len(dataset_scale_factor_lr)
    dataset_scale_factor_hr = [1] * len(dataset_scale_factor_hr)

# data transform
# transform = v2.Compose(
#     [utils_data.NormalizePercentile(p_low=params["p_low"], p_high=params["p_high"])]
# )
transform = None

# dataset
# whole dataset
dataset_all = utils_data.Dataset_iit(
    dim=params["dim"],
    path_index_file=path_index_file,
    path_dataset_lr=path_dataset_lr,
    path_dataset_hr=path_dataset_hr,
    dataset_index=dataset_index,
    path_dataset_text_embedding=None,
    transform=transform,
    scale_factor_lr=dataset_scale_factor_lr,
    scale_factor_hr=dataset_scale_factor_hr,
    output_type="ii",
)

# create training and validation dataset
dataloader_train, dataloader_validation = None, None
if params["enable_validation"]:
    # split whole dataset into training and validation dataset
    dataset_train, dataset_validation = random_split(
        dataset_all,
        [1.0 - params["frac_val"], params["frac_val"]],
        generator=torch.Generator().manual_seed(7),
    )

    dataloader_validation = DataLoader(
        dataset=dataset_validation,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=params["num_workers"],
    )
    num_batches_validation = len(dataloader_validation)
else:
    dataset_train = dataset_all
    num_batches_validation = 0

dataloader_train = DataLoader(
    dataset=dataset_train,
    batch_size=params["batch_size"],
    shuffle=params["data_shuffle"],
    num_workers=params["num_workers"],
)
num_batches_train = len(dataloader_train)

# ------------------------------------------------------------------------------
# data infomation
img_lr_shape = dataset_train[0]["lr"].shape
img_hr_shape = dataset_train[0]["hr"].shape

print(f"- Num of Batches (train| valid): {num_batches_train}|{num_batches_validation}")
print("- Input shape:", img_lr_shape)
print("- GT shape:", img_hr_shape)

# ------------------------------------------------------------------------------
# model
# ------------------------------------------------------------------------------
# 2D models
if params["model_name"] == "care":
    model = CARE(
        in_channels=1,
        out_channels=1,
        n_filter_base=16,
        kernel_size=5,
        batch_norm=False,
        dropout=0.0,
        residual=True,
        expansion=2,
        pos_out=False,
    )

if params["model_name"] == "dfcan":
    model = DFCAN(
        in_channels=1,
        scale_factor=params["scale_factor"],
        num_features=64,
        num_groups=4,
    )

model.to(device=device)
summary(model=model, input_size=(1,) + img_lr_shape)

# ------------------------------------------------------------------------------
# pre-trained model parameters
if params["saved_checkpoint"] is not None:
    print("- Load saved pre-trained model parameters:", params["saved_checkpoint"])
    state_dict = torch.load(
        params["saved_checkpoint"],
        map_location=device,
        weights_only=True,
    )["model_state_dict"]
    state_dict = utils_optim.on_load_checkpoint(state_dict)
    model.load_state_dict(state_dict)
    start_iter = params["saved_checkpoint"].split(".")[-2].split("_")[-1]
    start_iter = int(start_iter)
else:
    start_iter = 0

# ------------------------------------------------------------------------------
# optimization
# ------------------------------------------------------------------------------
# optimizer = torch.optim.Adam(params=model.parameters(), lr=params["lr"])
optimizer = torch.optim.AdamW(params=model.parameters(), lr=params["lr"])
log_writer = SummaryWriter(os.path.join(path_save_model, "log"))

LR_schedule = utils_optim.StepLR_iter(
    lr_start=params["lr"],
    optimizer=optimizer,
    decay_every_iter=params["lr_decay_every_iter"],
    lr_min=params["lr_min"],
    warm_up=params["warm_up"],
    decay_rate=params["lr_decay_rate"],
)
LR_schedule.init(start_iter)

# ------------------------------------------------------------------------------
# train
# ------------------------------------------------------------------------------
print(f"Batch size: {params['batch_size']} | Num of Batches: {num_batches_train}")
print(f"save model to {path_save_model}")

scaler = torch.GradScaler("cuda", enabled=params["enable_gradscaler"])

try:
    for i_epoch in range(params["num_epochs"]):
        pbar = tqdm.tqdm(
            total=num_batches_train,
            desc=f"Epoch {i_epoch + 1}|{params['num_epochs']}",
            leave=True,
            ncols=120,
        )

        # --------------------------------------------------------------------------
        for i_batch, data in enumerate(dataloader_train):
            i_iter = i_batch + i_epoch * num_batches_train + start_iter

            imgs_lr, imgs_hr = (data["lr"].to(device), data["hr"].to(device))

            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=params["enable_amp"]
            ):
                imgs_est = model(imgs_lr)

                if params["loss"] == "mse":
                    loss = torch.nn.MSELoss()(imgs_est, imgs_hr)
                if params["loss"] == "mae":
                    loss = torch.nn.L1Loss()(imgs_est, imgs_hr)

                if torch.isnan(loss):
                    print(" NaN!")

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # ----------------------------------------------------------------------
            # evaluation
            if params["dim"] == 3:
                imgs_est = utils_eva.linear_transform(
                    img_true=imgs_hr, img_test=imgs_est, axis=(2, 3, 4)
                )

            if params["dim"] == 2:
                imgs_est = utils_eva.linear_transform(
                    img_true=imgs_hr, img_test=imgs_est, axis=(2, 3)
                )

            ssim = utils_eva.SSIM_tb(
                img_true=imgs_hr, img_test=imgs_est, data_range=None, version_wang=False
            )
            psnr = utils_eva.PSNR_tb(
                img_true=imgs_hr, img_test=imgs_est, data_range=None
            )

            pbar.set_postfix(
                Metrics="Loss: {:>.6f}, PSNR: {:>.6f}, SSIM: {:>.6f}".format(
                    loss.cpu().detach().numpy(), psnr, ssim
                )
            )
            pbar.update(1)

            # ----------------------------------------------------------------------
            # update learning rate
            LR_schedule.update(i_iter=i_iter)

            # ----------------------------------------------------------------------
            # log
            if i_iter % params["plot_every_iter"] == 0:
                if log_writer is not None:
                    log_writer.add_scalar(
                        "Learning rate", optimizer.param_groups[-1]["lr"], i_iter
                    )
                    log_writer.add_scalar(params["loss"], loss, i_iter)
                    log_writer.add_scalar("PSNR", psnr, i_iter)
                    log_writer.add_scalar("SSIM", ssim, i_iter)

            if i_iter % params["save_every_iter"] == 0:
                print("\nsave model (epoch: {}, iter: {})".format(i_epoch, i_iter))
                model_dict = {
                    "model_state_dict": getattr(model, "_orig_mod", model).state_dict()
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
                pbar_val = tqdm.tqdm(
                    desc="validation",
                    total=num_batches_validation,
                    leave=True,
                    ncols=120,
                )
                # convert model to evaluation model
                model.eval()
                # ------------------------------------------------------------------
                running_val_ssim, running_val_psnr = 0, 0
                for i_batch_val, data_val in enumerate(dataloader_validation):
                    imgs_lr_val, imgs_hr_val = (
                        data_val["lr"].to(device),
                        data_val["hr"].to(device),
                    )

                    with torch.no_grad():
                        imgs_est_val = model(imgs_lr_val)

                    # evaluation
                    # linear transform
                    if params["dim"] == 2:
                        imgs_est_val = utils_eva.linear_transform(
                            img_true=imgs_hr_val, img_test=imgs_est_val, axis=(2, 3)
                        )
                    if params["dim"] == 3:
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

                    pbar_val.set_postfix(
                        Metrics="PSNR: {:>.6f}, SSIM: {:>.6f}".format(
                            running_val_psnr / (i_batch_val + 1),
                            running_val_ssim / (i_batch_val + 1),
                        )
                    )
                    pbar_val.update(1)

                if log_writer is not None:
                    log_writer.add_scalar(
                        "psnr_val", running_val_psnr / num_batches_validation, i_iter
                    )
                    log_writer.add_scalar(
                        "ssim_val", running_val_ssim / num_batches_validation, i_iter
                    )
                # convert model to train mode
                model.train(True)
                pbar_val.close()
        pbar.close()

    # ------------------------------------------------------------------------------
    # save and finish
    # ------------------------------------------------------------------------------
    print("\nsave model (epoch: {}, iter: {})".format(i_epoch, i_iter))

    # saving general checkpoint
    model_dict = {"model_state_dict": getattr(model, "_orig_mod", model).state_dict()}
    torch.save(
        model_dict,
        os.path.join(path_save_model, f"epoch_{i_epoch+1}_iter_{i_iter+1}.pt"),
    )

    log_writer.flush()
    log_writer.close()
    print("Training done.")

except KeyboardInterrupt:
    print("\ntraining stop, saving model ...")
    print("\nsave model (epoch: {}, iter: {})".format(i_epoch, i_iter))

    # saving general checkpoint
    model_dict = {"model_state_dict": getattr(model, "_orig_mod", model).state_dict()}
    torch.save(
        model_dict,
        os.path.join(path_save_model, f"epoch_{i_epoch+1}_iter_{i_iter+1}.pt"),
    )

    pbar.close()
    log_writer.flush()
    log_writer.close()
    print("Training done.")
