import torch, os, tqdm, json, pandas
from torchinfo import summary
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2

from models.unet_sd_c import UNetModel

import utils.data as utils_data
import utils.evaluation as utils_eva
import utils.optim as utils_optim
import utils.loss_functions as utils_loss

# ------------------------------------------------------------------------------
path_dataset_excel = "dataset_train_transformer.xlsx"
# data_frame = pandas.read_excel(path_dataset_excel, sheet_name="256x256")
data_frame = pandas.read_excel(path_dataset_excel, sheet_name="64x64")

# datasets_id = ["biosr-cpp-sr", "biosr-er-sr", "biosr-actin-sr", "biosr-mt-sr"]
# datasets_id = ["biosr-cpp-sr"]
# datasets_id = ["biosr-er-sr"]
# datasets_id = ["biosr-actin-sr"]
# datasets_id = ["srcaco2-h2b"]
# data_frame = data_frame[data_frame["id"].isin(datasets_id)]

path_dataset_lr = list(data_frame["path_lr"])
path_dataset_hr = list(data_frame["path_hr"])
path_index_file = list(data_frame["path_index"])
dataset_scale_factor_lr = list(data_frame["sf_lr"])
dataset_scale_factor_hr = list(data_frame["sf_hr"])

# ------------------------------------------------------------------------------
# parameters
# ------------------------------------------------------------------------------
params = {
    # device
    "device": "cuda:1",
    "num_workers": 0,
    "random_seed": 7,
    "data_shuffle": True,
    # mixed-precision ----------------------------------------------------------
    "enable_amp": True,
    "enable_gradscaler": True,
    # model parameters ---------------------------------------------------------
    "dim": 2,
    "model_name": "unet_sd_c",
    "in_channels": 1,
    "out_channels": 1,
    "channels": 320,
    "n_res_blocks": 2,
    "attention_levels": [1, 2, 3],
    "channel_multipliers": [1, 2, 4, 4],
    "n_heads": 8,
    "tf_layers": 1,
    "d_cond": 768,
    # "d_cond": None,
    "pixel_shuffle": False,
    "scale_factor": 4,
    # loss function ------------------------------------------------------------
    "loss": "mse",
    # "loss": "msew",
    # "loss": "mae",
    # learning rate ------------------------------------------------------------
    "lr": 0.00001,
    "batch_size": 4,
    # "num_epochs": 8,
    "num_epochs": 2,
    # "num_epochs": 100,
    "warm_up": 0,
    "lr_decay_every_iter": 10000 * 10,
    "lr_decay_rate": 0.5,
    # validation ---------------------------------------------------------------
    "enable_validation": True,
    "frac_val": 0.001,
    "validate_every_iter": 5000,
    # dataset ------------------------------------------------------------------
    "path_dataset_lr": path_dataset_lr,
    "path_dataset_hr": path_dataset_hr,
    "path_index_file": path_index_file,
    "dataset_scale_factor_lr": dataset_scale_factor_lr,
    "dataset_scale_factor_hr": dataset_scale_factor_hr,
    "p_low": 0.0,
    "p_high": 0.9,
    "text_version": "_bmc_v3",
    # checkpoints --------------------------------------------------------------
    # "suffix": "_biosrx2_123_adamw_bmc",
    "suffix": "_all_123_adamw_bmc_v3",
    # "suffix": "_biosrx2_123_adamw_bmc_actin_sr_crossx",
    # "suffix": "_biosrx2_123_adamw_bmc_mt_sr_crossx",
    "path_checkpoints": "checkpoints\conditional",
    "save_every_iter": 5000,
    "plot_every_iter": 100,
    # saved model --------------------------------------------------------
    # "saved_checkpoint": "E:\qiqilu\Project\\2024 Foundation model\code\checkpoints\conditional\\unet_sd_c_mse_bs_4_lr_1e-05_biosrx2_123\epoch_1_iter_77019.pt",
    "saved_checkpoint": None,
}

# ------------------------------------------------------------------------------
device = torch.device(params["device"])
torch.manual_seed(params["random_seed"])

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
print(f"save model to {path_save_model}")
utils_data.make_path(path_save_model)

# save parameters
with open(os.path.join(path_save_model, "parameters.json"), "w") as f:
    f.write(json.dumps(params, indent=1))

utils_data.print_dict(params)

# ------------------------------------------------------------------------------
# dataset
# ------------------------------------------------------------------------------
# data transform
# transform = v2.Compose(
#     [utils_data.NormalizePercentile(p_low=params["p_low"], p_high=params["p_high"])]
# )
transform = None


# dataset
# whole dataset
dataset_all = utils_data.Dataset_it2i(
    dim=params["dim"],
    path_index_file=params["path_index_file"],
    path_dataset_lr=params["path_dataset_lr"],
    path_dataset_hr=params["path_dataset_hr"],
    transform=transform,
    scale_factor_hr=params["dataset_scale_factor_hr"],
    scale_factor_lr=params["dataset_scale_factor_lr"],
    text_version=params["text_version"],
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
text_lr_shape = dataset_train[0]["lr_text"].shape
text_hr_shape = dataset_train[0]["hr_text"].shape

print(f"- Num of Batches (train| valid): {num_batches_train}|{num_batches_validation}")
print(
    f"- Input shape: ({img_lr_shape}, {text_lr_shape}, {text_hr_shape})",
)
print(f"- GT shape: {img_hr_shape}")

# ------------------------------------------------------------------------------
# model
# ------------------------------------------------------------------------------
# 2D models
if params["model_name"] == "unet_sd_c":
    model = UNetModel(
        in_channels=params["in_channels"],
        out_channels=params["out_channels"],
        channels=params["channels"],
        n_res_blocks=params["n_res_blocks"],
        attention_levels=params["attention_levels"],
        channel_multipliers=params["channel_multipliers"],
        n_heads=params["n_heads"],
        tf_layers=params["tf_layers"],
        d_cond=params["d_cond"],
        pixel_shuffle=params["pixel_shuffle"],
        scale_factor=params["scale_factor"],
    )

model.to(device=device)
# summary(model=model, input_size=((1,) + img_lr_shape, (1,), (1, 154, 768)))
summary(model=model, input_size=((1,) + img_lr_shape, (1,), (1, 512, 768)))

# ------------------------------------------------------------------------------
# pre-trained model parameters
if params["saved_checkpoint"] is not None:
    print("- Load saved pre-trained model parameters.")
    model.load_state_dict(
        torch.load(
            params["saved_checkpoint"],
            map_location=device,
            weights_only=True,
        )["model_state_dict"]
    )
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

# ------------------------------------------------------------------------------
# train
# ------------------------------------------------------------------------------
print(f"Batch size: {params['batch_size']} | Num of Batches: {num_batches_train}")

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
        i_iter = i_batch + i_epoch * num_batches_train + start_iter

        imgs_lr, text_lr, imgs_hr, text_hr = (
            data["lr"].to(device),
            data["lr_text"].to(device),
            data["hr"].to(device),
            data["hr_text"].to(device),
        )

        # create zero time embedding
        time_embed = torch.zeros(size=(params["batch_size"],)).to(device)

        # concate text embeddings
        if (params["d_cond"] == 0) or (params["d_cond"] is None):
            text_embed = None
        else:
            text_embed = torch.cat([text_lr, text_hr], dim=1)

        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=params["enable_amp"]
        ):
            imgs_est = model(imgs_lr, time_embed, text_embed)

            if params["loss"] == "mse":
                loss = torch.nn.MSELoss()(imgs_est, imgs_hr)
            if params["loss"] == "mae":
                loss = torch.nn.L1Loss()(imgs_est, imgs_hr)
            if params["loss"] == "msew":
                loss = utils_loss.MSE_w(img_est=imgs_est, img_gt=imgs_hr, scale=0.1)

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
                log_writer.add_scalar(params["loss"], loss, i_iter)
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
            pbar_val = tqdm.tqdm(
                desc="validation", total=num_batches_validation, leave=False, ncols=120
            )
            # convert model to evaluation model
            model.eval()
            # ------------------------------------------------------------------
            running_val_ssim, running_val_psnr = 0, 0
            for i_batch_val, data_val in enumerate(dataloader_validation):

                imgs_lr_val, text_lr_val, imgs_hr_val, text_hr_val = (
                    data_val["lr"].to(device),
                    data_val["lr_text"].to(device),
                    data_val["hr"].to(device),
                    data_val["hr_text"].to(device),
                )

                time_embed_val = torch.zeros(size=(params["batch_size"],)).to(device)
                text_embed_val = torch.cat([text_lr_val, text_hr_val], dim=1)

                with torch.no_grad():
                    imgs_est_val = model(imgs_lr_val, time_embed_val, text_embed_val)

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
model_dict = {"model_state_dict": model.state_dict()}
torch.save(
    model_dict,
    os.path.join(path_save_model, f"epoch_{i_epoch+1}_iter_{i_iter+1}.pt"),
)

log_writer.flush()
log_writer.close()
print("Training done.")
