import torch, os, tqdm, json
from torchinfo import summary
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2

from models.autoencoder import Autoencoder, Encoder, Decoder

import utils.data as utils_data
import utils.evaluation as utils_eva
import utils.optim as utils_optim
import pandas

# ------------------------------------------------------------------------------
# parameters
# ------------------------------------------------------------------------------
# load dataset list from excel file
path_dataset_excel = (
    "E:\qiqilu\Project\\2024 Foundation model\code\\dataset_train_vae.xlsx"
)
dataframe = pandas.read_excel(path_dataset_excel)

path_dataset = list(dataframe["path"])
path_index_file = list(dataframe["path_index"])
scale_factor = list(dataframe["scale_factor"])
path_dataset_val = None
scale_factor_val = None

# ------------------------------------------------------------------------------
params = {
    # device
    "device": "cuda:1",
    "num_workers": 0,
    "random_seed": 7,
    "data_shuffle": True,
    # mixed-precision ----------------------------------------------------------
    "enable_amp": False,
    "enable_gradscaler": False,
    # model parameters ---------------------------------------------------------
    "dim": 2,
    "in_channels": 1,
    "out_channels": 1,
    "model_name": "vae",
    "channels": 128,
    "channel_multiplier": [1, 2, 4, 4],
    "n_resnet_blocks": 2,
    "z_channels": 32,
    "emb_channels": 32,
    # loss function ------------------------------------------------------------
    "loss": "mae_kl",
    "kl_weight": 0.000001,
    # learning rate ------------------------------------------------------------
    "lr": 0.00001,
    "batch_size": 4,
    "num_epochs": 8,
    "warm_up": 0,
    "lr_decay_every_iter": 10000 * 6,
    "lr_decay_rate": 0.5,
    # validation ---------------------------------------------------------------
    "enable_validation": False,
    "validate_every_iter": 200,
    # dataset ------------------------------------------------------------------
    "path_dataset": path_dataset,
    "scale_factor": scale_factor,
    "path_index_file": path_index_file,
    "path_dataset_val": path_dataset_val,
    "scale_factor_val": scale_factor_val,
    # checkpoints --------------------------------------------------------------
    "suffix": "_biosrall_sf2_c32",
    "path_checkpoints": "checkpoints\\vae",
    "save_every_iter": 5000,
    "plot_every_iter": 100,
    # "load_checkpoint": "checkpoints\\vae\\vae_mae_kl_bs_4_lr_1e-05_biosrall_sf2\epoch_0_iter_10000.pt",
    "load_checkpoint": None,
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
transform = None

# training dataset
dataset_train = utils_data.Dataset_i(
    path_index_file=params["path_index_file"],
    path_dataset=params["path_dataset"],
    transform=transform,
    scale_factor=params["scale_factor"],
)

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
    dataset_validation = utils_data.Dataset_i(
        path_dataset=params["path_dataset_val"],
        transform=transform,
        scale_factor=params["scale_factor_val"],
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
# data infomation
img_shape = dataset_train[0]["img"].shape

print(f"- Num of Batches (train| valid): {num_batches_train}|{num_batches_validation}")
print("- Input shape:", img_shape)

# ------------------------------------------------------------------------------
# model
# ------------------------------------------------------------------------------
# 2D models
if params["model_name"] == "vae":
    encoder = Encoder(
        channels=params["channels"],
        channel_multipliers=params["channel_multiplier"],
        n_resnet_blocks=params["n_resnet_blocks"],
        in_channels=params["in_channels"],
        z_channels=params["z_channels"],
    ).to(device=device)

    decoder = Decoder(
        channels=params["channels"],
        channel_multipliers=params["channel_multiplier"],
        n_resnet_blocks=params["n_resnet_blocks"],
        out_channels=params["out_channels"],
        z_channels=params["z_channels"],
    ).to(device=device)

    model = Autoencoder(
        encoder=encoder,
        decoder=decoder,
        emb_channels=params["emb_channels"],
        z_channels=params["z_channels"],
    ).to(device=device)

# ------------------------------------------------------------------------------
# load pre-trained checkpoint
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# pre-trained model parameters
if params["load_checkpoint"] is not None:
    print("- Load saved pre-trained model parameters.")
    model.load_state_dict(
        torch.load(
            params["load_checkpoint"],
            map_location=device,
            weights_only=True,
        )["model_state_dict"]
    )
    start_iter = params["load_checkpoint"].split(".")[-2].split("_")[-1]
    start_iter = int(start_iter)
else:
    start_iter = 0

# ------------------------------------------------------------------------------
# optimization
# ------------------------------------------------------------------------------
optimizer = torch.optim.Adam(params=model.parameters(), lr=params["lr"])
# optimizer = torch.optim.Adam(params=model.parameters(), lr=params["lr"], betas=(0.5,0.9))
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

        imgs = data["img"].to(device)

        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=params["enable_amp"]
        ):
            posteriors = model.encode(img=imgs)
            enc = posteriors.sample()
            imgs_rec = model.decode(enc)

            # losses
            # reconstruciton loss
            rec_loss = torch.abs(imgs.contiguous() - imgs_rec.contiguous())
            nll_loss = rec_loss
            weights = 1
            weighted_nll_loss = nll_loss * weights
            weighted_nll_loss = (
                torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
            )
            # kl loss
            kl_loss = posteriors.kl()
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            # total loss
            if params["loss"] == "mae_kl":
                loss = weighted_nll_loss + params["kl_weight"] * kl_loss

            if torch.isnan(loss):
                print(" NaN!")

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # ----------------------------------------------------------------------
        # evaluation
        ssim = utils_eva.SSIM_tb(
            img_true=imgs, img_test=imgs_rec, data_range=None, version_wang=False
        )
        psnr = utils_eva.PSNR_tb(img_true=imgs, img_test=imgs_rec, data_range=None)

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
            print("validating ...")
            running_val_ssim, running_val_psnr = 0, 0
            for i_batch_val, sample_val in enumerate(dataloader_validation):
                imgs_val = sample_val["img"].to(device)

                with torch.no_grad():
                    enc_val = model.encode(img=imgs_val).sample()
                    imgs_rec_val = model.decode(enc_val)

                # evaluation
                ssim_val = utils_eva.SSIM_tb(
                    img_true=imgs_val,
                    img_test=imgs_rec_val,
                    data_range=None,
                    version_wang=False,
                )
                psnr_val = utils_eva.PSNR_tb(
                    img_true=imgs_val, img_test=imgs_rec_val, data_range=None
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
