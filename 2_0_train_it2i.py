"""
The code can be used for both model training and fine-tuning.
- (single-channel 2D image, text) to (single-channel 2D image,)
- (mutil-channel 2D image, text) to (single-channel 2D image,)

If you want to use all dataset to train the network,
- set params['datasets_id'] to [].
- set params['task'] to [].
- set params['finetune'] to False.
when given the path of the pre-trained model in params['saved_checkpoint'],
the network will be trained from the pre-trained model.

If you want to train from scratch,
- set params['saved_checkpoint'] to None.

If you want to finetune the network,
- set params['finetune'] to True.
- set params['saved_checkpoint'] to the path of the pre-trained model.
- set params['datasets_id'] to the dataset id you want to finetune.
- set params['task'] to the task you want to finetune.

"""

import torch, os, tqdm, json, pandas, datetime
from torchinfo import summary
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from models.unet_sd_c import UNetModel

import utils.data as utils_data
import utils.evaluation as utils_eva
import utils.optim as utils_optim
import utils.loss_functions as utils_loss


# ------------------------------------------------------------------------------
# parameters
# ------------------------------------------------------------------------------
params = {
    # device
    "device": "cuda:1",
    "random_seed": 7,
    "data_shuffle": True,
    "num_workers": 3,
    "pin_memory": True,
    "cudnn-auto-tunner": True,
    "complie": True,
    # "complie": False,
    # mixed-precision ----------------------------------------------------------
    "enable_amp": True,
    "enable_gradscaler": True,
    # model parameters ---------------------------------------------------------
    "dim": 2,
    "model_name": "unet_sd_c",
    "in_channels": 1,
    "out_channels": 1,
    "channels": 320,
    "n_res_blocks": 1,
    "attention_levels": [0, 1, 2, 3],
    "channel_multipliers": [1, 2, 4, 4],
    "n_heads": 8,
    "tf_layers": 1,
    "d_cond": 768,
    # "d_cond": None,
    "pixel_shuffle": False,
    "scale_factor": 4,
    # loss function ------------------------------------------------------------
    # "loss": "mse",
    "loss": "mae",
    # learning rate ------------------------------------------------------------
    "lr": 0.00001,
    "batch_size": 16,
    # "batch_size": 8,
    "num_epochs": 20,
    "warm_up": 0,
    "lr_decay_every_iter": 10000 * 10,
    "lr_decay_rate": 0.5,
    "lr_min": 0.0000001,
    # validation ---------------------------------------------------------------
    "enable_validation": True,
    "frac_val": 0.001,
    "validate_every_iter": 5000,
    # dataset ------------------------------------------------------------------
    "path_dataset_excel": "dataset_train_transformer-v2.xlsx",
    "sheet_name": "64x64",
    "augmentation": 3,
    "use_clean_data": True,
    # "data_clip": (0.0, 2.5),
    "data_clip": None,
    # "datasets_id": [],
    "datasets_id": [
        # "biotisr-mt-sr-1",
        # "biotisr-mt-sr-2",
        # "biotisr-mt-sr-3",
        # "biotisr-mt-sr-1-2",
        # "biotisr-mt-sr-2-2",
        # "biotisr-mt-sr-3-2",
        # "biotisr-mito-sr-1",
        # "biotisr-mito-sr-2",
        # "biotisr-mito-sr-3",
        # "biotisr-mito-sr-1-2",
        # "biotisr-mito-sr-2-2",
        # "biotisr-mito-sr-3-2",
        # "biotisr-factin-nonlinear-sr-1",
        # "biotisr-factin-nonlinear-sr-2",
        # "biotisr-factin-nonlinear-sr-3",
        # "biotisr-ccp-sr-1",
        # "biotisr-ccp-sr-1-2",
        # "biotisr-ccp-sr-2",
        # "biotisr-ccp-sr-2-2",
        # "biotisr-ccp-sr-3",
        # "biotisr-ccp-sr-3-2",
        # "biotisr-factin-sr-1",
        # "biotisr-factin-sr-2",
        # "biotisr-factin-sr-3",
        # "biotisr-lysosome-sr-1",
        # "biotisr-lysosome-sr-2",
        # "biotisr-lysosome-sr-3",
        # "biotisr-lysosome-sr-1-2",
        # "biotisr-lysosome-sr-2-2",
        # "biotisr-lysosome-sr-3-2",
        # "rcan3d-c2s-mt-dcv-mc",  # 16 bs, 400 epoch
        "rcan3d-c2s-mt-dcv-mc-3d",  # 16 bs, 2100 epoch
        # "rcan3d-c2s-npc-dcv-mc",
        # "rcan3d-c2s-sirdna-dcv-mc",
        # "care-projection-flywing-0",  # 6500 epoch
        # "care-projection-flywing-1",  # 6500 epoch
        # "care-projection-flywing-2",  # 6500 epoch
        # "care-projection-flywing-3",  # 6500 epoch
        # "care-denoising-flywing-0",  # 6500 epoch
        # "care-denoising-flywing-1",  # 6500 epoch
        # "care-denoising-flywing-2",  # 6500 epoch
        # "care-denoising-flywing-3",  # 6500 epoch
        # "care-projection-flywing-0-dn",
        # "care-projection-flywing-1-dn",
        # "care-projection-flywing-2-dn",
        # "care-projection-flywing-3-dn",
        # "biosr-factinnl-sr3-1",
        # "biosr-factinnl-sr3-2",
        # "biosr-factinnl-sr3-3",
        # "biosr-factinnl-sr3-4",
        # "biosr-factinnl-sr3-5",
        # "biosr-factinnl-sr3-6",  # 1000 epoch
        # "biosr-factinnl-sr3-7",
        # "biosr-factinnl-sr3-8",
        # "biosr-factinnl-sr3-9",  # 1000 epoch 16 bs
        # "care-iso-drosophila-3d",  # 50 epoch
        # "synprot-channe-0-64",  # 200 epoch
        # "synprot-channe-0-128",  # 700 epoch 16 batch size
        # "synprot-channe-1-64",  # 200 epoch
        # "synprot-channe-1-128",  # 700 epoch
        # "synprot-channe-0-64-granule",
        # "synprot-channe-0-128-granule",
        # "synprot-channe-1-64-granule",
        # "synprot-channe-1-128-granule",
        # "synprot-channe-0-64-reg",  # 200 epoch
        # "synprot-channe-0-128-reg",  # 400 epoch 8 batch size
        # "synprot-channe-1-64-reg",  # 200 epoch
        # "synprot-channe-1-128-reg",  # 400 epoch
        # "dl-smlm-microtubule-64",  # 600 epoch 16 batch size
        # "dl-smlm-microtubule-128",  # 1500 epoch
    ],
    "task": [],
    # "task": ["sr"],
    # "task": ["dn"],
    # "task": ["dcv"],
    # "task": ["iso"],
    "path_text": "text\\v2",
    # "embaedding_type": "",
    # "embaedding_type": "_ALL_256",
    "embaedding_type": "_ALL_160",
    # "embaedding_type": "_ALL_wot_160",
    # "embaedding_type": "_TSpixel_77",
    # "embaedding_type": "_TSmicro_77",
    # "embaedding_type": "_TS_77",
    # "embaedding_type": "_T_77",
    # checkpoints --------------------------------------------------------------
    # "suffix": "_all_newnorm_ALL-v2-160-res1-att0123-crossx",
    "suffix": "_all_newnorm_ALL-v2-160-res1-att0123",
    # "suffix": "_all_newnorm_ALL-v2-160-res1-att0123-wot",
    # "suffix": "_all_newnorm_ALL-v2-res1-att0123-T77",
    "path_checkpoints": "checkpoints\conditional",
    "save_every_iter": 5000,
    "plot_every_iter": 100,
    "print_loss": False,
    # saved model --------------------------------------------------------------
    # use when training from scrach --------------------------------------------
    # "finetune": False,
    # "saved_checkpoint": None,
    # use when finetuning ------------------------------------------------------
    "finetune": True,
    "finetune-strategy": "in-out",  # finetune the input and output part convolutional layers
    # "finetune-strategy": "in", # finetune the input part convolutional layers
    # "finetune-strategy": "out",# finetune the output part convolutional layers
    # "finetune-strategy": "all",  # finetune the output part convolutional layers
    "saved_checkpoint": "checkpoints\conditional\\unet_sd_c_mae_bs_16_lr_1e-05_all_newnorm_ALL-v2-160-res1-att0123\epoch_0_iter_700000.pt",  # fintune base
}

# ------------------------------------------------------------------------------
device = torch.device(params["device"])
torch.manual_seed(params["random_seed"])

# finetune parameters update ---------------------------------------------------
if params["finetune"] == True:
    params["path_text"] = params["path_text"] + "-finetune"
    params["path_checkpoints"] = os.path.join(params["path_checkpoints"], "finetune")
    params["suffix"] = (
        params["suffix"]
        + "-ft-"
        + params["finetune-strategy"]
        + "-"
        + params["datasets_id"][0]
    )
    params.update(
        {
            "save_every_iter": 1000,
            "plot_every_iter": 100,
            "frac_val": 0.05,
            "lr": 0.00001,
            # "lr": 0.0001,
            # "lr": 0.001,
            # "num_epochs": 2000,
            "num_epochs": 2100,
            # "num_epochs": 1000,
            # "num_epochs": 1500,
            # "num_epochs": 6500,
            # "num_epochs": 400,
            # "num_epochs": 600,
            # "num_epochs": 200,
            # "num_epochs": 175,
            # "num_epochs": 50,
            "lr_decay_every_iter": 10000,
            "validate_every_iter": 500,
            "use_clean_data": False,
        }
    )

path_dataset_text = os.path.join(
    params["path_text"], "dataset_text" + params["embaedding_type"]
)

params["path_checkpoints"] = utils_data.win2linux(params["path_checkpoints"])
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
os.makedirs(path_save_model, exist_ok=True)


# ------------------------------------------------------------------------------
# dataset
# ------------------------------------------------------------------------------
if params["finetune"] == False:
    data_frame = pandas.read_excel(
        params["path_dataset_excel"], sheet_name=params["sheet_name"]
    )
    # add augmented data
    if params["augmentation"] > 0:
        data_frame_aug = pandas.read_excel(
            params["path_dataset_excel"], sheet_name=params["sheet_name"] + "-aug"
        )
        data_frame = pandas.concat(
            [data_frame] + [data_frame_aug] * params["augmentation"]
        )
else:
    data_frame = pandas.read_excel(
        params["path_dataset_excel"], sheet_name=params["sheet_name"] + "-finetune"
    )


if params["task"]:
    data_frame = data_frame[data_frame["task"].isin(params["task"])]

if params["datasets_id"]:
    data_frame = data_frame[data_frame["id"].isin(params["datasets_id"])]

path_dataset_lr = list(data_frame["path_lr"])
path_dataset_hr = list(data_frame["path_hr"])
path_index_file = list(data_frame["path_index"])
dataset_index = list(data_frame["index"])
dataset_scale_factor_lr = list(data_frame["sf_lr"])
dataset_scale_factor_hr = list(data_frame["sf_hr"])

if params["finetune"] == True:
    # only when using one dataset to finetune
    in_channels = int(data_frame["in_channels"].iloc[0])
    out_channels = int(data_frame["out_channels"].iloc[0])
    params.update(
        {
            "in_channels": in_channels,
            "out_channels": out_channels,
        }
    )

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
    path_dataset_text_embedding=path_dataset_text,
    transform=transform,
    scale_factor_lr=dataset_scale_factor_lr,
    scale_factor_hr=dataset_scale_factor_hr,
    output_type="ii-text",
    use_clean_data=params["use_clean_data"],
    rotflip=False,
    clip=params["data_clip"],
)

# create training and validation dataset
dataloader_train, dataloader_val = None, None
if params["enable_validation"]:
    # split whole dataset into training and validation dataset
    dataset_train, dataset_validation = random_split(
        dataset_all,
        [1.0 - params["frac_val"], params["frac_val"]],
        generator=torch.Generator().manual_seed(7),
    )

    dataloader_val = DataLoader(
        dataset=dataset_validation,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=params["num_workers"],
        pin_memory=params["pin_memory"],
    )
    num_batch_val = len(dataloader_val)
else:
    dataset_train = dataset_all
    num_batch_val = 0

dataloader_train = DataLoader(
    dataset=dataset_train,
    batch_size=params["batch_size"],
    shuffle=params["data_shuffle"],
    num_workers=params["num_workers"],
    pin_memory=params["pin_memory"],
)
num_batches_train = len(dataloader_train)

# ------------------------------------------------------------------------------
# data infomation
img_lr_shape = dataset_train[0]["lr"].shape
img_hr_shape = dataset_train[0]["hr"].shape
text_shape = dataset_train[0]["text"].shape

print(f"[INFO] Num of Batches (train| valid): {num_batches_train}|{num_batch_val}")
print(
    f"[INFO] Input shape: ({img_lr_shape}, {text_shape})",
)
print(f"[INFO] GT shape: {img_hr_shape}")

# ------------------------------------------------------------------------------
# save parameters
# ------------------------------------------------------------------------------
with open(
    os.path.join(path_save_model, f"parameters-{datetime.date.today()}.json"), "w"
) as f:
    f.write(json.dumps(params, indent=1))

utils_data.print_dict(params)

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

with torch.autocast("cuda", torch.float16, enabled=params["enable_amp"]):
    dtype = torch.float16 if params["enable_amp"] else torch.float32
    summary(
        model=model,
        input_size=((1,) + img_lr_shape, (1,), (1, text_shape[0], 768)),
        dtypes=(dtype,) * 3,
        device=params["device"],
    )

model.to(device=device)

# complie
if params["complie"]:
    model = torch.compile(model)

torch.backends.cudnn.benchmark = params["cudnn-auto-tunner"]  # cuDNN auto-tunner
torch.set_float32_matmul_precision("high")

# ------------------------------------------------------------------------------
# pre-trained model parameters
if params["saved_checkpoint"] is not None:
    print("[INFO] Load saved pre-trained model parameters:", params["saved_checkpoint"])
    state_dict = torch.load(
        params["saved_checkpoint"], map_location=device, weights_only=True
    )["model_state_dict"]
    # del prefix for complied model
    state_dict = utils_optim.on_load_checkpoint(
        checkpoint=state_dict, complie_mode=params["complie"]
    )
    # --------------------------------------------------------------------------
    # if the input and output channel is different from the original model
    # modify the input and output convolutional layers weights
    if params["in_channels"] > 1:
        if params["complie"]:
            name_layer = "_orig_mod.input_blocks.0.0.weight"
        else:
            name_layer = "input_blocks.0.0.weight"
        state_dict[name_layer] = (
            torch.repeat_interleave(
                state_dict[name_layer], params["in_channels"], dim=1
            )
            / params["in_channels"]
        )
    if params["out_channels"] > 1:
        if params["complie"]:
            name_layer = "_orig_mod.out.2.weight"
            name_layer_bias = "_orig_mod.out.2.bias"
        else:
            name_layer = "out.2.weight"
            name_layer_bias = "out.2.bias"
        state_dict[name_layer] = (
            torch.repeat_interleave(
                state_dict[name_layer], params["out_channels"], dim=0
            )
            / params["out_channels"]
        )
        state_dict[name_layer_bias] = torch.repeat_interleave(
            state_dict[name_layer_bias], params["out_channels"], dim=0
        )

    # --------------------------------------------------------------------------
    model.load_state_dict(state_dict, strict=not params["finetune"])
    start_iter = params["saved_checkpoint"].split(".")[-2].split("_")[-1]
    start_iter = int(start_iter)
    del state_dict
else:
    start_iter = 0

if params["finetune"] == True:
    start_iter = 0
    model_parameters = model.finetune(strategy=params["finetune-strategy"])
    # print the name of parameters that are not frozen
    print(f"[INFO] Finetune model parameters:")
    for name, param in model_parameters:
        print(f"- ({name, param.shape})")
else:
    model_parameters = model.named_parameters()

print("[INFO] Number of trainable parameters:")
model_parameters = list(model_parameters)  # 20250728
print(f"[INFO] {sum(p[1].numel() for p in model_parameters if p[1].requires_grad)}")

# ------------------------------------------------------------------------------
# optimization
# ------------------------------------------------------------------------------
# optimizer = torch.optim.Adam(params=model_parameters, lr=params["lr"])
optimizer = torch.optim.AdamW(params=model_parameters, lr=params["lr"])
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
# trains
# ------------------------------------------------------------------------------
print(
    f"[INFO] Batch size: {params['batch_size']} | Num of Batches: {num_batches_train}"
)
print(f"[INFO] save model to {path_save_model}")

scaler = torch.GradScaler("cuda", enabled=params["enable_gradscaler"])
# create zero time embedding
time_embed = torch.zeros(size=(params["batch_size"],)).to(device)
time_embed_val = torch.zeros(size=(params["batch_size"],)).to(device)

enable_pbar_echo = True if params["num_epochs"] > 100 else False
enable_pbar_iter = True if params["num_epochs"] <= 100 else False
pbar_echo = tqdm.tqdm(
    total=params["num_epochs"],
    desc=f"[INFO] Epoches",
    leave=enable_pbar_echo,
    ncols=100,
)

try:
    for i_epoch in range(params["num_epochs"]):
        pbar_echo.update(1)

        pbar_iter = tqdm.tqdm(
            total=num_batches_train,
            desc=f"[INFO] Epoch {i_epoch + 1}|{params['num_epochs']}",
            leave=enable_pbar_iter,
            ncols=100,
        )

        # ----------------------------------------------------------------------
        for i_batch, data in enumerate(dataloader_train):
            i_iter = i_batch + i_epoch * num_batches_train + start_iter
            pbar_iter.update(1)
            # skip
            # if i_iter < start_iter:
            #     continue

            imgs_lr, imgs_hr = data["lr"].to(device), data["hr"].to(device)

            # text embeddings
            if (params["d_cond"] == 0) or (params["d_cond"] is None):
                text_embed = None
            else:
                text_embed = data["text"].to(device)

            with torch.autocast("cuda", torch.float16, enabled=params["enable_amp"]):

                # predict
                imgs_est = model(imgs_lr, time_embed, text_embed)

                if params["loss"] == "mse":
                    loss = torch.nn.MSELoss()(imgs_est, imgs_hr)
                if params["loss"] == "mae":
                    loss = torch.nn.L1Loss()(imgs_est, imgs_hr)
                if params["loss"] == "msew":
                    loss = utils_loss.MSE_w(img_est=imgs_est, img_gt=imgs_hr, scale=0.1)

            if torch.isnan(loss):
                print("-" * 50)
                print("\nLoss is NaN!")
                print(f"- input max/min: {imgs_lr.max()} {imgs_hr.min()}")
                print(f"- output max/min: {imgs_est.max()} {imgs_est.min()}")
                print(f"- estimation max/min: {imgs_est.max()} {imgs_est.min()}")
                print("-" * 50)
                pbar_iter.close()
                pbar_echo.close()
                log_writer.close()
                exit()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # ------------------------------------------------------------------
            # evaluation
            if params["print_loss"]:

                imgs_est = utils_eva.linear_transform(imgs_hr, imgs_est, axis=(2, 3))
                ssim = utils_eva.SSIM_tb(img_true=imgs_hr, img_test=imgs_est)
                psnr = utils_eva.PSNR_tb(img_true=imgs_hr, img_test=imgs_est)

                pbar_iter.set_postfix(
                    loss=f"{loss.cpu().detach().numpy():>.4f}, PSNR: {psnr:>.4f}, SSIM: {ssim:>.4f}"
                )

            # ------------------------------------------------------------------
            # update learning rate
            LR_schedule.update(i_iter=i_iter)

            # ------------------------------------------------------------------
            # log
            if i_iter % params["plot_every_iter"] == 0:
                if log_writer is not None:
                    log_writer.add_scalar(
                        "Learning rate", optimizer.param_groups[-1]["lr"], i_iter
                    )
                    log_writer.add_scalar(params["loss"], loss, i_iter)
                    if params["print_loss"]:
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

            # ------------------------------------------------------------------
            # validation
            if (i_iter % params["validate_every_iter"] == 0) and (
                params["enable_validation"] == True
            ):
                pbar_val = tqdm.tqdm(
                    desc="[INFO] VALIDATION", total=num_batch_val, ncols=100
                )
                model.eval()  # convert model to evaluation model

                # --------------------------------------------------------------
                running_val_ssim, running_val_psnr, running_val_mse = 0, 0, 0
                with torch.autocast(
                    "cuda", torch.float16, enabled=params["enable_amp"]
                ):
                    with torch.no_grad():
                        for i_batch_val, data_val in enumerate(dataloader_val):
                            imgs_lr_val, imgs_hr_val, text_embed_val = (
                                data_val["lr"].to(device),
                                data_val["hr"].to(device),
                                data_val["text"].to(device),
                            )

                            imgs_est_val = model(
                                imgs_lr_val, time_embed_val, text_embed_val
                            )

                            # evaluation
                            # linear transform
                            imgs_est_val = utils_eva.linear_transform(
                                img_true=imgs_hr_val, img_test=imgs_est_val, axis=(2, 3)
                            )

                            mse_val = utils_eva.MSE(imgs_hr_val, imgs_est_val)
                            ssim_val = utils_eva.SSIM_tb(imgs_hr_val, imgs_est_val)
                            psnr_val = utils_eva.PSNR_tb(imgs_hr_val, imgs_est_val)

                            if not np.isinf(psnr_val):
                                running_val_psnr += psnr_val
                                running_val_ssim += ssim_val
                                running_val_mse += mse_val

                            pbar_val.set_postfix(
                                PSNR="{:>.4f}, SSIM={:>.4f}, MSE={:>.4f}".format(
                                    running_val_psnr / (i_batch_val + 1),
                                    running_val_ssim / (i_batch_val + 1),
                                    running_val_mse / (i_batch_val + 1),
                                )
                            )
                            pbar_val.update(1)

                del imgs_lr_val, imgs_hr_val, text_embed_val

                if log_writer is not None:
                    log_writer.add_scalar(
                        "psnr_val", running_val_psnr / num_batch_val, i_iter
                    )
                    log_writer.add_scalar(
                        "ssim_val", running_val_ssim / num_batch_val, i_iter
                    )
                    log_writer.add_scalar(
                        "mse_val", running_val_mse / num_batch_val, i_iter
                    )
                pbar_val.close()
                # convert model to train mode
                model.train(True)
        pbar_iter.close()
    pbar_echo.close()

    # --------------------------------------------------------------------------
    # save and finish
    # --------------------------------------------------------------------------
    print(f"\n[INFO] save model (epoch: {i_epoch}, iter: {i_iter})")

    # saving general checkpoint
    model_dict = {"model_state_dict": getattr(model, "_orig_mod", model).state_dict()}
    torch.save(
        model_dict,
        os.path.join(path_save_model, f"epoch_{i_epoch}_iter_{i_iter+1}.pt"),
    )

    log_writer.flush()
    log_writer.close()
    print("[INFO] Training done.")

except KeyboardInterrupt:
    print("\n[INFO] Training stop, saving model ...")
    print(f"\n[INFO] Save model (epoch: {i_epoch}, iter: {i_iter})")

    # saving general checkpoint
    model_dict = {"model_state_dict": getattr(model, "_orig_mod", model).state_dict()}
    torch.save(
        model_dict,
        os.path.join(path_save_model, f"epoch_{i_epoch}_iter_{i_iter+1}.pt"),
    )

    pbar_iter.close()
    pbar_echo.close()
    log_writer.flush()
    log_writer.close()
    print("[INFO] Training done.")
