"""
Model Training.
- (2D image,) to (2D image,)
"""

import numpy as np
import torch, os, tqdm, json, pandas, datetime
from torchinfo import summary
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from models.care import CARE
from models.dfcan import DFCAN
from models.unifmir import UniModel

import utils.data as utils_data
import utils.evaluation as utils_eva
import utils.optim as utils_optim
import utils.loss_functions as loss_func

# ------------------------------------------------------------------------------
# parameters
# ------------------------------------------------------------------------------
params = {
    # device
    "device": "cuda:0",
    "random_seed": 7,
    "data_shuffle": True,
    "num_workers": 8,
    "complie": True,
    # mixed-precision ----------------------------------------------------------
    "enable_amp": False,
    "enable_gradscaler": False,
    # model parameters ---------------------------------------------------------
    "dim": 2,
    # "model_name": "care",
    "model_name": "dfcan",
    # "model_name": "unifmir",
    # loss function ------------------------------------------------------------
    # "loss": "mse",
    "loss": "mae",
    # "loss": "mae_mse",
    # learning rate ------------------------------------------------------------
    "lr": 0.0001,
    "batch_size": 16,
    "num_epochs": 300,
    "warm_up": 0,
    "lr_decay_every_iter": 10000 * 10,
    "lr_decay_rate": 0.5,
    "lr_min": 0.0000001,
    # validation ---------------------------------------------------------------
    "enable_validation": True,
    "frac_val": 0.01,
    "validate_every_iter": 5000,
    # dataset ------------------------------------------------------------------
    "path_dataset_excel": "dataset_train_transformer-v2.xlsx",
    "sheet_name": "64x64",
    "augmentation": 3,
    "use_clean_data": True,
    "data_clip": None,
    # "datasets_id": [],
    # "datasets_id": [
    #     "biosr-cpp",
    #     "biosr-er",
    #     "biosr-actin",
    #     "biosr-mt",
    # ],
    # "datasets_id": ["biosr-cpp"],
    # "datasets_id": ["biosr-er"],
    # "datasets_id": ["biosr-actin"],
    # "datasets_id": ["biosr-mt"],
    # "datasets_id": ["biosr-actinnl"],
    "datasets_id": [
        # "biotisr-mt-sr-1"
        # "biotisr-mt-sr-2"
        # "biotisr-mt-sr-3"
        # "biotisr-mito-sr-1"
        # "biotisr-mito-sr-2"
        # "biotisr-mito-sr-3"
        # "biotisr-factin-nonlinear-sr-1"
        # "biotisr-factin-nonlinear-sr-2"
        # "biotisr-factin-nonlinear-sr-3",
        # "biotisr-ccp-sr-1",
        # "biotisr-ccp-sr-1-2",
        # "biotisr-ccp-sr-1-4",
        # "biotisr-ccp-sr-1-8",
        # "biotisr-ccp-sr-1-16",
        # "biotisr-ccp-sr-1-64",
        "biotisr-ccp-sr-1-256",
        # "biotisr-ccp-sr-2",
        # "biotisr-ccp-sr-2-2",
        # "biotisr-ccp-sr-2-4",
        # "biotisr-ccp-sr-2-8",
        # "biotisr-ccp-sr-2-16",
        # "biotisr-ccp-sr-2-64",
        # "biotisr-ccp-sr-2-256",
        # "biotisr-ccp-sr-3",
        # "biotisr-ccp-sr-3-2",
        # "biotisr-ccp-sr-3-4",
        # "biotisr-ccp-sr-3-8",
        # "biotisr-ccp-sr-3-16",
        # "biotisr-ccp-sr-3-64",
        # "biotisr-ccp-sr-3-256",
        # "biotisr-factin-sr-1",
        # "biotisr-factin-sr-2",
        # "biotisr-factin-sr-3",
        # "biotisr-lysosome-sr-1",
        # "biotisr-lysosome-sr-1-2",
        # "biotisr-lysosome-sr-1-4",
        # "biotisr-lysosome-sr-1-8",
        # "biotisr-lysosome-sr-1-16",
        # "biotisr-lysosome-sr-1-64",
        # "biotisr-lysosome-sr-1-256",
        # "biotisr-lysosome-sr-2",
        # "biotisr-lysosome-sr-2-2",
        # "biotisr-lysosome-sr-2-4",
        # "biotisr-lysosome-sr-2-8",
        # "biotisr-lysosome-sr-2-16",
        # "biotisr-lysosome-sr-2-64",
        # "biotisr-lysosome-sr-2-256",
        # "biotisr-lysosome-sr-3",
        # "biotisr-lysosome-sr-3-2",
        # "biotisr-lysosome-sr-3-4",
        # "biotisr-lysosome-sr-3-8",
        # "biotisr-lysosome-sr-3-16",
        # "biotisr-lysosome-sr-3-64",
        # "biotisr-lysosome-sr-3-256",
    ],
    "task": [],
    # "task": ["sr"],
    # "task": ["dcv"],
    # "task": ["dn"],
    # "task": ["iso"],
    "scale_factor": 1,
    # checkpoints --------------------------------------------------------------
    "suffix": "_newnorm-v2",
    # "suffix": "_newnorm-v2-dn",
    # "suffix": "_newnorm-v2-all",
    "path_checkpoints": "checkpoints\conditional",
    "save_every_iter": 5000,
    "plot_every_iter": 100,
    "print_loss": False,
    # saved model --------------------------------------------------------
    "finetune": True,
    "saved_checkpoint": None,
}

# ------------------------------------------------------------------------------
device = torch.device(params["device"])
torch.manual_seed(params["random_seed"])

if params["finetune"]:
    params["suffix"] = params["suffix"] + "-ft-" + params["datasets_id"][0]
    params["path_checkpoints"] = os.path.join(params["path_checkpoints"], "finetune")
    params.update(
        {
            "save_every_iter": 500,
            "plot_every_iter": 100,
            "frac_val": 0.05,
            "lr": 0.0001,
            "num_epochs": 10,
            "lr_decay_every_iter": 10000,
            "validate_every_iter": 500,
            "use_clean_data": False,
        }
    )
    if params["model_name"] == "unifmir":
        params["saved_checkpoint"] = (
            "checkpoints\conditional\\unifmir_mae_bs_1_lr_0.0001_newnorm-v2-all\epoch_1_iter_4300000.pt"
        )
        params["num_epochs"] = 150

params["path_checkpoints"] = utils_data.win2linux(params["path_checkpoints"])
params["saved_checkpoint"] = utils_data.win2linux(params["saved_checkpoint"])

# special settings for UniFMIR model
if params["model_name"] == "unifmir":
    params.update(
        {
            "data_output_type": "ii-task",
            "batch_size": 1,
            "lr": 0.0001,
        }
    )
else:
    params.update({"data_output_type": "ii"})

# decrease the number fo valiation data when using all the datasets for training
if not params["datasets_id"] and not params["task"]:
    params["frac_val"] = 0.001

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


print("Number of datasets:", data_frame.shape[0])

path_dataset_lr = list(data_frame["path_lr"])
path_dataset_hr = list(data_frame["path_hr"])
path_index_file = list(data_frame["path_index"])
dataset_index = list(data_frame["index"])
dataset_scale_factor_lr = list(data_frame["sf_lr"])
dataset_scale_factor_hr = list(data_frame["sf_hr"])

if params["model_name"] == "unifmir":
    tasks = list(data_frame["task"])
else:
    tasks = None

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
    task=tasks,
    output_type=params["data_output_type"],
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
)
num_batches_train = len(dataloader_train)

# ------------------------------------------------------------------------------
# data infomation
img_lr_shape = dataset_train[0]["lr"].shape
img_hr_shape = dataset_train[0]["hr"].shape

print(f"- Num of Batches (train| valid): {num_batches_train}|{num_batch_val}")
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

if params["model_name"] == "unifmir":
    model = UniModel(
        in_channels=1,
        out_channels=1,
        tsk=0,
        img_size=(64, 64),
        patch_size=1,
        embed_dim=180 // 2,
        depths=[6, 6, 6],
        num_heads=[6, 6, 6],
        window_size=8,
        mlp_ratio=2,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0,
        attn_drop_rate=0,
        drop_path_rate=0.1,
        norm_layer=torch.nn.LayerNorm,
        patch_norm=True,
        use_checkpoint=False,
        num_feat=32,
        srscale=1,
    )

model.to(device=device)
try:
    summary(model=model, input_size=(1,) + img_lr_shape)
except:
    print("Fail to show the summary of model.")

# complie
if params["complie"]:
    model = torch.compile(model)

# ------------------------------------------------------------------------------
# pre-trained model parameters
if params["saved_checkpoint"] is not None:
    print("- Load saved pre-trained model parameters:", params["saved_checkpoint"])
    state_dict = torch.load(
        params["saved_checkpoint"],
        map_location=device,
        weights_only=True,
    )["model_state_dict"]
    state_dict = utils_optim.on_load_checkpoint(
        state_dict, complie_mode=params["complie"]
    )
    model.load_state_dict(state_dict)
    start_iter = params["saved_checkpoint"].split(".")[-2].split("_")[-1]
    start_iter = int(start_iter)
else:
    start_iter = 0

# ------------------------------------------------------------------------------
if params["finetune"]:
    if params["model_name"] == "unifmir":
        start_iter = 0
        model_parameters = model.finetune()
        print("- Finetune model parameters:")
        for name, param in model_parameters:
            print("  ", name, param.shape)
    else:
        print(
            "[INFO] Other models will be trained from scratch, except for the [unifmir] model."
        )
        model_parameters = list(model.named_parameters())
else:
    model_parameters = list(model.named_parameters())

torch.set_float32_matmul_precision("high")

print("-" * 80)
print("Number of trainable parameters: ")
print(sum(p[1].numel() for p in model_parameters if p[1].requires_grad))
print("-" * 80)

# ------------------------------------------------------------------------------
# optimization
# ------------------------------------------------------------------------------
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
            ncols=100,
        )

        # --------------------------------------------------------------------------
        for i_batch, data in enumerate(dataloader_train):
            i_iter = i_batch + i_epoch * num_batches_train + start_iter
            pbar.update(1)

            imgs_lr, imgs_hr = (data["lr"].to(device), data["hr"].to(device))

            if params["model_name"] == "unifmir":
                task = data["task"].to(device)

            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=params["enable_amp"]
            ):
                if params["model_name"] == "unifmir":
                    imgs_est = model(imgs_lr, task)
                else:
                    imgs_est = model(imgs_lr)

                if params["loss"] == "mse":
                    loss = torch.nn.MSELoss()(imgs_est, imgs_hr)
                if params["loss"] == "mae":
                    loss = torch.nn.L1Loss()(imgs_est, imgs_hr)
                if params["loss"] == "mae_mse":
                    loss = loss_func.mae_mse(imgs_est, imgs_hr)

                if torch.isnan(loss):
                    print(" NaN!")

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # ----------------------------------------------------------------------
            # evaluation
            if params["print_loss"]:
                imgs_est = utils_eva.linear_transform(imgs_hr, imgs_est, axis=(2, 3))
                ssim = utils_eva.SSIM_tb(img_true=imgs_hr, img_test=imgs_est)
                psnr = utils_eva.PSNR_tb(img_true=imgs_hr, img_test=imgs_est)

                if i_iter % 10 == 0:
                    pbar.set_postfix(
                        Loss=f"{loss.cpu().detach().numpy():>.4f}, PSNR: {psnr:>.4f}, SSIM: {ssim:>.4f}"
                    )
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

            # ----------------------------------------------------------------------
            # validation
            if (i_iter % params["validate_every_iter"] == 0) and (
                params["enable_validation"] == True
            ):
                pbar_val = tqdm.tqdm(desc="VALIDATION", total=num_batch_val, ncols=100)
                model.eval()  # convert model to evaluation model

                # ------------------------------------------------------------------
                running_val_ssim, running_val_psnr, running_val_mse = 0, 0, 0
                for i_batch_val, data_val in enumerate(dataloader_val):
                    imgs_lr_val, imgs_hr_val = (
                        data_val["lr"].to(device),
                        data_val["hr"].to(device),
                    )

                    if params["model_name"] == "unifmir":
                        task_val = data_val["task"].to(device)

                    with torch.no_grad():
                        if params["model_name"] == "unifmir":
                            imgs_est_val = model(imgs_lr_val, task_val)
                        else:
                            imgs_est_val = model(imgs_lr_val)

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

                    if i_batch_val % 10 == 0:
                        pbar_val.set_postfix(
                            PSNR="{:>.6f}, SSIM= {:>.6f}, MSE={:>.4f}".format(
                                running_val_psnr / (i_batch_val + 1),
                                running_val_ssim / (i_batch_val + 1),
                                running_val_mse / (i_batch_val + 1),
                            )
                        )
                    pbar_val.update(1)

                del imgs_lr_val, imgs_hr_val

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
