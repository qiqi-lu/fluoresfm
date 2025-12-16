import numpy as np
import torch, os, pandas, math, tqdm, datetime
import skimage.io as io
from models.unet import UNet
from models.care import CARE
from models.dfcan import DFCAN
from models.unifmir import UniModel

import utils.data as utils_data
import utils.evaluation as utils_eva
import utils.optim as utils_optim
from finetune_checkpoints import checkpoints_finetune

# ------------------------------------------------------------------------------
# parameters
# ------------------------------------------------------------------------------
checkpoints = checkpoints_finetune

params = {
    "device": "cuda:0",
    # dataset ------------------------------------------------------------------
    "dim": 2,
    "path_dataset_test": "dataset_test-v2.xlsx",
    "scale_factor": 1,
    "num_sample": 8,
    "percentiles": (0.03, 0.995),
    "patch_image": True,
    "patch_size": 256,
    # output -------------------------------------------------------------------
    "path_output": "results\\predictions",
}

# ------------------------------------------------------------------------------
params["path_output"] = utils_data.win2linux(params["path_output"])
utils_data.print_dict(params)

print("load dataset information ...")
datasets_frame = pandas.read_excel(params["path_dataset_test"])
device = torch.device(params["device"])
output_normalizer = utils_data.NormalizePercentile(0.03, 0.995)
input_normalizer = utils_data.NormalizePercentile(
    params["percentiles"][0], params["percentiles"][1]
)
num_checkpoints = len(checkpoints)
num_datasets = 1

print("-" * 50)
print("Number of checkpoints:", num_checkpoints)
print("number of datasets:", num_datasets)

# ------------------------------------------------------------------------------
#                                      PREDICT
# ------------------------------------------------------------------------------
for checkpoint in checkpoints:
    print("-" * 50)
    print(f"Checkpoint: {checkpoint}")
    model_name, model_suffix, model_path, dataset_finetune = checkpoint

    if dataset_finetune is not list:
        datasets = [dataset_finetune]
    else:
        datasets = dataset_finetune

    if model_name == "dfcan" and ("_sr" in model_suffix):
        params["scale_factor"] = 2
        params["patch_size"] = 32
    else:
        params["scale_factor"] = 1
        params["patch_size"] = 256

    if model_name == "dfcan" and params["scale_factor"] == 1:
        params["patch_size"] = 64
    else:
        params["patch_size"] = 256

    params.update(
        {
            "overlap": params["patch_size"] // 4,
            "batch_size": int(64 / params["patch_size"] * 32),
        }
    )

    bs = params["batch_size"]
    model_path = utils_data.win2linux(model_path)

    stitcher = utils_data.Patch_stitcher(
        patch_size=params["patch_size"],
        overlap=params["overlap"],
        padding_mode="reflect",
    )

    # ------------------------------------------------------------------------------
    # model
    # ------------------------------------------------------------------------------
    # 2D models
    if model_name == "unet":
        model = UNet(
            in_channels=1, out_channels=1, bilinear=False, residual=True, pos_out=False
        )

    if model_name == "care":
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

    if model_name == "dfcan":
        model = DFCAN(
            in_channels=1,
            scale_factor=params["scale_factor"],
            num_features=64,
            num_groups=4,
        )

    if model_name == "unifmir":
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

    model = model.to(device)

    # --------------------------------------------------------------------------
    # load model parameters
    # --------------------------------------------------------------------------
    print("loading model parameters...")
    state_dict = torch.load(model_path, map_location=device, weights_only=True)[
        "model_state_dict"
    ]
    # del prefix for complied model
    state_dict = utils_optim.on_load_checkpoint(checkpoint=state_dict)
    model.load_state_dict(state_dict)
    model.eval()

    # --------------------------------------------------------------------------
    # predict
    # --------------------------------------------------------------------------
    for id_dataset in datasets:
        print("-" * 80)
        # load dataset information
        try:
            ds = datasets_frame[datasets_frame["id"] == id_dataset].iloc[0]
            print(ds["id"])
        except:
            print(id_dataset, "Not Exist!")
            continue

        # save retuls to
        path_results = os.path.join(
            params["path_output"], ds["id"], model_name + model_suffix
        )
        os.makedirs(path_results, exist_ok=True)

        path_index = utils_data.win2linux(ds["path_index"])
        path_lr = utils_data.win2linux(ds["path_lr"])
        sf_lr, sf_hr = ds["sf_lr"], ds["sf_hr"]

        # check task
        task = 1
        if ds["task"] == "sr":
            task = 1
        elif ds["task"] == "dn":
            task = 2
        elif ds["task"] == "iso":
            task = 3
        elif ds["task"] == "dcv":
            task = 4
        else:
            raise ValueError("Unsupported Task.")
        task = torch.tensor(task, device=device)

        # load sample names in current dataset
        filenames = utils_data.read_txt(path_index)

        num_sample_total = len(filenames)
        if params["num_sample"] is not None:
            if params["num_sample"] > num_sample_total:
                num_sample_eva = num_sample_total
            else:
                num_sample_eva = params["num_sample"]
        else:
            num_sample_eva = params["num_sample"]
        print("- Number of test data:", num_sample_eva, "/", num_sample_total)

        # ----------------------------------------------------------------------
        for i_sample in range(num_sample_eva):
            filename = filenames[i_sample]
            print(f"- File Name: {filename}")

            # low-resolution image ---------------------------------------------
            img_lr = utils_data.read_image(os.path.join(path_lr, filename))
            img_lr = np.clip(img_lr, 0, None)
            img_lr = input_normalizer(img_lr)

            if params["scale_factor"] == 1:
                img_lr = utils_data.interp_sf(img_lr, sf=sf_lr)
            if (
                ds["id"] in ["deepbacs-sim-ecoli-sr", "deepbacs-sim-saureus-sr"]
                and model_name == "dfcan"
                and "_sr" in model_suffix
            ):
                img_lr = utils_data.interp_sf(img_lr, sf=-2)

            img_lr = torch.tensor(img_lr[None]).to(device)

            # prediction -------------------------------------------------------
            with torch.no_grad():
                if params["patch_image"] and (
                    params["patch_size"] < max(img_lr.shape[-2:])
                ):
                    # padding
                    img_lr_shape_ori = img_lr.shape
                    if params["patch_size"] > img_lr.shape[-1]:
                        pad_size = params["patch_size"] - img_lr.shape[-1]
                        img_lr = torch.nn.functional.pad(
                            img_lr, pad=(0, pad_size, 0, 0), mode="reflect"
                        )
                    if params["patch_size"] > img_lr.shape[-2]:
                        pad_size = params["patch_size"] - img_lr.shape[-2]
                        img_lr = torch.nn.functional.pad(
                            img_lr, pad=(0, 0, 0, pad_size), mode="reflect"
                        )

                    # patching image
                    img_lr_patches = stitcher.unfold(img=img_lr)

                    # --------------------------------------------------------------
                    num_iter = math.ceil(img_lr_patches.shape[0] / bs)
                    pbar = tqdm.tqdm(desc="PREDICT", total=num_iter, ncols=100)
                    img_est_patches = []
                    for i_iter in range(num_iter):
                        if model_name == "unifmir":
                            img_est_patch = model(
                                img_lr_patches[i_iter * bs : bs + i_iter * bs], task
                            )
                        else:
                            img_est_patch = model(
                                img_lr_patches[i_iter * bs : bs + i_iter * bs]
                            )
                        img_est_patches.append(img_est_patch)
                        pbar.update(1)
                    pbar.close()
                    img_est_patches = torch.cat(img_est_patches, dim=0)
                    # ----------------------------------------------------------
                    # fold the patches
                    original_image_shape = (
                        img_lr.shape[0],
                        img_lr.shape[1],
                        img_lr.shape[2] * params["scale_factor"],
                        img_lr.shape[3] * params["scale_factor"],
                    )

                    if params["scale_factor"] != 1:
                        overlap = params["overlap"] * params["scale_factor"]
                        patch_size = params["patch_size"] * params["scale_factor"]
                        stitcher = stitcher.set_params(
                            overlap=overlap, patch_size=patch_size
                        )

                    # ----------------------------------------------------------
                    # fold the patches
                    img_est = stitcher.fold_linear_ramp(
                        patches=img_est_patches,
                        original_image_shape=original_image_shape,
                    )
                    img_est = torch.tensor(img_est)

                    # unpadding
                    img_est = img_est[
                        ...,
                        : img_lr_shape_ori[-2] * params["scale_factor"],
                        : img_lr_shape_ori[-1] * params["scale_factor"],
                    ]
                else:
                    input_shape = img_lr.shape
                    # padding for care model, which is a unet model requires
                    # specific image size
                    if model_name == "care":
                        if input_shape[-1] % 4 > 0:
                            pad_size = 4 - input_shape[-1] % 4
                            img_lr = torch.nn.functional.pad(
                                img_lr, pad=(0, pad_size, 0, pad_size), mode="reflect"
                            )
                    # ----------------------------------------------------------
                    if model_name == "unifmir":
                        img_est = model(img_lr, task)
                    else:
                        img_est = model(img_lr)
                    # ----------------------------------------------------------
                    if model_name == "care":
                        if input_shape[-1] % 4 > 0:
                            img_est = img_est[
                                :, :, : input_shape[-2], : input_shape[-1]
                            ]
            img_est = img_est.float().cpu().detach().numpy()

            # ------------------------------------------------------------------
            if num_datasets < 3:
                if ds["path_hr"] != "Unknown":
                    dr = 2.5
                    clip = lambda x: np.clip(x, 0.0, dr)

                    img_hr = utils_data.read_image(
                        os.path.join(ds["path_hr"], filename)
                    )
                    if params["scale_factor"] == 1:
                        img_hr = utils_data.interp_sf(img_hr, sf=sf_hr)[0]
                    else:
                        img_hr = img_hr[0]

                    # imgs_est = utils_eva.linear_transform(
                    #     img_true=clip(img_hr), img_test=img_est
                    # )

                    dict_eva = {
                        "img_true": clip(output_normalizer(img_hr)),
                        "img_test": clip(output_normalizer(img_est))[0, 0],
                        "data_range": dr,
                    }

                    ssim = utils_eva.SSIM(**dict_eva)
                    psnr = utils_eva.PSNR(**dict_eva)
                    print(f"PSNR: {psnr:.4f}, SSIM: {ssim:.4f}")
                else:
                    print("There is no reference data.")

            # ------------------------------------------------------------------
            # save results
            io.imsave(
                os.path.join(path_results, filename),
                arr=img_est[0],
                check_contrast=False,
            )
    del model

print("-" * 80)
print("Done.")
print("Current time: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("-" * 80)
