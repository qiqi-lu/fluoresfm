import numpy as np
import torch, os, json, tqdm
import skimage.io as io
from torchvision.transforms import v2
from models.clip_embedder import CLIPTextEmbedder
from models.biomedclip_embedder import BiomedCLIPTextEmbedder

from models.unet_sd_c import UNetModel
from models.autoencoder import Encoder, Decoder, Autoencoder, gaussian_sample

import utils.data as utils_data
import utils.evaluation as utils_eva


# ------------------------------------------------------------------------------
# parameters
# ------------------------------------------------------------------------------
params = {
    "device": "cuda:0",
    # embedder -----------------------------------------------------------------
    "embedder": "biomedclip",
    # model parameters (vae) ---------------------------------------------------
    "vae": {
        "in_channels": 1,
        "out_channels": 1,
        "channels": 128,
        "channel_multiplier": [1, 2, 4, 4],
        "n_resnet_blocks": 2,
        "z_channels": 4,
        "emb_channels": 4,
        "checkpoint": "checkpoints\\vae\\vae_mae_kl_bs_4_lr_1e-05_biosrall_sf2_c4\epoch_2_iter_150000.pt",
        # "checkpoint": "checkpoints\\vae\\vae_mae_kl_bs_4_lr_1e-05_biosrall_sf2_c16\epoch_8_iter_524576.pt",
    },
    # model parameters (unet) --------------------------------------------------
    "model_name": "unet_sd_c",
    # "suffix": "biosrx2_123_adamw_bmc_v2_vae_c16_img_msew",
    "suffix": "biosrx2_123_adamw_bmc_v2_vae_c4",
    "path_model": "checkpoints\conditional\\unet_vae\\unet_sd_c_mse_bs_4_lr_1e-05_biosrx2_123_adamw_bmc_v2_vae_c4\epoch_2_iter_310000.pt",
    # "path_model": "checkpoints\conditional\\unet_vae\\unet_sd_c_mse_bs_4_lr_1e-05_biosrx2_123_adamw_bmc_v2_vae_c16\epoch_2_iter_135000.pt",
    # "path_model": "checkpoints\conditional\\unet_vae\\unet_sd_c_mse_bs_4_lr_1e-05_biosrx2_123_adamw_bmc_v2_vae_c16_10\epoch_6_iter_305000.pt",
    # "path_model": "checkpoints\conditional\\unet_vae\\unet_sd_c_mse_bs_1_lr_1e-05_biosrx2_123_adamw_bmc_v2_vae_c16_img\epoch_2_iter_485000.pt",
    # "path_model": "checkpoints\conditional\\unet_vae\\unet_sd_c_msew_bs_1_lr_1e-05_biosrx2_123_adamw_bmc_v2_vae_c16_img_w\epoch_1_iter_360000.pt",
    # "path_model": "checkpoints\conditional\\unet_vae\\unet_sd_c_mae_bs_1_lr_1e-05_biosrx2_123_adamw_bmc_v2_vae_c16_img_mae\epoch_2_iter_520000.pt",
    "in_channels": 4,
    "out_channels": 4,
    "channels": 320,
    "n_res_blocks": 2,
    "attention_levels": [1, 2, 3],
    "channel_multipliers": [1, 2, 4, 4],
    "n_heads": 8,
    "tf_layers": 1,
    "d_cond": 768,
    # dataset ------------------------------------------------------------------
    "dim": 2,
    "dataset_name": "F_actin_noise_level_1",
    "path_dataset_lr": "E:\qiqilu\datasets\BioSR\\transformed\F_actin\\test\channel_0\WF_noise_level_1",
    "path_dataset_hr": "E:\qiqilu\datasets\BioSR\\transformed\F_actin\\test\channel_0\SIM",
    "path_index_file": "E:\qiqilu\datasets\BioSR\\transformed\F_actin\\test.txt",
    "interpolation": 2,
    "p_low": 0.0,
    "p_high": 0.9999,
    "text_input": None,
    "text_target": None,
    # "text_target": "clathrin-coated pits in fixed COS-7 cell line acquired using a linear structured illumination microscope with excitation numerical aperture of 1.41, detection numerical aperture of 1.3, wavelength of 488 nm, pixel size of 31.3 x 31.3 nm.",
    # "text_target": "microtubules in fixed COS-7 cell line acquired using a wide-field microscope with excitation numerical aperture of 1.41, detection numerical aperture of 1.3, wavelength of 488 nm, pixel size of 31.3 x 31.3 nm.",
    # "text_target": "microtubules",
    # "text_target": "linear structured illumination microscope",
    # "text_target": "clathrin-coated pits",
    "patch_image": True,
    "patch_size": 256,
    "overlap": 64,
}

# ------------------------------------------------------------------------------
# choose sample id
idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# idxs = [0, 1, 2]

# ------------------------------------------------------------------------------
device = torch.device(params["device"])
path_results = os.path.join(
    "outputs", "unet_c", params["dataset_name"], params["model_name"] + params["suffix"]
)  # save retuls to

utils_data.print_dict(params)
utils_data.make_path(path_results)

# ------------------------------------------------------------------------------
# datasets
# ------------------------------------------------------------------------------
path_data = utils_data.read_txt(path_txt=params["path_index_file"])
normalizer = utils_data.NormalizePercentile(
    p_low=params["p_low"], p_high=params["p_high"]
)

print("- Number of test data:", len(path_data))

# ------------------------------------------------------------------------------
# model
# ------------------------------------------------------------------------------
# Embedder
if params["embedder"] == "clip":
    embedder = CLIPTextEmbedder(device=torch.device("cpu")).eval()
elif params["embedder"] == "biomedclip":
    embedder = BiomedCLIPTextEmbedder(
        path_json="checkpoints/clip//biomedclip/open_clip_config.json",
        path_bin="checkpoints/clip//biomedclip/open_clip_pytorch_model.bin",
        context_length=256,
        device=torch.device("cpu"),
    ).eval()
else:
    raise ValueError(f"Embedder '{params['embedder']}' does not exist.")

# ------------------------------------------------------------------------------
# VAE model
encoder = Encoder(
    channels=params["vae"]["channels"],
    channel_multipliers=params["vae"]["channel_multiplier"],
    n_resnet_blocks=params["vae"]["n_resnet_blocks"],
    in_channels=params["vae"]["in_channels"],
    z_channels=params["vae"]["z_channels"],
).to(device=device)

decoder = Decoder(
    channels=params["vae"]["channels"],
    channel_multipliers=params["vae"]["channel_multiplier"],
    n_resnet_blocks=params["vae"]["n_resnet_blocks"],
    out_channels=params["vae"]["out_channels"],
    z_channels=params["vae"]["z_channels"],
).to(device=device)

autoencoder = Autoencoder(
    encoder=encoder,
    decoder=decoder,
    emb_channels=params["vae"]["emb_channels"],
    z_channels=params["vae"]["z_channels"],
).to(device=device)

# ------------------------------------------------------------------------------
# Unet models
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
    ).to(device)

# ------------------------------------------------------------------------------
# load model parameters
# ------------------------------------------------------------------------------
# load parameters of VAE model
autoencoder.load_state_dict(
    torch.load(params["vae"]["checkpoint"], map_location=device, weights_only=True)[
        "model_state_dict"
    ]
)
autoencoder.eval()

# load parameters of UNet model
model.load_state_dict(
    torch.load(params["path_model"], map_location=device, weights_only=True)[
        "model_state_dict"
    ]
)
model.eval()

# ------------------------------------------------------------------------------
# predict
# ------------------------------------------------------------------------------
for i in idxs:
    img_file_name = path_data[i]
    print(f"- File Name: {img_file_name}")

    # load low-resolution image (input) ----------------------------------------
    img_lr = io.imread(os.path.join(params["path_dataset_lr"], img_file_name))
    img_lr = img_lr.astype(np.float32)

    # normalization
    img_lr = normalizer(img_lr)
    img_lr = torch.tensor(img_lr)[None].to(device)

    # interpolat low-resolution image
    if params["interpolation"] > 1:
        img_lr = torch.nn.functional.interpolate(
            img_lr,
            scale_factor=(params["interpolation"], params["interpolation"]),
            mode="nearest",
        )

    # load low-resolution text
    if params["text_input"] is None:
        pt_lr = os.path.join(
            params["path_dataset_lr"], img_file_name.split(".")[0] + ".txt"
        )
        if os.path.exists(pt_lr):
            with open(pt_lr) as f:
                text_lr = f.read()
        else:
            with open(os.path.join(params["path_dataset_lr"], "text.txt")) as f:
                text_lr = f.read()
    else:
        text_lr = params["text_input"]

    # --------------------------------------------------------------------------
    if params["path_dataset_hr"] is not None:
        # load high-resolution image (reference)
        img_hr = utils_data.read_image(
            os.path.join(params["path_dataset_hr"], img_file_name), expend_channel=False
        )
        img_hr = normalizer(img_hr)
        img_hr = torch.tensor(img_hr)[None].to(device)

        # load high-resolution text
        if params["text_target"] is None:
            pt_hr = os.path.join(
                params["path_dataset_hr"], img_file_name.split(".")[0] + ".txt"
            )
            if os.path.exists(pt_hr):
                with open(pt_hr) as f:
                    text_hr = f.read()
            else:
                with open(os.path.join(params["path_dataset_hr"], "text.txt")) as f:
                    text_hr = f.read()
        else:
            text_hr = params["text_target"]

    else:
        print("there is no reference image.")
        if params["text_target"] is not None:
            text_hr = params["text_target"]
        else:
            raise ValueError("Target text is required.")

    with torch.no_grad():
        text_hr = embedder(text_hr)
        text_lr = embedder(text_lr)
    text_embed = torch.cat([text_lr, text_hr], dim=1).to(device)
    time_embed = torch.zeros(size=(1,)).to(device)

    # --------------------------------------------------------------------------
    # prediction
    with torch.no_grad():
        if params["patch_image"]:
            # patching image
            img_lr_patches = utils_data.unfold(
                img=img_lr,
                patch_size=params["patch_size"],
                overlap=params["overlap"],
                padding_mode="reflect",
            )
            num_patches = img_lr_patches.shape[0]

            pbar = tqdm.tqdm(
                desc="Processing patches", total=num_patches, leave=True, ncols=100
            )

            # predict
            img_est_patches = []
            enc_est_patches = []
            for i in range(num_patches):
                enc_lr_patch = autoencoder.encode(img=img_lr_patches[i][None]).sample()
                # enc_lr_patch = autoencoder.encode(
                #     img=img_lr_patches[i][None]
                # ).posterior()
                # enc_est_patch = (
                #     model(enc_lr_patch / 10.0, time_embed, text_embed) * 10.0
                # )
                enc_est_patch = model(enc_lr_patch, time_embed, text_embed)
                # enc_est_patch = gaussian_sample(enc_est_patch)
                img_est_patch = autoencoder.decode(enc_est_patch)

                enc_est_patches.append(enc_est_patch)
                img_est_patches.append(img_est_patch)
                pbar.update(1)
            pbar.close()

            img_est_patches = torch.cat(img_est_patches, dim=0)
            enc_est_patches = torch.cat(enc_est_patches, dim=0)

            # fold the patches
            # img_est = utils_data.fold(
            #     patches=img_est_patches,
            #     original_image_shape=img_lr.shape,
            #     overlap=params["overlap"],
            # )

            img_est = utils_data.fold_scale(
                patches=img_est_patches,
                original_image_shape=img_lr.shape,
                overlap=params["overlap"],
                crop_center=True,
                enable_scale=False,
            )
        else:
            # directly process the whole image, which may result in OOM problem
            enc_lr = autoencoder.encode(img=img_lr).sample()
            enc_est = model(enc_lr, time_embed, text_embed)
            img_est = autoencoder.decode(enc_est)

    img_est = torch.clip(img_est, min=0.0)

    # --------------------------------------------------------------------------
    # calculate metrics
    if params["path_dataset_hr"] is not None:
        if params["dim"] == 3:
            imgs_est = utils_eva.linear_transform(
                img_true=img_hr, img_test=img_est, axis=(2, 3, 4)
            )
        if params["dim"] == 2:
            imgs_est = utils_eva.linear_transform(
                img_true=img_hr, img_test=img_est, axis=(2, 3)
            )

        ssim = utils_eva.SSIM_tb(
            img_true=img_hr, img_test=imgs_est, data_range=None, version_wang=False
        )
        psnr = utils_eva.PSNR_tb(img_true=img_hr, img_test=imgs_est, data_range=None)

        print(ssim, psnr)
    else:
        print("There is no reference data.")

    # --------------------------------------------------------------------------
    # save results
    io.imsave(
        os.path.join(path_results, img_file_name),
        arr=img_est.cpu().detach().numpy()[0],
        check_contrast=False,
    )

    # # --------------------------------------------------------------------------
    os.makedirs(os.path.join(path_results, "enc_hr"), exist_ok=True)
    os.makedirs(os.path.join(path_results, "enc_est"), exist_ok=True)

    if params["path_dataset_hr"] is not None:
        # encode predition
        img_hr_patches = utils_data.unfold(
            img_hr, patch_size=params["patch_size"], overlap=params["overlap"]
        )

        num_patches = img_hr_patches.shape[0]
        enc_hr_patches = []
        with torch.no_grad():
            for i in range(num_patches):
                enc_hr_patch = autoencoder.encode(img=img_hr_patches[i][None]).sample()
                enc_hr_patches.append(enc_hr_patch)
            enc_hr_patches = torch.cat(enc_hr_patches, dim=0)

        io.imsave(
            os.path.join(path_results, "enc_hr", img_file_name),
            arr=enc_hr_patches.cpu().detach().numpy(),
            check_contrast=False,
        )

    io.imsave(
        os.path.join(path_results, "enc_est", img_file_name),
        arr=enc_est_patches.cpu().detach().numpy(),
        check_contrast=False,
    )
