import numpy as np
import torch, os, json
import skimage.io as io
from torchvision.transforms import v2

from models.autoencoder import Encoder, Decoder, Autoencoder

import utils.data as utils_data
import utils.evaluation as utils_eva


# ------------------------------------------------------------------------------
# parameters
# ------------------------------------------------------------------------------
params = {
    "device": "cuda:0",
    # checkpoints --------------------------------------------------------------
    "model_name": "vae",
    "suffix": "_biosrall_sf_c4",
    "path_model": "checkpoints\\vae\\vae_mae_kl_bs_4_lr_1e-05_biosrall_sf2_c4\epoch_3_iter_260000.pt",
    # "suffix": "_biosrall_sf_c16",
    # "path_model": "checkpoints\\vae\\vae_mae_kl_bs_4_lr_1e-05_biosrall_sf2_c16\epoch_8_iter_524576.pt",
    # model parameters ---------------------------------------------------------
    "in_channels": 1,
    "out_channels": 1,
    "channels": 128,
    "channel_multiplier": [1, 2, 4, 4],
    "n_resnet_blocks": 2,
    "z_channels": 4,
    "emb_channels": 4,
    # dataset ------------------------------------------------------------------
    "dim": 2,
    # "dataset_name": "F_actin_noise_level_9",
    # "path_dataset": "E:\qiqilu\datasets\BioSR\\transformed\F_actin\\test\channel_0\WF_noise_level_9",
    "dataset_name": "F_actin_SIM",
    "path_dataset": "E:\qiqilu\datasets\BioSR\\transformed\F_actin\\test\channel_0\SIM",
    "path_index_file": "E:\qiqilu\datasets\BioSR\\transformed\F_actin\\test.txt",
    "scale_factor": 1,
    "p_low": 0.0,
    "p_high": 0.9999,
}

# ------------------------------------------------------------------------------
idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# idxs = [0, 1, 2]

# ------------------------------------------------------------------------------
utils_data.print_dict(params)
device = torch.device(params["device"])
# save retuls to
path_results = os.path.join(
    "outputs", "vae", params["dataset_name"], params["model_name"] + params["suffix"]
)
utils_data.make_path(path_results)

# ------------------------------------------------------------------------------
# datasets
# ------------------------------------------------------------------------------
path_data = utils_data.read_txt(path_txt=params["path_index_file"])
num_data = len(path_data)
normalizer = utils_data.NormalizePercentile(
    p_low=params["p_low"], p_high=params["p_high"]
)

print("- Number of test data:", num_data)

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
# load model parameters
# ------------------------------------------------------------------------------
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

    # load image
    img_input = utils_data.read_image(
        os.path.join(params["path_dataset"], img_file_name), expend_channel=False
    )
    # normalization
    img_input = normalizer(img_input)
    img_input = torch.tensor(img_input)[None].to(device)

    # interpolat low-resolution image
    if params["scale_factor"] > 1:
        img_input = torch.nn.functional.interpolate(
            img_input,
            scale_factor=(params["scale_factor"], params["scale_factor"]),
            mode="nearest",
        )

    # patching
    img_input_patches = utils_data.unfold(
        img=img_input,
        patch_size=256,
        overlap=64,
        padding_mode="constant",
    )

    with torch.no_grad():
        posteriors = model.encode(img=img_input_patches)
        enc = posteriors.sample()
        img_rec = model.decode(enc)
        img_rec = torch.clip(img_rec, min=0.0)

    # unpatching
    img_rec = utils_data.fold(img_rec, original_image_shape=img_input.shape, overlap=64)

    # --------------------------------------------------------------------------
    # calculate metrics
    ssim = utils_eva.SSIM_tb(
        img_true=img_input, img_test=img_rec, data_range=None, version_wang=False
    )
    psnr = utils_eva.PSNR_tb(img_true=img_input, img_test=img_rec, data_range=None)

    print(ssim, psnr)

    # --------------------------------------------------------------------------
    # save results
    io.imsave(
        os.path.join(path_results, img_file_name),
        arr=img_rec.cpu().detach().numpy()[0],
        check_contrast=False,
    )
