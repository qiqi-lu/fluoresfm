import numpy as np
import torch, os, json, tqdm, pandas, math
import skimage.io as io
from torchvision.transforms import v2
from models.clip_embedder import CLIPTextEmbedder
from models.biomedclip_embedder import BiomedCLIPTextEmbedder

from models.unet_sd_c import UNetModel

import utils.data as utils_data
import utils.evaluation as utils_eva
import utils.optim as utils_optim

# ------------------------------------------------------------------------------
# parameters
# ------------------------------------------------------------------------------
params = {
    "device": "cuda:1",
    "enable_amp": True,
    # text embedder ------------------------------------------------------------
    "embedder": "biomedclip",
    "path_json": "checkpoints/clip//biomedclip/open_clip_config.json",
    "path_bin": "checkpoints/clip//biomedclip/open_clip_pytorch_model.bin",
    # model parameters ---------------------------------------------------------
    "model_name": "unet_sd_c",
    # --------------------------------------------------------------------------
    "suffix": "_all",
    "path_model": "checkpoints\\conditional\\unet_sd_c_mae_bs_4_lr_1e-05_all\epoch_0_iter_775000.pt",
    # --------------------------------------------------------------------------
    # "suffix": "_sr",
    # "path_model": "checkpoints\\conditional\\unet_sd_c_mae_bs_4_lr_1e-05_sr\epoch_1_iter_1095000.pt",
    # "suffix": "_dcv",
    # "path_model": "checkpoints\\conditional\\unet_sd_c_mae_bs_4_lr_1e-05_dcv\epoch_6_iter_915000.pt",
    # "suffix": "_dn",
    # "path_model": "checkpoints\\conditional\\unet_sd_c_mae_bs_4_lr_1e-05_dn\epoch_0_iter_370000.pt",
    # "suffix": "_iso",
    # "path_model": "checkpoints\\conditional\\unet_sd_c_mae_bs_4_lr_1e-05_iso\epoch_4_iter_830000.pt",
    # --------------------------------------------------------------------------
    # "suffix": "_sr_crossx",
    # "path_model": "checkpoints\conditional\\unet_sd_c_mae_bs_4_lr_1e-05_sr_crossx\epoch_0_iter_565000.pt",
    # "suffix": "_dcv_crossx",
    # "path_model": "checkpoints\conditional\\unet_sd_c_mae_bs_4_lr_1e-05_dcv_crossx\epoch_16_iter_1015000.pt",
    # "suffix": "_dn_crossx",
    # "path_model": "checkpoints\conditional\\unet_sd_c_mae_bs_4_lr_1e-05_dn_crossx\epoch_0_iter_660000.pt",
    # "suffix": "_iso_crossx",
    # "path_model": "checkpoints\conditional\\unet_sd_c_mae_bs_4_lr_1e-05_iso_crossx\epoch_12_iter_1220000.pt",
    # --------------------------------------------------------------------------
    # "suffix": "_all_cross",
    # "path_model": "checkpoints\conditional\\unet_sd_c_mae_bs_4_lr_1e-05_all_crossx\epoch_0_iter_825000.pt",
    # --------------------------------------------------------------------------
    "text_type": "single",
    # "text_type": "paired",
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
    # dataset ------------------------------------------------------------------
    "dim": 2,
    "path_dataset_test": "dataset_test.xlsx",
    "id_dataset": [
        "biosr-cpp-sr-1",
        "biosr-cpp-sr-2",
        "biosr-cpp-sr-3",
        "biosr-cpp-sr-4",
        "biosr-cpp-sr-5",
        "biosr-cpp-sr-6",
        "biosr-cpp-sr-7",
        "biosr-cpp-sr-8",
        "biosr-cpp-sr-9",
        "biosr-er-sr-1",
        "biosr-er-sr-2",
        "biosr-er-sr-3",
        "biosr-er-sr-4",
        "biosr-er-sr-5",
        "biosr-er-sr-6",
        "biosr-mt-sr-1",
        "biosr-mt-sr-2",
        "biosr-mt-sr-3",
        "biosr-mt-sr-4",
        "biosr-mt-sr-5",
        "biosr-mt-sr-6",
        "biosr-mt-sr-7",
        "biosr-mt-sr-8",
        "biosr-mt-sr-9",
        "biosr-actin-sr-1",
        "biosr-actin-sr-2",
        "biosr-actin-sr-3",
        "biosr-actin-sr-4",
        "biosr-actin-sr-5",
        "biosr-actin-sr-6",
        "biosr-actin-sr-7",
        "biosr-actin-sr-8",
        "biosr-actin-sr-9",
        "biosr-actin-sr-10",
        "biosr-actin-sr-11",
        "biosr-actin-sr-12",
        "deepbacs-sim-ecoli-sr",
        "deepbacs-sim-saureus-sr",
        "w2s-c0-sr-1",
        "w2s-c0-sr-2",
        "w2s-c0-sr-3",
        "w2s-c0-sr-4",
        "w2s-c0-sr-5",
        "w2s-c0-sr-6",
        "w2s-c0-sr-7",
        "w2s-c1-sr-1",
        "w2s-c1-sr-2",
        "w2s-c1-sr-3",
        "w2s-c1-sr-4",
        "w2s-c1-sr-5",
        "w2s-c1-sr-6",
        "w2s-c1-sr-7",
        "w2s-c2-sr-1",
        "w2s-c2-sr-2",
        "w2s-c2-sr-3",
        "w2s-c2-sr-4",
        "w2s-c2-sr-5",
        "w2s-c2-sr-6",
        "w2s-c2-sr-7",
        "srcaco2-h2b-sr-8",
        "srcaco2-h2b-sr-4",
        "srcaco2-h2b-sr-2",
        "srcaco2-survivin-sr-8",
        "srcaco2-survivin-sr-4",
        "srcaco2-survivin-sr-2",
        "srcaco2-tubulin-sr-8",
        "srcaco2-tubulin-sr-4",
        "srcaco2-tubulin-sr-2",
        "vmsim-mito-sr",
        "vmsim-er-sr",
        # ----------------------------------------------------------------------
        "biosr-cpp-dn-1",
        "biosr-cpp-dn-2",
        "biosr-cpp-dn-3",
        "biosr-cpp-dn-4",
        "biosr-cpp-dn-5",
        "biosr-cpp-dn-6",
        "biosr-cpp-dn-7",
        "biosr-cpp-dn-8",
        "biosr-er-dn-1",
        "biosr-er-dn-2",
        "biosr-er-dn-3",
        "biosr-er-dn-4",
        "biosr-er-dn-5",
        "biosr-mt-dn-1",
        "biosr-mt-dn-2",
        "biosr-mt-dn-3",
        "biosr-mt-dn-4",
        "biosr-mt-dn-5",
        "biosr-mt-dn-6",
        "biosr-mt-dn-7",
        "biosr-mt-dn-8",
        "biosr-actin-dn-1",
        "biosr-actin-dn-2",
        "biosr-actin-dn-3",
        "biosr-actin-dn-4",
        "biosr-actin-dn-5",
        "biosr-actin-dn-6",
        "biosr-actin-dn-7",
        "biosr-actin-dn-8",
        "biosr-actin-dn-9",
        "biosr-actin-dn-10",
        "biosr-actin-dn-11",
        "biosr-actinnl-dn-1",
        "biosr-actinnl-dn-2",
        "biosr-actinnl-dn-3",
        "biosr-actinnl-dn-4",
        "biosr-actinnl-dn-5",
        "biosr-actinnl-dn-6",
        "biosr-actinnl-dn-7",
        "biosr-actinnl-dn-8",
        "biosr+-ccp-dn-1",
        "biosr+-ccp-dn-2",
        "biosr+-ccp-dn-3",
        "biosr+-ccp-dn-4",
        "biosr+-ccp-dn-5",
        "biosr+-ccp-dn-6",
        "biosr+-ccp-dn-7",
        "biosr+-ccp-dn-8",
        "biosr+-er-dn-1",
        "biosr+-er-dn-2",
        "biosr+-er-dn-3",
        "biosr+-er-dn-4",
        "biosr+-er-dn-5",
        "biosr+-er-dn-6",
        "biosr+-actin-dn-1",
        "biosr+-actin-dn-2",
        "biosr+-actin-dn-3",
        "biosr+-actin-dn-4",
        "biosr+-actin-dn-5",
        "biosr+-actin-dn-6",
        "biosr+-actin-dn-7",
        "biosr+-actin-dn-8",
        "biosr+-actin-dn-9",
        "biosr+-actin-dn-10",
        "biosr+-actin-dn-11",
        "biosr+-mt-dn-1",
        "biosr+-mt-dn-2",
        "biosr+-mt-dn-3",
        "biosr+-mt-dn-4",
        "biosr+-mt-dn-5",
        "biosr+-mt-dn-6",
        "biosr+-mt-dn-7",
        "biosr+-mt-dn-8",
        "biosr+-myosin-dn-1",
        "biosr+-myosin-dn-2",
        "biosr+-myosin-dn-3",
        "biosr+-myosin-dn-4",
        "biosr+-myosin-dn-5",
        "biosr+-myosin-dn-6",
        "biosr+-myosin-dn-7",
        "biosr+-myosin-dn-8",
        "care-planaria-dn-1",
        "care-planaria-dn-2",
        "care-planaria-dn-3",
        "care-tribolium-dn-1",
        "care-tribolium-dn-2",
        "care-tribolium-dn-3",
        "deepbacs-ecoli-dn",
        "deepbacs-ecoli2-dn",
        "w2s-c0-dn-1",
        "w2s-c0-dn-2",
        "w2s-c0-dn-3",
        "w2s-c0-dn-4",
        "w2s-c0-dn-5",
        "w2s-c0-dn-6",
        "w2s-c1-dn-1",
        "w2s-c1-dn-2",
        "w2s-c1-dn-3",
        "w2s-c1-dn-4",
        "w2s-c1-dn-5",
        "w2s-c1-dn-6",
        "w2s-c2-dn-1",
        "w2s-c2-dn-2",
        "w2s-c2-dn-3",
        "w2s-c2-dn-4",
        "w2s-c2-dn-5",
        "w2s-c2-dn-6",
        "srcaco2-h2b-dn-8",
        "srcaco2-h2b-dn-4",
        "srcaco2-h2b-dn-2",
        "srcaco2-survivin-dn-8",
        "srcaco2-survivin-dn-4",
        "srcaco2-survivin-dn-2",
        "srcaco2-tubulin-dn-8",
        "srcaco2-tubulin-dn-4",
        "srcaco2-tubulin-dn-2",
        "fmd-confocal-bpae-b-avg2",
        "fmd-confocal-bpae-b-avg4",
        "fmd-confocal-bpae-b-avg8",
        "fmd-confocal-bpae-b-avg16",
        "fmd-confocal-bpae-g-avg2",
        "fmd-confocal-bpae-g-avg4",
        "fmd-confocal-bpae-g-avg8",
        "fmd-confocal-bpae-g-avg16",
        "fmd-confocal-bpae-r-avg2",
        "fmd-confocal-bpae-r-avg4",
        "fmd-confocal-bpae-r-avg8",
        "fmd-confocal-bpae-r-avg16",
        "fmd-confocal-fish-avg2",
        "fmd-confocal-fish-avg4",
        "fmd-confocal-fish-avg8",
        "fmd-confocal-fish-avg16",
        "fmd-confocal-mice-avg2",
        "fmd-confocal-mice-avg4",
        "fmd-confocal-mice-avg8",
        "fmd-confocal-mice-avg16",
        "fmd-twophoton-mice-avg2",
        "fmd-twophoton-mice-avg4",
        "fmd-twophoton-mice-avg8",
        "fmd-twophoton-mice-avg16",
        "fmd-twophoton-bpae-b-avg2",
        "fmd-twophoton-bpae-b-avg4",
        "fmd-twophoton-bpae-b-avg8",
        "fmd-twophoton-bpae-b-avg16",
        "fmd-twophoton-bpae-g-avg2",
        "fmd-twophoton-bpae-g-avg4",
        "fmd-twophoton-bpae-g-avg8",
        "fmd-twophoton-bpae-g-avg16",
        "fmd-twophoton-bpae-r-avg2",
        "fmd-twophoton-bpae-r-avg4",
        "fmd-twophoton-bpae-r-avg8",
        "fmd-twophoton-bpae-r-avg16",
        "fmd-wf-bpae-b-avg2",
        "fmd-wf-bpae-b-avg4",
        "fmd-wf-bpae-b-avg8",
        "fmd-wf-bpae-b-avg16",
        "fmd-wf-bpae-g-avg2",
        "fmd-wf-bpae-g-avg4",
        "fmd-wf-bpae-g-avg8",
        "fmd-wf-bpae-g-avg16",
        "fmd-wf-bpae-r-avg2",
        "fmd-wf-bpae-r-avg4",
        "fmd-wf-bpae-r-avg8",
        "fmd-wf-bpae-r-avg16",
        # ----------------------------------------------------------------------
        "biosr-cpp-dcv-1",
        "biosr-cpp-dcv-2",
        "biosr-cpp-dcv-3",
        "biosr-cpp-dcv-4",
        "biosr-cpp-dcv-5",
        "biosr-cpp-dcv-6",
        "biosr-cpp-dcv-7",
        "biosr-cpp-dcv-8",
        "biosr-cpp-dcv-9",
        "biosr-er-dcv-1",
        "biosr-er-dcv-2",
        "biosr-er-dcv-3",
        "biosr-er-dcv-4",
        "biosr-er-dcv-5",
        "biosr-er-dcv-6",
        "biosr-mt-dcv-1",
        "biosr-mt-dcv-2",
        "biosr-mt-dcv-3",
        "biosr-mt-dcv-4",
        "biosr-mt-dcv-5",
        "biosr-mt-dcv-6",
        "biosr-mt-dcv-7",
        "biosr-mt-dcv-8",
        "biosr-mt-dcv-9",
        "biosr-actin-dcv-1",
        "biosr-actin-dcv-2",
        "biosr-actin-dcv-3",
        "biosr-actin-dcv-4",
        "biosr-actin-dcv-5",
        "biosr-actin-dcv-6",
        "biosr-actin-dcv-7",
        "biosr-actin-dcv-8",
        "biosr-actin-dcv-9",
        "biosr-actin-dcv-10",
        "biosr-actin-dcv-11",
        "biosr-actin-dcv-12",
        "biosr-actinnl-dcv-1",
        "biosr-actinnl-dcv-2",
        "biosr-actinnl-dcv-3",
        "biosr-actinnl-dcv-4",
        "biosr-actinnl-dcv-5",
        "biosr-actinnl-dcv-6",
        "biosr-actinnl-dcv-7",
        "biosr-actinnl-dcv-8",
        "biosr-actinnl-dcv-9",
        "care-synthe-granules-dcv",
        "care-synthe-tubulin-dcv",
        "care-synthe-tubulin-gfp-dcv",
        "deepbacs-sim-ecoli-dcv",
        "deepbacs-sim-saureus-dcv",
        "w2s-c0-dcv-1",
        "w2s-c0-dcv-2",
        "w2s-c0-dcv-3",
        "w2s-c0-dcv-4",
        "w2s-c0-dcv-5",
        "w2s-c0-dcv-6",
        "w2s-c0-dcv-7",
        "w2s-c1-dcv-1",
        "w2s-c1-dcv-2",
        "w2s-c1-dcv-3",
        "w2s-c1-dcv-4",
        "w2s-c1-dcv-5",
        "w2s-c1-dcv-6",
        "w2s-c1-dcv-7",
        "w2s-c2-dcv-1",
        "w2s-c2-dcv-2",
        "w2s-c2-dcv-3",
        "w2s-c2-dcv-4",
        "w2s-c2-dcv-5",
        "w2s-c2-dcv-6",
        "w2s-c2-dcv-7",
        "vmsim-mito-dcv",
        "vmsim-er-dcv",
        # ----------------------------------------------------------------------
        "care-drosophila-iso",
        "care-retina0-iso",
        "care-retina1-iso",
        "care-liver-iso",
    ],
    "num_sample": 8,
    # "num_sample": None,
    "p_low": 0.0,
    "p_high": 0.9999,
    "patch_image": True,
    "patch_size": 384,
    "overlap": 64,
    "batch_size": 1,
    # output -------------------------------------------------------------------
    "path_output": "outputs\\unet_c",
}

if os.name == "posix":
    params["path_model"] = utils_data.win2linux(params["path_model"])
    params["path_output"] = utils_data.win2linux(params["path_output"])
    params["path_json"] = utils_data.win2linux(params["path_json"])
    params["path_bin"] = utils_data.win2linux(params["path_bin"])

# ------------------------------------------------------------------------------
datasets_frame = pandas.read_excel(params["path_dataset_test"])
dataset_info = datasets_frame[datasets_frame["id"].isin(params["id_dataset"])]
num_datasets = dataset_info.shape[0]

utils_data.print_dict(params)
device = torch.device(params["device"])
print("number of datasets:", num_datasets)

# ------------------------------------------------------------------------------
# model
# ------------------------------------------------------------------------------
# Text Embedder
if params["embedder"] == "clip":
    embedder = CLIPTextEmbedder(device=torch.device("cpu")).eval()

elif params["embedder"] == "biomedclip":
    embedder = BiomedCLIPTextEmbedder(
        path_json=params["path_json"],
        path_bin=params["path_bin"],
        context_length=256,
        device=torch.device("cpu"),
    ).eval()
else:
    raise ValueError(f"Embedder '{params['embedder']}' does not exist.")

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
    ).to(device)

# normalization
normalizer = utils_data.NormalizePercentile(
    p_low=params["p_low"], p_high=params["p_high"]
)

# ------------------------------------------------------------------------------
# load model parameters
# ------------------------------------------------------------------------------
# model.load_state_dict(
#     torch.load(params["path_model"], map_location=device, weights_only=True)[
#         "model_state_dict"
#     ]
# )

state_dict = torch.load(params["path_model"], map_location=device, weights_only=True)[
    "model_state_dict"
]
# del prefix for complied model
state_dict = utils_optim.on_load_checkpoint(checkpoint=state_dict)
model.load_state_dict(state_dict)

model.eval()

# ------------------------------------------------------------------------------
# predict
# ------------------------------------------------------------------------------
for i_dataset in range(num_datasets):
    ds = dataset_info.iloc[i_dataset]
    print(ds["id"])

    # save retuls to
    path_results = os.path.join(
        params["path_output"], ds["id"], params["model_name"] + params["suffix"]
    )
    utils_data.make_path(path_results)

    path_sample = utils_data.read_txt(path_txt=ds["path_index"])

    num_sample = len(path_sample)
    if params["num_sample"] is not None:
        if params["num_sample"] > num_sample:
            params["num_sample"] = num_sample
    else:
        params["num_sample"] = num_sample

    print("- Number of test data:", params["num_sample"], "/", num_sample)

    for i_sample in range(params["num_sample"]):
        sample_filename = path_sample[i_sample]
        print(f"- File Name: {sample_filename}")

        # load low-resolution image (input) ------------------------------------
        img_lr = utils_data.read_image(
            os.path.join(ds["path_lr"], sample_filename), expend_channel=False
        )
        # normalization
        img_lr = normalizer(img_lr)
        # interpolat low-resolution image
        img_lr = utils_data.interp_sf(img_lr, sf=ds["sf_lr"])
        img_lr = img_lr[None]
        img_lr = torch.tensor(img_lr).to(device)

        # load high-resolution image (reference) -------------------------------
        img_hr = None
        if ds["path_hr"] != "Unknown":
            img_hr = utils_data.read_image(
                os.path.join(ds["path_hr"], sample_filename), expend_channel=False
            )
            img_hr = normalizer(img_hr)
            img_hr = utils_data.interp_sf(img_hr, sf=ds["sf_hr"])
            img_hr = img_hr[None]
            img_hr = torch.tensor(img_hr).to(device)

        # load text and text embedding -----------------------------------------
        if params["text_type"] == "single":
            text = "Task: {}; sample: {}; structure: {}; fluorescence indicator: {}; input microscope: {}; input pixel size: {}; target microscope: {}; target pixel size: {}.".format(
                ds["task#"],
                ds["sample"],
                ds["structure#"],
                ds["fluorescence indicator"],
                ds["input microscope"],
                ds["input pixel size"],
                ds["target microscope"],
                ds["target pixel size"],
            )

            if (params["d_cond"] == 0) or (params["d_cond"] is None):
                text_embed = None
            else:
                with torch.no_grad():
                    text_embed = embedder(text).to(device)

        if params["text_type"] == "paired":
            text_lr, text_hr = ds["text_lr"], ds["text_hr"]
            # embedding
            with torch.no_grad():
                text_embed_lr, text_embed_hr = embedder(text_lr), embedder(text_hr)
            if (params["d_cond"] == 0) or (params["d_cond"] is None):
                text_embed = None
            else:
                text_embed = torch.cat([text_embed_lr, text_embed_hr], dim=1).to(device)

        # ----------------------------------------------------------------------
        # time_embed = torch.zeros(size=(1,)).to(device)
        time_embed = None

        # ----------------------------------------------------------------------
        # prediction
        bs = params["batch_size"]
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=params["enable_amp"]
        ):
            with torch.no_grad():
                if params["patch_image"] and (params["patch_size"] < img_lr.shape[-1]):
                    # patching image
                    img_lr_patches = utils_data.unfold(
                        img=img_lr,
                        patch_size=params["patch_size"],
                        overlap=params["overlap"],
                        padding_mode="reflect",
                    )
                    num_patches = img_lr_patches.shape[0]
                    num_iter = math.ceil(num_patches / bs)

                    pbar = tqdm.tqdm(desc="Predicting ...", total=num_iter, ncols=100)

                    img_est_patches = []
                    for i_iter in range(num_iter):
                        img_est_patch = model(
                            img_lr_patches[i_iter * bs : bs + i_iter * bs],
                            time_embed,
                            text_embed,
                        )
                        img_est_patches.append(img_est_patch)
                        pbar.update(1)
                    pbar.close()
                    img_est_patches = torch.cat(img_est_patches, dim=0)

                    # rescale according to the intensity of input patches
                    # scales = img_lr_patches.mean(dim=(-1, -2), keepdim=True)
                    # scales = scales / torch.max(scales, dim=0, keepdim=True).values
                    # img_est_patches *= scales

                    # fold the patches
                    img_est = utils_data.fold_scale(
                        patches=img_est_patches,
                        original_image_shape=img_lr.shape,
                        overlap=params["overlap"],
                        crop_center=True,
                        # enable_scale=False,
                        enable_scale=True,
                    )
                else:
                    img_est = model(img_lr, time_embed, text_embed)
        img_est = torch.clip(img_est, min=0.0)

        # ----------------------------------------------------------------------
        # calculate metrics
        if img_hr is not None:
            if params["dim"] == 3:
                imgs_est = utils_eva.linear_transform(
                    img_true=img_hr, img_test=img_est, axis=(2, 3, 4)
                )
            if params["dim"] == 2:
                imgs_est = utils_eva.linear_transform(
                    img_true=img_hr, img_test=img_est, axis=(2, 3)
                )

            ssim = utils_eva.SSIM_tb(
                img_true=img_hr,
                img_test=imgs_est,
                data_range=None,
                version_wang=False,
            )
            psnr = utils_eva.PSNR_tb(
                img_true=img_hr, img_test=imgs_est, data_range=None
            )

            print(ssim, psnr)
        else:
            print("There is no reference data.")

        # ----------------------------------------------------------------------
        # save results
        io.imsave(
            os.path.join(path_results, sample_filename),
            arr=img_est.cpu().detach().numpy()[0],
            check_contrast=False,
        )
