import numpy as np
import torch, os, json, pandas, math, tqdm
import skimage.io as io
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
    "device": "cuda:0",
    # model parameters ---------------------------------------------------------
    # "model_name": "care",
    # "suffix": "_sr",
    # "path_model": "checkpoints\conditional\care_mae_bs_4_lr_0.0001_sr\epoch_1_iter_1035000.pt",
    # "suffix": "_biosr_sr_actin",
    # "path_model": "checkpoints\conditional\care_mse_bs_4_lr_0.0001_biosr_sr_actin\epoch_10_iter_250000.pt",
    # "suffix": "_biosr_sr_cpp",
    # "path_model": "checkpoints\conditional\care_mse_bs_4_lr_0.0001_biosr_sr_cpp\epoch_13_iter_250000.pt",
    # "suffix": "_biosr_sr",
    # "path_model": "checkpoints\conditional\care_mse_bs_4_lr_0.0001_biosr_sr\epoch_8_iter_600000.pt",
    # --------------------------------------------------------------------------
    # "suffix": "_dcv",
    # "path_model": "checkpoints\conditional\care_mae_bs_4_lr_0.0001_dcv\epoch_19_iter_1255000.pt",
    # "suffix": "_biosr_dcv",
    # "path_model": "checkpoints\conditional\care_mse_bs_4_lr_0.0001_biosr_dcv\epoch_19_iter_390000.pt",
    # --------------------------------------------------------------------------
    # "suffix": "_dn",
    # "path_model": "checkpoints\conditional\care_mae_bs_4_lr_0.0001_dn\epoch_1_iter_710000.pt",
    # "suffix": "_iso",
    # "path_model": "checkpoints\conditional\care_mae_bs_4_lr_0.0001_iso\epoch_11_iter_1170000.pt",
    # --------------------------------------------------------------------------
    "model_name": "dfcan",
    # "suffix": "_biosr_sr_2",
    # "path_model": "checkpoints\conditional\dfcan_mse_bs_4_lr_0.0001_biosr_sr_2\epoch_7_iter_600000.pt",
    # "suffix": "_sr_2",
    # "path_model": "checkpoints\conditional\dfcan_mse_bs_4_lr_0.0001_sr_2\epoch_1_iter_675000.pt",
    # "suffix": "_dcv",
    # "path_model": "checkpoints\conditional\dfcan_mae_bs_4_lr_0.0001_dcv\epoch_16_iter_1015000.pt",
    # "suffix": "_dn",
    # "path_model": "checkpoints\conditional\dfcan_mae_bs_4_lr_0.0001_dn\epoch_0_iter_410000.pt",
    "suffix": "_iso",
    "path_model": "checkpoints\conditional\dfcan_mae_bs_4_lr_0.0001_iso\epoch_9_iter_890000.pt",
    # dataset ------------------------------------------------------------------
    "dim": 2,
    "path_dataset_test": "dataset_test.xlsx",
    "id_dataset": [
        # "biosr-cpp-sr-1",
        # "biosr-cpp-sr-2",
        # "biosr-cpp-sr-3",
        # "biosr-cpp-sr-4",
        # "biosr-cpp-sr-5",
        # "biosr-cpp-sr-6",
        # "biosr-cpp-sr-7",
        # "biosr-cpp-sr-8",
        # "biosr-cpp-sr-9",
        # "biosr-er-sr-1",
        # "biosr-er-sr-2",
        # "biosr-er-sr-3",
        # "biosr-er-sr-4",
        # "biosr-er-sr-5",
        # "biosr-er-sr-6",
        # "biosr-mt-sr-1",
        # "biosr-mt-sr-2",
        # "biosr-mt-sr-3",
        # "biosr-mt-sr-4",
        # "biosr-mt-sr-5",
        # "biosr-mt-sr-6",
        # "biosr-mt-sr-7",
        # "biosr-mt-sr-8",
        # "biosr-mt-sr-9",
        # "biosr-actin-sr-1",
        # "biosr-actin-sr-2",
        # "biosr-actin-sr-3",
        # "biosr-actin-sr-4",
        # "biosr-actin-sr-5",
        # "biosr-actin-sr-6",
        # "biosr-actin-sr-7",
        # "biosr-actin-sr-8",
        # "biosr-actin-sr-9",
        # "biosr-actin-sr-10",
        # "biosr-actin-sr-11",
        # "biosr-actin-sr-12",
        # "deepbacs-sim-ecoli-sr",
        # "deepbacs-sim-saureus-sr",
        # "w2s-c0-sr-1",
        # "w2s-c0-sr-2",
        # "w2s-c0-sr-3",
        # "w2s-c0-sr-4",
        # "w2s-c0-sr-5",
        # "w2s-c0-sr-6",
        # "w2s-c0-sr-7",
        # "w2s-c1-sr-1",
        # "w2s-c1-sr-2",
        # "w2s-c1-sr-3",
        # "w2s-c1-sr-4",
        # "w2s-c1-sr-5",
        # "w2s-c1-sr-6",
        # "w2s-c1-sr-7",
        # "w2s-c2-sr-1",
        # "w2s-c2-sr-2",
        # "w2s-c2-sr-3",
        # "w2s-c2-sr-4",
        # "w2s-c2-sr-5",
        # "w2s-c2-sr-6",
        # "w2s-c2-sr-7",
        # "srcaco2-h2b-sr-8",
        # "srcaco2-h2b-sr-4",
        # "srcaco2-h2b-sr-2",
        # "srcaco2-survivin-sr-8",
        # "srcaco2-survivin-sr-4",
        # "srcaco2-survivin-sr-2",
        # "srcaco2-tubulin-sr-8",
        # "srcaco2-tubulin-sr-4",
        # "srcaco2-tubulin-sr-2",
        # ----------------------------------------------------------------------
        # "biosr-cpp-dn-1",
        # "biosr-cpp-dn-2",
        # "biosr-cpp-dn-3",
        # "biosr-cpp-dn-4",
        # "biosr-cpp-dn-5",
        # "biosr-cpp-dn-6",
        # "biosr-cpp-dn-7",
        # "biosr-cpp-dn-8",
        # "biosr-er-dn-1",
        # "biosr-er-dn-2",
        # "biosr-er-dn-3",
        # "biosr-er-dn-4",
        # "biosr-er-dn-5",
        # "biosr-mt-dn-1",
        # "biosr-mt-dn-2",
        # "biosr-mt-dn-3",
        # "biosr-mt-dn-4",
        # "biosr-mt-dn-5",
        # "biosr-mt-dn-6",
        # "biosr-mt-dn-7",
        # "biosr-mt-dn-8",
        # "biosr-actin-dn-1",
        # "biosr-actin-dn-2",
        # "biosr-actin-dn-3",
        # "biosr-actin-dn-4",
        # "biosr-actin-dn-5",
        # "biosr-actin-dn-6",
        # "biosr-actin-dn-7",
        # "biosr-actin-dn-8",
        # "biosr-actin-dn-9",
        # "biosr-actin-dn-10",
        # "biosr-actin-dn-11",
        # "biosr-actinnl-dn-1",
        # "biosr-actinnl-dn-2",
        # "biosr-actinnl-dn-3",
        # "biosr-actinnl-dn-4",
        # "biosr-actinnl-dn-5",
        # "biosr-actinnl-dn-6",
        # "biosr-actinnl-dn-7",
        # "biosr-actinnl-dn-8",
        # "biosr+-ccp-dn-1",
        # "biosr+-ccp-dn-2",
        # "biosr+-ccp-dn-3",
        # "biosr+-ccp-dn-4",
        # "biosr+-ccp-dn-5",
        # "biosr+-ccp-dn-6",
        # "biosr+-ccp-dn-7",
        # "biosr+-ccp-dn-8",
        # "biosr+-er-dn-1",
        # "biosr+-er-dn-2",
        # "biosr+-er-dn-3",
        # "biosr+-er-dn-4",
        # "biosr+-er-dn-5",
        # "biosr+-er-dn-6",
        # "biosr+-actin-dn-1",
        # "biosr+-actin-dn-2",
        # "biosr+-actin-dn-3",
        # "biosr+-actin-dn-4",
        # "biosr+-actin-dn-5",
        # "biosr+-actin-dn-6",
        # "biosr+-actin-dn-7",
        # "biosr+-actin-dn-8",
        # "biosr+-actin-dn-9",
        # "biosr+-actin-dn-10",
        # "biosr+-actin-dn-11",
        # "biosr+-mt-dn-1",
        # "biosr+-mt-dn-2",
        # "biosr+-mt-dn-3",
        # "biosr+-mt-dn-4",
        # "biosr+-mt-dn-5",
        # "biosr+-mt-dn-6",
        # "biosr+-mt-dn-7",
        # "biosr+-mt-dn-8",
        # "biosr+-myosin-dn-1",
        # "biosr+-myosin-dn-2",
        # "biosr+-myosin-dn-3",
        # "biosr+-myosin-dn-4",
        # "biosr+-myosin-dn-5",
        # "biosr+-myosin-dn-6",
        # "biosr+-myosin-dn-7",
        # "biosr+-myosin-dn-8",
        # "care-planaria-dn-1",
        # "care-planaria-dn-2",
        # "care-planaria-dn-3",
        # "care-tribolium-dn-1",
        # "care-tribolium-dn-2",
        # "care-tribolium-dn-3",
        # "deepbacs-ecoli-dn",
        # "deepbacs-ecoli2-dn",
        # "w2s-c0-dn-1",
        # "w2s-c0-dn-2",
        # "w2s-c0-dn-3",
        # "w2s-c0-dn-4",
        # "w2s-c0-dn-5",
        # "w2s-c0-dn-6",
        # "w2s-c1-dn-1",
        # "w2s-c1-dn-2",
        # "w2s-c1-dn-3",
        # "w2s-c1-dn-4",
        # "w2s-c1-dn-5",
        # "w2s-c1-dn-6",
        # "w2s-c2-dn-1",
        # "w2s-c2-dn-2",
        # "w2s-c2-dn-3",
        # "w2s-c2-dn-4",
        # "w2s-c2-dn-5",
        # "w2s-c2-dn-6",
        # "srcaco2-h2b-dn-8",
        # "srcaco2-h2b-dn-4",
        # "srcaco2-h2b-dn-2",
        # "srcaco2-survivin-dn-8",
        # "srcaco2-survivin-dn-4",
        # "srcaco2-survivin-dn-2",
        # "srcaco2-tubulin-dn-8",
        # "srcaco2-tubulin-dn-4",
        # "srcaco2-tubulin-dn-2",
        # "fmd-confocal-bpae-b-avg2",
        # "fmd-confocal-bpae-b-avg4",
        # "fmd-confocal-bpae-b-avg8",
        # "fmd-confocal-bpae-b-avg16",
        # "fmd-confocal-bpae-g-avg2",
        # "fmd-confocal-bpae-g-avg4",
        # "fmd-confocal-bpae-g-avg8",
        # "fmd-confocal-bpae-g-avg16",
        # "fmd-confocal-bpae-r-avg2",
        # "fmd-confocal-bpae-r-avg4",
        # "fmd-confocal-bpae-r-avg8",
        # "fmd-confocal-bpae-r-avg16",
        # "fmd-confocal-fish-avg2",
        # "fmd-confocal-fish-avg4",
        # "fmd-confocal-fish-avg8",
        # "fmd-confocal-fish-avg16",
        # "fmd-confocal-mice-avg2",
        # "fmd-confocal-mice-avg4",
        # "fmd-confocal-mice-avg8",
        # "fmd-confocal-mice-avg16",
        # "fmd-twophoton-mice-avg2",
        # "fmd-twophoton-mice-avg4",
        # "fmd-twophoton-mice-avg8",
        # "fmd-twophoton-mice-avg16",
        # "fmd-twophoton-bpae-b-avg2",
        # "fmd-twophoton-bpae-b-avg4",
        # "fmd-twophoton-bpae-b-avg8",
        # "fmd-twophoton-bpae-b-avg16",
        # "fmd-twophoton-bpae-g-avg2",
        # "fmd-twophoton-bpae-g-avg4",
        # "fmd-twophoton-bpae-g-avg8",
        # "fmd-twophoton-bpae-g-avg16",
        # "fmd-twophoton-bpae-r-avg2",
        # "fmd-twophoton-bpae-r-avg4",
        # "fmd-twophoton-bpae-r-avg8",
        # "fmd-twophoton-bpae-r-avg16",
        # "fmd-wf-bpae-b-avg2",
        # "fmd-wf-bpae-b-avg4",
        # "fmd-wf-bpae-b-avg8",
        # "fmd-wf-bpae-b-avg16",
        # "fmd-wf-bpae-g-avg2",
        # "fmd-wf-bpae-g-avg4",
        # "fmd-wf-bpae-g-avg8",
        # "fmd-wf-bpae-g-avg16",
        # "fmd-wf-bpae-r-avg2",
        # "fmd-wf-bpae-r-avg4",
        # "fmd-wf-bpae-r-avg8",
        # "fmd-wf-bpae-r-avg16",
        # # ----------------------------------------------------------------------
        # "biosr-cpp-dcv-1",
        # "biosr-cpp-dcv-2",
        # "biosr-cpp-dcv-3",
        # "biosr-cpp-dcv-4",
        # "biosr-cpp-dcv-5",
        # "biosr-cpp-dcv-6",
        # "biosr-cpp-dcv-7",
        # "biosr-cpp-dcv-8",
        # "biosr-cpp-dcv-9",
        # "biosr-er-dcv-1",
        # "biosr-er-dcv-2",
        # "biosr-er-dcv-3",
        # "biosr-er-dcv-4",
        # "biosr-er-dcv-5",
        # "biosr-er-dcv-6",
        # "biosr-mt-dcv-1",
        # "biosr-mt-dcv-2",
        # "biosr-mt-dcv-3",
        # "biosr-mt-dcv-4",
        # "biosr-mt-dcv-5",
        # "biosr-mt-dcv-6",
        # "biosr-mt-dcv-7",
        # "biosr-mt-dcv-8",
        # "biosr-mt-dcv-9",
        # "biosr-actin-dcv-1",
        # "biosr-actin-dcv-2",
        # "biosr-actin-dcv-3",
        # "biosr-actin-dcv-4",
        # "biosr-actin-dcv-5",
        # "biosr-actin-dcv-6",
        # "biosr-actin-dcv-7",
        # "biosr-actin-dcv-8",
        # "biosr-actin-dcv-9",
        # "biosr-actin-dcv-10",
        # "biosr-actin-dcv-11",
        # "biosr-actin-dcv-12",
        # "biosr-actinnl-dcv-1",
        # "biosr-actinnl-dcv-2",
        # "biosr-actinnl-dcv-3",
        # "biosr-actinnl-dcv-4",
        # "biosr-actinnl-dcv-5",
        # "biosr-actinnl-dcv-6",
        # "biosr-actinnl-dcv-7",
        # "biosr-actinnl-dcv-8",
        # "biosr-actinnl-dcv-9",
        # "care-synthe-granules-dcv",
        # "care-synthe-tubulin-dcv",
        # "care-synthe-tubulin-gfp-dcv",
        # "deepbacs-sim-ecoli-dcv",
        # "deepbacs-sim-saureus-dcv",
        # "w2s-c0-dcv-1",
        # "w2s-c0-dcv-2",
        # "w2s-c0-dcv-3",
        # "w2s-c0-dcv-4",
        # "w2s-c0-dcv-5",
        # "w2s-c0-dcv-6",
        # "w2s-c0-dcv-7",
        # "w2s-c1-dcv-1",
        # "w2s-c1-dcv-2",
        # "w2s-c1-dcv-3",
        # "w2s-c1-dcv-4",
        # "w2s-c1-dcv-5",
        # "w2s-c1-dcv-6",
        # "w2s-c1-dcv-7",
        # "w2s-c2-dcv-1",
        # "w2s-c2-dcv-2",
        # "w2s-c2-dcv-3",
        # "w2s-c2-dcv-4",
        # "w2s-c2-dcv-5",
        # "w2s-c2-dcv-6",
        # "w2s-c2-dcv-7",
        # ----------------------------------------------------------------------
        "care-drosophila-iso",
        "care-retina0-iso",
        "care-retina1-iso",
        "care-liver-iso",
    ],
    "scale_factor": 1,
    "id_sample": [0, 1, 2, 3, 4, 5, 6, 7],
    "p_low": 0.0,
    "p_high": 0.9999,
    "patch_image": True,
    # "patch_image": False,
    "patch_size": 64,
    "overlap": 32,
    "batch_size": 64,
    # output -------------------------------------------------------------------
    "path_output": "outputs\\unet_c",
}

if params["model_name"] == "dfcan" and ("sr" in params["suffix"]):
    params.update(
        {
            "scale_factor": 2,
            "patch_size": 32,
            "overlap": 16,
        }
    )

if os.name == "posix":
    params["path_model"] = utils_data.win2linux(params["path_model"])
    params["path_output"] = utils_data.win2linux(params["path_output"])

# ------------------------------------------------------------------------------
datasets_frame = pandas.read_excel(params["path_dataset_test"])
dataset_info = datasets_frame[datasets_frame["id"].isin(params["id_dataset"])]
num_datasets = dataset_info.shape[0]

utils_data.print_dict(params)
device = torch.device(params["device"])

# ------------------------------------------------------------------------------
# model
# ------------------------------------------------------------------------------
# 2D models
if params["model_name"] == "unet":
    model = UNet(
        in_channels=1, out_channels=1, bilinear=False, residual=True, pos_out=False
    )

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

model = model.to(device)
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

    print("- Number of test data:", len(params["id_sample"]), "/", num_sample)

    if params["id_sample"]:
        idxs = params["id_sample"]
        if len(idxs) > num_sample:
            idxs = range(num_sample)
    else:
        idxs = range(num_sample)
    print(idxs)

    for i_sample in idxs:
        sample_filename = path_sample[i_sample]
        print(f"- File Name: {sample_filename}")

        # load low-resolution image --------------------------------------------
        img_lr = utils_data.read_image(
            os.path.join(ds["path_lr"], sample_filename), expend_channel=False
        )
        # normalization
        img_lr = normalizer(img_lr)
        # interpolat low-resolution image
        if params["scale_factor"] == 1:
            img_lr = utils_data.interp_sf(img_lr, sf=ds["sf_lr"])

        # special processing for some dataset
        if (
            ds["id"] in ["deepbacs-sim-ecoli-sr", "deepbacs-sim-saureus-sr"]
            and params["model_name"] == "dfcan"
        ):
            img_lr = utils_data.interp_sf(img_lr, sf=-2)

        img_lr = torch.tensor(img_lr[None]).to(device)

        # load high-resolution image (reference) -------------------------------
        img_hr = None
        if ds["path_hr"] != "Unknown":
            img_hr = utils_data.read_image(
                os.path.join(ds["path_hr"], sample_filename), expend_channel=False
            )
            img_hr = normalizer(img_hr)
            if params["scale_factor"] == 1:
                img_hr = utils_data.interp_sf(img_hr, sf=ds["sf_hr"])
            img_hr = img_hr[None]
            img_hr = torch.tensor(img_hr).to(device)

        # prediction -----------------------------------------------------------
        bs = params["batch_size"]
        with torch.no_grad():
            if not params["patch_image"]:
                input_shape = img_lr.shape
                # padding for care model, which is a unet model requires specific image size
                if params["model_name"] == "care":
                    if input_shape[-1] % 4 > 0:
                        pad_size = 4 - input_shape[-1] % 4
                        img_lr = torch.nn.functional.pad(
                            img_lr, pad=(0, pad_size, 0, pad_size), mode="reflect"
                        )
                img_est = model(img_lr)
                if params["model_name"] == "care":
                    if input_shape[-1] % 4 > 0:
                        img_est = img_est[:, :, : input_shape[-2], : input_shape[-1]]
            else:
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
                        img_lr_patches[i_iter * bs : bs + i_iter * bs]
                    )
                    img_est_patches.append(img_est_patch)
                    pbar.update(1)
                pbar.close()
                img_est_patches = torch.cat(img_est_patches, dim=0)

                # fold the patches
                img_est = utils_data.fold_scale(
                    patches=img_est_patches,
                    original_image_shape=(
                        img_lr.shape[0],
                        img_lr.shape[1],
                        img_lr.shape[2] * params["scale_factor"],
                        img_lr.shape[3] * params["scale_factor"],
                    ),
                    overlap=params["overlap"] * params["scale_factor"],
                    crop_center=True,
                    enable_scale=False,
                )

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
