import numpy as np
import torch, os, json, pandas, math, tqdm
import skimage.io as io
from models.unet import UNet
from models.care import CARE
from models.dfcan import DFCAN
from models.unifmir import UniModel

import utils.data as utils_data
import utils.evaluation as utils_eva
import utils.optim as utils_optim

# ------------------------------------------------------------------------------
# parameters
# ------------------------------------------------------------------------------
checkpoints = [
    # [
    #     "care",
    #     "_sr",
    #     "checkpoints\conditional\care_mae_bs_4_lr_0.0001_sr\epoch_1_iter_1035000.pt",
    # ],
    # [
    #     "care",
    #     "_biosr_sr_actin",
    #     "checkpoints\conditional\care_mse_bs_4_lr_0.0001_biosr_sr_actin\epoch_10_iter_250000.pt",
    # ],
    # [
    #     "care",
    #     "_biosr_sr_cpp",
    #     "checkpoints\conditional\care_mse_bs_4_lr_0.0001_biosr_sr_cpp\epoch_13_iter_250000.pt",
    # ],
    # [
    #     "care",
    #     "_biosr_sr_er",
    #     "checkpoints\conditional\care_mse_bs_4_lr_0.0001_biosr_sr_er\epoch_15_iter_250000.pt",
    # ],
    # [
    #     "care",
    #     "_biosr_sr_mt",
    #     "checkpoints\conditional\care_mse_bs_4_lr_0.0001_biosr_sr_mt\epoch_17_iter_250000.pt",
    # ],
    # [
    #     "care",
    #     "_biosr_sr",
    #     "checkpoints\conditional\care_mse_bs_4_lr_0.0001_biosr_sr\epoch_8_iter_600000.pt",
    # ],
    # # --------------------------------------------------------------------------
    # [
    #     "care",
    #     "_dcv",
    #     "checkpoints\conditional\care_mae_bs_4_lr_0.0001_dcv\epoch_19_iter_1255000.pt",
    # ],
    # [
    #     "care",
    #     "_biosr_dcv",
    #     "checkpoints\conditional\care_mse_bs_4_lr_0.0001_biosr_dcv\epoch_19_iter_390000.pt",
    # ],
    # [
    #     "care",
    #     "_dn",
    #     "checkpoints\conditional\care_mae_bs_4_lr_0.0001_dn\epoch_1_iter_710000.pt",
    # ],
    # [
    #     "care",
    #     "_iso",
    #     "checkpoints\conditional\care_mae_bs_4_lr_0.0001_iso\epoch_11_iter_1170000.pt",
    # ],
    # # --------------------------------------------------------------------------
    # [
    #     "dfcan",
    #     "_biosr_sr_2",
    #     "checkpoints\conditional\dfcan_mse_bs_4_lr_0.0001_biosr_sr_2\epoch_7_iter_600000.pt",
    # ],
    # [
    #     "dfcan",
    #     "_sr_2",
    #     "checkpoints\conditional\dfcan_mse_bs_4_lr_0.0001_sr_2\epoch_1_iter_675000.pt",
    # ],
    # [
    #     "dfcan",
    #     "_dcv",
    #     "checkpoints\conditional\dfcan_mae_bs_4_lr_0.0001_dcv\epoch_16_iter_1015000.pt",
    # ],
    # [
    #     "dfcan",
    #     "_dn",
    #     "checkpoints\conditional\dfcan_mae_bs_4_lr_0.0001_dn\epoch_0_iter_410000.pt",
    # ],
    # [
    #     "dfcan",
    #     "_iso",
    #     "checkpoints\conditional\dfcan_mae_bs_4_lr_0.0001_iso\epoch_9_iter_890000.pt",
    # ],
    # --------------------------------------------------------------------------
    # [
    #     "unifmir",
    #     "_all",
    #     "checkpoints\conditional\\unifmir_mae_bs_1_lr_0.0001_all\epoch_0_iter_2550000.pt",
    # ],
    [
        "unifmir",
        "_all-newnorm-v2",
        "checkpoints\conditional\\unifmir_mae_bs_1_lr_0.0001_newnorm-v2-all\epoch_0_iter_1000000.pt",
    ],
]

params = {
    "device": "cuda:0",
    # dataset ------------------------------------------------------------------
    "dim": 2,
    "path_dataset_test": "dataset_test-v2.xlsx",
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
        # # ----------------------------------------------------------------------
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
        # # --------------------------------------------------------------------
        # "care-drosophila-iso",
        # "care-retina0-iso",
        # "care-retina1-iso",
        # "care-liver-iso",
        # # # ------------------------------------------------------------------
        # "vmsim3-mito-sr",
        # "vmsim3-er-sr",
        # "vmsim5-mito-sr",
        # "vmsim3-mito-dcv",
        # "vmsim3-er-dcv",
        # "vmsim5-mito-dcv",
        # "vmsim488-bead-patch-dcv",
        # "vmsim568-bead-patch-dcv",
        # "vmsim647-bead-patch-dcv",
        # "sim-actin-3d-dcv",
        # "sim-actin-2d-patch-dcv",
        # "sim-microtubule-3d-dcv",
        # "sim-microtubule-2d-patch-dcv",
        # "bpae-dcv",
        # "bpae-dn",
        # "rcan3d-c2s-mt-dcv",
        "rcan3d-c2s-mt-sr",
        # "rcan3d-c2s-npc-dcv",
        "rcan3d-c2s-npc-sr",
        # "rcan3d-dn-actin-dn",
        # "rcan3d-dn-er-dn",
        # "rcan3d-dn-golgi-dn",
        # "rcan3d-dn-lysosome-dn",
        # "rcan3d-dn-mixtrixmito-dn",
        # "rcan3d-dn-mt-dn",
        # "rcan3d-dn-tomm20mito-dn",
        # "biotisr-ccp-sr-1",
        # "biotisr-ccp-sr-2",
        # "biotisr-ccp-sr-3",
        # "biotisr-factin-sr-1",
        # "biotisr-factin-sr-2",
        # "biotisr-factin-sr-3",
        # "biotisr-factin-nonlinear-sr-1",
        # "biotisr-factin-nonlinear-sr-2",
        # "biotisr-factin-nonlinear-sr-3",
        # "biotisr-lysosome-sr-1",
        # "biotisr-lysosome-sr-2",
        # "biotisr-lysosome-sr-3",
        # "biotisr-mt-sr-1",
        # "biotisr-mt-sr-2",
        # "biotisr-mt-sr-3",
        # "biotisr-mito-sr-1",
        # "biotisr-mito-sr-2",
        # "biotisr-mito-sr-3",
        # "biotisr-ccp-dcv-1",
        # "biotisr-ccp-dcv-2",
        # "biotisr-ccp-dcv-3",
        # "biotisr-factin-dcv-1",
        # "biotisr-factin-dcv-2",
        # "biotisr-factin-dcv-3",
        # "biotisr-factin-nonlinear-dcv-1",
        # "biotisr-factin-nonlinear-dcv-2",
        # "biotisr-factin-nonlinear-dcv-3",
        # "biotisr-lysosome-dcv-1",
        # "biotisr-lysosome-dcv-2",
        # "biotisr-lysosome-dcv-3",
        # "biotisr-mt-dcv-1",
        # "biotisr-mt-dcv-2",
        # "biotisr-mt-dcv-3",
        # "biotisr-mito-dcv-1",
        # "biotisr-mito-dcv-2",
        # "biotisr-mito-dcv-3",
        # "biotisr-ccp-dn-1",
        # "biotisr-ccp-dn-2",
        # "biotisr-factin-dn-1",
        # "biotisr-factin-dn-2",
        # "biotisr-factin-nonlinear-dn-1",
        # "biotisr-factin-nonlinear-dn-2",
        # "biotisr-lysosome-dn-1",
        # "biotisr-lysosome-dn-2",
        # "biotisr-mt-dn-1",
        # "biotisr-mt-dn-2",
        # "biotisr-mito-dn-1",
        # "biotisr-mito-dn-2",
    ],
    "scale_factor": 1,
    "num_sample": 8,
    # "num_sample": None,
    "percentiles": (0.0, 0.9999),
    "patch_image": True,
    "patch_size": 256,
    # output -------------------------------------------------------------------
    "path_output": "results\\predictions",
}

# ------------------------------------------------------------------------------
params.update(
    {
        "overlap": params["patch_size"] // 4,
        "batch_size": int(64 / params["patch_size"] * 32),
    }
)

if os.name == "posix":
    params["path_output"] = utils_data.win2linux(params["path_output"])

# ------------------------------------------------------------------------------
print("load dataset information ...")
utils_data.print_dict(params)

datasets_frame = pandas.read_excel(params["path_dataset_test"])
device = torch.device(params["device"])
output_normalizer = utils_data.NormalizePercentile(0.03, 0.995)
bs = params["batch_size"]
num_checkpoints = len(checkpoints)

# ------------------------------------------------------------------------------
print("-" * 50)
print("Number of checkpoints:", num_checkpoints)
print("number of datasets:", len(params["id_dataset"]))

# ------------------------------------------------------------------------------
# PREDICT
# ------------------------------------------------------------------------------
for checkpoint in checkpoints:
    print("-" * 50)
    print(f"Checkpoint: {checkpoint}")
    model_name, model_suffix, model_path = checkpoint

    if model_name == "dfcan" and ("_sr" in model_suffix):
        params.update(
            {
                "scale_factor": 2,
                "patch_size": 32,
                "overlap": 24,
            }
        )

    if os.name == "posix":
        model_path = utils_data.win2linux(model_path)

    # normalization
    if "norm" in model_suffix:
        params["percentiles"] = (0.03, 0.995)
    else:
        params["percentiles"] = (0.0, 0.9999)

    input_normalizer = utils_data.NormalizePercentile(
        params["percentiles"][0], params["percentiles"][1]
    )

    if "clip" in model_suffix:
        params.update({"data_clip": (0.0, 2.5)})

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
    for id_dataset in params["id_dataset"]:
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
        sf_lr = ds["sf_lr"]
        sf_hr = ds["sf_hr"]

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
                    img_lr_patches = utils_data.unfold(
                        img=img_lr,
                        patch_size=params["patch_size"],
                        overlap=params["overlap"],
                        padding_mode="reflect",
                    )
                    num_iter = math.ceil(img_lr_patches.shape[0] / bs)

                    # --------------------------------------------------------------
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

                    overlap = params["overlap"] * params["scale_factor"]

                    # ----------------------------------------------------------
                    # img_est = utils_data.fold_scale(
                    #     patches=img_est_patches,
                    #     original_image_shape=original_image_shape,
                    #     overlap=overlap,
                    #     crop_center=True,
                    #     enable_scale=True,
                    # )

                    img_est = utils_data.fold_linear_ramp(
                        patches=img_est_patches,
                        original_image_shape=original_image_shape,
                        overlap=overlap,
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
            # clip
            # img_est = torch.clip(img_est, min=0.0)
            img_est = img_est.float().cpu().detach().numpy()

            # ------------------------------------------------------------------
            if ds["path_hr"] != "Unknown":
                dr = 2.5
                clip = lambda x: np.clip(x, 0.0, dr)

                img_hr = utils_data.read_image(os.path.join(ds["path_hr"], filename))
                if params["scale_factor"] == 1:
                    img_hr = utils_data.interp_sf(img_hr, sf=sf_hr)
                img_hr = output_normalizer(img_hr)[0]

                imgs_est = utils_eva.linear_transform(
                    img_true=clip(img_hr), img_test=img_est
                )

                dict_eva = {
                    "img_true": clip(img_hr),
                    "img_test": clip(img_est)[0, 0],
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
