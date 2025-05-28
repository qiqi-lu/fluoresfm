import numpy as np
import torch, os, tqdm, pandas, math
import skimage.io as io

from models.clip_embedder import CLIPTextEmbedder
from models.biomedclip_embedder import BiomedCLIPTextEmbedder
from models.unet_sd_c import UNetModel

import utils.data as utils_data
import utils.evaluation as utils_eva
import utils.optim as utils_optim

# ------------------------------------------------------------------------------
# parameters
# ------------------------------------------------------------------------------
checkpoints = (
    # [
    #     "_all_newnorm-ALL-v2-160-small-bs16",
    #     "checkpoints\\conditional\\unet_sd_c_mae_bs_16_lr_1e-05_all_newnorm_ALL-v2-160-res1-att0123\epoch_0_iter_700000.pt",
    #     ("ALL", 160),
    # ],
    [
        "_all_newnorm-ALL-v2-160-small-bs16-in-T",
        "checkpoints\\conditional\\unet_sd_c_mae_bs_16_lr_1e-05_all_newnorm_ALL-v2-160-res1-att0123\epoch_0_iter_700000.pt",
        ("T", 160),
    ],
    [
        "_all_newnorm-ALL-v2-160-small-bs16-in-TS",
        "checkpoints\\conditional\\unet_sd_c_mae_bs_16_lr_1e-05_all_newnorm_ALL-v2-160-res1-att0123\epoch_0_iter_700000.pt",
        ("TS", 160),
    ],
    [
        "_all_newnorm-ALL-v2-160-small-bs16-in-TSmicro",
        "checkpoints\\conditional\\unet_sd_c_mae_bs_16_lr_1e-05_all_newnorm_ALL-v2-160-res1-att0123\epoch_0_iter_700000.pt",
        ("TSmicro", 160),
    ],
    [
        "_all_newnorm-ALL-v2-160-small-bs16-in-TSpixel",
        "checkpoints\\conditional\\unet_sd_c_mae_bs_16_lr_1e-05_all_newnorm_ALL-v2-160-res1-att0123\epoch_0_iter_700000.pt",
        ("TSpixel", 160),
    ],
    # [
    #     "_all_newnorm-ALL-v2-160-small-bs16-crossx",
    #     "checkpoints\\conditional\\unet_sd_c_mae_bs_16_lr_1e-05_all_newnorm_ALL-v2-160-res1-att0123-crossx\epoch_2_iter_700000.pt",
    #     ("ALL", 160),
    # ],
    # [
    #     "_all_newnorm-ALL-v2-160-small-bs8",
    #     "checkpoints\\conditional\\unet_sd_c_mae_bs_8_lr_1e-05_all_newnorm_ALL-v2-160-res1-att0123\epoch_1_iter_700000.pt",
    #     ("ALL", 160),
    # ],
    # [
    #     "_all_newnorm-ALL-v2-160-small-bs4",
    #     "checkpoints\\conditional\\unet_sd_c_mae_bs_4_lr_1e-05_all_newnorm_ALL-v2-160-res1-att0123\epoch_0_iter_700000.pt",
    #     ("ALL", 160),
    # ],
    # [
    #     "_all_newnorm-ALL-v2-small-bs16-T77",
    #     "checkpoints\\conditional\\unet_sd_c_mae_bs_16_lr_1e-05_all_newnorm_ALL-v2-res1-att0123-T77\epoch_2_iter_700000.pt",
    #     ("T", 77),
    # ],
    # [
    #     "_all_newnorm-ALL-v2-small-bs16-TS77",
    #     "checkpoints\\conditional\\unet_sd_c_mae_bs_16_lr_1e-05_all_newnorm_ALL-v2-res1-att0123-TS77\epoch_2_iter_700000.pt",
    #     ("TS", 77),
    # ],
    # [
    #     "_all_newnorm-ALL-v2-small-bs16-TSmicro77",
    #     "checkpoints\\conditional\\unet_sd_c_mae_bs_16_lr_1e-05_all_newnorm_ALL-v2-res1-att0123-TSmicro77\epoch_2_iter_700000.pt",
    #     ("TSmicro", 77),
    # ],
    # [
    #     "_all_newnorm-ALL-v2-small-bs16-TSpixel77",
    #     "checkpoints\\conditional\\unet_sd_c_mae_bs_16_lr_1e-05_all_newnorm_ALL-v2-res1-att0123-TSpixel77\epoch_2_iter_700000.pt",
    #     ("TSpixel", 77),
    # ],
    # [
    #     "_all",
    #     "checkpoints\\conditional\\unet_sd_c_mae_bs_4_lr_1e-05_all\epoch_0_iter_775000.pt",
    #     ("all", 256),
    # ],
    # [
    #     "_all_clean",
    #     "checkpoints\\conditional\\unet_sd_c_mae_bs_4_lr_1e-05_all_clean\epoch_1_iter_1550000.pt",
    #     ("all", 256),
    # ],
    # [
    #     "_all_TSpixel",
    #     "checkpoints\\conditional\\unet_sd_c_mae_bs_4_lr_1e-05_all_TSpixel\epoch_0_iter_700000.pt",
    #     ("TSpixel", 77),
    # ],
    # [
    #     "_all_newnorm",
    #     "checkpoints\\conditional\\unet_sd_c_mae_bs_4_lr_1e-05_all_newnorm\epoch_0_iter_1290000.pt",
    #     ("all", 256),
    # ],
    # [
    #     "_all_newnorm-ALL-v2",
    #     "checkpoints\\conditional\\unet_sd_c_mae_bs_4_lr_1e-05_all_newnorm_ALL-v2\epoch_0_iter_1200000.pt",
    #     ("ALL", 256),
    # ],
    # [
    #     "_all_newnorm-ALL-v2-160-s123-bs16",
    #     "checkpoints\\conditional\\unet_sd_c_mae_bs_16_lr_1e-05_all_newnorm_ALL-v2-160-res1-att123\epoch_1_iter_775000.pt",
    #     ("ALL", 160),
    # ],
    # [
    #     "_all_newnorm_TS",
    #     "checkpoints\\conditional\\unet_sd_c_mae_bs_4_lr_1e-05_all_newnorm_TS\epoch_0_iter_1070000.pt",
    #     ("TS", 77),
    # ],
    # [
    #     "_all_newnorm_TSpixel",
    #     "checkpoints\\conditional\\unet_sd_c_mae_bs_4_lr_1e-05_all_newnorm_TSpixel\epoch_0_iter_745000.pt",
    #     ("TSpixel", 77),
    # ],
    # [
    #     "_all_newnorm_TSmicro",
    #     "checkpoints\\conditional\\unet_sd_c_mae_bs_4_lr_1e-05_all_newnorm_TSmicro\epoch_0_iter_955000.pt",
    #     ("TSmicro", 77),
    # ],
    # --------------------------------------------------------------------------
    # [
    #     "_sr",
    #     "checkpoints\\conditional\\unet_sd_c_mae_bs_4_lr_1e-05_sr\epoch_1_iter_1095000.pt",
    #     ("all", 256),
    # ],
    # [
    #     "_dcv",
    #     "checkpoints\\conditional\\unet_sd_c_mae_bs_4_lr_1e-05_dcv\epoch_6_iter_915000.pt",
    #     ("all", 256),
    # ],
    # [
    #     "_dn",
    #     "checkpoints\\conditional\\unet_sd_c_mae_bs_4_lr_1e-05_dn\epoch_0_iter_370000.pt",
    #     ("all", 256),
    # ],
    # [
    #     "_iso",
    #     "checkpoints\\conditional\\unet_sd_c_mae_bs_4_lr_1e-05_iso\epoch_4_iter_830000.pt",
    #     ("all", 256),
    # ],
    # --------------------------------------------------------------------------
    # [
    #     "_sr_crossx",
    #     "checkpoints\conditional\\unet_sd_c_mae_bs_4_lr_1e-05_sr_crossx\epoch_0_iter_565000.pt",
    #     ("all", 256),
    # ],
    # [
    #     "_dcv_crossx",
    #     "checkpoints\conditional\\unet_sd_c_mae_bs_4_lr_1e-05_dcv_crossx\epoch_16_iter_1015000.pt",
    #     ("all", 256),
    # ],
    # [
    #     "_dn_crossx",
    #     "checkpoints\conditional\\unet_sd_c_mae_bs_4_lr_1e-05_dn_crossx\epoch_0_iter_660000.pt",
    #     ("all", 256),
    # ],
    # [
    #     "_iso_crossx",
    #     "checkpoints\conditional\\unet_sd_c_mae_bs_4_lr_1e-05_iso_crossx\epoch_12_iter_1220000.pt",
    #     ("all", 256),
    # ],
)

params = {
    "device": "cuda:1",
    "enable_amp": True,
    # "complie_model": False,
    "complie_model": True,
    # text embedder ------------------------------------------------------------
    "embedder": "biomedclip",
    "path_embedder_json": "checkpoints/clip//biomedclip/open_clip_config.json",
    "path_embedder_bin": "checkpoints/clip//biomedclip/open_clip_pytorch_model.bin",
    # model parameters ---------------------------------------------------------
    "model_name": "unet_sd_c",
    # --------------------------------------------------------------------------
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
    "path_dataset_test": "dataset_test-v2.xlsx",
    # "data_clip": (0.0, 2.5),
    "data_clip": None,
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
        "srcaco2-h2b-sr-2",
        "srcaco2-survivin-sr-2",
        "srcaco2-tubulin-sr-2",
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
        # # --------------------------------------------------------------------
        "care-drosophila-iso",
        "care-retina0-iso",
        "care-retina1-iso",
        "care-liver-iso",
        # # # --------------------------------------------------------------------
        "vmsim3-mito-sr",
        "vmsim3-er-sr",
        "vmsim5-mito-sr",
        "vmsim3-mito-dcv",
        "vmsim3-er-dcv",
        "vmsim5-mito-dcv",
        "vmsim488-bead-patch-dcv",
        "vmsim568-bead-patch-dcv",
        "vmsim647-bead-patch-dcv",
        "sim-actin-3d-dcv",
        "sim-actin-2d-patch-dcv",
        "sim-microtubule-3d-dcv",
        "sim-microtubule-2d-patch-dcv",
        "bpae-dcv",
        "bpae-dn",
        "rcan3d-c2s-mt-dcv",
        "rcan3d-c2s-npc-dcv",
        "rcan3d-c2s-mt-sr",
        "rcan3d-c2s-npc-sr",
        "rcan3d-dn-actin-dn",
        "rcan3d-dn-er-dn",
        "rcan3d-dn-golgi-dn",
        "rcan3d-dn-lysosome-dn",
        "rcan3d-dn-mixtrixmito-dn",
        "rcan3d-dn-mt-dn",
        "rcan3d-dn-tomm20mito-dn",
        "biotisr-ccp-sr-1",
        "biotisr-ccp-sr-2",
        "biotisr-ccp-sr-3",
        "biotisr-factin-sr-1",
        "biotisr-factin-sr-2",
        "biotisr-factin-sr-3",
        "biotisr-factin-nonlinear-sr-1",
        "biotisr-factin-nonlinear-sr-2",
        "biotisr-factin-nonlinear-sr-3",
        "biotisr-lysosome-sr-1",
        "biotisr-lysosome-sr-2",
        "biotisr-lysosome-sr-3",
        "biotisr-mt-sr-1",
        "biotisr-mt-sr-2",
        "biotisr-mt-sr-3",
        "biotisr-mito-sr-1",
        "biotisr-mito-sr-2",
        "biotisr-mito-sr-3",
        "biotisr-ccp-dcv-1",
        "biotisr-ccp-dcv-2",
        "biotisr-ccp-dcv-3",
        "biotisr-factin-dcv-1",
        "biotisr-factin-dcv-2",
        "biotisr-factin-dcv-3",
        "biotisr-factin-nonlinear-dcv-1",
        "biotisr-factin-nonlinear-dcv-2",
        "biotisr-factin-nonlinear-dcv-3",
        "biotisr-lysosome-dcv-1",
        "biotisr-lysosome-dcv-2",
        "biotisr-lysosome-dcv-3",
        "biotisr-mt-dcv-1",
        "biotisr-mt-dcv-2",
        "biotisr-mt-dcv-3",
        "biotisr-mito-dcv-1",
        "biotisr-mito-dcv-2",
        "biotisr-mito-dcv-3",
        "biotisr-ccp-dn-1",
        "biotisr-ccp-dn-2",
        "biotisr-factin-dn-1",
        "biotisr-factin-dn-2",
        "biotisr-factin-nonlinear-dn-1",
        "biotisr-factin-nonlinear-dn-2",
        "biotisr-lysosome-dn-1",
        "biotisr-lysosome-dn-2",
        "biotisr-mt-dn-1",
        "biotisr-mt-dn-2",
        "biotisr-mito-dn-1",
        "biotisr-mito-dn-2",
        # ----------------------------------------------------------------------
        "cellpose3-2photon-dn-1",
        "cellpose3-2photon-dn-4",
        "cellpose3-2photon-dn-16",
        "cellpose3-2photon-dn-64",
        "colon-tissue-dn-high",
        "colon-tissue-dn-low",
        "hl60-high-noise-c00",
        "hl60-high-noise-c25",
        "hl60-high-noise-c50",
        "hl60-high-noise-c75",
        "hl60-low-noise-c00",
        "hl60-low-noise-c25",
        "hl60-low-noise-c50",
        "hl60-low-noise-c75",
        "scaffold-a549-dn",
        "granuseg-dn-high",
        "granuseg-dn-low",
        "colon-tissue-dcv-high",
        "colon-tissue-dcv-low",
        "hl60-high-noise-c00-dcv",
        "hl60-high-noise-c25-dcv",
        "hl60-high-noise-c50-dcv",
        "hl60-high-noise-c75-dcv",
        "hl60-low-noise-c00-dcv",
        "hl60-low-noise-c25-dcv",
        "hl60-low-noise-c50-dcv",
        "hl60-low-noise-c75-dcv",
        "granuseg-dcv-high",
        "granuseg-dcv-low",
        "deepbacs-seg-saureus-dcv",
        "deepbacs-seg-bsubtiles-dn",
    ],
    "num_sample": 8,
    "percentiles": (0.03, 0.995),
    "patch_image": True,
    "patch_size": 256,
    # "patch_size": 128,
    # "patch_size": 64,
    # output -------------------------------------------------------------------
    "path_output": "results\\predictions",
}

assert params["patch_size"] >= 64, "[ERROR] Patch size should be >= 64."
params.update(
    {
        "overlap": params["patch_size"] // 4,
        "batch_size": int(64 / params["patch_size"] * 32),
    }
)

# ------------------------------------------------------------------------------
if os.name == "posix":
    params.update(
        {
            "path_output": utils_data.win2linux(params["path_output"]),
            "path_embedder_json": utils_data.win2linux(params["path_embedder_json"]),
            "path_embedder_bin": utils_data.win2linux(params["path_embedder_bin"]),
        }
    )


# ------------------------------------------------------------------------------
print("-" * 80)
print("load dataset information ...")
utils_data.print_dict(params)

datasets_frame = pandas.read_excel(params["path_dataset_test"])
device = torch.device(params["device"])
normalizer_eva = utils_data.NormalizePercentile(p_low=0.03, p_high=0.995)
num_checkpoints = len(checkpoints)
time_embed = None
bs = params["batch_size"]

print("-" * 80)
print("number of datasets:", len(params["id_dataset"]))
print("number of checkpoints:", num_checkpoints)

# ------------------------------------------------------------------------------
# predict
# ------------------------------------------------------------------------------
for checkpoint in checkpoints:
    print("-" * 80)
    [print(x) for x in checkpoint]
    print("-" * 80)

    suffix, path_checkpoint, text_type = checkpoint

    if os.name == "posix":
        path_checkpoint = utils_data.win2linux(path_checkpoint)

    # update parameters according to the checkpoint
    if "cross" in suffix:
        params["d_cond"] = None
    else:
        params["d_cond"] = 768

    if "small" in suffix:
        params.update(
            {
                "n_res_blocks": 1,
                "attention_levels": [0, 1, 2, 3],
            }
        )
    elif "s123" in suffix:
        params.update(
            {
                "n_res_blocks": 1,
                "attention_levels": [1, 2, 3],
            }
        )
    else:
        params.update(
            {
                "n_res_blocks": 2,
                "attention_levels": [1, 2, 3],
            }
        )

    if "clip" in suffix:
        params.update({"data_clip": (0.0, 2.5)})

    input_normallizer = utils_data.NormalizePercentile(
        params["percentiles"][0], params["percentiles"][1]
    )

    print(f'd_cond: {params["d_cond"]}, percentiles: {params["percentiles"]}')

    # --------------------------------------------------------------------------
    # model
    # --------------------------------------------------------------------------
    # Text Embedder
    if params["embedder"] == "clip":
        embedder = CLIPTextEmbedder(device=torch.device("cpu"))
    elif params["embedder"] == "biomedclip":
        embedder = BiomedCLIPTextEmbedder(
            path_json=params["path_embedder_json"],
            path_bin=params["path_embedder_bin"],
            context_length=text_type[1],
            device=torch.device("cpu"),
        )
    else:
        raise ValueError(f"Embedder '{params['embedder']}' does not exist.")
    embedder.eval()

    # --------------------------------------------------------------------------
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

    # --------------------------------------------------------------------------
    # load model parameters
    print("load model parameters...")
    state_dict = torch.load(path_checkpoint, map_location=device, weights_only=True)[
        "model_state_dict"
    ]
    # del prefix for complied model
    state_dict = utils_optim.on_load_checkpoint(checkpoint=state_dict)
    model.load_state_dict(state_dict)
    if params["complie_model"]:
        print("compile model...")
        model = torch.compile(model)  # complie will need more time for model compile.
    model.eval()

    # --------------------------------------------------------------------------
    for id_dataset in params["id_dataset"]:
        print("-" * 80)
        # get dataset infomation
        try:
            ds = datasets_frame[datasets_frame["id"] == id_dataset].iloc[0]
            print("Dataset:", ds["id"])
        except:
            print(f"{id_dataset} Not Exist")
            continue

        # save retuls to
        path_results = os.path.join(
            params["path_output"], ds["id"], params["model_name"] + suffix
        )
        os.makedirs(path_results, exist_ok=True)

        # load sample names in current dataset
        filenames = utils_data.read_txt(path_txt=ds["path_index"])
        num_sample_total = len(filenames)

        # set the number of samples to be evaluated
        if params["num_sample"] is not None:
            if params["num_sample"] > num_sample_total:
                num_sample_eva = num_sample_total
            else:
                num_sample_eva = params["num_sample"]
        else:
            num_sample_eva = num_sample_total
        print("- Number of test data:", num_sample_eva, "/", num_sample_total)

        # ----------------------------------------------------------------------
        # PREDICT
        for i_sample in range(num_sample_eva):
            print("-" * 30)
            sample_filename = filenames[i_sample]
            print(f"- File Name: {sample_filename}")

            # load low-resolution image (input) --------------------------------
            img_lr = utils_data.read_image(os.path.join(ds["path_lr"], sample_filename))
            img_lr = np.clip(img_lr, 0.0, None)
            img_lr = input_normallizer(img_lr)
            img_lr = utils_data.interp_sf(img_lr, sf=ds["sf_lr"])[None]
            img_lr = torch.tensor(img_lr).to(device)

            if params["data_clip"] is not None:
                img_lr = torch.clip(
                    img_lr, min=params["data_clip"][0], max=params["data_clip"][1]
                )

            # load text and text embedding -------------------------------------
            # single text embedding
            if text_type[0] in ["all", "ALL", "TSpixel", "TSmicro", "TS", "T"]:
                if text_type[0] == "all":
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
                elif text_type[0] == "ALL":
                    text = "Task: {}; sample: {}; structure: {}; fluorescence indicator: {}; input microscope: {}; input pixel size: {}; target microscope: {}; target pixel size: {}.".format(
                        ds["task#"],
                        ds["sample"],
                        ds["structure#"],
                        ds["fluorescence indicator"],
                        f'{ds["input microscope-device"]} {ds["input microscope-params"]}',
                        ds["input pixel size"],
                        f'{ds["target microscope-device"]} {ds["target microscope-params"]}',
                        ds["target pixel size"],
                    )
                elif text_type[0] == "TSpixel":
                    text = "Task: {}; struture: {}; input pixel size: {}; target pixel size: {}.".format(
                        ds["task#"],
                        ds["structure#"],
                        ds["input pixel size"],
                        ds["target pixel size"],
                    )
                elif text_type[0] == "TSmicro":
                    text = "Task: {}; struture: {}; input microscope: {}; target microscope: {}.".format(
                        ds["task#"],
                        ds["structure#"],
                        ds["input microscope-device"],
                        ds["target microscope-device"],
                    )
                elif text_type[0] == "TS":
                    text = "Task: {}; struture: {}".format(
                        ds["task#"], ds["structure#"]
                    )
                elif text_type[0] == "T":
                    text = "Task: {}.".format(ds["task#"])

                if (params["d_cond"] == 0) or (params["d_cond"] is None):
                    text_embed = None
                else:
                    with torch.no_grad():
                        text_embed = embedder(text).to(device)

            # paired text embedding
            if text_type[0] == "paired":
                text_lr, text_hr = ds["text_lr"], ds["text_hr"]
                # embedding
                if (params["d_cond"] == 0) or (params["d_cond"] is None):
                    text_embed = None
                else:
                    with torch.no_grad():
                        text_embed_lr, text_embed_hr = embedder(text_lr), embedder(
                            text_hr
                        )
                    text_embed = torch.cat([text_embed_lr, text_embed_hr], dim=1).to(
                        device
                    )

            # ----------------------------------------------------------------------
            # prediction
            with torch.autocast("cuda", torch.float16, enabled=params["enable_amp"]):
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
                        # ----------------------------------------------------------
                        pbar = tqdm.tqdm(desc="PREDICT", total=num_iter, ncols=80)
                        img_est_patches = torch.zeros_like(img_lr_patches)

                        for i_iter in range(num_iter):
                            img_est_patch = model(
                                img_lr_patches[i_iter * bs : bs + i_iter * bs],
                                time_embed,
                                text_embed,
                            )
                            img_est_patches[
                                i_iter * bs : bs + i_iter * bs
                            ] += img_est_patch
                            pbar.update(1)
                        pbar.close()

                        # ----------------------------------------------------------
                        # fold the patches
                        # img_est = utils_data.fold_scale(
                        #     patches=img_est_patches,
                        #     original_image_shape=img_lr.shape,
                        #     overlap=params["overlap"],
                        #     crop_center=True,
                        #     # enable_scale=True,
                        # )
                        img_est = utils_data.fold_linear_ramp(
                            patches=img_est_patches,
                            original_image_shape=img_lr.shape,
                            overlap=params["overlap"],
                        )
                        img_est = torch.tensor(img_est)

                        # unpadding
                        img_est = img_est[
                            ..., : img_lr_shape_ori[-2], : img_lr_shape_ori[-1]
                        ]
                    else:
                        img_est = model(img_lr, time_embed, text_embed)

            # clip
            img_est = img_est.float().cpu().detach().numpy()

            # ------------------------------------------------------------------
            if ds["path_hr"] != "Unknown":
                dr = 2.5
                clip = lambda x: np.clip(x, 0.0, dr)

                # load high-resolution image (reference)
                img_hr = utils_data.read_image(
                    os.path.join(ds["path_hr"], sample_filename)
                )
                img_hr = utils_data.interp_sf(img_hr, sf=ds["sf_hr"])[0]
                img_hr = normalizer_eva(img_hr)

                img_est = utils_eva.linear_transform(
                    img_true=clip(img_hr), img_test=img_est
                )

                # calculate metrics
                dict_eva = {
                    "img_true": clip(img_hr),
                    "img_test": clip(img_est)[0, 0],
                    "data_range": dr,
                }
                psnr = utils_eva.PSNR(**dict_eva)
                ssim = utils_eva.SSIM(**dict_eva)
                print(f"PSNR: {psnr:.4f}, SSIM: {ssim:.4f}")
            else:
                print("There is no reference data.")

            # ------------------------------------------------------------------
            # save results
            io.imsave(
                os.path.join(path_results, sample_filename),
                arr=img_est[0],
                check_contrast=False,
            )
