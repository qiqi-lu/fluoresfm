"""
Calculate the metrics of each image in each dataset.
Saved into a excel file.
Each row is a dataset, each column is a methods, each sheet is a metric.
"""

import os, pandas, traceback
import numpy as np
import utils.data as utils_data
import utils.evaluation as eva
from utils.data import rolling_ball_approximation, interp_sf
from dataset_analysis import datasets_need_bkg_sub

# ------------------------------------------------------------------------------
# parameters
# ------------------------------------------------------------------------------
params = {
    "dataset_names": [
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
        # "srcaco2-h2b-sr-2",
        # "srcaco2-survivin-sr-2",
        # "srcaco2-tubulin-sr-2",
        # # # ----------------------------------------------------------------------
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
        # # # --------------------------------------------------------------------
        # "sim-actin-3d-dcv",
        # "sim-actin-2d-patch-dcv",
        # "sim-microtubule-3d-dcv",
        # "sim-microtubule-2d-patch-dcv",
        # "bpae-dcv",
        # "vmsim3-mito-dcv",
        # "vmsim3-mito-dcv-crop",
        # "vmsim3-er-dcv",
        # "vmsim5-mito-dcv",
        # "vmsim5-mito-dcv-crop",
        # "vmsim488-bead-patch-dcv",
        # "vmsim568-bead-patch-dcv",
        # "vmsim647-bead-patch-dcv",
        # "rcan3d-c2s-mt-dcv",
        # "rcan3d-c2s-npc-dcv",
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
        # # ----------------------------------------------------------------------
        # "bpae-dn",
        # "rcan3d-dn-actin-dn",
        # "rcan3d-dn-er-dn",
        # "rcan3d-dn-golgi-dn",
        # "rcan3d-dn-lysosome-dn",
        # "rcan3d-dn-mixtrixmito-dn",
        # "rcan3d-dn-mt-dn",
        # "rcan3d-dn-tomm20mito-dn",
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
        # # ----------------------------------------------------------------------
        # "vmsim3-mito-sr",
        # "vmsim3-mito-sr-crop",
        # "vmsim3-er-sr",
        # "vmsim5-mito-sr",
        # "vmsim5-mito-sr-crop",
        # "rcan3d-c2s-mt-sr",
        # "rcan3d-c2s-npc-sr",
        # # ----------------------------- finetune -------------------------------
        # "biotisr-mt-sr-1",
        # "biotisr-mt-sr-2",
        # "biotisr-mt-sr-3",
        # "biotisr-mito-sr-1",
        # "biotisr-mito-sr-2",
        # "biotisr-mito-sr-3",
        # "biotisr-factin-nonlinear-sr-1",
        # "biotisr-factin-nonlinear-sr-2",
        # "biotisr-factin-nonlinear-sr-3",
        # "biotisr-ccp-sr-1",
        # "biotisr-ccp-sr-2",
        # "biotisr-ccp-sr-3",
        # "biotisr-factin-sr-1",
        # "biotisr-factin-sr-2",
        # "biotisr-factin-sr-3",
        # "biotisr-lysosome-sr-1",
        # "biotisr-lysosome-sr-2",
        # "biotisr-lysosome-sr-3",
        # "cellpose3-2photon-dn-1",
        # "cellpose3-2photon-dn-4",
        # "cellpose3-2photon-dn-16",
        # "cellpose3-2photon-dn-64",
    ],
    "num_sample": 8,
    # "num_sample": None,
    "path_dataset_test": "dataset_test-v2.xlsx",
    "path_results": "results\\predictions",
    "methods": (
        # ------------------ data distribution bias effect ---------------------
        # ("CARE:CCP", "care_biosr_sr_cpp-v2-newnorm"),
        # ("CARE:ER", "care_biosr_sr_er-v2-newnorm"),
        # ("CARE:MT", "care_biosr_sr_mt-v2-newnorm"),
        # ("CARE:F-actin", "care_biosr_sr_actin-v2-newnorm"),
        # ("CARE:Mix", "care_biosr_sr_mix-v2-newnorm"),
        # ------------------- comparison methods -------------------------------
        ("UniFMIR:all-v2", "unifmir_all-newnorm-v2"),
        # --------------------- batch size effect ------------------------------
        # (
        #     "UNet-c:all-newnorm-ALL-v2-160-small-bs4",
        #     "unet_sd_c_all_newnorm-ALL-v2-160-small-bs4",
        # ),
        # (
        #     "UNet-c:all-newnorm-ALL-v2-160-small-bs8",
        #     "unet_sd_c_all_newnorm-ALL-v2-160-small-bs8",
        # ),
        (
            "UNet-c:all-newnorm-ALL-v2-160-small-bs16",
            "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16",
        ),
        (
            "UNet-c:all-newnorm-ALL-v2-160-small-bs16-crossx",
            "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-crossx",
        ),
        # # -------------------- train with different text -----------------------
        (
            "UNet-c:all-newnorm-ALL-v2-small-bs16-T77",
            "unet_sd_c_all_newnorm-ALL-v2-small-bs16-T77",
        ),
        (
            "UNet-c:all-newnorm-ALL-v2-small-bs16-TS77",
            "unet_sd_c_all_newnorm-ALL-v2-small-bs16-TS77",
        ),
        # (
        #     "UNet-c:all-newnorm-ALL-v2-small-bs16-TSmicro77",
        #     "unet_sd_c_all_newnorm-ALL-v2-small-bs16-TSmicro77",
        # ),
        # (
        #     "UNet-c:all-newnorm-ALL-v2-small-bs16-TSpixel77",
        #     "unet_sd_c_all_newnorm-ALL-v2-small-bs16-TSpixel77",
        # ),
        # # -------------------- test with different text ------------------------
        (
            "UNet-c:all-newnorm-ALL-v2-160-small-bs16-in-T",
            "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-in-T",
        ),
        (
            "UNet-c:all-newnorm-ALL-v2-160-small-bs16-in-TS",
            "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-in-TS",
        ),
        # (
        #     "UNet-c:all-newnorm-ALL-v2-160-small-bs16-in-TSmicro",
        #     "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-in-TSmicro",
        # ),
        # (
        #     "UNet-c:all-newnorm-ALL-v2-160-small-bs16-in-TSpixel",
        #     "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-in-TSpixel",
        # ),
        # --------------------------- finetune ---------------------------------
        # (
        #     # "UNet-c:all-newnorm-ALL-v2-160-small-bs16-ft-io-cellpose3-2photon-dn-1",
        #     # "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-ft-inout-cellpose3-2photon-dn-1",
        #     # "UNet-c:all-newnorm-ALL-v2-160-small-bs16-ft-io-cellpose3-2photon-dn-4",
        #     # "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-ft-inout-cellpose3-2photon-dn-4",
        #     # "UNet-c:all-newnorm-ALL-v2-160-small-bs16-ft-io-cellpose3-2photon-dn-16",
        #     # "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-ft-inout-cellpose3-2photon-dn-16",
        #     # ------------------------------------------------------------------
        #     # "UNet-c:all-newnorm-ALL-v2-160-small-bs16-ft-io-biotisr-mt-sr-1",
        #     # "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-ft-inout-biotisr-mt-sr-1",
        #     # "UNet-c:all-newnorm-ALL-v2-160-small-bs16-ft-io-biotisr-mt-sr-2",
        #     # "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-ft-inout-biotisr-mt-sr-2",
        #     # "UNet-c:all-newnorm-ALL-v2-160-small-bs16-ft-io-biotisr-mt-sr-3",
        #     # "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-ft-inout-biotisr-mt-sr-3",
        #     # ------------------------------------------------------------------
        #     # "UNet-c:all-newnorm-ALL-v2-160-small-bs16-ft-io-biotisr-mito-sr-1",
        #     # "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-ft-inout-biotisr-mito-sr-1",
        #     # "UNet-c:all-newnorm-ALL-v2-160-small-bs16-ft-io-biotisr-mito-sr-2",
        #     # "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-ft-inout-biotisr-mito-sr-2",
        #     # "UNet-c:all-newnorm-ALL-v2-160-small-bs16-ft-io-biotisr-mito-sr-3",
        #     # "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-ft-inout-biotisr-mito-sr-3",
        #     # ------------------------------------------------------------------
        #     # "UNet-c:all-newnorm-ALL-v2-160-small-bs16-ft-io-biotisr-factin-nonlinear-sr-1",
        #     # "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-ft-inout-biotisr-factin-nonlinear-sr-1",
        #     # "UNet-c:all-newnorm-ALL-v2-160-small-bs16-ft-io-biotisr-factin-nonlinear-sr-2",
        #     # "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-ft-inout-biotisr-factin-nonlinear-sr-2",
        #     # "UNet-c:all-newnorm-ALL-v2-160-small-bs16-ft-io-biotisr-factin-nonlinear-sr-3",
        #     # "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-ft-inout-biotisr-factin-nonlinear-sr-3",
        #     # ------------------------------------------------------------------
        #     # "UNet-c:all-newnorm-ALL-v2-160-small-bs16-ft-io-biotisr-ccp-sr-1",
        #     # "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-ft-inout-biotisr-ccp-sr-1",
        #     # "UNet-c:all-newnorm-ALL-v2-160-small-bs16-ft-io-biotisr-ccp-sr-2",
        #     # "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-ft-inout-biotisr-ccp-sr-2",
        #     # "UNet-c:all-newnorm-ALL-v2-160-small-bs16-ft-io-biotisr-ccp-sr-3",
        #     # "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-ft-inout-biotisr-ccp-sr-3",
        #     # ------------------------------------------------------------------
        #     # "UNet-c:all-newnorm-ALL-v2-160-small-bs16-ft-io-biotisr-factin-sr-1",
        #     # "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-ft-inout-biotisr-factin-sr-1",
        #     # "UNet-c:all-newnorm-ALL-v2-160-small-bs16-ft-io-biotisr-factin-sr-2",
        #     # "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-ft-inout-biotisr-factin-sr-2",
        #     # "UNet-c:all-newnorm-ALL-v2-160-small-bs16-ft-io-biotisr-factin-sr-3",
        #     # "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-ft-inout-biotisr-factin-sr-3",
        #     # ------------------------------------------------------------------
        #     # "UNet-c:all-newnorm-ALL-v2-160-small-bs16-ft-io-biotisr-lysosome-sr-1",
        #     # "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-ft-inout-biotisr-lysosome-sr-1",
        #     # "UNet-c:all-newnorm-ALL-v2-160-small-bs16-ft-io-biotisr-lysosome-sr-2",
        #     # "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-ft-inout-biotisr-lysosome-sr-2",
        #     "UNet-c:all-newnorm-ALL-v2-160-small-bs16-ft-io-biotisr-lysosome-sr-3",
        #     "unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-ft-inout-biotisr-lysosome-sr-3",
        #     # ------------------------------------------------------------------
        # ),
        # (
        #     # "CARE:v2-newnorm-ft-io-cellpose3-2photon-dn-1",
        #     # "care-v2-newnorm-ft-inout-cellpose3-2photon-dn-1",
        #     # "CARE:v2-newnorm-ft-io-cellpose3-2photon-dn-4",
        #     # "care-v2-newnorm-ft-inout-cellpose3-2photon-dn-4",
        #     # "CARE:v2-newnorm-ft-io-cellpose3-2photon-dn-16",
        #     # "care-v2-newnorm-ft-inout-cellpose3-2photon-dn-16",
        #     # ------------------------------------------------------------------
        #     # "CARE:v2-newnorm-ft-io-biotisr-mt-sr-1",
        #     # "care-v2-newnorm-ft-inout-biotisr-mt-sr-1",
        #     # "CARE:v2-newnorm-ft-io-biotisr-mt-sr-2",
        #     # "care-v2-newnorm-ft-inout-biotisr-mt-sr-2",
        #     # "CARE:v2-newnorm-ft-io-biotisr-mt-sr-3",
        #     # "care-v2-newnorm-ft-inout-biotisr-mt-sr-3",
        #     # ------------------------------------------------------------------
        #     # "CARE:v2-newnorm-ft-io-biotisr-mito-sr-1",
        #     # "care-v2-newnorm-ft-inout-biotisr-mito-sr-1",
        #     # "CARE:v2-newnorm-ft-io-biotisr-mito-sr-2",
        #     # "care-v2-newnorm-ft-inout-biotisr-mito-sr-2",
        #     # "CARE:v2-newnorm-ft-io-biotisr-mito-sr-3",
        #     # "care-v2-newnorm-ft-inout-biotisr-mito-sr-3",
        #     # ------------------------------------------------------------------
        #     # "CARE:v2-newnorm-ft-io-biotisr-factin-nonlinear-sr-1",
        #     # "care-v2-newnorm-ft-inout-biotisr-factin-nonlinear-sr-1",
        #     # "CARE:v2-newnorm-ft-io-biotisr-factin-nonlinear-sr-2",
        #     # "care-v2-newnorm-ft-inout-biotisr-factin-nonlinear-sr-2",
        #     # "CARE:v2-newnorm-ft-io-biotisr-factin-nonlinear-sr-3",
        #     # "care-v2-newnorm-ft-inout-biotisr-factin-nonlinear-sr-3",
        #     # ------------------------------------------------------------------
        #     # "CARE:v2-newnorm-ft-io-biotisr-ccp-sr-1",
        #     # "care-v2-newnorm-ft-inout-biotisr-ccp-sr-1",
        #     # "CARE:v2-newnorm-ft-io-biotisr-ccp-sr-2",
        #     # "care-v2-newnorm-ft-inout-biotisr-ccp-sr-2",
        #     # "CARE:v2-newnorm-ft-io-biotisr-ccp-sr-3",
        #     # "care-v2-newnorm-ft-inout-biotisr-ccp-sr-3",
        #     # ------------------------------------------------------------------
        #     # "CARE:v2-newnorm-ft-io-biotisr-factin-sr-1",
        #     # "care-v2-newnorm-ft-inout-biotisr-factin-sr-1",
        #     # "CARE:v2-newnorm-ft-io-biotisr-factin-sr-2",
        #     # "care-v2-newnorm-ft-inout-biotisr-factin-sr-2",
        #     # "CARE:v2-newnorm-ft-io-biotisr-factin-sr-3",
        #     # "care-v2-newnorm-ft-inout-biotisr-factin-sr-3",
        #     # ------------------------------------------------------------------
        #     # "CARE:v2-newnorm-ft-io-biotisr-lysosome-sr-1",
        #     # "care-v2-newnorm-ft-inout-biotisr-lysosome-sr-1",
        #     # "CARE:v2-newnorm-ft-io-biotisr-lysosome-sr-2",
        #     # "care-v2-newnorm-ft-inout-biotisr-lysosome-sr-2",
        #     "CARE:v2-newnorm-ft-io-biotisr-lysosome-sr-3",
        #     "care-v2-newnorm-ft-inout-biotisr-lysosome-sr-3",
        #     # ------------------------------------------------------------------
        # ),
        # (
        #     # "DFCAN:v2-newnorm-ft-io-cellpose3-2photon-dn-1",
        #     # "dfcan-v2-newnorm-ft-inout-cellpose3-2photon-dn-1",
        #     # "DFCAN:v2-newnorm-ft-io-cellpose3-2photon-dn-4",
        #     # "dfcan-v2-newnorm-ft-inout-cellpose3-2photon-dn-4",
        #     # "DFCAN:v2-newnorm-ft-io-cellpose3-2photon-dn-16",
        #     # "dfcan-v2-newnorm-ft-inout-cellpose3-2photon-dn-16",
        #     # ------------------------------------------------------------------
        #     # "DFCAN:v2-newnorm-ft-io-biotisr-mt-sr-1",
        #     # "dfcan-v2-newnorm-ft-inout-biotisr-mt-sr-1",
        #     # "DFCAN:v2-newnorm-ft-io-biotisr-mt-sr-2",
        #     # "dfcan-v2-newnorm-ft-inout-biotisr-mt-sr-2",
        #     # "DFCAN:v2-newnorm-ft-io-biotisr-mt-sr-3",
        #     # "dfcan-v2-newnorm-ft-inout-biotisr-mt-sr-3",
        #     # ------------------------------------------------------------------
        #     # "DFCAN:v2-newnorm-ft-io-biotisr-mito-sr-1",
        #     # "dfcan-v2-newnorm-ft-inout-biotisr-mito-sr-1",
        #     # "DFCAN:v2-newnorm-ft-io-biotisr-mito-sr-2",
        #     # "dfcan-v2-newnorm-ft-inout-biotisr-mito-sr-2",
        #     # "DFCAN:v2-newnorm-ft-io-biotisr-mito-sr-3",
        #     # "dfcan-v2-newnorm-ft-inout-biotisr-mito-sr-3",
        #     # ------------------------------------------------------------------
        #     # "DFCAN:v2-newnorm-ft-io-biotisr-factin-nonlinear-sr-1",
        #     # "dfcan-v2-newnorm-ft-inout-biotisr-factin-nonlinear-sr-1",
        #     # "DFCAN:v2-newnorm-ft-io-biotisr-factin-nonlinear-sr-2",
        #     # "dfcan-v2-newnorm-ft-inout-biotisr-factin-nonlinear-sr-2",
        #     # "DFCAN:v2-newnorm-ft-io-biotisr-factin-nonlinear-sr-3",
        #     # "dfcan-v2-newnorm-ft-inout-biotisr-factin-nonlinear-sr-3",
        #     # ------------------------------------------------------------------
        #     # "DFCAN:v2-newnorm-ft-io-biotisr-ccp-sr-1",
        #     # "dfcan-v2-newnorm-ft-inout-biotisr-ccp-sr-1",
        #     # "DFCAN:v2-newnorm-ft-io-biotisr-ccp-sr-2",
        #     # "dfcan-v2-newnorm-ft-inout-biotisr-ccp-sr-2",
        #     # "DFCAN:v2-newnorm-ft-io-biotisr-ccp-sr-3",
        #     # "dfcan-v2-newnorm-ft-inout-biotisr-ccp-sr-3",
        #     # ------------------------------------------------------------------
        #     # "DFCAN:v2-newnorm-ft-io-biotisr-factin-sr-1",
        #     # "dfcan-v2-newnorm-ft-inout-biotisr-factin-sr-1",
        #     # "DFCAN:v2-newnorm-ft-io-biotisr-factin-sr-2",
        #     # "dfcan-v2-newnorm-ft-inout-biotisr-factin-sr-2",
        #     # "DFCAN:v2-newnorm-ft-io-biotisr-factin-sr-3",
        #     # "dfcan-v2-newnorm-ft-inout-biotisr-factin-sr-3",
        #     # ------------------------------------------------------------------
        #     # "DFCAN:v2-newnorm-ft-io-biotisr-lysosome-sr-1",
        #     # "dfcan-v2-newnorm-ft-inout-biotisr-lysosome-sr-1",
        #     # "DFCAN:v2-newnorm-ft-io-biotisr-lysosome-sr-2",
        #     # "dfcan-v2-newnorm-ft-inout-biotisr-lysosome-sr-2",
        #     "DFCAN:v2-newnorm-ft-io-biotisr-lysosome-sr-3",
        #     "dfcan-v2-newnorm-ft-inout-biotisr-lysosome-sr-3",
        #     # ------------------------------------------------------------------
        # ),
        # (
        #     # "UniFMIR:v2-newnorm-ft-io-cellpose3-2photon-dn-1",
        #     # "unifmir-v2-newnorm-ft-inout-cellpose3-2photon-dn-1",
        #     # "UniFMIR:v2-newnorm-ft-io-cellpose3-2photon-dn-4",
        #     # "unifmir-v2-newnorm-ft-inout-cellpose3-2photon-dn-4",
        #     # "UniFMIR:v2-newnorm-ft-io-cellpose3-2photon-dn-16",
        #     # "unifmir-v2-newnorm-ft-inout-cellpose3-2photon-dn-16",
        #     # ------------------------------------------------------------------
        #     # "UniFMIR:v2-newnorm-ft-io-biotisr-mt-sr-1",
        #     # "unifmir-v2-newnorm-ft-inout-biotisr-mt-sr-1",
        #     # "UniFMIR:v2-newnorm-ft-io-biotisr-mt-sr-2",
        #     # "unifmir-v2-newnorm-ft-inout-biotisr-mt-sr-2",
        #     # "UniFMIR:v2-newnorm-ft-io-biotisr-mt-sr-3",
        #     # "unifmir-v2-newnorm-ft-inout-biotisr-mt-sr-3",
        #     # ------------------------------------------------------------------
        #     # "UniFMIR:v2-newnorm-ft-io-biotisr-mito-sr-1",
        #     # "unifmir-v2-newnorm-ft-inout-biotisr-mito-sr-1",
        #     # "UniFMIR:v2-newnorm-ft-io-biotisr-mito-sr-2",
        #     # "unifmir-v2-newnorm-ft-inout-biotisr-mito-sr-2",
        #     # "UniFMIR:v2-newnorm-ft-io-biotisr-mito-sr-3",
        #     # "unifmir-v2-newnorm-ft-inout-biotisr-mito-sr-3",
        #     # ------------------------------------------------------------------
        #     # "UniFMIR:v2-newnorm-ft-io-biotisr-factin-nonlinear-sr-1",
        #     # "unifmir-v2-newnorm-ft-inout-biotisr-factin-nonlinear-sr-1",
        #     # "UniFMIR:v2-newnorm-ft-io-biotisr-factin-nonlinear-sr-2",
        #     # "unifmir-v2-newnorm-ft-inout-biotisr-factin-nonlinear-sr-2",
        #     # "UniFMIR:v2-newnorm-ft-io-biotisr-factin-nonlinear-sr-3",
        #     # "unifmir-v2-newnorm-ft-inout-biotisr-factin-nonlinear-sr-3",
        #     # ------------------------------------------------------------------
        #     # "UniFMIR:v2-newnorm-ft-io-biotisr-ccp-sr-1",
        #     # "unifmir-v2-newnorm-ft-inout-biotisr-ccp-sr-1",
        #     # "UniFMIR:v2-newnorm-ft-io-biotisr-ccp-sr-2",
        #     # "unifmir-v2-newnorm-ft-inout-biotisr-ccp-sr-2",
        #     # "UniFMIR:v2-newnorm-ft-io-biotisr-ccp-sr-3",
        #     # "unifmir-v2-newnorm-ft-inout-biotisr-ccp-sr-3",
        #     # ------------------------------------------------------------------
        #     # "UniFMIR:v2-newnorm-ft-io-biotisr-factin-sr-1",
        #     # "unifmir-v2-newnorm-ft-inout-biotisr-factin-sr-1",
        #     # "UniFMIR:v2-newnorm-ft-io-biotisr-factin-sr-2",
        #     # "unifmir-v2-newnorm-ft-inout-biotisr-factin-sr-2",
        #     # "UniFMIR:v2-newnorm-ft-io-biotisr-factin-sr-3",
        #     # "unifmir-v2-newnorm-ft-inout-biotisr-factin-sr-3",
        #     # ------------------------------------------------------------------
        #     # "UniFMIR:v2-newnorm-ft-io-biotisr-lysosome-sr-1",
        #     # "unifmir-v2-newnorm-ft-inout-biotisr-lysosome-sr-1",
        #     # "UniFMIR:v2-newnorm-ft-io-biotisr-lysosome-sr-2",
        #     # "unifmir-v2-newnorm-ft-inout-biotisr-lysosome-sr-2",
        #     "UniFMIR:v2-newnorm-ft-io-biotisr-lysosome-sr-3",
        #     "unifmir-v2-newnorm-ft-inout-biotisr-lysosome-sr-3",
        #     # ------------------------------------------------------------------
        # ),
    ),
    "percentiles": (0.03, 0.995),
}

normalizer = utils_data.NormalizePercentile(
    p_low=params["percentiles"][0], p_high=params["percentiles"][1]
)

# ------------------------------------------------------------------------------
params["path_results"] = utils_data.win2linux(params["path_results"])
titles = [i[0] for i in params["methods"]]  # table title
methods = [i[1] for i in params["methods"]]

# ------------------------------------------------------------------------------
data_frame = pandas.read_excel(params["path_dataset_test"])
datasets_not_processed = []  # the datasets that are not processed by the script.
dict_clip = {"a_min": 0.0, "a_max": 2.5}
data_range = dict_clip["a_max"] - dict_clip["a_min"]


def bkg_subtraction(image):
    radius, sf = 25, 16
    image_rb, bg = rolling_ball_approximation(image, radius=radius, sf=sf)
    image_rb = np.clip(image_rb, 0, None)
    return image_rb


for id_dataset in params["dataset_names"]:
    print("-" * 80)
    print("Datasset:", id_dataset)

    if id_dataset in datasets_need_bkg_sub:
        sub_bkg = True
        print("Background subtraction is applied.")
    else:
        sub_bkg = False

    # get the information of current dataset
    ds = data_frame[data_frame["id"] == id_dataset].iloc[0]
    path_results = os.path.join(params["path_results"], ds["id"])
    path_metrics_file = os.path.join(path_results, f"metrics-v2.xlsx")

    # --------------------------------------------------------------------------
    # collect all the datasets that are not processed successfully.
    if not os.path.exists(path_results):
        datasets_not_processed.append(id_dataset)
        continue

    # check if the dataset has ground truth
    if ds["path_hr"] == "Unknown":
        print("The ground truth is inexistent.")
        continue

    # read filenames of test images
    sample_filenames = utils_data.read_txt(ds["path_index"])

    # set the number of samples used for test
    if params["num_sample"] is not None:
        if params["num_sample"] > len(sample_filenames):
            num_sample_analysis = len(sample_filenames)
        else:
            num_sample_analysis = params["num_sample"]
    else:
        num_sample_analysis = len(sample_filenames)
    print(f"Number of test samples: {num_sample_analysis}/{len(sample_filenames)}")

    # --------------------------------------------------------------------------
    # load results
    # --------------------------------------------------------------------------
    try:
        df_psnr = pandas.read_excel(path_metrics_file, sheet_name="PSNR", index_col=0)
    except:
        df_psnr = pandas.DataFrame()

    try:
        df_ssim = pandas.read_excel(path_metrics_file, sheet_name="SSIM", index_col=0)
    except:
        df_ssim = pandas.DataFrame()

    try:
        df_zncc = pandas.read_excel(path_metrics_file, sheet_name="ZNCC", index_col=0)
    except:
        df_zncc = pandas.DataFrame()

    try:
        df_nrmse = pandas.read_excel(path_metrics_file, sheet_name="NRMSE", index_col=0)
    except:
        df_nrmse = pandas.DataFrame()

    try:
        df_msssim = pandas.read_excel(
            path_metrics_file, sheet_name="MSSSIM", index_col=0
        )
    except:
        df_msssim = pandas.DataFrame()

    # --------------------------------------------------------------------------
    writer = pandas.ExcelWriter(path_metrics_file, engine="xlsxwriter")
    try:
        metrics_dataset = []
        for i_sample in range(num_sample_analysis):
            sample_name = sample_filenames[i_sample]
            img_est_multi_meth = []  # collect estimated images

            print(f"load results of {sample_name} ...")
            img_gt = utils_data.read_image(os.path.join(ds["path_hr"], sample_name))
            img_gt = utils_data.interp_sf(img_gt, sf=ds["sf_hr"])[0]
            img_gt = normalizer(img_gt)
            img_gt = np.clip(img_gt, **dict_clip)

            # ------------------------------------------------------------------
            # get raw image and apply normalization
            img_raw = utils_data.read_image(os.path.join(ds["path_lr"], sample_name))
            img_raw = utils_data.utils_data.interp_sf(img_raw, sf=ds["sf_lr"])[0]
            # img_raw = eva.linear_transform(img_true=img_gt, img_test=img_raw)
            img_raw = normalizer(img_raw)
            img_raw = np.clip(img_raw, **dict_clip)
            img_est_multi_meth.append(img_raw)

            # ------------------------------------------------------------------
            # get estimated images form different methods
            for meth in methods:
                img_tmp = utils_data.read_image(
                    os.path.join(path_results, meth, sample_name)
                )

                if sub_bkg:
                    img_tmp = bkg_subtraction(img_tmp)

                # img_tmp = eva.linear_transform(img_true=img_gt, img_test=img_tmp)[0]
                img_tmp = normalizer(img_tmp[0])
                img_tmp = np.clip(img_tmp, **dict_clip)
                img_est_multi_meth.append(img_tmp)

            # ------------------------------------------------------------------
            # evaluate all the images
            metrics_method = []
            for img in img_est_multi_meth:
                metrics_metric = []
                dict_tmp = {"img_true": img_gt, "img_test": img}
                psnr = eva.PSNR(data_range=data_range, **dict_tmp)
                ssim = eva.SSIM(data_range=data_range, **dict_tmp)
                zncc = eva.ZNCC(**dict_tmp)
                nrmse = eva.NRMSE(**dict_tmp)
                try:
                    msssim = eva.MSSSIM(data_range=data_range, **dict_tmp)
                except:
                    try:
                        msssim = eva.MSSSIM(
                            img_true=interp_sf(x=img_gt[None], sf=2)[0],
                            img_test=interp_sf(x=img[None], sf=2)[0],
                            data_range=data_range,
                        )
                    except Exception as e:
                        traceback.print_exc()
                        msssim = 0
                metrics_metric.extend([psnr, ssim, zncc, nrmse, msssim])
                metrics_method.append(metrics_metric)  # (num_meth, num_metric)
            metrics_dataset.append(metrics_method)  # (num_sample, num_meth, num_metric)

        # (num_sample, num_meth, num_metric)
        metrics_dataset = np.array(metrics_dataset)

        # --------------------------------------------------------------------------
        all_titles = ["raw"] + titles
        for i_df, df in enumerate([df_psnr, df_ssim, df_zncc, df_nrmse, df_msssim]):
            for i_title, title in enumerate(all_titles):
                # if current method is not in the dataframe, add it at the last column
                if title not in list(df.columns):
                    df.insert(df.shape[-1], title, value="")
                # insert the metrics of current method into the dataframe
                df[title] = metrics_dataset[:, i_title, i_df]

        # --------------------------------------------------------------------------
        # save all the metrics value into a excel file
        df_psnr.to_excel(writer, sheet_name="PSNR", index_label="id")
        df_ssim.to_excel(writer, sheet_name="SSIM", index_label="id")
        df_zncc.to_excel(writer, sheet_name="ZNCC", index_label="id")
        df_nrmse.to_excel(writer, sheet_name="NRMSE", index_label="id")
        df_msssim.to_excel(writer, sheet_name="MSSSIM", index_label="id")
        writer.close()
    except Exception as e:
        df_psnr.to_excel(writer, sheet_name="PSNR", index_label="id")
        df_ssim.to_excel(writer, sheet_name="SSIM", index_label="id")
        df_zncc.to_excel(writer, sheet_name="ZNCC", index_label="id")
        df_nrmse.to_excel(writer, sheet_name="NRMSE", index_label="id")
        df_msssim.to_excel(writer, sheet_name="MSSSIM", index_label="id")
        writer.close()
        traceback.print_exc()

print("-" * 80)
if datasets_not_processed:
    print("The following datasets are not processed:")
    for dataset in datasets_not_processed:
        print(dataset)
