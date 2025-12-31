from utils.data import normalization, win2linux, grayto255
from skimage import io
import numpy as np
from models.biomedclip_embedder import BiomedCLIP
import torch, os


def classcification_image_retrieval(params_in: dict):
    """
    Structure type classification based on image retrieval.
    ### Parameters
    - `path_image`: str, the path to the image.
    - `path_database`: str, the path to the database.
    - `num_patches`: int, the number of patches to extract from the image.
    - `top_k`: int, the number of top k patches to retrieve.

    """
    # load parameters ----------------------------------------------------------
    params = {
        "path_image": "",
        "path_database": "",
        "path_embedder": "",
        "num_patches": 10,
        "top_k": 10,
    }

    # check and update the parameters
    for key, value in params.items():
        if key not in params_in:
            print(f"[ERROR] Parameter {key} is missing.")
            return 0
        else:
            params[key] = params_in[key]

    path_image = params["path_image"]
    path_database = params["path_database"]
    num_patches = params["num_patches"]
    top_k = params["top_k"]
    path_embedder = params["path_embedder"]

    path_embedder_json = os.path.join(path_embedder, "open_clip_config.json")
    path_embedder_bin = os.path.join(path_embedder, "open_clip_pytorch_model.bin")
    # --------------------------------------------------------------------------

    # crop 224x224 patch from the image
    patch_size = 224
    step = 112
    normalizer = lambda x: normalization(x, p_low=0.03, p_high=0.995)

    # read image
    img_raw = io.imread(path_image)
    img_raw = np.squeeze(img_raw)
    if len(img_raw.shape) == 2:
        img = img_raw
    if len(img_raw.shape) == 3:
        # use the center slice
        nslice = img_raw.shape[0]
        img = img_raw[nslice // 2]

    # preprocess image
    img = normalizer(img)
    img = np.clip(img, 0, 2.0)
    img = grayto255(img)

    # skip small images
    if img.shape[-2] < patch_size or img.shape[-1] < patch_size:
        print(
            "[ERROR] Image is too small for recognize the structure type (larger than 224x224 is required)."
        )
        return 0
    centers = []
    # grid centers
    for cy in range(0, img.shape[-2] - patch_size, step):
        for cx in range(0, img.shape[-1] - patch_size, step):
            centers.append((cy + patch_size // 2, cx + patch_size // 2))
    # random centers
    for _ in range(len(centers) // 2):
        cy = np.random.randint(0, img.shape[-2] - patch_size)
        cx = np.random.randint(0, img.shape[-1] - patch_size)
        centers.append((cy + patch_size // 2, cx + patch_size // 2))

    patches = []
    for center in centers:
        patch = img[
            center[0] - patch_size // 2 : center[0] + patch_size // 2,
            center[1] - patch_size // 2 : center[1] + patch_size // 2,
        ]
        patches.append(patch)
    patches = np.array(patches)

    # calculate the intensity std of each patch, exclude the flat patch
    avg_intensity = np.std(patches, axis=(1, 2))
    # sort for large to small
    idx = np.argsort(avg_intensity)[::-1]
    # get the top num_patches_per_image patches
    idx_select = idx[:num_patches]

    # save the patches to a folder
    path_save_to = os.path.join(path_image + "_patches")

    # embedding the patches ----------------------------------------------------
    biomedcliper = BiomedCLIP(
        path_json=path_embedder_json,
        path_bin=path_embedder_bin,
        context_length=160,
        device=torch.device("cuda:0"),
    )
