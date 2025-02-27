import numpy as np
import skimage.io as io
from math import ceil
import torch, json, os, pydicom
import methods.convolution as conv
import utils.evaluation as eva
import utils.data as utils_data
from torch.utils.data import Dataset, DataLoader
import random
import sys

sys.path.insert(1, "E:\qiqilu\Project\\2024 Foundation model\code")
from models.PSFmodels import BWModel


def print_dict(dict):
    print("-" * 90)
    print(json.dumps(dict, indent=2))
    print("-" * 90)


def ave_pooling(x, scale_factor=1):
    """Average pooling for 2D/3D image."""
    x = torch.tensor(x, dtype=torch.float32)
    if len(x.shape) == 2:
        x = torch.nn.functional.avg_pool2d(x[None, None], kernel_size=scale_factor)
    if len(x.shape) == 3:
        x = torch.nn.functional.avg_pool3d(x[None, None], kernel_size=scale_factor)
    return x.numpy()[0, 0]


def add_noise(x, poisson=0, sigma_gauss=0, scale_factor=1):
    """Add Poisson and Gaussian noise."""
    x = np.maximum(x, 0.0)

    # add poisson noise
    x_poi = np.random.poisson(lam=x) if poisson == 1 else x

    # downsampling
    if scale_factor > 1:
        x_poi = ave_pooling(x_poi, scale_factor=scale_factor)

    # add gaussian noise
    if sigma_gauss > 0:
        max_signal = np.max(x_poi)
        x_poi_norm = x_poi / max_signal
        x_poi_gaus = x_poi_norm + np.random.normal(
            loc=0, scale=sigma_gauss / max_signal, size=x_poi_norm.shape
        )
        x_n = x_poi_gaus * max_signal
    else:
        x_n = x_poi

    return x_n.astype(np.float32)


def center_crop(x, size=None, verbose=False):
    """Crop the center region of the 3D image x."""
    if size is not None:
        dim = len(x.shape)
        if dim == 3:  # (Nz, Ny, Nx)
            Nz, Ny, Nx = x.shape
            out = x[
                Nz // 2 - size[0] // 2 : Nz // 2 + size[0] // 2 + 1,
                Ny // 2 - size[1] // 2 : Ny // 2 + size[1] // 2 + 1,
                Nx // 2 - size[2] // 2 : Nx // 2 + size[2] // 2 + 1,
            ]

        if dim == 2:  # (Ny, Nx)
            Ny, Nx = x.shape
            out = x[
                Ny // 2 - size[1] // 2 : Ny // 2 + size[1] // 2 + 1,
                Nx // 2 - size[2] // 2 : Nx // 2 + size[2] // 2 + 1,
            ]

        if dim == 5:  # (Nb, Nc, Nz, Ny, Nx)
            _, _, Nz, Ny, Nx = x.shape
            out = x[
                :,
                :,
                Nz // 2 - size[0] // 2 : Nz // 2 + size[0] // 2 + 1,
                Ny // 2 - size[1] // 2 : Ny // 2 + size[1] // 2 + 1,
                Nx // 2 - size[2] // 2 : Nx // 2 + size[2] // 2 + 1,
            ]

        if dim == 4:  # (Nb, Nc, Ny, Nx)
            _, _, Ny, Nx = x.shape
            out = x[
                :,
                :,
                Ny // 2 - size[1] // 2 : Ny // 2 + size[1] // 2 + 1,
                Nx // 2 - size[2] // 2 : Nx // 2 + size[2] // 2 + 1,
            ]

        if verbose:
            print(f"crop from {x.shape} to a shape of {out.shape}")
    else:
        out = x
    return out


def even2odd_shape(x, verbose=False):
    """Convert the image x to a odd-shape image."""
    dim = len(x.shape)
    assert dim in [2, 3], "Only 2D or 3D image are supported."
    if dim == 3:
        i, j, k = x.shape
        if i % 2 == 0:
            i = i - 1
        if j % 2 == 0:
            j = j - 1
        if k % 2 == 0:
            k = k - 1

        shape_to = (i, j, k)

        x_inter = torch.nn.functional.interpolate(
            torch.tensor(x)[None, None], size=shape_to, mode="trilinear"
        )
    if dim == 2:
        i, j = x.shape
        if i % 2 == 0:
            i = i - 1
        if j % 2 == 0:
            j = j - 1

        shape_to = (i, j)
        x_inter = torch.nn.functional.interpolate(
            torch.tensor(x)[None, None], size=shape_to, mode="bilinear"
        )

    if verbose:
        print(f"convert psf shape from {x.shape} to {shape_to}.")
    x_inter = x_inter / x_inter.sum()
    return x_inter.numpy()[0, 0]


def generate_digital_phantom(
    path_output,
    shape=(128, 128, 128),
    num_simulation=1,
    is_with_background=False,
    *args,
    **kwargs,
):
    """
    Generate digital phantom consisting of three types of structure:
    dots, solid spher and ellipsoidal surfaces.
    """
    delta = 0.7  # parameter related to std of gaussian filter, not std
    Rsphere = 9  # diameter of solid spheres
    Ldot = 9
    inten_bkg = 30

    print(
        "- Output Path: {}\n- data shape: {}\n- num of simulation: {}\n- backgroun signal (1/0): {}".format(
            path_output, shape, num_simulation, is_with_background
        )
    )

    if shape[0] == 1:
        data_dim = 2
        more_obj = (shape[1] / 128) * (shape[2] / 128)

        n_spheres = 5 * more_obj
        n_ellipsoidal = 5 * more_obj
        n_dots = 10 * more_obj

    elif shape[0] > 1:
        data_dim = 3
        more_obj = (shape[0] / 128) * (shape[1] / 128) * (shape[2] / 128)

        n_spheres = 200 * more_obj
        n_ellipsoidal = 200 * more_obj
        n_dots = 50 * more_obj

    # set the min number of object
    n_spheres, n_ellipsoidal, n_dots = (
        np.maximum(ceil(n_spheres), 10),
        np.maximum(ceil(n_ellipsoidal), 10),
        np.maximum(ceil(n_dots), 10),
    )

    # create Gaussian filter
    Ggrid = range(-2, 2 + 1)
    if data_dim == 3:
        [Z, Y, X] = np.meshgrid(Ggrid, Ggrid, Ggrid)
        GaussM = np.exp(-(X**2 + Y**2 + Z**2)) / (2 * delta**2)

    if data_dim == 2:
        [Y, X] = np.meshgrid(Ggrid, Ggrid)
        GaussM = np.exp(-(X**2 + Y**2)) / (2 * delta**2)

    # normalize so thant total area (sum of all weights) is 1
    GaussM = GaussM / np.sum(GaussM)

    if not os.path.exists(os.path.join(path_output, "gt", "images")):
        os.makedirs(os.path.join(path_output, "gt", "images"), exist_ok=True)
    file_list_txt = open(os.path.join(path_output, "gt", "list.txt"), "w")

    # spheroid
    for tt in range(num_simulation):
        print(f"simulation {tt}")

        if data_dim == 3:
            A = np.zeros(shape=shape)
            Nz, Ny, Nx = A.shape

            rrange = np.fix(Rsphere / 2)
            xrange, yrange, zrange = (
                Nx - 2 * Rsphere,
                Ny - 2 * Rsphere,
                Nz - 2 * Rsphere,
            )  # avoid out of image range

            # --------------------------------------------------------------
            for _ in range(n_spheres):
                x = np.floor(xrange * np.random.rand() + Rsphere)
                y = np.floor(yrange * np.random.rand() + Rsphere)
                z = np.floor(zrange * np.random.rand() + Rsphere)

                r = np.floor(rrange * np.random.rand() + rrange)
                inten = 800 * np.random.rand() + 50  # intensity range (50, 850)

                x, y, z, r = int(x), int(y), int(z), int(r)
                for i in range(x - r, x + r + 1):
                    for j in range(y - r, y + r + 1):
                        for k in range(z - r, z + r + 1):
                            if ((i - x) ** 2 + (j - y) ** 2 + (k - z) ** 2) < r**2 and (
                                0 <= i < Nx and 0 <= j < Ny and 0 <= k < Nz
                            ):
                                A[k, j, i] = inten

            # --------------------------------------------------------------
            for _ in range(n_ellipsoidal):
                x = np.floor(xrange * np.random.rand() + Rsphere)
                y = np.floor(yrange * np.random.rand() + Rsphere)
                z = np.floor(zrange * np.random.rand() + Rsphere)

                r1 = np.floor(rrange * np.random.rand() + rrange)
                r2 = np.floor(rrange * np.random.rand() + rrange)
                r3 = np.floor(rrange * np.random.rand() + rrange)

                x, y, z, r1, r2, r3 = (
                    int(x),
                    int(y),
                    int(z),
                    int(r1),
                    int(r2),
                    int(r3),
                )

                inten = 800 * np.random.rand() + 50  # intensity range (50, 850)

                for i in range(x - r1, x + r1 + 1):
                    for j in range(y - r2, y + r2 + 1):
                        for k in range(z - r3, z + r3 + 1):
                            if (
                                (
                                    ((i - x) ** 2) / r1**2
                                    + ((j - y) ** 2) / r2**2
                                    + ((k - z) ** 2) / r3**2
                                )
                                <= 1.3
                                and (
                                    ((i - x) ** 2) / r1**2
                                    + ((j - y) ** 2) / r2**2
                                    + ((k - z) ** 2) / r3**2
                                )
                                >= 0.8
                                and (0 <= i < Nx and 0 <= j < Ny and 0 <= k < Nz)
                            ):
                                A[k, j, i] = inten

            # --------------------------------------------------------------
            dotrangex = Nx - Ldot - 1
            dotrangey = Ny - Ldot - 1
            dotrangez = Nz - Ldot - 1

            for _ in range(n_dots):
                x = np.floor((Nx - 3) * np.random.rand() + 1)
                y = np.floor((Ny - 3) * np.random.rand() + 1)
                z = np.floor((Nz - 3) * np.random.rand() + 1)

                x, y, z = int(x), int(y), int(z)

                r = 1
                inten = 800 * np.random.rand() + 50  # intensity range (50, 850)

                A[z : z + 2, y : y + 2, x : x + 2] = inten

            for _ in range(n_dots):
                x = np.floor(dotrangex * np.random.rand() + 1)
                y = np.floor((Ny - 3) * np.random.rand() + 1)
                z = np.floor((Nz - 3) * np.random.rand() + 1)

                r = 1

                inten = 800 * np.random.rand() + 50
                k = np.floor(np.random.rand() * Ldot) + 1

                x, y, z, k = int(x), int(y), int(z), int(k)

                A[z : z + 2, y : y + 2, x : x + k + 1] = inten

            for _ in range(n_dots):
                x = np.floor((Nx - 3) * np.random.rand() + 1)
                y = np.floor(dotrangey * np.random.rand() + 1)
                z = np.floor((Nz - 3) * np.random.rand() + 1)

                r = 1
                inten = 800 * np.random.rand() + 50
                k = np.floor(np.random.rand() * 9) + 1

                x, y, z, k = int(x), int(y), int(z), int(k)

                A[z : z + 2, y : y + k + 1, x : x + 2] = inten + 50 * np.random.rand()

            for _ in range(n_dots):
                x = np.floor((Nx - 3) * np.random.rand() + 1)
                y = np.floor((Ny - 3) * np.random.rand() + 1)
                z = np.floor(dotrangez * np.random.rand() + 1)

                r = 1
                inten = 800 * np.random.rand() + 50
                k = np.floor(np.random.rand() * Ldot) + 1

                x, y, z, k = int(x), int(y), int(z), int(k)
                A[z : z + k + 1, y : y + 2, x : x + 2] = inten

            for _ in range(n_dots):
                x = np.floor(dotrangex * np.random.rand() + 1)
                y = np.floor((Ny - 3) * np.random.rand() + 1)
                z = np.floor(dotrangez * np.random.rand() + 1)

                r = 1
                inten = 800 * np.random.rand() + 50
                k1 = np.floor(np.random.rand() * Ldot) + 1
                k2 = np.floor(np.random.rand() * Ldot) + 1

                x, y, z, k1, k2 = int(x), int(y), int(z), int(k1), int(k2)

                A[z : z + k2 + 1, y : y + 2, x : x + k1 + 1] = inten

            for _ in range(n_dots):
                x = np.floor(dotrangex * np.random.rand() + 1)
                y = np.floor(dotrangey * np.random.rand() + 1)
                z = np.floor((Nz - 3) * np.random.rand() + 1)

                r = 1
                inten = 800 * np.random.rand() + 50
                k1 = np.floor(np.random.rand() * Ldot) + 1
                k2 = np.floor(np.random.rand() * Ldot) + 1
                x, y, z, k1, k2 = int(x), int(y), int(z), int(k1), int(k2)

                A[z : z + 2, y : y + k2 + 1, x : x + k1 + 1] = inten

            for _ in range(n_dots):
                x = np.floor((Nx - 3) * np.random.rand() + 1)
                y = np.floor(dotrangey * np.random.rand() + 1)
                z = np.floor(dotrangez * np.random.rand() + 1)

                r = 1
                inten = 800 * np.random.rand() + 50
                k1 = np.floor(np.random.rand() * Ldot) + 1
                k2 = np.floor(np.random.rand() * Ldot) + 1

                x, y, z, k1, k2 = int(x), int(y), int(z), int(k1), int(k2)
                A[z : z + k2 + 1, y : y + k1 + 1, x : x + 2] = inten

            for _ in range(n_dots):
                x = np.floor(dotrangex * np.random.rand() + 1)
                y = np.floor(dotrangey * np.random.rand() + 1)
                z = np.floor(dotrangez * np.random.rand() + 1)

                r = 1
                inten = 800 * np.random.rand() + 50
                k1 = np.floor(np.random.rand() * Ldot) + 1
                k2 = np.floor(np.random.rand() * Ldot) + 1
                k3 = np.floor(np.random.rand() * Ldot) + 1

                x, y, z, k1, k2, k3 = (
                    int(x),
                    int(y),
                    int(z),
                    int(k1),
                    int(k2),
                    int(k3),
                )

                A[z : z + k3 + 1, y : y + k2 + 1, x : x + k1 + 1] = inten

            if is_with_background:
                A = A + inten_bkg

            A_torch = torch.Tensor(A)[None, None]
            GaussM_torch = torch.Tensor(GaussM)[None, None]

            A_conv = torch.nn.functional.conv3d(
                input=A_torch, weight=GaussM_torch, padding="same"
            )

        if data_dim == 2:
            A = np.zeros(shape=(shape[1], shape[2]))
            Ny, Nx = A.shape

            rrange = np.fix(Rsphere / 2)
            xrange, yrange = (
                Nx - 2 * Rsphere,
                Ny - 2 * Rsphere,
            )  # avoid out of image range

            # --------------------------------------------------------------
            for _ in range(n_spheres):
                x = np.floor(xrange * np.random.rand() + Rsphere)
                y = np.floor(yrange * np.random.rand() + Rsphere)

                r = np.floor(rrange * np.random.rand() + rrange)
                inten = 800 * np.random.rand() + 50

                x, y, r = int(x), int(y), int(r)

                for i in range(x - r, x + r + 1):
                    for j in range(y - r, y + r + 1):
                        if ((i - x) ** 2 + (j - y) ** 2) < r**2:
                            A[j, i] = inten

            # --------------------------------------------------------------
            for _ in range(n_ellipsoidal):
                x = np.floor(xrange * np.random.rand() + Rsphere)
                y = np.floor(yrange * np.random.rand() + Rsphere)

                r1 = np.floor(rrange * np.random.rand() + rrange)
                r2 = np.floor(rrange * np.random.rand() + rrange)

                inten = 800 * np.random.rand() + 50

                x, y, r1, r2 = int(x), int(y), int(r1), int(r2)

                for i in range(x - r1, x + r1 + 1):
                    for j in range(y - r2, y + r2 + 1):
                        if (
                            ((i - x) ** 2) / r1**2 + ((j - y) ** 2) / r2**2
                        ) <= 1.3 and (
                            ((i - x) ** 2) / r1**2 + ((j - y) ** 2) / r2**2
                        ) >= 0.8:
                            A[j, i] = inten

            # --------------------------------------------------------------
            dotrangex = Nx - Ldot - 1
            dotrangey = Ny - Ldot - 1

            for _ in range(n_dots):
                x = np.floor((Nx - 3) * np.random.rand() + 1)
                y = np.floor((Ny - 3) * np.random.rand() + 1)

                r = 1
                inten = 800 * np.random.rand() + 50

                x, y = int(x), int(y)
                A[y : y + 2, x : x + 2] = inten

            for _ in range(n_dots):
                x = np.floor(dotrangex * np.random.rand() + 1)
                y = np.floor((Ny - 3) * np.random.rand() + 1)

                r = 1
                inten = 800 * np.random.rand() + 50
                k = np.floor(np.random.rand() * Ldot) + 1

                x, y, k = int(x), int(y), int(k)
                A[y : y + 2, x : x + k + 1] = inten

            for _ in range(n_dots):
                x = np.floor((Nx - 3) * np.random.rand() + 1)
                y = np.floor(dotrangey * np.random.rand() + 1)

                r = 1
                inten = 800 * np.random.rand() + 50
                k = np.floor(np.random.rand() * 9) + 1
                x, y, k = int(x), int(y), int(k)

                A[y : y + k + 1, x : x + 2] = inten + 50 * np.random.rand()

            for _ in range(n_dots):
                x = np.floor(dotrangex * np.random.rand() + 1)
                y = np.floor(dotrangey * np.random.rand() + 1)

                r = 1
                inten = 800 * np.random.rand() + 50
                k1 = np.floor(np.random.rand() * Ldot) + 1
                k2 = np.floor(np.random.rand() * Ldot) + 1
                x, y, k1, k2 = int(x), int(y), int(k1), int(k2)
                A[y : y + k2 + 1, x : x + k1 + 1] = inten

            if is_with_background:
                A = A + inten_bkg

            A_torch = torch.Tensor(A)[None, None]
            GaussM_torch = torch.Tensor(GaussM)[None, None]

            A_conv = torch.nn.functional.conv2d(
                input=A_torch, weight=GaussM_torch, padding="same"
            )

        # ------------------------------------------------------------------
        A_conv = A_conv.cpu().detach().numpy()[0, 0]
        A_conv = np.array(A_conv, dtype=np.float32)

        io.imsave(
            fname=os.path.join(path_output, "gt", "images", f"{tt}.tif"),
            arr=A_conv,
            check_contrast=False,
        )
        file_list_txt.write(f"{tt}.tif\n")
    file_list_txt.close()


def generate_synthetic_data(
    path_output,
    path_phantom,
    path_psf,
    psf_crop_shape=None,
    std_gauss=0,
    poisson=1,
    ratio=1,
    scale_factor=1,
    conv_padding_mode="reflect",
    suffix="",
    *args,
    **kwargs,
):

    # load psf
    print(f"load psf from: {path_psf}")
    psf = io.imread(path_psf).astype(np.float32)

    # --------------------------------------------------------------------------
    # load digital phantom
    phantom_list = utils_data.read_txt(os.path.join(path_phantom, "list.txt"))
    print(phantom_list)

    img_gt_single = io.imread(os.path.join(path_phantom, "images", phantom_list[0]))
    img_gt_single = img_gt_single.astype(np.float32)
    img_gt_shape = img_gt_single.shape
    print(f"GT shape: {img_gt_shape}")

    # --------------------------------------------------------------------------
    data_name = "data_{}_{}_{}_gauss_{}_poiss_{}_ratio_{}_{}".format(
        img_gt_shape[0],
        img_gt_shape[1],
        img_gt_shape[2],
        std_gauss,
        poisson,
        ratio,
        suffix,
    )

    path_raw = os.path.join(path_output, "raw", data_name)
    utils_data.make_path(os.path.join(path_raw, "images"))
    print(f"save to: {path_raw}")

    # --------------------------------------------------------------------------
    # generate synthetic data (blur and noise)
    # interpolate psf with even shape to odd shape
    psf_odd = even2odd_shape(psf)
    psf_crop = center_crop(psf_odd, size=psf_crop_shape)  # crop psf
    psf_crop = psf_crop / psf_crop.sum()

    # save cropped psf
    io.imsave(
        os.path.join(path_raw, "psf.tif"),
        arr=psf_crop,
        check_contrast=False,
    )

    # --------------------------------------------------------------------------
    for name in phantom_list:
        img_gt = io.imread(os.path.join(path_phantom, "images", name))

        # scale to control SNR
        img_gt = img_gt.astype(np.float32) * ratio

        # blur
        img_blur = conv.convolution(
            img_gt, psf_crop, padding_mode=conv_padding_mode, domain="fft"
        )

        # add noise
        img_blur_n = add_noise(
            img_blur,
            poisson=poisson,
            sigma_gauss=std_gauss,
            scale_factor=scale_factor,
        )

        # SNR
        print(f"{name}, SNR: {eva.SNR(img_blur, img_blur_n)}")
        io.imsave(
            os.path.join(path_raw, "images", name),
            arr=img_blur_n,
            check_contrast=False,
        )

    # save parameters
    params_dict = {
        "path_phantom": path_phantom,
        "path_psf": path_psf,
        "img_gt_shape": img_gt_shape,
        "psf_crop_shape": psf_crop_shape,
        "std_gauss": std_gauss,
        "poisson": poisson,
        "ratio": ratio,
        "scale_factor": scale_factor,
        "conv_padding_mode": conv_padding_mode,
    }

    with open(os.path.join(path_raw, "parameters.json"), "w") as f:
        f.write(json.dumps(params_dict, indent=1))


def generate_synthetic_data_multiPSF(
    path_output,
    path_phantom,
    path_psf=None,
    psf_size=(127, 127, 127),
    wvl_range=(300, 800),
    aa_range=(1.7, 71.8),
    reflective_index=[1.0, 1.3, 1.47, 1.51],
    psf_crop_shape=None,
    std_gauss=0,
    poisson=1,
    ratio=1,
    scale_factor=1,
    conv_padding_mode="constant",
    suffix="",
    verbese=False,
):
    # --------------------------------------------------------------------------
    # load digital phantom
    phantom_list = utils_data.read_txt(os.path.join(path_phantom, "list.txt"))

    img_gt_single = io.imread(os.path.join(path_phantom, "images", phantom_list[0]))
    img_gt_shape = img_gt_single.shape
    print(f"GT shape: {img_gt_shape}")

    # --------------------------------------------------------------------------
    # make save path
    data_name = "data_{}_{}_{}_gauss_{}_poiss_{}_ratio_{}{}".format(
        img_gt_shape[0],
        img_gt_shape[1],
        img_gt_shape[2],
        std_gauss,
        poisson,
        ratio,
        suffix,
    )
    path_raw = os.path.join(path_output, "raw", data_name)
    print(f"save to: {path_raw}")

    utils_data.make_path(os.path.join(path_raw, "images"))
    utils_data.make_path(os.path.join(path_raw, "psf"))

    psf_list_txt = open(os.path.join(path_raw, "psf", "list.txt"), "w")

    if path_psf is None:
        gen = BWModel(
            kernel_size=psf_size,
            kernel_norm=True,
            num_integral=100,
            over_sampling=2,
            pixel_size_z=1,
        )

    for name in phantom_list:
        print(name)
        # get a PSF randomly
        if path_psf is not None:
            path_psf_single = random.choice(path_psf)
            psf_list_txt.write(f"{path_psf_single}\n")
            psf = io.imread(path_psf_single).astype(np.float32)
        else:
            n_immersion = random.choice(reflective_index)
            aa = np.random.uniform(low=aa_range[0], high=aa_range[1])
            NA = n_immersion * np.sin(aa * np.pi / 180.0)
            wvl_vaccum = np.random.uniform(low=wvl_range[0], high=wvl_range[1])

            wvl = wvl_vaccum / 100 / n_immersion
            n = NA / n_immersion

            psf = (
                gen(torch.tensor([wvl, n])[None, None]).numpy()[0, 0].astype(np.float32)
            )
            psf_list_txt.write(f"{wvl_vaccum} {NA} {n_immersion}\n")
            print(f"{wvl_vaccum} {NA} {n_immersion}")

        # interpolate psf with even shape to odd shape
        psf_odd = even2odd_shape(psf, verbose=verbese)
        psf_crop = center_crop(
            psf_odd, size=psf_crop_shape, verbose=verbese
        )  # crop psf
        psf_crop = psf_crop / psf_crop.sum()  # normalization

        # save cropped psf
        io.imsave(
            os.path.join(path_raw, "psf", name), arr=psf_crop, check_contrast=False
        )

        # ----------------------------------------------------------------------
        # read ground truth phantom
        img_gt = io.imread(os.path.join(path_phantom, "images", name))

        # scale to control SNR
        img_gt = img_gt.astype(np.float32) * ratio

        # blur
        img_blur = conv.convolution(
            img_gt, psf_crop, padding_mode=conv_padding_mode, domain="fft"
        )

        # add noise
        img_blur_n = add_noise(
            img_blur,
            poisson=poisson,
            sigma_gauss=std_gauss,
            scale_factor=scale_factor,
        )

        if verbese:
            # measure SNR
            print(f"{name}, SNR: {eva.SNR(img_blur, img_blur_n)}")
        # save blurred image
        io.imsave(
            os.path.join(path_raw, "images", name),
            arr=img_blur_n,
            check_contrast=False,
        )

    # save parameters
    params_dict = {
        "path_phantom": path_phantom,
        "path_psf": path_psf,
        "img_gt_shape": img_gt_shape,
        "psf_crop_shape": psf_crop_shape,
        "std_gauss": std_gauss,
        "poisson": poisson,
        "ratio": ratio,
        "scale_factor": scale_factor,
        "conv_padding_mode": conv_padding_mode,
        "psf_size": psf_size,
        "wvl_range": wvl_range,
        "aa_range": aa_range,
        "reflective_index": reflective_index,
    }

    with open(os.path.join(path_raw, "parameters.json"), "w") as f:
        f.write(json.dumps(params_dict, indent=1))
    psf_list_txt.close()


class MicroDataset(Dataset):
    """
    Fluorescen Microscope Images.
    """

    def __init__(
        self,
        path_dataset_lr,
        path_dataset_hr=None,
        transform=None,
        mode="train",
    ):
        super().__init__()
        self.transform = transform
        self.path_sample_lr, self.path_sample_hr, self.path_sample_psf = [], [], []

        if not isinstance(path_dataset_lr, list):
            path_dataset_lr = [path_dataset_lr]
        if not isinstance(path_dataset_hr, list):
            path_dataset_hr = [path_dataset_hr]

        print("-" * 90)
        print(mode)
        print(f"- Number of datastes: {len(path_dataset_lr)}")

        for path_lr, path_hr in zip(path_dataset_lr, path_dataset_hr):
            sample_names = read_txt(os.path.join(path_lr, mode + ".txt"))

            if len(path_dataset_lr) < 10:
                print(f"- Dataset:\n- LR: {path_lr}\n- HR: {path_hr}")
                print(f"- Number of samples: {len(sample_names)}")

            for sample_name in sample_names:
                self.path_sample_lr.append(os.path.join(path_lr, "images", sample_name))
                self.path_sample_hr.append(os.path.join(path_hr, "images", sample_name))
                if os.path.exists(os.path.join(path_lr, "psf.tif")):
                    self.path_sample_psf.append(os.path.join(path_lr, "psf.tif"))
                else:
                    self.path_sample_psf.append(
                        os.path.join(path_lr, "psf", sample_name)
                    )

        print(f"- total number of samples: {len(self.path_sample_lr)}")
        print("-" * 90)

    def __len__(self):
        return len(self.path_sample_lr)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # low-resolution image
        img_lr = read_image(img_path=self.path_sample_lr[idx], expend_channel=True)
        img_lr = torch.tensor(img_lr)

        # high-resolution image
        if self.path_sample_hr[idx] == self.path_sample_lr[idx]:
            img_hr = img_lr
        else:
            img_hr = read_image(img_path=self.path_sample_hr[idx], expend_channel=True)
            img_hr = torch.tensor(img_hr)

        # psf
        if os.path.exists(self.path_sample_psf[idx]):
            psf = read_image(img_path=self.path_sample_psf[idx], expend_channel=True)
            psf = torch.tensor(psf)
        else:
            psf = None

        # transformation
        if self.transform is not None:
            img_lr = self.transform(img_lr)
            img_hr = self.transform(img_hr)

        _, tail = os.path.split(self.path_sample_lr[idx])

        return {
            "lr": img_lr,
            "hr": img_hr,
            "file_name": tail,
            "psf_lr": psf,
        }


class RealFMDataset(Dataset):
    """
    Real Fluorescen Microscope Images.
    """

    def __init__(
        self,
        path_index_file,
        path_dataset_lr,
        path_dataset_hr=None,
        transform=None,
        z_padding=0,
        interpolation=True,
    ):
        super().__init__()
        self.transform = transform
        self.path_sample_lr, self.path_sample_hr = [], []
        self.z_padding = z_padding
        self.interpolation = interpolation

        if not isinstance(path_dataset_lr, list):
            path_dataset_lr = [path_dataset_lr]
        if not isinstance(path_dataset_hr, list):
            path_dataset_hr = [path_dataset_hr]
        if not isinstance(path_index_file, list):
            path_index_file = [path_index_file]

        print("-" * 90)
        print(f"- Number of datastes: {len(path_dataset_lr)}")

        for path_index, path_lr, path_hr in zip(
            path_index_file, path_dataset_lr, path_dataset_hr
        ):
            sample_names = read_txt(path_index)

            if len(path_dataset_lr) < 10:
                print(f"- Dataset:\n- LR: {path_lr}\n- HR: {path_hr}")
                print(f"- Number of samples: {len(sample_names)}")

            for sample_name in sample_names:
                self.path_sample_lr.append(os.path.join(path_lr, sample_name))
                self.path_sample_hr.append(os.path.join(path_hr, sample_name))

        print(f"- total number of samples: {len(self.path_sample_lr)}")
        print("-" * 90)

    def __len__(self):
        return len(self.path_sample_lr)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # low-resolution image
        img_lr = read_image(img_path=self.path_sample_lr[idx], expend_channel=True)
        img_lr = torch.tensor(img_lr)

        # high-resolution image
        img_hr = read_image(img_path=self.path_sample_hr[idx], expend_channel=True)
        img_hr = torch.tensor(img_hr)

        # transformation
        if self.transform is not None:
            img_lr = self.transform(img_lr)
            img_hr = self.transform(img_hr)

        img_lr, img_hr = self.to3d(img_lr), self.to3d(img_hr)

        if self.interpolation:
            img_lr = self.interp(img_lr, img_hr)

        img_lr = self.padz(img_lr)
        img_hr = self.padz(img_hr)

        _, tail = os.path.split(self.path_sample_lr[idx])
        return {
            "lr": img_lr,
            "hr": img_hr,
            "file_name": tail,
            "psf_lr": torch.tensor(0),
        }

    def to3d(self, x):
        if len(x.shape) == 3:
            x = torch.unsqueeze(x, axis=1)
        return x

    def padz(self, x):
        if self.z_padding > 0:
            x = torch.nn.functional.pad(x, (0, 0, 0, 0, self.z_padding, self.z_padding))
        return x

    def interp(self, x, y):
        x, y = torch.unsqueeze(x, dim=0), torch.unsqueeze(y, dim=0)
        x_inter = torch.nn.functional.interpolate(
            x, size=(y.shape[-3], y.shape[-2], y.shape[-1])
        )
        return x_inter[0]


class Dataset_i2i(Dataset):
    """output (image, image)"""

    def __init__(
        self,
        dim,
        path_index_file,
        path_dataset_lr,
        path_dataset_hr,
        transform=None,
        interpolation=True,
    ):
        super().__init__()
        self.dim = dim
        self.interpolation = interpolation
        self.transform = transform

        # string to list of string
        if not isinstance(path_dataset_lr, list):
            path_dataset_lr = [path_dataset_lr]
        if not isinstance(path_dataset_hr, list):
            path_dataset_hr = [path_dataset_hr]
        if not isinstance(path_index_file, list):
            path_index_file = [path_index_file]

        # collect all the path of image
        print("-" * 90)
        print(f"- Number of datastes: {len(path_dataset_lr)}")

        self.path_sample_lr, self.path_sample_hr = [], []

        for path_index, path_lr, path_hr in zip(
            path_index_file, path_dataset_lr, path_dataset_hr
        ):
            # load all the file names in current dataset
            sample_names = read_txt(path_index)
            # connect the path of images
            for sample_name in sample_names:
                self.path_sample_lr.append(os.path.join(path_lr, sample_name))
                self.path_sample_hr.append(os.path.join(path_hr, sample_name))

            if len(path_dataset_lr) <= 3:
                print(f"- Dataset:\n- LR: {path_lr}\n- HR: {path_hr}")
                print(f"- Number of samples: {len(sample_names)}")

        print(f"- total number of samples: {len(self.path_sample_lr)}")
        print("-" * 90)

    def __len__(self):
        return len(self.path_sample_lr)

    def to3d(self, x):
        # convert 2D image with 3D shape.
        if len(x.shape) == 3:
            x = torch.unsqueeze(x, axis=-3)
        return x

    def interp(self, x, y):
        x, y = torch.unsqueeze(x, dim=0), torch.unsqueeze(y, dim=0)

        if self.dim == 2:
            x_inter = torch.nn.functional.interpolate(
                x, size=(y.shape[-2], y.shape[-1]), mode="nearest"
            )
        if self.dim == 3:
            x_inter = torch.nn.functional.interpolate(
                x, size=(y.shape[-3], y.shape[-2], y.shape[-1]), mode="nearest"
            )
        return x_inter[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # low-resolution image
        img_lr = read_image(img_path=self.path_sample_lr[idx], expend_channel=False)
        img_lr = torch.tensor(img_lr)

        # high-resolution image
        img_hr = read_image(img_path=self.path_sample_hr[idx], expend_channel=False)
        img_hr = torch.tensor(img_hr)

        # interpolation low-quality image when the image size of them is different
        if self.interpolation:
            img_lr = self.interp(img_lr, img_hr)

        # transformation
        if self.transform is not None:
            img_lr = self.transform(img_lr)
            img_hr = self.transform(img_hr)

        if self.dim == 3:
            img_lr, img_hr = self.to3d(img_lr), self.to3d(img_hr)

        return {"lr": img_lr, "hr": img_hr}


class Dataset_i(Dataset):
    """output (image)"""

    def __init__(self, path_index_file, path_dataset, scale_factor, transform=None):
        super().__init__()
        self.transform = transform

        # string to list of string
        if not isinstance(path_dataset, list):
            path_dataset = [path_dataset]
        if not isinstance(path_index_file, list):
            path_index_file = [path_index_file]

        # collect all the path of image
        print("-" * 90)
        print(f"- Number of datastes: {len(path_dataset)}")

        self.path_sample = []
        self.sf_sample = []

        for path_index, path_data, sf in zip(
            path_index_file, path_dataset, scale_factor
        ):
            # load all the file names in current dataset
            sample_names = read_txt(path_index)
            # connect the path of images
            for sample_name in sample_names:
                self.path_sample.append(os.path.join(path_data, sample_name))
                self.sf_sample.append(sf)

            if len(path_dataset) <= 3:
                print(f"- Dataset: {path_data}")
                print(f"- Number of samples: {len(sample_names)}")

        print(f"- total number of samples: {len(self.path_sample)}")
        print("-" * 90)

    def __len__(self):
        return len(self.path_sample)

    def interp(self, x, scale_factor):
        x_inter = torch.nn.functional.interpolate(
            x[None], scale_factor=(scale_factor, scale_factor), mode="nearest"
        )
        return x_inter[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # low-resolution image
        img = read_image(img_path=self.path_sample[idx], expend_channel=False)
        img = torch.tensor(img)

        # interpolation low-quality image when the image size of them is different
        if self.sf_sample[idx] != 1:
            img = self.interp(img, scale_factor=self.sf_sample[idx])

        # transformation
        if self.transform is not None:
            img = self.transform(img)

        return {"img": img}


class Dataset_it2i(Dataset):
    """output (image, text, image, text)"""

    def __init__(
        self,
        dim,
        path_index_file,
        path_dataset_lr,
        path_dataset_hr,
        transform=None,
        scale_factor_lr=1,
        scale_factor_hr=1,
        text_version="",
    ):
        super().__init__()
        self.dim = dim
        self.transform = transform

        # string to list of string
        if not isinstance(path_dataset_lr, list):
            path_dataset_lr = [path_dataset_lr]
        if not isinstance(path_dataset_hr, list):
            path_dataset_hr = [path_dataset_hr]
        if not isinstance(path_index_file, list):
            path_index_file = [path_index_file]

        # collect all the path of image
        print("-" * 90)
        print(f"- Number of datastes: {len(path_dataset_lr)}")

        self.path_sample_lr, self.path_sample_hr = [], []
        self.path_text_lr, self.path_text_hr = [], []
        self.scale_factor_lr = []
        self.scale_factor_hr = []

        for sf_lr, sf_hr, path_index, path_lr, path_hr in zip(
            scale_factor_lr,
            scale_factor_hr,
            path_index_file,
            path_dataset_lr,
            path_dataset_hr,
        ):
            # load all the file names in current dataset
            sample_names = read_txt(path_index)

            # connect the path of images
            for sample_name in sample_names:
                self.scale_factor_lr.append(sf_lr)
                self.scale_factor_hr.append(sf_hr)
                # low-resolution images
                self.path_sample_lr.append(os.path.join(path_lr, sample_name))

                # high-resolution images
                self.path_sample_hr.append(os.path.join(path_hr, sample_name))

                # text of low-resolution images
                pt_lr = os.path.join(
                    path_lr, sample_name.split(".")[0] + text_version + ".npy"
                )
                if os.path.exists(pt_lr):
                    self.path_text_lr.append(pt_lr)
                else:
                    self.path_text_lr.append(
                        os.path.join(path_lr, "text" + text_version + ".npy")
                    )

                # text of high-resolution images
                pt_hr = os.path.join(
                    path_hr, sample_name.split(".")[0] + text_version + ".npy"
                )
                if os.path.exists(pt_hr):
                    self.path_text_hr.append(pt_hr)
                else:
                    self.path_text_hr.append(
                        os.path.join(path_hr, "text" + text_version + ".npy")
                    )

            if len(path_dataset_lr) <= 3:
                print(f"- Dataset:\n- LR: {path_lr}\n- HR: {path_hr}")
                print(f"- Number of samples: {len(sample_names)}")

        print(f"- total number of samples: {len(self.path_sample_lr)}")
        print("-" * 90)

    def __len__(self):
        return len(self.path_sample_lr)

    def to3d(self, x):
        # convert 2D image with 3D shape.
        if len(x.shape) == 3:
            x = torch.unsqueeze(x, axis=-3)
        return x

    def interp(self, x, y):
        x, y = torch.unsqueeze(x, dim=0), torch.unsqueeze(y, dim=0)

        if self.dim == 2:
            x_inter = torch.nn.functional.interpolate(
                x, size=(y.shape[-2], y.shape[-1]), mode="nearest"
            )
        if self.dim == 3:
            x_inter = torch.nn.functional.interpolate(
                x, size=(y.shape[-3], y.shape[-2], y.shape[-1]), mode="nearest"
            )
        return x_inter[0]

    def interp_sf(self, x, sf):
        x = torch.unsqueeze(x, dim=0)
        if sf > 0:
            x_inter = torch.nn.functional.interpolate(
                x, scale_factor=sf, mode="nearest"
            )
        if sf < 0:
            x_inter = torch.nn.functional.avg_pool2d(x, kernel_size=-sf, stride=-sf)
        return x_inter[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # low-resolution image and text
        img_lr = read_image(img_path=self.path_sample_lr[idx], expend_channel=False)
        img_lr = torch.tensor(img_lr)

        txt_lr = np.load(self.path_text_lr[idx])
        txt_lr = torch.tensor(txt_lr, dtype=torch.float32)[0]

        # high-resolution image and text
        img_hr = read_image(img_path=self.path_sample_hr[idx], expend_channel=False)
        img_hr = torch.tensor(img_hr)

        txt_hr = np.load(self.path_text_hr[idx])
        txt_hr = torch.tensor(txt_hr, dtype=torch.float32)[0]

        # interpolation low-quality image when the image size of them is different
        if self.scale_factor_lr[idx] != 1:
            img_lr = self.interp_sf(img_lr, self.scale_factor_lr[idx])

        if self.scale_factor_hr[idx] != 1:
            img_hr = self.interp_sf(img_hr, self.scale_factor_hr[idx])

        # transformation
        if self.transform is not None:
            img_lr = self.transform(img_lr)
            img_hr = self.transform(img_hr)

        if self.dim == 3:
            img_lr, img_hr = self.to3d(img_lr), self.to3d(img_hr)

        return {"lr": img_lr, "lr_text": txt_lr, "hr": img_hr, "hr_text": txt_hr}


def win2linux(win_path):
    linux_path = win_path.replace("\\", "/")
    if len(linux_path) > 1 and linux_path[1] == ":":
        drive_letter = linux_path[0].lower()
        linux_path = "/mnt/" + drive_letter + linux_path[2:]
    return linux_path


class Dataset_iit(Dataset):
    """output (image, image, text) or (image, image)"""

    def __init__(
        self,
        dim,
        path_index_file,  # image name file
        path_dataset_lr,
        path_dataset_hr,
        dataset_index,
        path_dataset_text_embedding=None,
        transform=None,
        scale_factor_lr=1,
        scale_factor_hr=1,
        output_type="iit",
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.transform = transform
        self.output_type = output_type

        # string to list of string
        if not isinstance(path_dataset_lr, list):
            path_dataset_lr = [path_dataset_lr]
        if not isinstance(path_dataset_hr, list):
            path_dataset_hr = [path_dataset_hr]
        if not isinstance(path_index_file, list):
            path_index_file = [path_index_file]

        # collect all the path of image
        num_dataset = len(path_dataset_lr)
        print("-" * 90)
        print(f"- Number of datastes: {num_dataset}")

        self.path_sample_lr, self.path_sample_hr = [], []
        self.scale_factor_lr, self.scale_factor_hr = [], []

        if output_type == "iit":
            self.path_sample_text = []

        for i in range(num_dataset):
            sf_lr = scale_factor_lr[i]
            sf_hr = scale_factor_hr[i]
            path_index = path_index_file[i]
            path_lr = path_dataset_lr[i]
            path_hr = path_dataset_hr[i]

            if os.name == "posix":
                path_index = win2linux(path_index)
                path_lr = win2linux(path_lr)
                path_hr = win2linux(path_hr)

            # load all the file names in current dataset
            sample_names = read_txt(os.path.join(path_index))

            # connect the path of images
            for sample_name in sample_names:
                self.scale_factor_lr.append(sf_lr)
                self.scale_factor_hr.append(sf_hr)

                # low-resolution images
                self.path_sample_lr.append(os.path.join(path_lr, sample_name))

                # high-resolution images
                self.path_sample_hr.append(os.path.join(path_hr, sample_name))

                if output_type == "iit":
                    # text of images
                    self.path_sample_text.append(
                        os.path.join(
                            path_dataset_text_embedding, str(dataset_index[i]) + ".npy"
                        )
                    )

            if len(path_dataset_lr) <= 3:
                print(f"- Dataset:\n- LR: {path_lr}\n- HR: {path_hr}")
                print(f"- Number of samples: {len(sample_names)}")

        print(f"- total number of samples: {len(self.path_sample_lr)}")
        print("-" * 90)

    def __len__(self):
        return len(self.path_sample_lr)

    def to3d(self, x):
        # convert 2D image with 3D shape.
        if len(x.shape) == 3:
            x = torch.unsqueeze(x, axis=-3)
        return x

    def interp(self, x, y):
        x, y = torch.unsqueeze(x, dim=0), torch.unsqueeze(y, dim=0)

        if self.dim == 2:
            x_inter = torch.nn.functional.interpolate(
                x, size=(y.shape[-2], y.shape[-1]), mode="nearest"
            )
        if self.dim == 3:
            x_inter = torch.nn.functional.interpolate(
                x, size=(y.shape[-3], y.shape[-2], y.shape[-1]), mode="nearest"
            )
        return x_inter[0]

    def interp_sf(self, x, sf):
        x = torch.unsqueeze(x, dim=0)
        if sf > 0:
            x_inter = torch.nn.functional.interpolate(
                x, scale_factor=sf, mode="nearest"
            )
        if sf < 0:
            x_inter = torch.nn.functional.avg_pool2d(x, kernel_size=-sf, stride=-sf)
        return x_inter[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # low-resolution image
        img_lr = read_image(img_path=self.path_sample_lr[idx], expend_channel=False)
        img_lr = torch.tensor(img_lr)

        # high-resolution image
        img_hr = read_image(img_path=self.path_sample_hr[idx], expend_channel=False)
        img_hr = torch.tensor(img_hr)

        # text
        if self.output_type == "iit":
            text = np.load(self.path_sample_text[idx])
            text = torch.tensor(text, dtype=torch.float32)[0]

        # interpolation low-quality image when the image size of them is different
        if self.scale_factor_lr[idx] != 1:
            img_lr = self.interp_sf(img_lr, self.scale_factor_lr[idx])

        if self.scale_factor_hr[idx] != 1:
            img_hr = self.interp_sf(img_hr, self.scale_factor_hr[idx])

        # transformation
        if self.transform is not None:
            img_lr = self.transform(img_lr)
            img_hr = self.transform(img_hr)

        if self.dim == 3:
            img_lr, img_hr = self.to3d(img_lr), self.to3d(img_hr)

        if self.output_type == "iit":
            return {"lr": img_lr, "hr": img_hr, "text": text}
        if self.output_type == "ii":
            return {"lr": img_lr, "hr": img_hr}


def interpx(x, shape):
    x = torch.tensor(x)
    x = torch.unsqueeze(x, dim=0)
    x = torch.unsqueeze(x, dim=1)
    x_inter = torch.nn.functional.interpolate(x, size=shape)
    return x_inter.numpy()[0, 0]


def interp_sf(x, sf):
    """
    x : [Nc, Ny, Nx]
    """
    x = torch.tensor(x)
    x = torch.unsqueeze(x, dim=0)
    if sf > 0:
        x_inter = torch.nn.functional.interpolate(x, scale_factor=sf, mode="nearest")
    if sf < 0:
        x_inter = torch.nn.functional.avg_pool2d(x, kernel_size=-sf, stride=-sf)
    return x_inter[0].numpy()


def read_image(img_path, expend_channel=False):
    """
    Read image and convert to a numpy array.
    """

    if os.name == "posix":
        img_path = win2linux(img_path)

    # check file type, get extension of file
    _, ext = os.path.splitext(img_path)

    # DICOM data
    if ext == ".dcm":
        img_dcm = pydicom.dcmread(img_path)
        img = img_dcm.pixel_array

    # TIFF data
    if ext == ".tif":
        img = io.imread(img_path)

    # add channel dim
    if expend_channel:
        img = np.expand_dims(img, axis=0)

    return img.astype(np.float32)


def read_txt(path_txt):
    """
    Read txt file consisting of info in each line.
    """
    if os.name == "posix":
        path_txt = win2linux(path_txt)

    with open(path_txt) as f:
        lines = f.read().splitlines()

    if lines[-1] == "":
        lines.pop()

    return lines


def make_path(path):
    """
    makedirs(path).
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def max_norm(x):
    x_max = torch.amax(x, dim=(2, 3, 4), keepdim=True)
    return x / x_max


class NormalizeMinMax(object):
    """
    Normlaize image using min and max value.
    """

    def __init__(self, vmin=None, vmax=None, backward=False):
        self.vmin = vmin
        self.vmax = vmax
        self.backward = backward

    def __call__(self, image):
        if self.backward == False:
            if self.vmin is None:
                self.vmin = image.min()
            if self.vmax is None:
                self.vmax = image.max()

            image = (image - self.vmin) / (self.vmax - self.vmin)

        else:
            assert self.vmin is not None, "Error: 'vmin' is required."
            assert self.vmax is not None, "Error: 'vmax' is required."

            image = image * (self.vmax - self.vmin) + self.vmin

        return image


class NormalizePercentile(object):
    """
    Percentile-based normalization.
    """

    def __init__(self, p_low=0.0, p_high=1.0):
        self.p_low = p_low
        self.p_high = p_high

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            vmin = np.percentile(a=image, q=self.p_low * 100)
            vmax = np.percentile(a=image, q=self.p_high * 100)
            if vmax == 0:
                vmax = np.max(image)

        if isinstance(image, torch.Tensor):
            vmin = torch.quantile(input=image, q=self.p_low)
            vmax = torch.quantile(input=image, q=self.p_high)
            if vmax == 0:
                vmax = torch.max(image)

        amp = vmax - vmin
        if amp == 0:
            amp = 1
        image = (image - vmin) / amp

        return image


def normalization(image, p_low, p_high):
    vmin = np.percentile(a=image, q=p_low * 100)
    vmax = np.percentile(a=image, q=p_high * 100)
    if vmax == 0:
        vmax = np.max(image)
    amp = vmax - vmin
    if amp == 0:
        amp = 1
    image = (image - vmin) / amp

    return image


def unfold(
    img: torch.Tensor, patch_size: int = 64, overlap: int = 0, padding_mode="constant"
):
    img_shape = img.shape
    dim = len(img_shape)
    if dim == 4:
        Ny, Nx = img_shape[-2], img_shape[-1]
        step = patch_size - overlap
        # number of patch along each dim
        num_patch_x = ceil((Nx - patch_size) / step) + 1
        num_patch_y = ceil((Ny - patch_size) / step) + 1
        print(f"({num_patch_y},{num_patch_x})")

        # the size of image after padding
        Nx_pad = num_patch_x * step + overlap
        Ny_pad = num_patch_y * step + overlap
        # padding image
        img_pad = torch.nn.functional.pad(
            img, pad=(0, Nx_pad - Nx, 0, Ny_pad - Ny), mode=padding_mode
        )
        # patching
        patches = []
        for i in range(num_patch_y):
            for j in range(num_patch_x):
                # extract patches
                patch = img_pad[
                    :,
                    :,
                    i * step : i * step + patch_size,
                    j * step : j * step + patch_size,
                ]
                patches.append(patch)
        patches = torch.cat(patches, dim=0)
    else:
        raise ValueError("Only support 2D (batch, channel, height, width) image.")
    print(f"unfold image {img_shape} to patches {patches.shape}")
    return patches


def fold(patches: torch.Tensor, original_image_shape, overlap: int = 0):
    """
    Stitch square patches.
    """
    patch_size = patches.shape[-1]
    step = patch_size - overlap
    batch_size, num_channel = original_image_shape[0:2]

    if len(original_image_shape) == 4:
        Ny, Nx = original_image_shape[-2], original_image_shape[-1]

        # number of patch along each dim
        num_patch_y = ceil((Ny - patch_size) / step) + 1
        num_patch_x = ceil((Nx - patch_size) / step) + 1

        # reshape patches
        patches = torch.reshape(
            patches,
            (
                num_patch_y * num_patch_x,
                batch_size,
                num_channel,
                patch_size,
                patch_size,
            ),
        )

        # calculate the image shape after padding
        img_shape = (
            batch_size,
            num_channel,
            num_patch_y * step + overlap,
            num_patch_x * step + overlap,
        )
        img_pad = torch.zeros(img_shape)  # place holder

        for i in range(num_patch_y):
            for j in range(num_patch_x):
                patch_pad = torch.zeros_like(img_pad)

                patch_pad[
                    :,
                    :,
                    i * step : i * step + patch_size,
                    j * step : j * step + patch_size,
                ] = patches[i * num_patch_x + j]

                overlap_region = torch.where(
                    (img_pad > 0.0) & (patch_pad > 0.0), 0.5, 1.0
                )
                img_pad = (img_pad + patch_pad) * overlap_region

        img_fold = img_pad[..., :Ny, :Nx]
    return img_fold


def fold_scale(
    patches: torch.Tensor,
    original_image_shape,
    overlap: int = 0,
    crop_center: bool = False,
    enable_scale: bool = False,
):
    """
    Stitch square patches.
    """
    patch_size = patches.shape[-1]
    step = patch_size - overlap
    batch_size, num_channel = original_image_shape[0:2]

    if len(original_image_shape) == 4:
        Ny, Nx = original_image_shape[-2], original_image_shape[-1]

        # number of patch along each dim
        num_patch_y = ceil((Ny - patch_size) / step) + 1
        num_patch_x = ceil((Nx - patch_size) / step) + 1

        # reshape patches
        patches = torch.reshape(
            patches,
            (
                num_patch_y * num_patch_x,
                batch_size,
                num_channel,
                patch_size,
                patch_size,
            ),
        )

        # calculate the image shape after padding
        img_shape = (
            batch_size,
            num_channel,
            num_patch_y * step + overlap,
            num_patch_x * step + overlap,
        )
        img_pad = torch.zeros(img_shape)  # place holder

        edge = overlap // 4
        for i in range(num_patch_y):
            for j in range(num_patch_x):
                patch_pad = torch.zeros_like(img_pad)
                patch = patches[i * num_patch_x + j]

                # crop center region -------------------------------------------
                if overlap > 0 and crop_center:
                    patch_crop = torch.zeros_like(patch)
                    if i == 0:
                        if j == 0:
                            patch_crop[..., 0:-edge, 0:-edge] = patch[
                                ..., 0:-edge, 0:-edge
                            ]
                        elif j == (num_patch_x - 1):
                            patch_crop[..., 0:-edge, edge:] = patch[..., 0:-edge, edge:]
                        else:
                            patch_crop[..., 0:-edge, edge:-edge] = patch[
                                ..., 0:-edge, edge:-edge
                            ]
                    elif i == (num_patch_y - 1):
                        if j == 0:
                            patch_crop[..., edge:, 0:-edge] = patch[..., edge:, 0:-edge]
                        elif j == (num_patch_x - 1):
                            patch_crop[..., edge:, edge:] = patch[..., edge:, edge:]
                        else:
                            patch_crop[..., edge:, edge:-edge] = patch[
                                ..., edge:, edge:-edge
                            ]
                    else:
                        if j == 0:
                            patch_crop[..., edge:-edge, 0:-edge] = patch[
                                ..., edge:-edge, 0:-edge
                            ]
                        elif j == (num_patch_x - 1):
                            patch_crop[..., edge:-edge, edge:] = patch[
                                ..., edge:-edge, edge:
                            ]
                        else:
                            patch_crop[..., edge:-edge, edge:-edge] = patch[
                                ..., edge:-edge, edge:-edge
                            ]
                else:
                    patch_crop = patch
                # --------------------------------------------------------------

                patch_pad[
                    :,
                    :,
                    i * step : i * step + patch_size,
                    j * step : j * step + patch_size,
                ] = patch_crop

                overlap_region = torch.where(
                    (img_pad > 0.0) & (patch_pad > 0.0), 0.5, 1.0
                )

                # scale --------------------------------------------------------
                if enable_scale:
                    overlap_region_a = torch.where(
                        (img_pad > 0.0) & (patch_pad > 0.0), img_pad, 0.0
                    )
                    overlap_region_b = torch.where(
                        (img_pad > 0.0) & (patch_pad > 0.0), patch_pad, 0.0
                    )
                    if torch.sum(overlap_region_b) > 0.00001:
                        scale = torch.sum(overlap_region_a) / torch.sum(
                            overlap_region_b
                        )
                    else:
                        scale = 1.0
                else:
                    scale = 1.0
                # --------------------------------------------------------------

                img_pad = (img_pad + patch_pad * scale) * overlap_region

        img_fold = img_pad[..., :Ny, :Nx]
    return img_fold
