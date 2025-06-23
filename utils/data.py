"""
Functions used for data processing.
"""

import torch, json, os, pydicom, random, sys, scipy

sys.path.insert(1, "E:\qiqilu\Project\\2024 Foundation model\code")
sys.path.insert(1, "/mnt/e/qiqilu/Project/2024 Foundation model/code")
from models.PSFmodels import BWModel

import numpy as np
import skimage.io as io
from skimage import restoration
from math import ceil
import methods.convolution as conv
import utils.evaluation as eva
import utils.data as utils_data
from torch.utils.data import Dataset


def rolling_ball_approximation(image, radius, sf=4):
    """
    Background subtraction using rolling ball algorithm.\n
    This algorithm's complexity is polynomial in the radius, with degree equal
    to the image dimensionality (a 2D image is N^2, a 3D image is N^3, etc.),
    so it can take a long time as the radius grows beyond 30.
    So this function downsample the image and then uoscaling the background
    to reduce the size of the processed image.

    ### Parameters:
    - `image` : numpy array, image.
    - `radius` : int, radius of the rolling ball.
    - `sf` : int, downsample factor. Default is 4.
    ### Returns:
    - `image` : numpy array, background subtracted image.
    """
    if type(image) == torch.Tensor:
        image = image.numpy()
    image = np.array(image)
    if len(image.shape) == 2:
        image = image[None]
    image = image.astype(np.float32)
    # downsample the input image of x4
    image_down = interp_sf(image, sf=-sf)
    # apply rolling ball algorithm to the downsampled image
    bg_down = restoration.rolling_ball(image_down[0], radius=radius)
    # upscaling the background to the original size
    bg = interp_sf(bg_down[None], sf=sf, mode="bicubic")
    # subtract the background from the original image
    # bg = np.clip(bg, 0, None)
    image_roll = image - bg
    return image_roll, bg


class RotFlip(object):
    """
    Rotation and flip.
    Input a image and a random number, and do a specific operation on the image according to the random number.
    """

    def __init__(self):
        pass

    def __call__(self, img, random_num):
        """
        ### Inputs:
        - `img` : numpy array, image. [B, C, H, W].
        - `random_num` : int, random number within [0, 6].
        ### Returns:
        - `img` : numpy array, augmented image. [B, C, H, W].
        """
        if random_num == 1:
            img = torch.rot90(img, k=1, dims=[1, 2])
        elif random_num == 2:
            img = torch.rot90(img, k=2, dims=[1, 2])
        elif random_num == 3:
            img = torch.rot90(img, k=3, dims=[1, 2])
        elif random_num == 4:
            img = torch.flip(img, dims=[1])
        elif random_num == 5:
            img = torch.flip(img, dims=[2])
        elif random_num == 6:
            img = torch.flip(img, dims=[1, 2])
        else:
            pass
        return img


def convert_to_8bit(img, data_range=None):
    """
    Convert image to 8-bit.

    ### Parameters:
    - `img` : numpy array, image.
    - `data_range` : tuple, (vmin, vmax), the range of the image. If None, the range is calculated from the image.

    ### Returns:
    - `img` : numpy array, image.
    """
    img = tensor_to_array(img)
    if data_range is None:
        vmin = img.min()
        vmax = img.max()
    else:
        vmin, vmax = data_range
    img = np.clip(img, vmin, vmax)
    img = (img - vmin) / (vmax - vmin) * 255.0
    img = img.astype(np.uint8)
    return img


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
    if win_path == None:
        return None
    elif os.name == "posix":
        linux_path = win_path.replace("\\", "/")
        if len(linux_path) > 1 and linux_path[1] == ":":
            drive_letter = linux_path[0].lower()
            linux_path = "/mnt/" + drive_letter + linux_path[2:]
        return linux_path
    else:
        return win_path


class Dataset_iit(Dataset):
    """output (image, image, text), (image, image, task) or (image, image)"""

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
        task=None,
        output_type="ii-text",
        use_clean_data=False,
        rotflip=False,
        clip=None,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.transform = transform
        self.output_type = output_type
        self.rotflip = rotflip
        self.clip = clip

        if self.rotflip:
            self.Rot = utils_data.RotFlip()
            self.random_generator = torch.Generator()
            self.random_generator.manual_seed(7)

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

        if output_type == "ii-text":
            self.path_sample_text = []
        if output_type == "ii-task":
            self.sampel_task = []

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
                if path_dataset_text_embedding is not None:
                    path_dataset_text_embedding = win2linux(path_dataset_text_embedding)

            # load all the file names in current dataset
            if not use_clean_data:
                sample_names = read_txt(os.path.join(path_index))
            else:
                sample_names = read_txt(
                    os.path.join(path_index.split(".")[0] + "_clean.txt")
                )

            # connect the path of images
            for sample_name in sample_names:
                self.scale_factor_lr.append(sf_lr)
                self.scale_factor_hr.append(sf_hr)

                # low-resolution images
                self.path_sample_lr.append(os.path.join(path_lr, sample_name))

                # high-resolution images
                self.path_sample_hr.append(os.path.join(path_hr, sample_name))

                if output_type == "ii-text":
                    # text of images
                    self.path_sample_text.append(
                        os.path.join(
                            path_dataset_text_embedding, str(dataset_index[i]) + ".npy"
                        )
                    )
                if output_type == "ii-task":
                    # collect task of each sample
                    if task[i] == "sr":
                        id_task = 1
                    elif task[i] == "dn":
                        id_task = 2
                    elif task[i] == "iso":
                        id_task = 3
                    elif task[i] == "dcv":
                        id_task = 4
                    self.sampel_task.append(id_task)

            if len(path_dataset_lr) <= 3:
                print(f"- Dataset:\n- LR: {path_lr}\n- HR: {path_hr}")
                print(f"- Number of samples: {len(sample_names)}")

        print(f"- total number of samples: {self.__len__()}")
        print("-" * 90)

        if self.rotflip:
            num_sample = self.__len__()
            self.random_num = torch.randint(
                low=0, high=6, size=(num_sample,), generator=self.random_generator
            )

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
        img_lr = read_image(img_path=self.path_sample_lr[idx])
        img_lr = torch.tensor(img_lr)

        # high-resolution image
        img_hr = read_image(img_path=self.path_sample_hr[idx])
        img_hr = torch.tensor(img_hr)

        # text
        if self.output_type == "ii-text":
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

        # augmentation
        if self.rotflip:
            img_lr = self.Rot(img=img_lr, random_num=self.random_num[idx])
            img_hr = self.Rot(img=img_hr, random_num=self.random_num[idx])

        if self.clip is not None:
            img_lr = torch.clamp(img_lr, min=self.clip[0], max=self.clip[1])
            img_hr = torch.clamp(img_hr, min=self.clip[0], max=self.clip[1])

        # output
        if self.output_type == "ii-text":
            return {"lr": img_lr, "hr": img_hr, "text": text}
        elif self.output_type == "ii":
            return {"lr": img_lr, "hr": img_hr}
        elif self.output_type == "ii-task":
            return {"lr": img_lr, "hr": img_hr, "task": self.sampel_task[idx]}


def interpx(x, shape):
    x = torch.tensor(x)
    x = torch.unsqueeze(x, dim=0)
    x = torch.unsqueeze(x, dim=1)
    x_inter = torch.nn.functional.interpolate(x, size=shape)
    return x_inter.numpy()[0, 0]


def interp_sf(x, sf, mode="nearest"):
    """
    Interpolate the image based on the scale factor.
    When `sf` > 1, the image is unsampled.
    When `sf` < 1, the image is downsampled.

    ### Args:
    - `x` : numpy array, image to be interpolated. [C, H, W]
    - `sf` : float, scale factor.

    ### Returns:
    - `x_inter` : numpy array, interpolated image. [C, H, W]
    """
    assert len(x.shape) == 3, "The image shape should be [C, H, W]."
    x = torch.unsqueeze(torch.tensor(x), dim=0)
    if sf > 0:
        x_inter = torch.nn.functional.interpolate(x, scale_factor=sf, mode=mode)
    if sf < 0:
        x_inter = torch.nn.functional.avg_pool2d(x, kernel_size=-sf, stride=-sf)
    return x_inter[0].numpy()


def read_image(img_path: str, expend_channel: bool = False) -> np.ndarray:
    """
    Read image and convert to a numpy array. Supported data formats: `.dcm`, and `.tif`.

    ### Parameters:
    - `img_path` : str, path of the image.
    - `expend_channel` : bool, whether to expand the channel dimension at axis 0. Default is False.

    ### Returns:
    - `img` : numpy array, image. [C, H, W] or [1, C, H, W] if `expend_channel` is True.
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

    ### Parameters:
    - `p_low` : float, lower percentile.
    - `p_high` : float, upper percentile.
    """

    def __init__(self, p_low=0.0, p_high=1.0):
        self.p_low = p_low
        self.p_high = p_high

    def __call__(self, image):
        """
        ### Inputs:
        - `image` : numpy array, image to be normalized. [C, H, W] or [1, C, H, W].

        ### Returns:
        - `image` : numpy array, normalized image. [C, H, W] or [1, C, H, W].
        """
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


class NormalizePercentile_patch(object):
    """
    Normalize each patch in a batch, and save the normalization parameters for each patch.
    ### Parameters:
    - `p_low` : float, lower percentile.
    - `p_high` : float, upper percentile.
    """

    def __init__(self, p_low=0.0, p_high=1.0):
        self.p_low = p_low
        self.p_high = p_high
        self.vmin = None
        self.vmax = None
        self.amp = None

    def __call__(self, image):
        """
        ### Inputs:
        - `image` : torch tensor, image to be normalized. [B, C, H, W]
        ### Returns:
        - `image` : torch tensor, normalized image. [B, C, H, W]
        """
        assert len(image.shape) == 4, "Error: 'image' should be a 4D tensor."
        if isinstance(image, np.ndarray):
            self.vmin = np.percentile(
                a=image, q=self.p_low * 100, axis=(1, 2, 3), keepdims=True
            )
            self.vmax = np.percentile(
                a=image, q=self.p_high * 100, axis=(1, 2, 3), keepdims=True
            )
            self.amp = self.vmax - self.vmin
            self.amp = np.where(self.amp == 0, 1, self.amp)

        if isinstance(image, torch.Tensor):
            num_batch = image.shape[0]
            image_flatten = image.view(num_batch, -1)
            self.vmin = torch.quantile(
                input=image_flatten, q=self.p_low, dim=1, keepdim=True
            ).view(num_batch, 1, 1, 1)
            self.vmax = torch.quantile(
                input=image_flatten, q=self.p_high, dim=1, keepdim=True
            ).view(num_batch, 1, 1, 1)
            self.amp = self.vmax - self.vmin
            self.amp = torch.where(self.amp == 0, 1, self.amp)

        image_norm = (image - self.vmin) / self.amp
        return image_norm

    def backward(self, image_norm):
        """
        ### Inputs:
        - `image_norm` : torch tensor, normalized image. [B, C, H, W]
        ### Returns:
        - `image` : torch tensor, denormalized image. [B, C, H, W]
        """
        assert len(image_norm.shape) == 4, "Error: 'image_norm' should be a 4D tensor."
        assert self.vmin is not None, "Error: 'vmin' is required."
        assert self.vmax is not None, "Error: 'vmax' is required."
        assert self.amp is not None, "Error: 'amp' is required."
        image = image_norm * self.amp + self.vmin
        return image


def tensor_to_array(img):
    if not isinstance(img, np.ndarray):
        img = img.cpu().detach().numpy()
    return img


def normalization(image, p_low, p_high, clip=False):
    """
    Normalize image using percentile-based normalization.
    - image: numpy array or torch tensor.
    - p_low: low percentile.
    - p_high: high percentile.
    - clip: clip the image to [0, 1].
    """
    image = tensor_to_array(image).astype(np.float32)

    vmin = np.percentile(a=image, q=p_low * 100)
    vmax = np.percentile(a=image, q=p_high * 100)
    if vmax == 0:
        vmax = np.max(image)
    amp = vmax - vmin
    if amp == 0:
        amp = 1
    image = (image - vmin) / amp

    if clip:
        image = np.clip(image, 0, 1)

    return image


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

    ### Parameters:
    - `patches` : torch tensor, patches to be stitched. [N, 1, C, patchsize, patchsize].
    - `original_image_shape` : tuple, shape of the original image. (1, C, Ny, Nx).
    - `overlap` : int, overlap between patches. Default is 0.
    - `crop_center` : bool, whether to crop the center of the patch to stitch.
    - `enable_scale` : bool, whether to enable scaling.

    ### Returns:
    - `img_fold` : torch tensor, stitched image. [1, C, Ny, Nx].
    """
    patch_size = patches.shape[-1]
    step = patch_size - overlap
    batch_size, num_channel = original_image_shape[0:2]

    if len(original_image_shape) == 4:
        Ny, Nx = original_image_shape[-2:]  # image shape

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
        img_pad_shape = (
            batch_size,
            num_channel,
            num_patch_y * step + overlap,
            num_patch_x * step + overlap,
        )
        img_pad = torch.zeros(img_pad_shape, device=patches.device)  # place holder
        patch_pad = torch.zeros_like(img_pad)

        edge = torch.tensor(overlap // 4, device=patches.device)

        patch_zero = torch.zeros_like(patches[0])
        patch_mask_lu = patch_zero
        patch_mask_lu[..., 0:-edge, 0:-edge] = 1.0

        patch_mask_mu = patch_zero
        patch_mask_mu[..., 0:-edge, edge:-edge] = 1.0

        patch_mask_ru = patch_zero
        patch_mask_ru[..., 0:-edge, edge:] = 1.0

        patch_mask_lm = patch_zero
        patch_mask_lm[..., edge:-edge, 0:-edge] = 1.0

        patch_mask_mm = patch_zero
        patch_mask_mm[..., edge:-edge, edge:-edge] = 1.0

        patch_mask_rm = patch_zero
        patch_mask_rm[..., edge:-edge, edge:] = 1.0

        patch_mask_lb = patch_zero
        patch_mask_lb[..., edge:, 0:-edge] = 1.0

        patch_mask_mb = patch_zero
        patch_mask_mb[..., edge:, edge:-edge] = 1.0

        patch_mask_rb = patch_zero
        patch_mask_rb[..., edge:, edge:] = 1.0

        # # calculate the mean of each patch
        # patch_mean = torch.mean(patches, dim=(1, 2, 3, 4))
        # # sort the mean and get the index of them
        # patch_mean_sort, sort_index = torch.sort(patch_mean, descending=True)
        # start_y = sort_index[0] // num_patch_x
        # start_x = sort_index[0] % num_patch_x
        # index_x = list(range(start_x, num_patch_x)) + list(range(start_x - 1, -1, -1))
        # index_y = list(range(start_y, num_patch_y)) + list(range(start_y - 1, -1, -1))

        index_y = range(num_patch_y)
        index_x = range(num_patch_x)

        # pbar = tqdm.tqdm(desc="unfold", total=num_patch_y, ncols=80)
        for i in index_y:
            for j in index_x:
                patch_pad *= 0.0
                patch = patches[i * num_patch_x + j]
                patch_crop = patch
                # crop center region -------------------------------------------
                if overlap > 0 and crop_center:
                    if i == 0:
                        if j == 0:
                            patch_crop *= patch_mask_lu
                        elif j == (num_patch_x - 1):
                            patch_crop *= patch_mask_ru
                        else:
                            patch_crop *= patch_mask_mu
                    elif i == (num_patch_y - 1):
                        if j == 0:
                            patch_crop *= patch_mask_lb
                        elif j == (num_patch_x - 1):
                            patch_crop *= patch_mask_rb
                        else:
                            patch_crop *= patch_mask_mb
                    else:
                        if j == 0:
                            patch_crop *= patch_mask_lm
                        elif j == (num_patch_x - 1):
                            patch_crop *= patch_mask_rm
                        else:
                            patch_crop *= patch_mask_mm
                # --------------------------------------------------------------
                patch_pad[
                    :,
                    :,
                    i * step : i * step + patch_size,
                    j * step : j * step + patch_size,
                ] = patch_crop

                overlap_mask = torch.where(
                    (img_pad > 0.0) & (patch_pad > 0.0), 1.0, 0.0
                )
                # scale --------------------------------------------------------
                scale = torch.tensor(1.0, device=patches.device)
                if enable_scale and ((i + j) > 0):
                    # sum_a = torch.sum(img_pad * overlap_mask)
                    # sum_b = torch.sum(patch_pad * overlap_mask)

                    # if sum_b > 0.00001:
                    #     scale = sum_a / sum_b

                    xx = img_pad * overlap_mask
                    yy = patch_pad * overlap_mask
                    if torch.sum(xx * xx) > 0.00001:
                        scale = torch.sum(xx * yy) / torch.sum(xx * xx)

                # ------------------------------------------------------------
                overlap_region = overlap_mask * (-0.5) + 1.0
                img_pad += patch_pad * scale
                img_pad *= overlap_region

                # # ------------------------------------------------------------
                # if enable_scale and ((i + j) > 0):
                #     img_pad_ol = img_pad * overlap_mask
                #     patch_pad_ol = patch_pad * overlap_mask
                #     # get the value larger than 0 into a vector
                #     img_pad_ol = img_pad_ol[img_pad_ol > 0.0]
                #     patch_pad_ol = patch_pad_ol[patch_pad_ol > 0.0]
                #     a, b = linear_transform_ab(img_pad_ol, patch_pad_ol)

                #     # convert to 01
                #     patch_pad_01 = torch.where(patch_pad > 0.0, 1.0, 0.0)
                #     patch_pad_scale = (a + b * patch_pad) * patch_pad_01
                # else:
                #     patch_pad_scale = patch_pad
                # # ------------------------------------------------------------
                # overlap_region = overlap_mask * (-0.5) + 1.0
                # img_pad += patch_pad_scale
                # img_pad *= overlap_region
                # # ------------------------------------------------------------

            # pbar.update(1)
        # pbar.close()
        img_fold = img_pad[..., :Ny, :Nx]
    return img_fold


def linear_transform_ab(img_true, img_test):
    """
    Get the a and b for linear transform.
    `img_test_transform = a + b * img_test`

    ### Parameters:
    - `img_true` : ground truth image. [Ny, Nx] or N.
    - `img_test` : test image. [Ny, Nx] or N.
    ### Returns:
    - `a` : float, a.
    - `b` : float, b.
    """
    # calculate mean and std
    if type(img_true) == torch.Tensor:
        mean = lambda x: torch.mean(x)
        square = lambda x: torch.square(x)
    if type(img_true) == np.ndarray:
        mean = lambda x: np.mean(x)
        square = lambda x: np.square(x)

    mean_true = mean(img_true)
    mean_test = mean(img_test)
    # calculate slope and intercept
    b = mean((img_test - mean_test) * (img_true - mean_true)) / mean(
        square(img_test - mean_test) + 0.00001
    )
    a = mean_true - b * mean_test
    return a, b


class Patch_stitcher(object):
    def __init__(self, patch_size: int = 64, overlap: int = 0, padding_mode="constant"):
        self.ps = patch_size
        self.ol = overlap
        self.padding_mode = padding_mode
        self.generate_mask()
        print("StitchPatch initialized.")
        print(f"patch size: {self.ps}, overlap: {self.ol}")

    def set_params(self, patch_size: int, overlap: int):
        if patch_size != self.ps or overlap != self.ol:
            self.ps = patch_size
            self.ol = overlap
            self.generate_mask()
            print("StitchPatch parameters updated.")
            print(f"patch size: {self.ps}, overlap: {self.ol}")

    def unfold(self, img: torch.Tensor):
        img_shape = img.shape
        dim = len(img_shape)
        if dim == 4:
            Ny, Nx = img_shape[-2], img_shape[-1]
            step = self.ps - self.ol
            # number of patch along each dim
            num_patch_x = ceil((Nx - self.ps) / step) + 1
            num_patch_y = ceil((Ny - self.ps) / step) + 1

            # the size of image after padding
            Nx_pad = num_patch_x * step + self.ol
            Ny_pad = num_patch_y * step + self.ol
            # padding image
            img_pad = torch.nn.functional.pad(
                img, pad=(0, Nx_pad - Nx, 0, Ny_pad - Ny), mode=self.padding_mode
            )
            # patching
            patches = torch.zeros(
                size=(num_patch_x * num_patch_y, img_shape[1], self.ps, self.ps),
                device=img_pad.device,
                dtype=img_pad.dtype,
            )
            for i in range(num_patch_y):
                for j in range(num_patch_x):
                    # extract patches
                    patches[i * num_patch_x + j] = img_pad[
                        0,
                        :,
                        i * step : i * step + self.ps,
                        j * step : j * step + self.ps,
                    ]
        else:
            raise ValueError("Only support 2D (batch, channel, height, width) image.")
        print(
            f"unfold image {img_shape} to patches {patches.shape}",
            f"({num_patch_y},{num_patch_x})",
        )
        return patches

    def generate_mask(self):
        self.patch_mask_lu = np.pad(
            np.ones((1, 1, self.ps - self.ol, self.ps - self.ol)),
            ((0, 0), (0, 0), (0, self.ol + 1), (0, self.ol + 1)),
            "linear_ramp",
        )[..., 0:-1, 0:-1]

        self.patch_mask_mu = np.pad(
            np.ones((1, 1, self.ps - self.ol, self.ps - 2 * self.ol)),
            ((0, 0), (0, 0), (0, self.ol + 1), (self.ol + 1, self.ol + 1)),
            "linear_ramp",
        )[..., 0:-1, 1:-1]

        self.patch_mask_ru = np.pad(
            np.ones((1, 1, self.ps - self.ol, self.ps - self.ol)),
            ((0, 0), (0, 0), (0, self.ol + 1), (self.ol + 1, 0)),
            "linear_ramp",
        )[..., 0:-1, 1:]

        self.patch_mask_lm = np.pad(
            np.ones((1, 1, self.ps - 2 * self.ol, self.ps - self.ol)),
            ((0, 0), (0, 0), (self.ol + 1, self.ol + 1), (0, self.ol + 1)),
            "linear_ramp",
        )[..., 1:-1, 0:-1]

        self.patch_mask_mm = np.pad(
            np.ones((1, 1, self.ps - 2 * self.ol, self.ps - 2 * self.ol)),
            ((0, 0), (0, 0), (self.ol + 1, self.ol + 1), (self.ol + 1, self.ol + 1)),
            "linear_ramp",
        )[..., 1:-1, 1:-1]

        self.patch_mask_rm = np.pad(
            np.ones((1, 1, self.ps - 2 * self.ol, self.ps - self.ol)),
            ((0, 0), (0, 0), (self.ol + 1, self.ol + 1), (self.ol + 1, 0)),
            "linear_ramp",
        )[..., 1:-1, 1:]

        self.patch_mask_lb = np.pad(
            np.ones((1, 1, self.ps - self.ol, self.ps - self.ol)),
            ((0, 0), (0, 0), (self.ol + 1, 0), (0, self.ol + 1)),
            "linear_ramp",
        )[..., 1:, 0:-1]

        self.patch_mask_mb = np.pad(
            np.ones((1, 1, self.ps - self.ol, self.ps - 2 * self.ol)),
            ((0, 0), (0, 0), (self.ol + 1, 0), (self.ol + 1, self.ol + 1)),
            "linear_ramp",
        )[..., 1:, 1:-1]

        self.patch_mask_rb = np.pad(
            np.ones((1, 1, self.ps - self.ol, self.ps - self.ol)),
            ((0, 0), (0, 0), (self.ol + 1, 0), (self.ol + 1, 0)),
            "linear_ramp",
        )[..., 1:, 1:]

        # ----------------------------------------------------------------------
        # one column patches
        self.patch_mask_lu_01 = np.pad(
            np.ones((1, 1, self.ps - self.ol, self.ps)),
            ((0, 0), (0, 0), (0, self.ol + 1), (0, 0)),
            "linear_ramp",
        )[..., 0:-1, :]
        self.patch_mask_lm_01 = np.pad(
            np.ones((1, 1, self.ps - 2 * self.ol, self.ps)),
            ((0, 0), (0, 0), (self.ol + 1, self.ol + 1), (0, 0)),
            "linear_ramp",
        )[..., 1:-1, :]
        self.patch_mask_lb_01 = np.pad(
            np.ones((1, 1, self.ps - self.ol, self.ps)),
            ((0, 0), (0, 0), (self.ol + 1, 0), (0, 0)),
            "linear_ramp",
        )[..., 1:, :]
        # ----------------------------------------------------------------------
        # one row patches
        self.patch_mask_lu_10 = np.pad(
            np.ones((1, 1, self.ps, self.ps - self.ol)),
            ((0, 0), (0, 0), (0, 0), (0, self.ol + 1)),
            "linear_ramp",
        )[..., 0:-1]
        self.patch_mask_mu_10 = np.pad(
            np.ones((1, 1, self.ps, self.ps - 2 * self.ol)),
            ((0, 0), (0, 0), (0, 0), (self.ol + 1, self.ol + 1)),
            "linear_ramp",
        )[..., 1:-1]
        self.patch_mask_ru_10 = np.pad(
            np.ones((1, 1, self.ps, self.ps - self.ol)),
            ((0, 0), (0, 0), (0, 0), (self.ol + 1, 0)),
            "linear_ramp",
        )[..., 1:]
        # ----------------------------------------------------------------------
        # only one patch
        self.patch_mask_lu_11 = np.ones((1, 1, self.ps, self.ps))

    def fold_linear_ramp(self, patches, original_image_shape):
        """
        Stitch square patches.

        ### Parameters:
        - `patches` : patches to be stitched. [N, 1, C, patchsize, patchsize].
        - `original_image_shape` : tuple, shape of the original image. (1, C, Ny, Nx).
        - `overlap` : int, overlap between patches. Default is 0.

        ### Returns:
        - `img_fold` : torch tensor, stitched image. [1, C, Ny, Nx].
        """
        patches = tensor_to_array(patches)
        input_patch_size = patches.shape[-1]

        if input_patch_size != self.ps:
            print("[Warning] the patch size of input is not equal to the init setting.")
            print("[Warning] recreate the masks.")
            self.ps = input_patch_size
            self.generate_mask()

        step = self.ps - self.ol

        assert (
            len(original_image_shape) == 4
        ), "Only support image with shape of [Nb, Nc, Ny, Nx]."
        bs, nc, Ny, Nx = original_image_shape  # image shape

        # number of patch along each dim
        num_patch_y = ceil((Ny - self.ps) / step) + 1
        num_patch_x = ceil((Nx - self.ps) / step) + 1
        num_pacth = num_patch_y * num_patch_x

        # reshape patches
        patches = np.reshape(patches, (num_pacth, bs, nc, self.ps, self.ps))

        # calculate the image shape after padding
        img_pad_shape = (
            bs,
            nc,
            num_patch_y * step + self.ol,
            num_patch_x * step + self.ol,
        )
        img_pad = np.zeros(img_pad_shape)  # place holder
        patch_pad = np.zeros_like(img_pad)

        patch_mask_lu = self.patch_mask_lu
        patch_mask_ru = self.patch_mask_ru
        patch_mask_mu = self.patch_mask_mu

        patch_mask_lb = self.patch_mask_lb
        patch_mask_rb = self.patch_mask_rb
        patch_mask_mb = self.patch_mask_mb

        patch_mask_lm = self.patch_mask_lm
        patch_mask_rm = self.patch_mask_rm
        patch_mask_mm = self.patch_mask_mm
        # ----------------------------------------------------------------------
        # update masks for special cases
        if num_patch_x == 1 and num_patch_y > 1:
            patch_mask_lu = self.patch_mask_lu_01
            patch_mask_lm = self.patch_mask_lm_01
            patch_mask_lb = self.patch_mask_lb_01

        if num_patch_y == 1 and num_patch_x > 1:
            patch_mask_lu = self.patch_mask_lu_10
            patch_mask_mu = self.patch_mask_mu_10
            patch_mask_ru = self.patch_mask_ru_10

        if num_patch_x == 1 and num_patch_y == 1:
            patch_mask_lu = self.patch_mask_lu_11

        # ----------------------------------------------------------------------
        for i in range(num_patch_y):
            for j in range(num_patch_x):
                patch = patches[i * num_patch_x + j]
                patch_crop = patch
                # --------------------------------------------------------------
                # weighting
                if self.ol > 0:
                    if i == 0:
                        if j == 0:
                            patch_crop *= patch_mask_lu
                        elif j == (num_patch_x - 1):
                            patch_crop *= patch_mask_ru
                        else:
                            patch_crop *= patch_mask_mu
                    elif i == (num_patch_y - 1):
                        if j == 0:
                            patch_crop *= patch_mask_lb
                        elif j == (num_patch_x - 1):
                            patch_crop *= patch_mask_rb
                        else:
                            patch_crop *= patch_mask_mb
                    else:
                        if j == 0:
                            patch_crop *= patch_mask_lm
                        elif j == (num_patch_x - 1):
                            patch_crop *= patch_mask_rm
                        else:
                            patch_crop *= patch_mask_mm
                # --------------------------------------------------------------
                patch_pad[
                    :,
                    :,
                    i * step : i * step + self.ps,
                    j * step : j * step + self.ps,
                ] += patch_crop
        # sum
        img_fold = patch_pad[..., :Ny, :Nx]
        return img_fold


if __name__ == "__main__":
    # test RotFlip class
    # rot = RotFlip()
    # img = torch.randn(1, 4, 4)
    # img_rot = rot(img, 7)
    # print(img[0])
    # print(img_rot[0])

    # test normalizepercentile_patch
    norm = NormalizePercentile_patch(0.03, 0.995)
    img = torch.randn(4, 1, 64, 64)
    img_norm = norm(img)
    img_norm_reverse = norm.backward(img_norm)

    print(img_norm.shape)
    print(norm.vmin.flatten(), norm.vmax.flatten(), norm.amp.flatten())
    print(img.flatten().max())
    print(img_norm.flatten().max())
    print(img_norm_reverse.flatten().max())
