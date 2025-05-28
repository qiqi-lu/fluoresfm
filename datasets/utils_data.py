import numpy as np
import torch
import matplotlib.pyplot as plt
import skimage.io as io


def read_txt(path_txt):
    """
    Read txt file saving file name in each line.
    """
    with open(path_txt) as f:
        lines = f.read().splitlines()

    if lines[-1] == "":
        lines.pop()
    return lines


# def normalization(image, p_low, p_high):
#     vmin = np.percentile(a=image, q=p_low * 100)
#     vmax = np.percentile(a=image, q=p_high * 100)
#     if vmax == 0:
#         vmax = np.max(image)
#     amp = vmax - vmin
#     if amp == 0:
#         amp = 1
#     image = (image - vmin) / amp

#     return image


def normalization(image, p_low, p_high):
    vmin = np.percentile(a=image, q=p_low * 100)
    vmax = np.percentile(a=image, q=p_high * 100)
    if vmax == 0:
        image *= 0.0
    else:
        amp = vmax - vmin
        if amp == 0:
            amp = 1
        image = (image - vmin) / amp

    return image, vmin, vmax


def interp_sf(x, sf=(1, 1), mode="bilinear"):
    """
    x : [Ny, Nx]
    """
    x = torch.tensor(x)
    x = torch.unsqueeze(x, dim=0)
    x = torch.unsqueeze(x, dim=0)
    x_inter = torch.nn.functional.interpolate(x, scale_factor=sf, mode=mode)
    return x_inter[0, 0].numpy()


def mean_pooling(x, kernel_size=2, stride=2):
    """
    x: [Ny, Nx]
    """
    x = torch.tensor(x)[None, None]
    x_pool = torch.nn.functional.avg_pool2d(x, kernel_size=kernel_size, stride=stride)
    return x_pool[0, 0].numpy()


def ave_pooling_2d(x, scale_factor=1):
    """Average pooling for 2D image.

    ### Parameters:
    - x: [1, Ny, Nx] or [Ny, Nx]
    - scale_factor: int, scale factor for pooling.

    ### Returns:
    - x: [1, Ny, Nx] or [Ny, Nx]
    """
    x = torch.tensor(x, dtype=torch.float32)
    if len(x.shape) == 2:
        x = torch.nn.functional.avg_pool2d(x[None, None], kernel_size=scale_factor)
        return x.numpy()[0, 0]
    elif len(x.shape) == 3:
        x = torch.nn.functional.avg_pool2d(x[None], kernel_size=scale_factor)
        return x.numpy()[0]
    elif len(x.shape) == 4:
        x = torch.nn.functional.avg_pool2d(x, kernel_size=scale_factor)
        return x.numpy()
    else:
        raise ValueError("Invalid input shape.")


def linear_trasnform(x, y):
    """
    transform x to have a similar range as y.
    - x: [Ny, Nx]
    - y: [Ny, Nx]
    """
    y = y.astype(np.float32)
    x = x.astype(np.float32)
    x_size = x.shape[-1]
    y_size = y.shape[-1]
    sf = int(y_size / x_size)
    y = mean_pooling(y, kernel_size=sf, stride=sf)
    # linear regression between x and y
    mean_y = np.mean(y)
    mean_x = np.mean(x)
    b = np.mean((x - mean_x) * (y - mean_y)) / np.mean(np.square(x - mean_x))
    a = mean_y - b * mean_x
    x_transform = a + b * x
    return x_transform


def zero_padding(img, patch_size, step_size):
    if len(img.shape) == 2:
        Ny, Nx = img.shape
    if len(img.shape) == 3:
        Nz, Ny, Nx = img.shape

    tail_x = (Nx - patch_size) % step_size
    if tail_x > step_size * 3 / 4:
        x_pad = step_size - tail_x
    tail_y = (Ny - patch_size) % step_size
    if tail_y > step_size * 3 / 4:
        y_pad = step_size - tail_y

    if len(img.shape) == 2:
        img = np.pad(img, pad_width=((0, y_pad), (0, x_pad)))
    if len(img.shape) == 3:
        img = np.pad(img, pad_width=((0, 0), (0, y_pad), (0, x_pad)))
    return img


if __name__ == "__main__":
    img_lr = io.imread(
        "E:\qiqilu\datasets\BioSR\\transformed\F_actin\\train\channel_0\WF_noise_level_1\\3.tif"
    )
    img_hr = io.imread(
        "E:\qiqilu\datasets\BioSR\\transformed\F_actin\\train\channel_0\SIM\\3.tif"
    )

    pl, ph = 0.0, 0.9999
    img_lr_t = normalization(img_lr, p_low=pl, p_high=ph)
    img_hr_t = normalization(img_hr, p_low=pl, p_high=ph)

    print(img_lr_t.max(), img_hr_t.max())
    print(img_lr_t.min(), img_hr_t.min())

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 6), dpi=300)
    axes[0, 0].imshow(img_lr)
    axes[0, 1].imshow(img_hr)
    axes[1, 0].imshow(img_lr_t, vmin=0, vmax=1)
    axes[1, 1].imshow(img_hr_t, vmin=0, vmax=1)

    plt.savefig("tmp.png")
