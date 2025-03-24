import numpy as np
import torch


def read_txt(path_txt):
    """
    Read txt file saving file name in each line.
    """
    with open(path_txt) as f:
        lines = f.read().splitlines()

    if lines[-1] == "":
        lines.pop()
    return lines


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


def interp_sf(x, sf=(1, 1), mode="bilinear"):
    """
    x : [Ny, Nx]
    """
    x = torch.tensor(x)
    x = torch.unsqueeze(x, dim=0)
    x = torch.unsqueeze(x, dim=0)
    x_inter = torch.nn.functional.interpolate(x, scale_factor=sf, mode=mode)
    return x_inter[0, 0].numpy()
