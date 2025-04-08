import numpy as np


def cal_radar_range(data, percent=(0.1, 0.95), precision=0.5):
    # x(m)|----|a-----------b|-----|y(n)
    a = data.max(axis=-1)
    b = data.min(axis=-1)
    m = percent[1]
    n = percent[0]

    x = (a - b + b * m - a * n) / (m - n)
    y = (b * m - a * n) / (m - n)
    x = np.ceil(x / precision) * precision
    y = np.floor(y / precision) * precision
    return x, y


def crop_edge(image, width=25):
    """
    Crop the edge of the image.

    ### Args:
    - `image`: numpy array, shape (H, W).
    - `width`: int, the width of the edge to crop.

    ### Returns:
    - `image`: numpy array, shape (H-2*width, W-2*width).
    """
    if width > 0:
        image = image[width:-width, width:-width]
    return image
