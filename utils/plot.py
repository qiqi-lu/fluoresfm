import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import measure


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


def colorize(image, vmin=0, vmax=1, color=(0, 255, 0)):
    """
    Create an RGB image from a single-channel image using a
    specific color.

    ### Parameters:
    - `image`: numpy array, shape (H, W), single channel image.
    - `vmin`: float, the minimum value of the image.
    - `vmax`: float, the maximum value of the image.
    - `color`: tuple, the color to use for the image.

    ### Returns:
    - `image`: numpy array, shape (H, W, 3), RGB image.
    """
    # Rescale the image
    image_clip = np.clip(image, vmin, vmax)
    image_clip = (image_clip - vmin) / (vmax - vmin)
    image_clip_3 = np.repeat(image_clip[..., None], 3, axis=-1)
    image_clip_3 = image_clip_3 * color
    return image_clip_3.astype(np.uint8)


def adjust_aspect_ratio(img, ratio=0.5):
    """
    Crop the image to have a specific aspect ratio.
    The larger dimentsion will be the 1.
    """
    if ratio != 1:
        h, w = img.shape[-2:]
        h_new = int(w * ratio)
        if h > h_new:
            img = img[:h_new]
    return img


def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt

    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)

    start_value = mappable.get_clim()[0]
    last_value = mappable.get_clim()[1]
    mid_value = (start_value + last_value) / 2
    cbar.set_ticks([start_value, mid_value, last_value])
    cbar.set_ticklabels([f"{start_value:.2f}", f"{mid_value:.2f}", f"{last_value:.2f}"])

    plt.sca(last_axes)
    return cbar


def add_scale_bar(
    ax, image, pixel_size, bar_length, bar_height=0.01, bar_color="white", pos=(20, 20)
):
    """
    Add a scale bar to the given axes.

    Parameters:
    ax (matplotlib.axes.Axes): The axes to which the scale bar will be added.
    image (np.ndarray): The 2D image array. The shape of the image array should be (H, W).
    pixel_size (float): The size of each pixel in the image (um).
    bar_length (float): The desired length of the scale bar (um).
    bar_height (float, optional): The height of the scale bar as a fraction of the image height. Default is 0.02.
    bar_color (str, optional): The color of the scale bar. Default is 'white'.
    pos (tuple, optional): The position of the scale bar in the image (pixels). Default is (20, 20).
    """
    # Calculate the number of pixels corresponding to the bar length
    bar_pixels = bar_length / pixel_size

    # Get the image dimensions
    image_height, image_width = image.shape[-2:]

    # Calculate the physical height of the bar
    bar_physical_height = bar_height * image_height

    # Add the scale bar rectangle
    rect = plt.Rectangle(
        pos, bar_pixels, bar_physical_height, color=bar_color, zorder=10
    )
    ax.add_patch(rect)


def add_significant_bars(ax, x1, x2, y, p_value, dict_line={}, dict_asterisks={}):
    """
    Add significant bars to the given axes.

    ### Parameters:
    - `ax`: matplotlib.axes.Axes, the axes to which the significant bars will be added.
    - `x1`: float, the x-coordinate of the left edge of the bar.
    - `x2`: float, the x-coordinate of the right edge of the bar.
    - `y`: float, the y-coordinate of the bar.
    - `p_value`: float, the p-value of the comparison.
    - `significant_level`: float, the significance level. Default is 0.05.
    """
    if p_value <= 0.0001:
        asterisks = "****"
    elif p_value <= 0.001:
        asterisks = "***"
    elif p_value <= 0.01:
        asterisks = "**"
    elif p_value <= 0.05:
        asterisks = "*"
    else:
        asterisks = "ns"

    offset = 0.05
    dict_l = {"color": "black", "linewidth": 1}
    dict_a = {"ha": "left", "va": "bottom", "fontsize": 10, "color": "black"}

    if dict_line:
        dict_l.update(dict_line)
    if dict_asterisks:
        dict_a.update(dict_asterisks)

    # ax.plot([x1, x1, x2, x2], [y, y + offset, y + offset, y], **dict_l)
    ax.plot([x1, x2], [y, y], **dict_l)
    # ax.text((x1 + x2) / 2, y * 1.001, asterisks, **dict_a)
    ax.text(x1, y * 0.98, asterisks, **dict_a)


def get_outlines(mask):
    """Get the outlines of each labels"""
    outlines = []
    for label in np.unique(mask):
        if label == 0:
            continue
        binary_mask = (mask == label).astype(np.uint8)
        # Find contours
        contours = measure.find_contours(binary_mask, 0.5)
        for contour in contours:
            contour = np.array(contour).astype(np.int32)
            outlines.append(contour)
    return outlines
