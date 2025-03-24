import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt


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
