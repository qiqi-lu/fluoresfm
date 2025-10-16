"""
Check the function in utils.
"""

# check linear_transform function.
import torch, os
import matplotlib.pyplot as plt
import skimage.io as skio
from utils.data import interp_sf, NormalizePercentile, Patch_stitcher
from utils.evaluation import linear_transform, PSNR, SSIM

# load test image
path_img_true = "/mnt/e/qiqilu/datasets/BioTISR/transformed/Mitochondria/test/channel_0/SIM_2d/Cell_041_0.tif"
path_img_test = (
    "results/predictions/biotisr-mito-dcv-1/unet_sd_c_all_newnorm/Cell_041_0.tif"
)
path_save_to = os.path.join("results", "figures", "utils")
os.makedirs(path_save_to, exist_ok=True)

normalizer = NormalizePercentile(p_low=0.03, p_high=0.995)
# ------------------------------------------------------------------------------
# test image preprocessing
# ------------------------------------------------------------------------------
print("-" * 80)
print("Test image preprocessing")
# load the images
img_true = skio.imread(path_img_true)
img_test = skio.imread(path_img_test)[0]

img_true = interp_sf(img_true, sf=-2)[0]  # downsample the image

# normalization
img_true = normalizer(img_true)
img_test = normalizer(img_test)

# img_test_transform = linear_transform(img_true, img_test, axis=None)
img_test_transform = img_test

img_true = img_true.clip(0.0, 2.5)
img_test = img_test.clip(0.0, 2.5)

dict_fig = {"dpi": 300, "constrained_layout": True}
dict_img = {"cmap": "hot", "vmin": 0.0, "vmax": 1.0}
dict_text = {"color": "white", "fontsize": 6}
dict_line = dict(linestyle="--", linewidth=0.5, color="black")
dict_hist = dict(bins=250, alpha=0.5)

# ------------------------------------------------------------------------------
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(9, 9), **dict_fig)
axes[0, 0].set_title("img_true")
axes[0, 0].imshow(img_true, **dict_img)
axes[0, 1].set_title("img_test")
axes[0, 1].imshow(img_test_transform, **dict_img)
# show psnr and ssim
psnr = PSNR(img_true, img_test_transform)
ssim = SSIM(img_true, img_test_transform)
axes[0, 1].text(0, 25, f"{psnr:.4f} | {ssim:.4f}", **dict_text)


# show the linear regression between img_true and img_test
axes[0, 2].scatter(img_true.ravel(), img_test_transform.ravel(), s=0.001, c="red")
axes[0, 2].set_title("img_true vs img_test")
# set the x axis and y axis to the same range
axes[0, 2].set_xlim(-0.5, 2.0)
axes[0, 2].set_ylim(-0.5, 2.0)
# show the equal line
axes[0, 2].plot([-0.5, 2.0], [-0.5, 2.0], **dict_line)
# 0 position line
axes[0, 2].plot([0.0, 0.0], [-0.5, 2.0], **dict_line)
axes[0, 2].plot([-0.5, 2.0], [0.0, 0.0], **dict_line)

# show the hitogram of each image
axes[1, 0].hist(img_true.ravel(), color="blue", **dict_hist)
axes[1, 0].set_title("img_true")
axes[1, 1].hist(img_test_transform.ravel(), color="red", **dict_hist)
axes[1, 1].set_title("img_test")
# show the histogram in one plot
axes[1, 2].hist(img_true.ravel(), color="blue", **dict_hist)
axes[1, 2].hist(img_test_transform.ravel(), color="red", **dict_hist)
axes[1, 2].set_title("img_true vs img_test")

# plot the profile of a line
x, y = 380, 200
dict_line = dict(color="white", linewidth=0.5, linestyle="--")
axes[0, 0].plot([x, x], [y, y + 200], **dict_line)
axes[0, 1].plot([x, x], [y, y + 200], **dict_line)
# plot the profile
axes[2, 0].plot(img_true[y : y + 200, x], color="red")
axes[2, 1].plot(img_test_transform[y : y + 200, x], color="blue")
# plot the profile in one plot
axes[2, 2].plot(img_true[y : y + 200, x], color="red")
axes[2, 2].plot(img_test_transform[y : y + 200, x], color="blue")
# mse of line
mse = (
    (img_true[y : y + 200, x] - img_test_transform[y : y + 200, x]) ** 2
).mean() / img_true[y : y + 200, x].mean()
axes[2, 2].text(0, 1, f"MSE: {mse:.4f}", color="black", fontsize=6)

plt.savefig(os.path.join(path_save_to, "input_preprocessing.png"))

# ------------------------------------------------------------------------------
# test unfold and fold function.
# ------------------------------------------------------------------------------
print("-" * 80)
print("Test unfold and fold function.")
stitcher = Patch_stitcher(overlap=64, patch_size=256, padding_mode="reflect")

img_true = torch.tensor(img_true).float()[None, None]
original_shape = img_true.shape
img_true_fold = stitcher.unfold(img_true)
img_true_fold = stitcher.fold_linear_ramp(
    img_true_fold, original_image_shape=original_shape
)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 3), **dict_fig)
axes[0].imshow(img_true[0, 0], **dict_img)
axes[0].set_title("img_true")
axes[1].imshow(img_true_fold[0, 0], **dict_img)
axes[1].set_title("img_true_fold")
plt.savefig(os.path.join(path_save_to, "unfold_fold.png"))
