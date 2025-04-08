"""
Check the function in utils.
"""

# check linear_transform function.
import matplotlib.pyplot as plt
import skimage.io as skio
from utils.data import normalization, interp_sf
from utils.evaluation import linear_transform

# load test image
path_img_true = "/mnt/e/qiqilu/datasets/BioTISR/transformed/Mitochondria/test/channel_0/SIM_2d/Cell_041_0.tif"
path_img_test = "outputs/unet_c/external_dataset/biotisr-mito-dcv-3/unet_sd_c_all_newnorm/Cell_041_0.tif"

# ------------------------------------------------------------------------------
# load the images
img_true = skio.imread(path_img_true)[0]
img_test = skio.imread(path_img_test)[0]

# downsample the image
img_true = interp_sf(img_true, sf=-2)

# normalization
img_true = normalization(img_true, p_low=0.001, p_high=0.999, clip=False)

# linear transform
img_test_transform = linear_transform(img_true, img_test, axis=None, normed_true=True)
# img_test_transform = normalization(
#     img_test_transform, p_low=0.001, p_high=0.999, clip=False
# )

# ------------------------------------------------------------------------------
fig, axes = plt.subplots(nrows=3, ncols=3, dpi=300, figsize=(9, 9))
axes[0, 0].imshow(img_true, cmap="hot", vmin=0.0, vmax=1.0)
axes[0, 0].set_title("img_true")
axes[0, 1].imshow(img_test_transform, cmap="hot", vmin=0.0, vmax=1.0)
axes[0, 1].set_title("img_test")
# show the linear regression between img_true and img_test
axes[0, 2].scatter(img_true.ravel(), img_test_transform.ravel(), s=0.001, c="red")
axes[0, 2].set_title("img_true vs img_test")
# set the x axis and y axis to the same range
axes[0, 2].set_xlim(0.0, 2.0)
axes[0, 2].set_ylim(0.0, 2.0)
# show the equal line
axes[0, 2].plot([0.0, 2.0], [0.0, 2.0], "k--")

# show the hitogram of each image
axes[1, 0].hist(img_true.ravel(), bins=250, color="blue", alpha=0.5)
axes[1, 0].set_title("img_true")
axes[1, 1].hist(img_test_transform.ravel(), bins=250, color="red", alpha=0.5)
axes[1, 1].set_title("img_test")
# show the histogram in one plot
axes[1, 2].hist(img_true.ravel(), bins=250, color="blue", alpha=0.5)
axes[1, 2].hist(img_test_transform.ravel(), bins=250, color="red", alpha=0.5)
axes[1, 2].set_title("img_true vs img_test")

# plot the profile of a line
x = 210
y = 200
axes[0, 0].plot([x, x], [y, y + 200], "--", color="white", linewidth=0.5)
axes[0, 1].plot([x, x], [y, y + 200], "--", color="white", linewidth=0.5)
# plot the profile
axes[2, 0].plot(img_true[y : y + 200, x], color="red")
axes[2, 1].plot(img_test_transform[y : y + 200, x], color="blue")
# plot the profile in one plot
axes[2, 2].plot(img_true[y : y + 200, x], color="red")
axes[2, 2].plot(img_test_transform[y : y + 200, x], color="blue")

plt.savefig("test_linear_transform.png")
