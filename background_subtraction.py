import numpy as np
import matplotlib.pyplot as plt
import os, tqdm
from skimage import io

from utils.data import rolling_ball_approximation, win2linux

# path_images = "E:\qiqilu\datasets\BioTISR\\transformed\Mitochondria\\test\channel_0\WF_noise_level_1_2d"
path_images = "E:\qiqilu\datasets\W2S\\transformed\\test\channel_0\SIM"
# path_images = "E:\qiqilu\datasets\BioTISR\\transformed\CCPs\\test\channel_0\SIM_2d"
# path_images = (
#     "E:\qiqilu\datasets\SIMActin\\transformed\\test\channel_0\SIM_2d_proj_p1024_s420_2d"
# )
# path_images = "E:\qiqilu\datasets\VMSIM\\transformed\Fig3\\test\mito\SIM_cropedge_32"
# path_images = (
#     "E:\qiqilu\datasets\BioTISR\\transformed\Mitochondria\\test\channel_0\SIM_2d"
# )
# path_images = "results\predictions\\biotisr-mito-sr-1-rb\\care_newnorm-v2-all"
# path_images = "results\predictions\\biotisr-mito-sr-1-rb\\dfcan_newnorm-v2-all"
# path_images = "results\predictions\\biotisr-mito-sr-1-rb\\unifmir_all-newnorm-v2"
# path_images = "results\predictions\\biotisr-mito-sr-1-rb\\unet_sd_c_all_newnorm-ALL-v2-160-small-bs16"
# path_images = "results\predictions\\biotisr-mito-sr-1\\unet_sd_c_all_newnorm-ALL-v2-160-small-bs16"
# path_images = "results\predictions\\vmsim3-mito-sr-crop\\unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-crossx"
# path_images = "results\predictions\\biotisr-mito-sr-1-rb\\unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-crossx"
# path_images = "results\predictions\\biotisr-mito-sr-1\\unet_sd_c_all_newnorm-ALL-v2-160-small-bs16-crossx"
# path_images = "results\predictions\\biotisr-mito-sr-1\\unet_sd_c_all_newnorm-ALL-v2-160-small-bs16"

path_images = win2linux(path_images)

# sub_type = "constant"
sub_type = "rolling-ball"

sub_value = 0.005
# radius, sf = 25, 16
# radius, sf = 25, 4
radius, sf = 5, 4

if sub_type == "constant":
    path_save_to = path_images + "-c-" + str(sub_value)
elif sub_type == "rolling-ball":
    path_save_to = path_images + "-rb-" + str(radius) + "-" + str(sf)
else:
    raise ValueError("sub_type must be 'constant' or 'rolling-ball'")

# path_save_to = path_images
os.makedirs(path_save_to, exist_ok=True)


# ------------------------------------------------------------------------------
# get all the tif image  from the folder
image_names = [f for f in os.listdir(path_images) if f.endswith(".tif")]
# sort the image according to the number in filename
image_names = sorted(image_names, key=lambda x: int(x.split(".")[0]))

print("Number of images:", len(image_names))

# for each image, read the image and perform rolling ball approximation
pbar = tqdm.tqdm(total=len(image_names), desc="Background subtraction")
for i, image_name in enumerate(image_names):
    image = io.imread(os.path.join(path_images, image_name))
    # image = np.clip(image, 0, None)
    if sub_type == "constant":
        image_rb = image - sub_value
        bg = np.ones_like(image) * sub_value
    elif sub_type == "rolling-ball":
        image_rb, bg = rolling_ball_approximation(image, radius=radius, sf=sf)
    image_rb = np.clip(image_rb, 0, None)

    if i == 0:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 3), dpi=300)
        # show image, iamge_rb and bg
        axes[0].imshow(image[0], cmap="hot")
        axes[0].set_title("Original Image")
        axes[1].imshow(image_rb[0], cmap="hot")
        axes[1].set_title("Rolling Ball Approximation")
        axes[2].imshow(bg[0], cmap="hot")
        axes[2].set_title("Background")

        plt.savefig("bkg_subtraction.png")

    # save the image_rb and bg to the folder path_save_to
    io.imsave(os.path.join(path_save_to, image_name), image_rb, check_contrast=False)
    pbar.update(1)
pbar.close()
