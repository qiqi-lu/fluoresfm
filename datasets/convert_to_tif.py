"""
Convert all the images in the folder to tif format.
"""

import skimage.io as io
import numpy as np
import os, tqdm

path_images = ["Cellpose\\transformed\mouse-cortical-neuron\\train\channel_0\masks"]

# ------------------------------------------------------------------------------
for path_image in path_images:
    path_save_to = path_image + "_tif"
    os.makedirs(path_save_to, exist_ok=True)

    print("-" * 80)
    print(path_image)
    # get all the images in the folder
    img_filenames = os.listdir(path_image)
    print("Number of images:", len(img_filenames))

    # loop all the images
    pbar = tqdm.tqdm(total=len(img_filenames), desc="TO TIFF", ncols=100)
    for name in img_filenames:
        img = io.imread(os.path.join(path_image, name))
        img = img.astype(np.float32)
        # img = np.transpose(img, (2, 0, 1))
        img = img[None]
        img = np.clip(img, a_min=0.0, a_max=None)

        # save to tif format, use the same name as the original image, but with tif extension.
        io.imsave(
            fname=os.path.join(path_save_to, name.split(".")[0] + ".tif"),
            arr=img,
            check_contrast=False,
        )
        pbar.update(1)
    pbar.close()
