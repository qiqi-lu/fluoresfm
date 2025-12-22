"""
Generate test image used for evaluation the processing cost.
"""

import numpy as np
import os
import skimage.io as io

image_size_list = [256, 512, 1024]
num_image = 10
path_data_root = os.path.join("data", "CostEva")

for image_size in image_size_list:
    path_data_save = os.path.join(path_data_root, f"size_{image_size}")
    os.makedirs(path_data_save, exist_ok=True)
    print("-" * 80)
    print(f"Generate test image with size {image_size}")
    print(f"Save to {path_data_save}")

    with open(os.path.join(path_data_save, "all.txt"), "w") as f:
        for i in range(num_image):
            image = np.random.rand(image_size, image_size)[None].astype(np.float32)
            image_name = f"image_{i}.tif"
            path_image_save = os.path.join(path_data_save, image_name)
            io.imsave(path_image_save, image, check_contrast=False)
            f.write(image_name + "\n")
print("-" * 80)
