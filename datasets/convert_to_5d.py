"""
Use Nellie to segment the strutures in the image.
"""

import os, tqdm
import skimage.io as skio
import numpy as np
from pyometiff import OMETIFFWriter, OMETIFFReader

path_image = "BioTISR/transformed/Mitochondria/test/channel_0/WF_noise_level_1"
save_to = path_image + "_5d"
os.makedirs(save_to, exist_ok=True)

# get all the tif file in path_image
list_image = os.listdir(path_image)
list_image = [x for x in list_image if x.endswith(".tif")]
# sort the list_image
list_image.sort()

# ------------------------------------------------------------------------------
print("-" * 100)
print(path_image)
print("num of images:", len(list_image))

pbar = tqdm.tqdm(total=len(list_image), desc="NELLIE", ncols=100)
for name in list_image:
    img = skio.imread(os.path.join(path_image, name))
    img = img.astype(np.float32)

    # add channel and z axis at (1,2) dims
    img = np.expand_dims(img, axis=1)
    img = np.expand_dims(img, axis=1)

    dimension_order = "TCZYX"
    metadata = {
        "PhysicalSizeX": 1,
        "PhysicalSizeUnit": "um",
        "PhysicalSizeY": 1,
        "PhysicalSizeUnit": "um",
        "PhysicalSizeZ": 1,
        "PhysicalSizeUnit": "um",
        "PhysicalSizeC": 1,
        "PhysicalSizeUnit": "um",
        "PhysicalSizeT": 1,
        "PhysicalSizeUnit": "s",
    }

    writer = OMETIFFWriter(
        os.path.join(save_to, name),
        dimension_order=dimension_order,
        array=img,
        metadata=metadata,
        # explicit_tiffdata=False,
    )
    writer.write()
    pbar.update(1)
pbar.close()
