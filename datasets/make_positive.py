import numpy as np
import skimage.io as io
import os, tqdm

path_root = "E:\qiqilu\datasets\RCAN3D\\transformed\C2S_MT\\test\channel_0\confocal"

filenames = os.listdir(path_root)
pbar = tqdm.tqdm(desc="POS", total=len(filenames), ncols=80)
for file in filenames:
    img = io.imread(os.path.join(path_root, file))
    img = np.clip(img, a_min=0.0, a_max=None)
    io.imsave(os.path.join(path_root, file), img, check_contrast=False)
    pbar.update(1)
pbar.close()
