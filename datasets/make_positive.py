import numpy as np
import skimage.io as io
from utils_data import read_txt
import os, tqdm

path_txt = "VMSIM\\transformed\Fig5\\test.txt"
path_root = "VMSIM\\transformed\Fig5\\test\mito\SIM"

filenames = read_txt(path_txt)
pbar = tqdm.tqdm(desc="POS", total=len(filenames), ncols=80)
for file in filenames:
    img = io.imread(os.path.join(path_root, file))
    img = np.clip(img, a_min=0.0, a_max=None)
    io.imsave(os.path.join(path_root, file), img, check_contrast=False)
    pbar.update(1)
pbar.close()
