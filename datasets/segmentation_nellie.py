"""
Use Nellie to segment the organelle in the image.
"""

import os, tqdm
from nellie.im_info.verifier import FileInfo, ImInfo
from nellie.segmentation.filtering import Filter
from nellie.segmentation.labelling import Label
from nellie.segmentation.networking import Network
from nellie.utils.base_logger import logger
from utils_data import read_txt
import skimage.io as io

# ------------------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logger.disable()

# path_images = "BioTISR/transformed/Lysosomes/test/channel_0/WF_noise_level_3_2d"
# path_images = "BioTISR/transformed/Lysosomes/test/channel_0/SIM_2d"

# path_filenames = "BioTISR\\transformed\\Lysosomes\\test_2d.txt"
# path_images = "BioTISR/transformed/Lysosomes/test/channel_0/WF_noise_level_3_2d"
# path_images = "BioTISR/transformed/Lysosomes/test/channel_0/WF_noise_level_2_2d"
# path_images = "BioTISR/transformed/Lysosomes/test/channel_0/WF_noise_level_1_2d"

path_filenames = "RCAN3D\\transformed\DN_Tomm20_Mito\\test_2d_selected.txt"
# path_images = "RCAN3D\\transformed\DN_Tomm20_Mito\\test\channel_0\GT_2d"
path_images = "RCAN3D\\transformed\DN_Tomm20_Mito\\test\channel_0\\raw_2d"

# path_filenames = "BioTISR\\transformed\\Mitochondria\\test_2d.txt"
# path_images = "BioTISR/transformed/Mitochondria/test/channel_0/WF_noise_level_3_2d"
# path_images = "BioTISR/transformed/Mitochondria/test/channel_0/WF_noise_level_2_2d"
# path_images = "BioTISR/transformed/Mitochondria/test/channel_0/WF_noise_level_1_2d"
# path_images = "BioTISR\\transformed\\Mitochondria\\test\channel_0\SIM_2d_ave2"
# path_images = "E:\qiqilu\Project\\2024 Foundation model\code\\results\predictions\\biotisr-mito-dcv-3\\unet_sd_c_all_newnorm-ALL-v2-160-small-bs16"
# path_images = "E:\qiqilu\Project\\2024 Foundation model\code\\results\predictions\\biotisr-mito-dn-2\\unet_sd_c_all_newnorm-ALL-v2-160-small-bs16"
# path_images = "E:\qiqilu\Project\\2024 Foundation model\code\\results\predictions\\biotisr-mito-dn-1\\unet_sd_c_all_newnorm-ALL-v2-160-small-bs16"

# get all the filenames
filenames = read_txt(path_filenames)[:8]
path_masks = path_images + "_mask_nellie"
os.makedirs(path_masks, exist_ok=True)

# ------------------------------------------------------------------------------
print("-" * 80)
print("num of images:", len(filenames))
pbar = tqdm.tqdm(total=len(filenames), desc="SEGMENTATION", ncols=80)
for filename in filenames:
    path_file = os.path.join(path_images, filename)
    file_info = FileInfo(path_file)
    file_info.find_metadata()
    file_info.load_metadata()
    file_info.change_axes("TYX")
    file_info.change_dim_res("T", 1)
    file_info.change_dim_res("Y", 0.2)
    file_info.change_dim_res("X", 0.2)

    # print(f"{file_info.metadata_type=}")
    # print(f"{file_info.axes=}")
    # print(f"{file_info.shape=}")
    # print(f"{file_info.dim_res=}")
    # print(f"{file_info.good_axes=}")
    # print(f"{file_info.good_dims=}")

    im_info = ImInfo(file_info)
    preprocessing = Filter(im_info, remove_edges=False)
    preprocessing.run()
    segmentation = Label(im_info, otsu_thresh_intensity=False, threshold=None)
    segmentation.run()
    # networking = Network(im_info)
    # networking.run()

    # move the segmentation to the mask folder
    path_saved = im_info.pipeline_paths["im_instance_label"]
    mask = io.imread(path_saved)
    mask = mask.astype("uint16")[None]
    io.imsave(os.path.join(path_masks, filename), mask, check_contrast=False)

    pbar.update(1)
pbar.close()
