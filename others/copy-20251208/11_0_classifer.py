from models.biomedclip_embedder import BiomedCLIP
import torch, os
from constants import structure_types
from utils.data import win2linux
import skimage.io as io
import numpy as np
from PIL import Image

biomedcliper = BiomedCLIP(
    path_json="checkpoints/clip/biomedclip/open_clip_config.json",
    path_bin="checkpoints/clip/biomedclip/open_clip_pytorch_model.bin",
    context_length=160,
    device=torch.device("cuda:0"),
)

print("-" * 80)
num_structure_types = len(structure_types)
print("[INFO] Number of structure types: {}".format(num_structure_types))
print(structure_types)
print("-" * 80)

# path_img_folder = "E:\qiqilu\datasets\BioSR\\transformed\ER\\test\channel_0\SIM"
path_img_folder = (
    "E:\qiqilu\datasets\BioSR\\transformed\ER\\test\channel_0\WF_noise_level_6",
    "E:\qiqilu\datasets\BioSR\\transformed\CCPs\\test\channel_0\WF_noise_level_6",
    "E:\qiqilu\datasets\BioSR\\transformed\F_actin\\test\channel_0\WF_noise_level_6",
    "E:\qiqilu\datasets\BioSR\\transformed\MTs\\test\channel_0\WF_noise_level_6",
    "E:\qiqilu\datasets\FMD\\transformed\Confocal_BPAE\\test\channel_B\\avg2\\repeat_1",
    "E:\qiqilu\datasets\FMD\\transformed\Confocal_FISH\\test\channel_0\\avg2\\repeat_1",
    "E:\qiqilu\datasets\DeepBacs\\transformed\Denoising_Ecoli\\train\channel_0\low_SNR",
)

imgs = []
imgs_path = []
for path in path_img_folder:
    path = win2linux(path)
    # get all the image names in the folder end with .tif
    img_names = os.listdir(path)
    img_names = [img_name for img_name in img_names if img_name.endswith(".tif")]
    num_imgs = len(img_names)
    print("[INFO] Number of images: {}".format(num_imgs))
    # get the embeddings of the images
    for img_name in img_names:
        path_img = os.path.join(path, img_name)
        imgs_path.append(path_img)
        # img = io.imread(path_img)
        img = biomedcliper.preprocess(Image.open(path_img))
        # crop the center 224x224 of the image
        img_shape = img.shape
        center = (img_shape[1] // 2, img_shape[2] // 2)
        # center = (400, 400)
        img = img[
            :, center[0] - 112 : center[0] + 112, center[1] - 112 : center[1] + 112
        ]
        imgs.append(img)
num_imgs = len(imgs)
assert num_imgs == len(
    imgs_path
), "The number of images and the number of image paths are not equal."
print("[INFO] Total number of images: {}".format(num_imgs))

imgs = np.stack(imgs)

print(imgs.shape)

# ------------------------------------------------------------------------------
# zero-shot image classification
# ------------------------------------------------------------------------------
# template = "this is a photo of "
# template = "this is a fluorescence image with a structure of "
# # template = ""
# labels = [template + structure_type for structure_type in structure_types]

# logits, sorted_indices = biomedcliper.classification(imgs, labels)

# # print the top 5 predictions
# for i in range(num_imgs):
#     print("-" * 80)
#     print("[INFO] Image: {}".format(imgs_path[i]))
#     for j in range(5):
#         idx = sorted_indices[i, j].item()
#         score = logits[i, idx].item()
#         print("[INFO] Top {}: {} - {}".format(j + 1, structure_types[idx], score))

# ------------------------------------------------------------------------------
# image retrieval
# ------------------------------------------------------------------------------
imgs_features = biomedcliper.image_embedding(imgs)
# calculate correlation between different images
corr = imgs_features @ imgs_features.t()
print(corr.shape)
print(corr)
# print the top 5 similar images
for i in range(num_imgs):
    print("-" * 80)
    print("[INFO] Image: {}".format(imgs_path[i]))
    for j in range(5):
        idx = torch.argsort(corr[i], descending=True)[j + 1].item()
        score = corr[i, idx].item()
        print("[INFO] Top {}: {} - {}".format(j + 1, imgs_path[idx], score))
print("-" * 80)
