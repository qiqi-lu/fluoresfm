"""
Construct datasets used for image calssfication.
Extract patches with a shape of 224x224 for the training images as training data
or templete banks.
Extraxt patches with a shape of 224x224 for the testing images as testing data.
Use the embeder to extract the features of the patches.

the json file follows the format of:
https://learn.microsoft.com/en-us/azure/machine-learning/reference-automl-images-schema?view=azureml-api-2
"""

import os, pandas, tqdm, json
import numpy as np
import skimage.io as io
from utils.data import win2linux, read_txt
from dataset_analysis import dataset_names_all
from utils.data import normalization, grayto255

dataset_names = []

# internal training datasets ---------------------------------------------------
# group = "train"
# for key, value in dataset_names_all["internal_dataset"].items():
#     dataset_names.extend(value)
# path_excel = os.path.join("dataset_train_transformer-v2.xlsx")
# path_json = os.path.join("data", "dataset_classfication_train.json")
# num_patches_per_class = 3000

# internal testing datasets ----------------------------------------------------
# group = "test"
# for key, value in dataset_names_all["internal_dataset"].items():
#     dataset_names.extend(value)
# path_excel = os.path.join("dataset_test-v2.xlsx")
# path_json = os.path.join("data", "dataset_classfication_test_in.json")
# num_patches_per_class = 500

# external testing datasets ----------------------------------------------------
group = "test"
for key, value in dataset_names_all["external_dataset"].items():
    dataset_names.extend(value)
path_excel = os.path.join("dataset_test-v2.xlsx")
path_json = os.path.join("data", "dataset_classfication_test_ex.json")
num_patches_per_class = 200
# ------------------------------------------------------------------------------

num_datasets = len(dataset_names)
patch_size = 224
step = 112
normalizer = lambda x: normalization(x, p_low=0.03, p_high=0.995)


print("-" * 80)
print("[INFO] Number of datasets: {}".format(num_datasets))
print("[INFO] Path to the excel file: {}".format(path_excel))
print("[INFO] Path to the json file: {}".format(path_json))

# ------------------------------------------------------------------------------
# construct training dataset
print("-" * 80)
if group == "train":
    data_frame = pandas.read_excel(path_excel, sheet_name="64x64")
if group == "test":
    data_frame = pandas.read_excel(path_excel)

# collect all the path of images
images_path_list = []
images_type_list = []
num_images_list = []
images_index_list = []

for datasetname in dataset_names:
    if datasetname not in data_frame["id"].values:
        continue
    info = data_frame[data_frame["id"] == datasetname].iloc[0]
    path_dataset = info["path_lr_raw"]
    if path_dataset == "Unknown":
        continue
    if path_dataset in images_path_list:
        continue
    images_path_list.append(path_dataset)
    images_type_list.append(info["structure#"])

    # check the number of images in current dataset
    if group == "train":
        images_index_list.append(info["path_index"])
        filenames = os.listdir(win2linux(path_dataset))
        filenames = [filename for filename in filenames if filename.endswith(".tif")]
        num_files = len(filenames)
        num_images_list.append(num_files)
    if group == "test":
        images_index_list.append(info["path_index_raw"])
        path_txt = win2linux(info["path_index"])
        filenames = read_txt(path_txt)
        filenames = filenames[:8]
        num_files = len(filenames)
        num_images_list.append(num_files)


df_images = pandas.DataFrame(
    {
        "images_index": images_index_list,
        "path": images_path_list,
        "type": images_type_list,
        "num_images": num_images_list,
    }
)

# get all class name
types_all = []
for type in images_type_list:
    if type not in types_all:
        types_all.append(type)
print("[INFO] All types: {}".format(types_all))

# ------------------------------------------------------------------------------
json_file = open(path_json, "w")
num_patches = 0

pbar = tqdm.tqdm(desc="[INFO] Extract patches", total=len(images_path_list), ncols=80)
# loop through each class
for class_name in types_all:
    # calculate the number of image in current class
    info = df_images[df_images["type"] == class_name]
    num_images_all = info["num_images"].sum()
    print("\n" + "-" * 80)
    print(f"[INFO] Class: {class_name}")
    print(f"[INFO] Number of images: {num_images_all}")
    num_patches_per_image = num_patches_per_class // num_images_all
    num_patches_per_image = max(num_patches_per_image, 1)
    print(f"[INFO] Number of patches per image: {num_patches_per_image}")

    pathes = info["path"].values
    indexfiles = info["images_index"].values
    for i_data, path in enumerate(pathes):
        pbar.update(1)
        path = win2linux(path)
        indexfile = win2linux(indexfiles[i_data])
        path_save_to = path + "_patches_224x224"
        os.makedirs(path_save_to, exist_ok=True)
        if group == "train":
            filenames = os.listdir(path)
            filenames = [
                filename for filename in filenames if filename.endswith(".tif")
            ]
        if group == "test":
            filenames = read_txt(indexfile)
            filenames = filenames[:8]

        for filename in filenames:
            path_img = os.path.join(path, filename)
            img_raw = io.imread(path_img).astype(np.float32)
            img_raw = np.squeeze(img_raw)

            if len(img_raw.shape) == 2:
                img = img_raw
            if len(img_raw.shape) == 3:
                # use the center slice
                nslice = img_raw.shape[0]
                img = img_raw[nslice // 2]

            # preprocess image
            img = normalizer(img)
            img = np.clip(img, 0, 2.0)
            img = grayto255(img)

            # skip small images
            if img.shape[-2] < patch_size or img.shape[-1] < patch_size:
                continue
            centers = []
            # grid centers
            for cy in range(0, img.shape[-2] - patch_size, step):
                for cx in range(0, img.shape[-1] - patch_size, step):
                    centers.append((cy + patch_size // 2, cx + patch_size // 2))
            # random centers
            for _ in range(len(centers) // 2):
                cy = np.random.randint(0, img.shape[-2] - patch_size)
                cx = np.random.randint(0, img.shape[-1] - patch_size)
                centers.append((cy + patch_size // 2, cx + patch_size // 2))

            while len(centers) < num_patches_per_image * 2:
                # random centers
                cy = np.random.randint(0, img.shape[-2] - patch_size)
                cx = np.random.randint(0, img.shape[-1] - patch_size)
                centers.append((cy + patch_size // 2, cx + patch_size // 2))

            patches = []
            for center in centers:
                patch = img[
                    center[0] - patch_size // 2 : center[0] + patch_size // 2,
                    center[1] - patch_size // 2 : center[1] + patch_size // 2,
                ]
                patches.append(patch)
            patches = np.array(patches)

            # ------------------------------------------------------------------
            # calculate the intensity std of each patch, exclude the flat patch
            avg_intensity = np.std(patches, axis=(1, 2))
            # sort for large to small
            idx = np.argsort(avg_intensity)[::-1]
            # get the top num_patches_per_image patches
            idx_select = idx[:num_patches_per_image]
            # ------------------------------------------------------------------

            # get the patches candidate
            for i, id in enumerate(idx_select):
                patch = patches[id]

                # save the patch
                path_save_to_img = os.path.join(
                    path_save_to, filename.split(".")[0] + f"_{i}.tif"
                )
                io.imsave(path_save_to_img, patch, check_contrast=False)
                num_patches += 1
                # cosntruct json dict
                dict_sample = {
                    "image_url": path_save_to_img,
                    "image_details": {
                        "format": "tif",
                        "width": patch_size,
                        "height": patch_size,
                    },
                    "label": class_name,
                }
                json.dump(dict_sample, json_file)
                json_file.write("\n")
pbar.close()
# close the json file
json_file.close()
print("-" * 80)
print(f"[INFO] Total number of patches: {num_patches}")
