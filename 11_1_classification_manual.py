"""
Copy all the patches into a new folder for manual classification.
WSL will not refresh the image during code execution.
"""

import os, shutil, json, tqdm, pandas
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

group = "in"
# path_json = os.path.join("data", f"dataset_classfication_test_{group}.json")
# path_save_to = os.path.join("data", f"patches_{group}")
# os.makedirs(path_save_to, exist_ok=True)

# with open(path_json, "r") as json_file:
#     json_list = json_file.readlines()

# print(f"[INFO] Number of patches: {len(json_list)}")
# pbar = tqdm.tqdm(desc="[INFO] Copying patches", total=len(json_list), ncols=80)
# for i_patch, json_str in enumerate(json_list):
#     json_dict = json.loads(json_str)
#     path_image = json_dict["image_url"]
#     path_save = os.path.join(path_save_to, f"{i_patch}.tif")
#     shutil.copy(path_image, path_save)
#     pbar.update(1)
# pbar.close()


path_patches = os.path.join("data", f"patches_{group}")
# get all the file names in the folder end with tif
file_names = os.listdir(path_patches)
file_names = [file_name for file_name in file_names if file_name.endswith(".tif")]

id_show = []
# shuffle the filenames
np.random.seed(0)
np.random.shuffle(file_names)

# get all the filenames id
for i in range(len(file_names)):
    id_image = file_names[i].split(".")[0]
    id_show.append(id_image)

label_manual = [-1] * len(file_names)

dataframe = pandas.DataFrame(columns=["id_true", "label"])
dataframe["id_true"] = id_show
# save the dataframe to a excel file
path_save_excel = os.path.join(path_patches, f"labels.xlsx")

for i in range(len(file_names)):
    # load image
    path_image = os.path.join(path_patches, file_names[i])
    image = io.imread(path_image)

    # show the image
    plt.figure(figsize=(3, 3))
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.title(f"{i}")
    plt.savefig(os.path.join(path_patches, "00show.png"))
    plt.close()

    # get the label
    label = input("Enter the label: ")
    label_manual[i] = label
    dataframe["label"] = label_manual
    dataframe.to_excel(path_save_excel, index=True)
