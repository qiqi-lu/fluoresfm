"""
Collect all the information of
- internal datasets
- external datasets
- datasets used for model fine-tuning
- datasets used for segmentation evaluation
and save the information in an excel file.
"""

import pandas, os
from dataset_analysis import (
    datasets_seg_show,
    dataset_names_all,
    dataset_names_radar,
    datasets_finetune,
)
from utils.data import win2linux, read_txt
import skimage.io as skio
import tqdm
import numpy as np

# get all the datasets id

id_datasets = {}
tasks = ["sr", "dcv", "dn"]

print("-" * 80)
id_datasets["internal"] = []
id_datasets["external"] = []
id_datasets["internal_radar"] = []
id_datasets["external_radar"] = []
id_datasets["finetune"] = []
id_datasets["segmentation"] = []

for task in tasks:
    # internal datasets
    id_datasets["internal"].extend(dataset_names_all["internal_dataset"][task])
    # external datasets
    id_datasets["external"].extend(dataset_names_all["external_dataset"][task])
# internal datasets (radar)
id_datasets["internal_radar"].extend(dataset_names_radar["internal_dataset"])
# external datasets (radar)
id_datasets["external_radar"].extend(dataset_names_radar["external_dataset"])
# datasets used for fine-tuning
for key, value in datasets_finetune.items():
    id_datasets["finetune"].extend(value)
id_datasets["segmentation"].extend(datasets_seg_show)

# print the number of each group
print("-" * 80)
for key, value in id_datasets.items():
    print(f"[INFO] Number of {key}: {len(value)}")

# ------------------------------------------------------------------------------
print("-" * 80)
# generate information table
# load informaiton excel file
path_excel_test = os.path.join("dataset_test-v2.xlsx")
path_excel_train = os.path.join("dataset_train_transformer-v2.xlsx")
path_excel_info = os.path.join("results", "dataset_info.xlsx")

df_test = pandas.read_excel(path_excel_test)
df_train = pandas.read_excel(path_excel_train, sheet_name="64x64")
df_finetune = pandas.read_excel(path_excel_train, sheet_name="64x64-finetune")

# conbine the list in the id_datasets into a single list
all_id = []
for key, value in id_datasets.items():
    for id in value:
        if id not in all_id:
            all_id.append(id)
print(f"[INFO] Number of datasets: {len(all_id)}")

info_table_titles = [
    "ID",
    "IN",
    "IN (Fig. 2a)",
    "EX",
    "FT",
    "SEG",
    "Task",
    "Imaging object",
    "Image size-train (RAW)",
    "Image size-train (GT)",
    "n-train",
    "Image size-test (RAW)",
    "Image size-test (GT)",
    "n-test",
    "task#",
    "sample",
    "structure#",
    "fluorescence indicator",
    "input microscope-device",
    "input microscope-params",
    "input pixel size",
    "target microscope-device",
    "target microscope-params",
    "target pixel size",
]
info_table = pandas.DataFrame(columns=info_table_titles)

table_ids = []
table_in = []
table_in_radar = []
table_ex = []
table_ft = []
table_seg = []
table_task = []
table_obj = []
table_size_raw = []
table_size_gt = []
table_n = []
table_size_raw_train = []
table_size_gt_train = []
table_n_train = []
table_task_num = []
table_sample = []
table_structure_num = []
table_fluorescence_indicator = []
table_input_microscope_device = []
table_input_microscope_params = []
table_input_pixel_size = []
table_target_microscope_device = []
table_target_microscope_params = []
table_target_pixel_size = []


pbar = tqdm.tqdm(total=len(all_id), desc="[INFO] Collecting information", ncols=80)
for id in all_id:
    # get the information of the dataset
    df = df_test[df_test["id"] == id].iloc[0]

    # check which group the current dataset belongs to -------------------------
    table_ids.append(df["id"])
    if df["id"] in id_datasets["internal"]:
        id_in_list = id_datasets["internal"].index(df["id"])
        table_in.append(id_in_list)
    else:
        table_in.append("")

    if df["id"] in id_datasets["internal_radar"]:
        id_in_list = id_datasets["internal_radar"].index(df["id"])
        table_in_radar.append(id_in_list)
    else:
        table_in_radar.append("")

    if df["id"] in id_datasets["external"]:
        id_in_list = id_datasets["external"].index(df["id"])
        table_ex.append(id_in_list)
    else:
        table_ex.append("")

    if df["id"] in id_datasets["finetune"]:
        id_in_list = id_datasets["finetune"].index(df["id"])
        table_ft.append(id_in_list)
    else:
        table_ft.append("")

    if df["id"] in id_datasets["segmentation"]:
        id_in_list = id_datasets["segmentation"].index(df["id"])
        table_seg.append(id_in_list)
    else:
        table_seg.append("")

    # --------------------------------------------------------------------------
    table_task.append(df["task"].upper())
    table_obj.append(df["structure"])

    # --------------------------------------------------------------------------
    # count the number of image in current dataset used for test (higerh than 8
    # will be set to 8)
    path_index = win2linux(df["path_index"])
    filenames = read_txt(path_index)
    if len(filenames) > 8:
        table_n.append(8)
    else:
        table_n.append(len(filenames))

    # --------------------------------------------------------------------------
    # get the image size and number of image in test set
    # get the image size (LR)
    path_img_raw = win2linux(df["path_lr"])
    # only read the first image
    img = skio.imread(os.path.join(path_img_raw, filenames[0]))
    table_size_raw.append(f"{img.shape[-2]} x {img.shape[-1]}")

    # get the image size (HR)
    if df["path_hr"] != "Unknown":
        path_img = win2linux(df["path_hr"])
        img = skio.imread(os.path.join(path_img, filenames[0]))
        table_size_gt.append(f"{img.shape[-2]} x {img.shape[-1]}")
    else:
        table_size_gt.append("")
    # --------------------------------------------------------------------------
    # get the image size and number of images in training set
    # check if the dataset is in the training set
    if id in df_train["id"].values or id in df_finetune["id"].values:
        if id in df_train["id"].values:
            dftr = df_train[df_train["id"] == id].iloc[0]
        else:
            dftr = df_finetune[df_finetune["id"] == id].iloc[0]

        # get the image size (LR)
        if dftr["path_lr_raw"] == "Unknown":
            table_size_raw_train.append("")
            table_size_gt_train.append("")
            table_n_train.append("")
        else:
            path_img_raw_train = win2linux(dftr["path_lr_raw"])

            # get all the filenames in the folder end with .tif
            filenames_train = os.listdir(path_img_raw_train)
            filenames_train = [
                filename for filename in filenames_train if filename.endswith(".tif")
            ]

            num_images_train = len(filenames_train)

            # get the image size (LR)
            img = skio.imread(os.path.join(path_img_raw_train, filenames_train[0]))
            img = np.squeeze(img)

            if len(img.shape) == 3:
                # if the image is a stack of images, sum of all slices and samples
                # recalculate the number of image
                num_images_train = 0
                for filename in filenames_train:
                    img_tmp = skio.imread(os.path.join(path_img_raw_train, filename))
                    img_tmp = np.squeeze(img_tmp)
                    num_sclice = img_tmp.shape[0]
                    num_images_train += num_sclice

            size_raw = f"{img.shape[-2]} x {img.shape[-1]}"
            if dftr["task"] == "sr":
                if dftr["sf_lr"] == 1:
                    size_gt = f"{img.shape[-2]} x {img.shape[-1]}"
                else:
                    size_gt = f"{img.shape[-2]*2} x {img.shape[-1]*2}"
            else:
                size_gt = f"{img.shape[-2]} x {img.shape[-1]}"

            table_size_raw_train.append(size_raw)
            table_size_gt_train.append(size_gt)
            table_n_train.append(num_images_train)
    else:
        table_size_raw_train.append("")
        table_size_gt_train.append("")
        table_n_train.append("")

    # --------------------------------------------------------------------------
    # get the information of the dataset
    table_task_num.append(df["task#"])
    table_sample.append(df["sample"])
    table_structure_num.append(df["structure#"])
    table_fluorescence_indicator.append(df["fluorescence indicator"])
    table_input_microscope_device.append(df["input microscope-device"])
    table_input_microscope_params.append(df["input microscope-params"])
    table_input_pixel_size.append(df["input pixel size"])
    table_target_microscope_device.append(df["target microscope-device"])
    table_target_microscope_params.append(df["target microscope-params"])
    table_target_pixel_size.append(df["target pixel size"])

    pbar.update(1)
pbar.close()

info_table["ID"] = table_ids
info_table["IN"] = table_in
info_table["IN (Fig. 2a)"] = table_in_radar
info_table["EX"] = table_ex
info_table["FT"] = table_ft
info_table["SEG"] = table_seg
info_table["Task"] = table_task
info_table["Imaging object"] = table_obj
info_table["Image size-train (RAW)"] = table_size_raw_train
info_table["Image size-train (GT)"] = table_size_gt_train
info_table["n-train"] = table_n_train
info_table["Image size-test (RAW)"] = table_size_raw
info_table["Image size-test (GT)"] = table_size_gt
info_table["n-test"] = table_n
info_table["task#"] = table_task_num
info_table["sample"] = table_sample
info_table["structure#"] = table_structure_num
info_table["fluorescence indicator"] = table_fluorescence_indicator
info_table["input microscope-device"] = table_input_microscope_device
info_table["input microscope-params"] = table_input_microscope_params
info_table["input pixel size"] = table_input_pixel_size
info_table["target microscope-device"] = table_target_microscope_device
info_table["target microscope-params"] = table_target_microscope_params
info_table["target pixel size"] = table_target_pixel_size


info_table.to_excel(path_excel_info, index=False)
