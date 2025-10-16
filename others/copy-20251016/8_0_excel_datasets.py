"""
Collect all the information of the internal datasets, external datasets, and
datasets used for model fine-tuning, segmentation evaluation.
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

# get all the datasets id

id_datasets = {}
tasks = ["sr", "dcv", "dn"]

print("-" * 80)
# internal datasets
id_datasets["internal"] = []
for task in tasks:
    for id in dataset_names_all["internal_dataset"][task]:
        id_datasets["internal"].append(id)
print(f"Number of internal datasets: {len(id_datasets['internal'])}")

# internal datasets (radar)
id_datasets["internal_radar"] = []
for id in dataset_names_radar["internal_dataset"]:
    id_datasets["internal_radar"].append(id)
print(f"Number of internal datasets (radar): {len(id_datasets['internal_radar'])}")

# external datasets
id_datasets["external"] = []
for task in tasks:
    for id in dataset_names_all["external_dataset"][task]:
        id_datasets["external"].append(id)
print(f"Number of external datasets: {len(id_datasets['external'])}")

# external datasets (radar)
id_datasets["external_radar"] = []
for id in dataset_names_radar["external_dataset"]:
    id_datasets["external_radar"].append(id)
print(f"Number of external datasets (radar): {len(id_datasets['external_radar'])}")

# datasets used for fine-tuning
id_datasets["finetune"] = []
for key, value in datasets_finetune.items():
    for id in value:
        id_datasets["finetune"].append(id)
print(f"Number of datasets used for fine-tuning: {len(id_datasets['finetune'])}")

# datasets used for segmentation evaluation
id_datasets["segmentation"] = []
for id in datasets_seg_show:
    id_datasets["segmentation"].append(id)
print(
    f"Number of datasets used for segmentation evaluation: {len(id_datasets['segmentation'])}"
)

# ------------------------------------------------------------------------------
print("-" * 80)
# generate information table
# load informaiton excel file
path_test_excel = os.path.join("dataset_test-v2.xlsx")
df_test = pandas.read_excel(path_test_excel)
# create a new excel file to save the information
path_info_excel = os.path.join("results", "dataset_info.xlsx")

columns_titles = [
    "id",
    "seg_model",
    "task",
    "structure",
    "path_index",
    "path_lr",
    "path_hr",
]

# conbine the list in the id_datasets into a single list
all_id = []
for key, value in id_datasets.items():
    for id in value:
        if id not in all_id:
            all_id.append(id)
print(f"Number of datasets: {len(all_id)}")

info_table_titles = [
    "ID",
    "IN",
    "IN (Fig. 2a)",
    "EX",
    "FT",
    "SEG",
    "Task",
    "Imaging object",
    "Image size (RAW)",
    "Image size (GT)",
    "n",
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

(
    table_ids,
    table_in,
    table_in_radar,
    table_ex,
    table_ft,
    table_seg,
    table_task,
    table_obj,
    table_size_raw,
    table_size_gt,
    table_n,
    table_task_num,
    table_sample,
    table_structure_num,
    table_fluorescence_indicator,
    table_input_microscope_device,
    table_input_microscope_params,
    table_input_pixel_size,
    table_target_microscope_device,
    table_target_microscope_params,
    table_target_pixel_size,
) = ([], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [])

pbar = tqdm.tqdm(total=len(all_id), desc="Collecting information", ncols=80)
for id in all_id:
    # get the information of the dataset
    df = df_test[df_test["id"] == id].iloc[0]
    # get the information of the dataset
    table_ids.append(df["id"])
    if df["id"] in id_datasets["internal"]:
        # the id in the list
        id_in_list = id_datasets["internal"].index(df["id"])
        table_in.append(id_in_list)
    else:
        table_in.append("")

    if df["id"] in id_datasets["internal_radar"]:
        # the id in the list
        id_in_list = id_datasets["internal_radar"].index(df["id"])
        table_in_radar.append(id_in_list)
    else:
        table_in_radar.append("")

    if df["id"] in id_datasets["external"]:
        # the id in the list
        id_in_list = id_datasets["external"].index(df["id"])
        table_ex.append(id_in_list)
    else:
        table_ex.append("")

    if df["id"] in id_datasets["finetune"]:
        # the id in the list
        id_in_list = id_datasets["finetune"].index(df["id"])
        table_ft.append(id_in_list)
    else:
        table_ft.append("")

    if df["id"] in id_datasets["segmentation"]:
        # the id in the list
        id_in_list = id_datasets["segmentation"].index(df["id"])
        table_seg.append(id_in_list)
    else:
        table_seg.append("")

    table_task.append(df["task"].upper())
    table_obj.append(df["structure"])

    # get all the filenames used for test in current dataset
    path_index = win2linux(df["path_index"])
    filenames = read_txt(path_index)
    if len(filenames) > 8:
        table_n.append(8)
    else:
        table_n.append(len(filenames))

    # get the image size
    path_img_raw = win2linux(df["path_lr"])
    img = skio.imread(os.path.join(path_img_raw, filenames[0]))
    table_size_raw.append(f"{img.shape[-2]} x {img.shape[-1]}")

    # get the image size
    if df["path_hr"] != "Unknown":
        path_img = win2linux(df["path_hr"])
        img = skio.imread(os.path.join(path_img, filenames[0]))
        table_size_gt.append(f"{img.shape[-2]} x {img.shape[-1]}")
    else:
        table_size_gt.append("N/A")

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
info_table["Image size (RAW)"] = table_size_raw
info_table["Image size (GT)"] = table_size_gt
info_table["n"] = table_n
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


info_table.to_excel(path_info_excel, index=False)
