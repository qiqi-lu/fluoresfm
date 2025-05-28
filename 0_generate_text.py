"""
[information in xlsx -----> text]
Generate the text information used for training.
"""

import pandas, os, tqdm

# path_dataset_xlx = "dataset_train_transformer.xlsx"
path_dataset_xlx = "dataset_train_transformer-v2.xlsx"

# text_type = "ALL"  # all the information
# text_type = "TSpixel"  # only task, structure, and input/output pixel size
# text_type = "TSmicro"  # only task, structure, and input/output microscope
# text_type = "TS"  # only task, structure
text_type = "T"  # only task

# ------------------------------------------------------------------------------
path_save_to = os.path.join("text", "v2", f"dataset_text_{text_type}.txt")
print("Path dataset info:", path_dataset_xlx)
print("Path save to:", path_save_to)

# ------------------------------------------------------------------------------
# get dataset information
print("-" * 50)
datasets_frame = pandas.read_excel(path_dataset_xlx, sheet_name="64x64")
num_patches = list(datasets_frame["number of patches"])
num_datset = len(num_patches)
print("Number of dataset:", num_datset)

text_parts = [
    "task#",
    "sample",
    "structure#",
    "fluorescence indicator",
    "input microscope",
    "input microscope-device",
    "input microscope-params",
    "input pixel size",
    "target microscope",
    "target microscope-device",
    "target microscope-params",
    "target pixel size",
]

text_data = datasets_frame[text_parts]

# num_datset = 10

# ------------------------------------------------------------------------------
# generate text
pbar = tqdm.tqdm(total=num_datset, ncols=100, desc="GENERATE TEXT")
with open(path_save_to, "w") as text_file:
    for i in range(num_datset):
        # conbine text
        if text_type == "ALL":
            text_single = "Task: {}; sample: {}; structure: {}; fluorescence indicator: {}; input microscope: {}; input pixel size: {}; target microscope: {}; target pixel size: {}.\n".format(
                text_data["task#"][i],
                text_data["sample"][i],
                text_data["structure#"][i],
                text_data["fluorescence indicator"][i],
                f'{text_data["input microscope-device"][i]} {text_data["input microscope-params"][i]}',
                text_data["input pixel size"][i],
                f'{text_data["target microscope-device"][i]} {text_data["target microscope-params"][i]}',
                text_data["target pixel size"][i],
            )
        elif text_type == "TSpixel":
            text_single = "Task: {}; structure: {}; input pixel size: {}; target pixel size: {}.\n".format(
                text_data["task#"][i],
                text_data["structure#"][i],
                text_data["input pixel size"][i],
                text_data["target pixel size"][i],
            )
        elif text_type == "TSmicro":
            text_single = "Task: {}; structure: {}; input microscope: {}; target microscope: {}.\n".format(
                text_data["task#"][i],
                text_data["structure#"][i],
                text_data["input microscope-device"][i],
                text_data["target microscope-device"][i],
            )
        elif text_type == "TS":
            text_single = "Task: {}; structure: {}.\n".format(
                text_data["task#"][i],
                text_data["structure#"][i],
            )
        elif text_type == "T":
            text_single = "Task: {}.\n".format(
                text_data["task#"][i],
            )
        else:
            raise ValueError("Invalid text type.")

        text_file.write(text_single)
        pbar.update(1)
pbar.close()
