"""
Generate the text information used for training.
The information is saved in the xlsx file.
[information in xlsx -----> text]
"""

import pandas, os, tqdm

# ------------------------------------------------------------------------------
finetune = False
# finetune = True  # generate the text for finetune datasets
# ------------------------------------------------------------------------------

path_dataset_xlx = "dataset_train_transformer-v2.xlsx"

# text_type = "ALL"  # all the information
# text_type = "ALL_wot"  # all the information without the information about the target
# text_type = "TSpixel"  # only task, structure, and input/output pixel size
# text_type = "TSmicro"  # only task, structure, and input/output microscope
# text_type = "TS"  # only task, structure
# text_type = "T"  # only task
text_type = (
    "STRUCTURAL-TSMM"  # structural prompts [task, structure, input/output microscope]
)

# ------------------------------------------------------------------------------
path_text = os.path.join("text", "v2")
if finetune:
    path_text = path_text + "-finetune"

os.makedirs(path_text, exist_ok=True)
path_save_to = os.path.join(path_text, f"dataset_text_{text_type}.txt")

print("-" * 50)
print(f"[INFO] Path dataset info: {path_dataset_xlx}")
print(f"[INFO] Path save to:      {path_save_to}")

# ------------------------------------------------------------------------------
# get dataset information
print("-" * 50)
if finetune:
    print("[INFO] Finetune")
    datasets_frame = pandas.read_excel(path_dataset_xlx, sheet_name="64x64-finetune")
else:
    datasets_frame = pandas.read_excel(path_dataset_xlx, sheet_name="64x64")

num_datset = len(datasets_frame)
print(f"[INFO] Number of dataset: {num_datset}")

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
pbar = tqdm.tqdm(total=num_datset, ncols=100, desc="[INFO] GENERATE TEXT")
with open(path_save_to, "w") as text_file:
    # generate text for each dataset
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
        elif text_type == "ALL_wot":  # exclude the informaiton about the target
            text_single = "Task: {}; sample: {}; structure: {}; fluorescence indicator: {}; input microscope: {}; input pixel size: {}.\n".format(
                text_data["task#"][i],
                text_data["sample"][i],
                text_data["structure#"][i],
                text_data["fluorescence indicator"][i],
                f'{text_data["input microscope-device"][i]} {text_data["input microscope-params"][i]}',
                text_data["input pixel size"][i],
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
        elif text_type == "STRUCTURAL-TSMM":
            text_single = "{};{};{};{}\n".format(
                text_data["task#"][i],
                text_data["structure#"][i],
                text_data["input microscope-device"][i],
                text_data["target microscope-device"][i],
            )
        else:
            raise ValueError("[ERROR] Invalid text type.")

        text_file.write(text_single)
        pbar.update(1)
pbar.close()


print("[INFO] Next is to use embeder to generate the embeddings.")
