"""
Generate the text information used for training.
"""

import pandas, os, tqdm


# path_save_to = "text/dataset_text.txt" # all the information
# path_save_to = "text/dataset_text_TSpixel.txt" # only task, structure, and input/output pixel size
# path_save_to = "text/dataset_text_TSmicro.txt" # only task, structure, and input/output microscope
path_save_to = "text/dataset_text_TS.txt"  # only task, structure


# ------------------------------------------------------------------------------
path_dataset_xlx = "dataset_train_transformer.xlsx"
datasets_frame = pandas.read_excel(path_dataset_xlx, sheet_name="64x64")
num_patches = list(datasets_frame["number of patches"])
num_datset = len(num_patches)
print(num_datset)

text_parts = [
    "task#",
    "sample",
    "structure#",
    "fluorescence indicator",
    "input microscope",
    "input microscope-device",
    "input pixel size",
    "target microscope",
    "target microscope-device",
    "target pixel size",
]

text_data = datasets_frame[text_parts]

# num_datset = 10

pbar = tqdm.tqdm(total=num_datset, ncols=100, desc="GENERATE TEXT")
with open(path_save_to, "w") as text_file:
    for i in range(num_datset):
        # conbine text
        # text_single = "Task: {}; sample: {}; structure: {}; fluorescence indicator: {}; input microscope: {}; input pixel size: {}; target microscope: {}; target pixel size: {}.\n".format(
        #     text_data["task#"][i],
        #     text_data["sample"][i],
        #     text_data["structure#"][i],
        #     text_data["fluorescence indicator"][i],
        #     text_data["input microscope"][i],
        #     text_data["input pixel size"][i],
        #     text_data["target microscope"][i],
        #     text_data["target pixel size"][i],
        # )

        # text_single = "Task: {}; structure: {}; input pixel size: {}; target pixel size: {}.\n".format(
        #     text_data["task#"][i],
        #     text_data["structure#"][i],
        #     text_data["input pixel size"][i],
        #     text_data["target pixel size"][i],
        # )

        text_single = "Task: {}; structure: {}.\n".format(
            text_data["task#"][i],
            text_data["structure#"][i],
        )

        # text_single = "Task: {}; structure: {}; input microscope: {}; target microscope: {}.\n".format(
        #     text_data["task#"][i],
        #     text_data["structure#"][i],
        #     text_data["input microscope-device"][i],
        #     text_data["target microscope-device"][i],
        # )

        text_file.write(text_single)
        pbar.update(1)
pbar.close()
