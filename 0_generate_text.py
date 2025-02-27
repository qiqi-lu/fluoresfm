import pandas, os, tqdm

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
    "input pixel size",
    "target microscope",
    "target pixel size",
]

text_data = datasets_frame[text_parts]

# num_datset = 10

pbar = tqdm.tqdm(total=num_datset, ncols=100, desc="combining text")
with open("dataset_text.txt", "w") as text_file:
    for i in range(num_datset):
        # conbine text
        text_single = "Task: {}; sample: {}; structure: {}; fluorescence indicator: {}; input microscope: {}; input pixel size: {}; target microscope: {}; target pixel size: {}.\n".format(
            text_data["task#"][i],
            text_data["sample"][i],
            text_data["structure#"][i],
            text_data["fluorescence indicator"][i],
            text_data["input microscope"][i],
            text_data["input pixel size"][i],
            text_data["target microscope"][i],
            text_data["target pixel size"][i],
        )
        text_file.write(text_single)
        pbar.update(1)
pbar.close()
