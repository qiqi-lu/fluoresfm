"""
Extract task, structure, and microscope metadata from the trianing and testing
datasets to construct dictionary used for structural categorical embedding.

Each task, each structure, and each microscope will be assigned a unique id.

"""

import pandas

excel_train = "dataset_train_transformer-v2.xlsx"
excel_test = "dataset_test-v2.xlsx"

keys = ["task#", "structure#", "input microscope-device", "target microscope-device"]

# ------------------------------------------------------------------------------
all_items = []
# get the dictionary
for key in keys:
    for excel in [excel_train, excel_test]:
        df = pandas.read_excel(excel)
        items = df[key]
        items = items.unique()
        items = items.tolist()
        all_items.extend(items)
        print("-" * 50)
        print("[INFO] {} - {}".format(key, excel))
        print(items)

print("-" * 80)
print("[INFO] Final dictionary:")
# get unique items from the list and keep the order of items
all_items = list(dict.fromkeys(all_items))
# print item with index
for i, item in enumerate(all_items):
    print("{} - {}".format(i, item))

# print the list
print("-" * 80)
print("[INFO] Final dictionary:")
print(all_items)
print("[WARNING] Copy the list above and paste it into constants.py")
