"""
Emberdding each patches into a vector using BiomedCLIP.
"""

import torch, os, json, tqdm
import numpy as np
from PIL import Image
from models.biomedclip_embedder import BiomedCLIP
from utils.data import win2linux

# path_json = os.path.join("data", "dataset_classfication_train.json")
# path_json = os.path.join("data", "dataset_classfication_test_in.json")
path_json = os.path.join("data", "dataset_classfication_test_ex.json")

# read each in json ------------------------------------------------------------
with open(path_json, "r") as json_file:
    json_list = json_file.readlines()

images_url = []
labels = []
for json_str in json_list:
    json_dict = json.loads(json_str)
    images_url.append(json_dict["image_url"])
    labels.append(json_dict["label"])

# get the unique item, do not use set because the order is important
structure_types = []
for label in labels:
    if label not in structure_types:
        structure_types.append(label)

counts_each_type = []
for label in structure_types:
    counts_each_type.append(labels.count(label))

num_samples = len(images_url)
num_structure_types = len(structure_types)
print("-" * 80)
print("[INFO] Number of samples: {}".format(num_samples))
print("[INFO] Labels: {}".format(structure_types))
print("[INFO] Number of structure types: {}".format(num_structure_types))
print("[INFO] Number of samples each type: {}".format(counts_each_type))
print("-" * 80)

# embadding each image ---------------------------------------------------------
biomedcliper = BiomedCLIP(
    path_json="checkpoints/clip/biomedclip/open_clip_config.json",
    path_bin="checkpoints/clip/biomedclip/open_clip_pytorch_model.bin",
    context_length=160,
    device=torch.device("cuda:0"),
)

images_embedding = []
pbar = tqdm.tqdm(desc="[INFO] Embedding images", total=num_samples, ncols=80)
for url in images_url:
    img = biomedcliper.preprocess(Image.open(win2linux(url)))[None]
    img_embed = biomedcliper.image_embedding(img)
    images_embedding.append(img_embed)
    pbar.update(1)
pbar.close()

images_embedding = torch.cat(images_embedding, dim=0)
images_embedding = images_embedding.detach().cpu().numpy()

print("[INFO] Embedding shape: {}".format(images_embedding.shape))

data = {
    "images_embedding": images_embedding,
    "labels": labels,
    "structure_types": structure_types,
}

# save the embeddings ----------------------------------------------------------
path_save_to = path_json.split(".")[0] + ".npy"
np.save(path_save_to, data, allow_pickle=True)


# load and check ---------------------------------------------------------------
data = np.load(path_save_to, allow_pickle=True).item()
images_embedding = data["images_embedding"]
labels = data["labels"]
structure_types = data["structure_types"]
print("-" * 80)
print("[INFO] Embedding shape: {}".format(images_embedding.shape))
print("[INFO] Number of Labels: {}".format(len(labels)))
print("[INFO] Structure types: {}".format(structure_types))
