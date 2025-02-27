from models.clip_embedder import CLIPTextEmbedder
from models.biomedclip_embedder import BiomedCLIPTextEmbedder
import torch
import utils.data as utils_data
import os
import numpy as np
import pandas, tqdm

# ------------------------------------------------------------------------------
path_dataset_txt = "dataset_text.txt"
with open(path_dataset_txt) as f:
    dataset_text = f.read().splitlines()

num_dataset = len(dataset_text)
print("Number of datasets:", num_dataset)

device = torch.device("cuda:0")
# device = torch.device("cpu")

# ------------------------------------------------------------------------------
# load embedder
# embedder = CLIPTextEmbedder(
#     version="openai/clip-vit-large-patch14", device=torch.device("cpu"), max_length=77
# )

embedder = BiomedCLIPTextEmbedder(
    path_json="checkpoints/clip//biomedclip/open_clip_config.json",
    path_bin="checkpoints/clip//biomedclip/open_clip_pytorch_model.bin",
    # context_length=192,
    context_length=256,
    device=device,
)

embedder.eval()

# num_dataset = 10
# ------------------------------------------------------------------------------
pbar = tqdm.tqdm(total=num_dataset, ncols=100, desc="embedding")
for i in range(num_dataset):
    prompt = dataset_text[i]
    cond = embedder(prompt)  # (1, 77, 768)/(1, 256, 768)
    np.save(os.path.join("dataset_text", f"{i}.npy"), cond.cpu().detach().numpy())
    # print("-" * 98)
    # print(dataset_path[i])
    # print(prompt)
    # print(cond.shape)
    pbar.update(1)
pbar.close()
