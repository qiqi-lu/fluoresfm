"""
[text -----> embedding]
Convert text information to embedding.
!! Use 0_generate_text.py to generate text information first.
Embedding each line in the `txt` file to a numpy array, and saved as `npy` file.
"""

# from models.clip_embedder import CLIPTextEmbedder
from models.biomedclip_embedder import BiomedCLIPTextEmbedder
import torch, os, tqdm
import numpy as np

# ------------------------------------------------------------------------------
finetune = True
# ------------------------------------------------------------------------------

text_type = ("ALL", 160)
# text_type = ("TSpixel", 77)
# text_type = ("TSmicro", 77)
# text_type = ("TS", 77)
# text_type = ("T", 77)

# ------------------------------------------------------------------------------
device = torch.device("cuda:0")
path_text = os.path.join("text", "v2")

if finetune == True:
    path_text += "-finetune"

path_dataset_txt = os.path.join(path_text, f"dataset_text_{text_type[0]}.txt")
context_length = text_type[1]
path_save_to = path_dataset_txt.split(".")[0] + "_" + str(context_length)
os.makedirs(path_save_to, exist_ok=True)

print("-" * 50)
print("Path dataset txt:", path_dataset_txt)
print("Context length:", context_length)
print("Path save to:", path_save_to)

# ------------------------------------------------------------------------------
# load dataset text
with open(path_dataset_txt) as f:
    dataset_text = f.read().splitlines()
# pop the last line if it is \n.
if dataset_text[-1] == "":
    dataset_text.pop(-1)

num_dataset = len(dataset_text)
print("Number of datasets:", num_dataset)

# ------------------------------------------------------------------------------
# load embedder
# embedder = CLIPTextEmbedder(
#     version="openai/clip-vit-large-patch14", device=device, max_length=77
# )

embedder = BiomedCLIPTextEmbedder(
    path_json="checkpoints/clip//biomedclip/open_clip_config.json",
    path_bin="checkpoints/clip//biomedclip/open_clip_pytorch_model.bin",
    context_length=context_length,
    device=device,
)

embedder.eval()

# ------------------------------------------------------------------------------
pbar = tqdm.tqdm(total=num_dataset, ncols=100, desc="EMBEDDING")
for i in range(num_dataset):
    prompt = dataset_text[i]
    cond = embedder(prompt)
    np.save(os.path.join(path_save_to, f"{i}.npy"), cond.cpu().detach().numpy())
    pbar.update(1)
pbar.close()
