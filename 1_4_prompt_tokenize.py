"""
Convert the structural prompt to a tokenized prompt.
Each task, structure, and microscope will be assigned a unique id.

"""

from utils.data import read_txt, IDTokenizer
import os, tqdm
import numpy as np
from constants import task_struc_micro_voc

finetune = False
# finetune = True  # only process the text for the finetune datasets.
# ------------------------------------------------------------------------------

text_type = "STRUCTURAL-TSMM"

# ------------------------------------------------------------------------------
path_text = os.path.join("text", "v2")
if finetune == True:
    path_text += "-finetune"
path_dataset_txt = os.path.join(path_text, f"dataset_text_{text_type}.txt")
path_save_to = path_dataset_txt.split(".")[0] + "_tokenized"
os.makedirs(path_save_to, exist_ok=True)

print("-" * 80)
print(f"[INFO] Path dataset txt: {path_dataset_txt}")
print(f"[INFO] Path save to:     {path_save_to}")
# ------------------------------------------------------------------------------
# load dataset text
dataset_text = read_txt(path_dataset_txt)
num_dataset = len(dataset_text)
print(f"[INFO] Num. of datasets: {num_dataset}")

list_length = len(task_struc_micro_voc)
print(f"[INFO] Vocabulary length: {list_length}")

tokenizer = IDTokenizer(all_tokens=task_struc_micro_voc)

# encode
pbar = tqdm.tqdm(total=num_dataset, ncols=80, desc="[INFO] TOKENIZING")
for i_dataset in range(num_dataset):
    text = dataset_text[i_dataset]
    text_tokenized = tokenizer.encode(text.split(";"))
    text_tokenized = np.array(text_tokenized, dtype=np.int16)  # shape = (n,)
    np.save(os.path.join(path_save_to, f"{i_dataset}.npy"), text_tokenized)
    pbar.update(1)
pbar.close()
