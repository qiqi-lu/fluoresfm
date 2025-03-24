from huggingface_hub import login

login(token="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")

import datasets

local_dir = "HEST-1k/zip"  # hest will be dowloaded to this folder

ids_to_query = ["TENX96", "TENX99"]  # list of ids to query

list_patterns = [f"*{id}[_.]**" for id in ids_to_query]
dataset = datasets.load_dataset(
    "MahmoodLab/hest",
    cache_dir=local_dir,
    patterns=list_patterns,
    trust_remote_code=True,
)
