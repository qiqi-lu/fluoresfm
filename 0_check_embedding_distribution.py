"""
Check the embedding of each text.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

path_embedding = os.path.join("text", "v2", "dataset_text_ALL_256")
index = 0

# plot the histogram of embedding
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
emb = np.load(os.path.join(path_embedding, f"{index}.npy"))
axes[0].hist(emb.flatten(), bins=100)
emd_2 = np.load(os.path.join(path_embedding, f"{index+1}.npy"))
axes[1].hist(emd_2.flatten(), bins=100)
plt.savefig("tmp.png")
