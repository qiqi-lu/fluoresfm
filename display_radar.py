import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


# load results
dataset_names = [
    # "CCPs_noise_level_1",
    "CCPs_noise_level_9",
    # "ER_noise_level_1",
    "ER_noise_level_6",
    # "F_actin_noise_level_1",
    "F_actin_noise_level_12",
    # "MTs_noise_level_1",
    "MTs_noise_level_9",
]

all_met = []
for data in dataset_names:
    met = []
    for m in ["PSNR", "SSIM", "ZNCC"]:
        df = pd.read_excel(
            os.path.join("outputs", "unet_c", data, "metrics.xlsx"), sheet_name=m
        )
        met.append(list(df.mean())[1:])
    all_met.append(met)
all_met = np.array(all_met)

headers = df.columns.values.tolist()[1:]

fig, axes = plt.subplots(
    nrows=1,
    ncols=3,
    figsize=(9, 3),
    dpi=300,
    subplot_kw=dict(polar=True),
    constrained_layout=True,
)

num_vars = len(dataset_names)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
for i, m in enumerate(["PSNR", "SSIM", "ZNCC"]):
    for j, meth in enumerate(headers):
        axes[i].plot(
            angles + angles[:1],
            list(all_met[:, i, j]) + list(all_met[:, i, j])[:1],
            # color="red",
            linewidth=1,
        )

for i in [0, 1, 2]:
    axes[i].set_ylim([all_met[:, i].min() * 0.9, all_met[:, i].max() * 1.0])


plt.savefig("tmp.png")
