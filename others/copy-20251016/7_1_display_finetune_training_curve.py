"""
Plot the training curve when finetune the model.
The data is saved in the tensorbaod log file.
"""

import os, pandas
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from utils.data import win2linux
import matplotlib.pyplot as plt
import numpy as np

# Path to your TensorBoard log file (or directory containing multiple events)
path_logs = [
    "checkpoints\conditional\\finetune\\unet_sd_c_mae_bs_16_lr_1e-05_all_newnorm_ALL-v2-160-res1-att0123-ft-in-out-biotisr-mt-sr-1\log\events.out.tfevents.1749197814.DESKTOP-A157IVB.2901001.0",
    "checkpoints\conditional\\finetune\\unifmir_mae_bs_1_lr_0.0001_newnorm-v2-ft-biotisr-mt-sr-1\log\events.out.tfevents.1749483893.DESKTOP-A157IVB.1895265.0",
    "checkpoints\conditional\\finetune\dfcan_mae_bs_16_lr_0.0001_newnorm-v2-ft-biotisr-mt-sr-1\log\events.out.tfevents.1749220589.DESKTOP-A157IVB.3588779.0",
    "checkpoints\conditional\\finetune\care_mae_bs_16_lr_0.0001_newnorm-v2-ft-biotisr-mt-sr-1\log\events.out.tfevents.1749220555.DESKTOP-A157IVB.3583895.0",
]

path_figure = os.path.join("results", "figures", "analysis", "finetune")

step_all, values_all = [], []
tag = "mse_val"
for path_log in path_logs:
    path_log = win2linux(path_log)
    # Load the event file
    event_acc = EventAccumulator(path_log)
    event_acc.Reload()  # Reload data from the log file
    # Get all scalar tags (e.g., "loss", "accuracy")
    scalar_tags = event_acc.Tags()["scalars"]
    scalar_events = event_acc.Scalars(tag)

    # Parse the data (each event contains step, value, and timestamp)
    step, values = [], []
    for event in scalar_events:
        step.append(event.step)
        values.append(event.value)
    print("number of steps:", len(step))
    print("number of values:", len(values))
    step_all.append(step)
    values_all.append(values)

# get the min step
min_step = min([len(step) for step in step_all])

methods_title = ["FluoResFM (ft)", "UniFMIR (ft)", "DFCAN", "CARE"]
methods_color = ["#C23637", "#0068A9", "#647086", "#8E99AB"]

plt.rcParams["svg.fonttype"] = "none"
dict_fig = {"dpi": 300, "constrained_layout": True}
# ------------------------------------------------------------------------------
nr, nc = 1, 1
fig, axes = plt.subplots(nr, nc, figsize=(3 * nc, 3 * nr), **dict_fig)

for i in range(len(methods_title)):
    axes.plot(
        step_all[i][:min_step],
        values_all[i][:min_step],
        label=methods_title[i],
        color=methods_color[i],
    )

axes.set_xlabel("iterations")
axes.set_ylabel("MSE (val)")
axes.set_title("Finetune curve")
axes.legend()

# Save the figure
fig.savefig(os.path.join(path_figure, "training_curve.png"))
fig.savefig(os.path.join(path_figure, "training_curve.svg"))

# save source data
data_save = []
for i in range(len(methods_title)):
    data_save.append(values_all[i][:min_step])
data_save = np.array(data_save)
data_save = data_save.T
df = pandas.DataFrame(data_save, columns=methods_title)
# save to excel
df.to_excel(os.path.join(path_figure, "training_curve.xlsx"), index=False)
