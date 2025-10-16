import torch
import numpy as np
import models.PSFmodels as PSFModel
import utils.evaluation as utils_eva
from methods.convolution import convolution as conv_fft
import utils.loss_functions as loss_func
import os
import utils.data as utils_data
import skimage.io as io
import matplotlib.pyplot as plt


params = {
    "dataset_name": "SimuMix_457",
    "path_dataset_lr": "E:\qiqilu\datasets\SimuMix\\raw\data_128_128_128_gauss_0_poiss_1_ratio_1_457",
    # "dataset_name": "SimuMix_528",
    # "path_dataset_lr": "E:\qiqilu\datasets\SimuMix\\raw\data_128_128_128_gauss_0_poiss_1_ratio_1_528",
    # "dataset_name": "SimuMix_751",
    # "path_dataset_lr": "E:\qiqilu\datasets\SimuMix\\raw\data_128_128_128_gauss_0_poiss_1_ratio_1_751",
    "path_dataset_hr": "E:\qiqilu\datasets\SimuMix\gt",
    "p_low": 0.0,
    "p_high": 0.9,
}

path_results = os.path.join("outputs", params["dataset_name"])
path_figures = os.path.join("outputs", "figures")
utils_data.make_path(path_figures)

norm = utils_data.NormalizePercentile(p_low=params["p_low"], p_high=params["p_high"])
filenames = utils_data.read_txt(os.path.join(params["path_dataset_lr"], "test.txt"))

idx = 2
img_hr = io.imread(os.path.join(params["path_dataset_hr"], "images", filenames[idx]))
img_lr = io.imread(os.path.join(params["path_dataset_lr"], "images", filenames[idx]))
psf_lr = io.imread(os.path.join(params["path_dataset_lr"], "psf.tif"))
img_hr = norm(img_hr)
img_lr = norm(img_lr)
img_lr = utils_eva.linear_transform(img_true=img_hr, img_test=img_lr)

# ------------------------------------------------------------------------------
loss_map = np.zeros(shape=(21, 21))

range_a = np.linspace(start=2, stop=8, num=21)
range_b = np.linspace(start=0.5, stop=1.5, num=21)
gm_g = PSFModel.BWModel(
    kernel_size=(127, 127, 127),
    kernel_norm=True,
    num_integral=1000,
    over_sampling=2,
    pixel_size_z=1,
)

# range_a = np.linspace(start=0.1, stop=10, num=21)
# range_b = np.linspace(start=0.1, stop=10, num=21)
# gm_g = PSFModel.GaussianModel(
#     kernel_size=(127, 127, 127), kernel_norm=True, num_params=2
# )

# for i, i_val in enumerate(range_a):
#     print(i)
#     for j, j_val in enumerate(range_b):
#         # init
#         psf_init = gm_g(torch.tensor([i_val, j_val]).reshape(shape=(1, 1, 2)))[0, 0]
#         psf_init = psf_init.clone().detach().numpy()
#         img_reblur_init = conv_fft(img_hr, psf_init, padding_mode="constant")
#         img_reblur_init = utils_eva.linear_transform(
#             img_true=img_hr, img_test=img_reblur_init
#         )
#         loss = loss_func.ZNCC(
#             torch.tensor(img_lr[None, None]), torch.tensor(img_reblur_init[None, None])
#         )
#         loss_map[i, j] = loss
# np.save(os.path.join(path_figures, "loss_map_bw"), loss_map)

loss_map = np.load(os.path.join(path_figures, "loss_map_bw.npy"))
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(3, 3), dpi=300)
axes.set_axis_off()
axes.imshow(loss_map)
pos_min = np.argmin(a=loss_map)
print(pos_min // 21, pos_min % 21 - 1)
print(range_a[pos_min // 21], range_b[pos_min % 21 - 1])
axes.plot(pos_min % 21 - 1, pos_min // 21, "*", color="red")
# axes.plot(0.75, 2.5, "*", color="black")
# axes.plot(1.4 / 1.5, 457 / 100 / 3 * 2, "*", color="black")
plt.savefig(os.path.join(path_figures, "loss_map"))
