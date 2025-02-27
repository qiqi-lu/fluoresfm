import skimage.io as io
import matplotlib.pyplot as plt
import utils.data as utils_data
import numpy as np
import scipy.optimize as optim


psf_path = "E:\qiqilu\datasets\SimuMix\psf\BW_1.5_1.4_100_100_100\\404.tif"

psf = io.imread(psf_path)
psf_size = 51
psf = utils_data.center_crop(psf, size=(psf_size, psf_size, psf_size), verbose=True)
print(psf.shape)

x_ticks = np.linspace(start=-(psf_size // 2), stop=psf_size // 2, num=psf_size)


x_shift_0 = np.array([1, 1, 1, 1, 1, 1, 1, 1])


std = [1.0, 0.5]
y = psf[15, 25, :]
# y = y / y.max()
# y = np.random.normal(y, scale=0.1)
# y = np.maximum(y, 0.0)
basis_kernel = np.exp(-(x_ticks**2) / (2 * std[0] ** 2))


x_ticks = np.linspace(start=-(psf_size // 2), stop=psf_size // 2, num=psf_size)


def curve(x, std):
    size = x.shape[0]
    kernels = x[: size // 2][:, None] * np.exp(
        -((x_ticks - x[size // 2 :][:, None]) ** 2) / (2 * std**2)
    )

    c = np.sum(kernels, axis=0)
    return c


def curve2(x, std):
    size = x.shape[0]
    x1 = x[: size // 2]
    x2 = x[size // 2 :]

    kernels1 = curve(x1, std[0])
    kernels2 = curve(x2, std[1])

    return kernels1 + kernels2


def func(x):
    est = curve2(x, std)
    error = 0.5 * np.mean((est - y) ** 2)

    return error


# x0 = list(np.ones(shape=psf_size // 2 + 1))
# x0.extend(
#     list(np.linspace(start=-(psf_size // 2), stop=psf_size // 2, num=psf_size // 2 + 1))
# )

x0 = list(np.ones(shape=psf_size // 4 + 1))
x0.extend(list(x_ticks[1::4]))
x0.extend(x0)

# x0 = list(np.ones(shape=10))
# x0.extend(list((0,) * 10))

bound = list(((0.0, None),) * (len(x0) // 2))
bound.extend(list(((None, None),) * (len(x0) // 2)))

# x0 = np.array([1, 1, 0, 0])
res = optim.minimize(func, x0, bounds=bound)
# res = optim.minimize(func, x0)
print(res)
print(res.x)


fig, axis = plt.subplots(nrows=2, ncols=2, figsize=(6, 6), dpi=300)
axis[0, 0].plot(x_ticks, y)
axis[0, 0].plot(x_ticks, curve2(res.x, std))
axis[0, 1].plot(x_ticks, basis_kernel)

plt.savefig("tmp.png")
