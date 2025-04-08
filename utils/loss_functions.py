import torch


def mae_mse(x, y, scale=(0.5, 0.5)):
    mae = torch.nn.L1Loss()(x, y)
    mse = torch.nn.MSELoss()(x, y)
    loss = scale[0] * mae + scale[1] * mse
    return loss


def ZNCC(x, y, mean_one=False):
    if len(x.shape) == 5:  # 3d
        dims = (2, 3, 4)
    elif len(x.shape) == 4:
        dims = (2, 3)
    else:
        dims = None

    mu_x = torch.mean(x, dim=dims, keepdim=True)
    mu_y = torch.mean(y, dim=dims, keepdim=True)
    sigma_x = torch.std(x, dim=dims, keepdim=True)
    sigma_y = torch.std(y, dim=dims, keepdim=True)

    zncc = torch.mean((x - mu_x) * (y - mu_y) / (sigma_x * sigma_y))

    if mean_one:
        # mean_loss = torch.mean(torch.square(mu_x - 1.0))
        mean_loss = torch.mean(torch.square(mu_x - mu_y))
    else:
        mean_loss = 0

    return (1 - zncc) + mean_loss


def AE(x, y):
    if len(x.shape) == 5:  # 3d
        dims = (2, 3, 4)
    elif len(x.shape) == 4:
        dims = (2, 3)
    else:
        dims = None

    error = torch.mean(torch.sum(torch.abs(x - y), dim=dims))
    return error


def absolute_error(x, y):
    e = torch.abs(x.contiguous() - y.contiguous())
    ae = torch.sum(e) / e.shape[0]
    return ae


def IMGPSF_sl(img_est, img_tar, psf_est, psf_tar):
    zncc_img = ZNCC(img_est, img_tar)
    ae_psf = AE(psf_est, psf_tar)

    loss = zncc_img + ae_psf
    return loss


def fftn_conv(signal, kernel):
    signal_shape = signal.shape[2:]
    kernel_shape = kernel.shape[2:]

    dim_fft = tuple(range(2, signal.ndim))

    signal_fr = torch.fft.fftn(signal.float(), s=signal_shape, dim=dim_fft)
    kernel_fr = torch.fft.fftn(kernel.float(), s=signal_shape, dim=dim_fft)

    kernel_fr.imag *= -1
    output_fr = signal_fr * kernel_fr
    output = torch.fft.ifftn(output_fr, dim=dim_fft)
    output = output.real

    if signal.ndim == 5:
        output = output[
            :,
            :,
            0 : signal_shape[0] - kernel_shape[0] + 1,
            0 : signal_shape[1] - kernel_shape[1] + 1,
            0 : signal_shape[2] - kernel_shape[2] + 1,
        ]

    if signal.ndim == 4:
        output = output[
            :,
            :,
            0 : signal_shape[0] - kernel_shape[0] + 1,
            0 : signal_shape[1] - kernel_shape[1] + 1,
        ]

    return output


def convolution(x, PSF, padding_mode="reflect"):
    ks = PSF.shape[2:]

    if len(ks) == 3:
        x_pad = torch.nn.functional.pad(
            input=x,
            pad=(
                ks[2] // 2,
                ks[2] // 2,
                ks[1] // 2,
                ks[1] // 2,
                ks[0] // 2,
                ks[0] // 2,
            ),
            mode=padding_mode,
        )

    if len(ks) == 2:
        x_pad = torch.nn.functional.pad(
            input=x,
            pad=(ks[1] // 2, ks[1] // 2, ks[0] // 2, ks[0] // 2),
            mode=padding_mode,
        )

    x_conv = fftn_conv(signal=x_pad, kernel=PSF)
    return x_conv


def MSE_ss(img_in, img_est, psf_est, lt=False):
    img_est_conv = convolution(img_est, PSF=psf_est, padding_mode="constant")
    if lt == True:
        img_est_conv = linear_transform(
            img_true=img_in, img_test=img_est_conv, axis=(2, 3, 4)
        )
    loss = torch.nn.MSELoss()(img_in, img_est_conv)
    return loss


def MSE_w(img_est, img_gt, scale=0.1, eps=0.001):
    loss = torch.mean(torch.square(torch.sub(img_est, img_gt)), dim=(-1, -2, -3))
    weight = scale / (torch.mean(img_gt, dim=(-1, -2, -3)) + eps)
    loss_w = torch.mean(loss * weight)
    return loss_w


def MSE_ss_norm(img_in, img_est, psf_est):
    img_est_conv = convolution(img_est, PSF=psf_est, padding_mode="constant")
    mean_in = torch.mean(img_in, dim=(2, 3, 4), keepdim=True)
    mean_est = torch.mean(img_est_conv, dim=(2, 3, 4), keepdim=True)
    loss = torch.nn.MSELoss()(img_in, img_est_conv * mean_in / mean_est)
    return loss


def MAE_ss_norm(img_in, img_est, psf_est):
    img_est_conv = convolution(img_est, PSF=psf_est, padding_mode="constant")
    mean_in = torch.mean(img_in, dim=(2, 3, 4), keepdim=True)
    mean_est = torch.mean(img_est_conv, dim=(2, 3, 4), keepdim=True)
    loss = torch.nn.L1Loss()(img_in, img_est_conv * mean_in / mean_est)
    return loss


def MAE_ss(img_in, img_est, psf_est):
    img_est_conv = convolution(img_est, PSF=psf_est, padding_mode="constant")
    loss = torch.nn.L1Loss()(img_in, img_est_conv)
    return loss


def Poisson(img_in, img_est, psf_est):
    img_est_conv = convolution(img_est, PSF=psf_est, padding_mode="constant")
    loss = torch.nn.PoissonNLLLoss(log_input=False)(img_est_conv + 0.000001, img_in)
    return loss


def ZNCC_ss(img_in, img_est, psf_est):
    img_est_conv = convolution(img_est, PSF=psf_est, padding_mode="constant")
    loss = ZNCC(img_in, img_est_conv)
    return loss


def linear_transform(img_true, img_test, axis=None):
    if axis is None:
        keepdim = False
    else:
        keepdim = True

    mean_true = torch.mean(img_true, dim=axis, keepdims=keepdim)
    mean_test = torch.mean(img_test, dim=axis, keepdims=keepdim)

    b = torch.sum(
        (img_test - mean_test) * (img_true - mean_true), dim=axis, keepdim=keepdim
    ) / torch.sum(torch.square(img_test - mean_test), dim=axis, keepdim=keepdim)

    a = mean_true - b * mean_test

    img_test_transform = a + b * img_test
    return img_test_transform
