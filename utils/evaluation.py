import numpy as np
import skimage.metrics as skim
import math


def tensor_to_array(img):
    if not isinstance(img, np.ndarray):
        img = img.cpu().detach().numpy()
    return img


def SNR(img_true, img_test, type_cal=0):
    """
    Signal-to-noise ratio.
    """
    if type_cal == 0:
        img_true_ss = np.sum(np.square(img_true))
        error_ss = np.sum(np.square(img_true - img_test))
    if type_cal == 1:
        img_true_ss = np.var(img_true)
        error_ss = np.var(img_test - img_true)
    snr = 10 * np.log10(img_true_ss / error_ss)
    return snr


def MSE(img_true, img_test):
    """
    Mean square error.
    """
    err = np.mean((img_test - img_true) ** 2)
    return err


def MAE(img_true, img_test):
    """
    Mean absolute error.
    """
    err = np.mean(np.abs(img_test - img_true))
    return err


def RMSE(img_true, img_test):
    """
    Root mean square error.
    """
    rmse = np.mean(np.square(img_true - img_test)) / np.mean(np.square(img_true)) * 100
    return rmse


def SSIM(
    img_true,
    img_test,
    data_range=None,
    multichannel=False,
    channle_axis=None,
    version_wang=False,
):
    """
    Structrual similarity index.

    ### Parameters:
    - `img_true`: ground truth image.
    - `img_test`: test image.
    - `data_range`: the dynamic range of the images.
    - `multichannel`: whether the image is multi-channel.
    - `channle_axis`: the axis of the channel.
    - `version_wang`: whether to use the Wang et al. version of SSIM.

    ### Returns:
    - `ssim`: structural similarity index.
    """
    if data_range == None:
        data_range = img_true.max() - img_true.min()
    if data_range == 0:
        data_range = 1

    if version_wang == False:
        ssim = skim.structural_similarity(
            im1=img_true,
            im2=img_test,
            multichannel=multichannel,
            data_range=data_range,
            channel_axis=channle_axis,
        )

    if version_wang == True:
        ssim = skim.structural_similarity(
            im1=img_true,
            im2=img_test,
            multichannel=multichannel,
            data_range=data_range,
            channel_axis=channle_axis,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
        )
    return ssim


def SSIM_tb(img_true, img_test, data_range=None, version_wang=False):
    """
    SSIM for a batch of tensor.
    Support 3d and 2d single/multi-channel images.

    ### Parameters:
    - `img_true`: ground truth image. [B, C, [depth], H, W]
    - `img_test`: test image. [B, C, [depth], H, W]
    - `data_range`: the dynamic range of the images. default is None.
    - `version_wang`: whether to use the Wang et al. version of SSIM. default is False.

    ### Returns:
    - `ssim`: structural similarity index.
    """
    # tensor to numpy array
    img_true = tensor_to_array(img_true)
    img_test = tensor_to_array(img_test)

    ssims = []

    for i_sample in range(img_true.shape[0]):  # loop through each sample
        x, y = img_test[i_sample], img_true[i_sample]

        if len(y.shape) == 4:  # 3D image
            if y.shape[0] == 1:  # one channel 3D image
                if y.shape[1] >= 7:
                    # SSIM only supports 3D images with more than 7 slices.
                    ssims.append(
                        SSIM(
                            img_true=y[0],
                            img_test=x[0],
                            data_range=data_range,
                            multichannel=False,
                            channle_axis=None,
                            version_wang=version_wang,
                        )
                    )
                else:
                    # if the image is 3D but with less than 7 slices,
                    # calculate SSIM for each slice. And take the mean.
                    tmp = []
                    for i_slice in range(y.shape[1]):  # loop through each slice
                        tmp.append(
                            SSIM(
                                img_true=y[0][i_slice],
                                img_test=x[0][i_slice],
                                data_range=data_range,
                                multichannel=False,
                                channle_axis=None,
                                version_wang=version_wang,
                            )
                        )
                    ssims.append(np.mean(tmp))
            else:  # multi-channel 3D image
                if y.shape[1] > 7:  # multi-channel 3D image with more than 7 slices.
                    ssims.append(
                        SSIM(
                            img_true=y,
                            img_test=x,
                            data_range=data_range,
                            multichannel=True,
                            channle_axis=0,
                            version_wang=version_wang,
                        )
                    )
                else:
                    # if the image is 3D but with less than 7 slices,
                    # calculate SSIM for each sclice. And take the mean.
                    tmp = []
                    for i_slice in range(y.shape[1]):
                        tmp.append(
                            SSIM(
                                img_true=y[:, i_slice, ...],
                                img_test=x[:, i_slice, ...],
                                data_range=data_range,
                                multichannel=True,
                                channle_axis=0,
                                version_wang=version_wang,
                            )
                        )
                    ssims.append(np.mean(tmp))

        if len(y.shape) == 3:  # 2D
            if y.shape[0] == 1:  # single-channel
                ssims.append(SSIM(img_true=y[0], img_test=x[0], data_range=data_range))
            else:  # mutli-channel
                ssims.append(
                    SSIM(
                        img_true=y,
                        img_test=x,
                        data_range=data_range,
                        multichannel=True,
                        channle_axis=0,
                        version_wang=False,
                    )
                )

    return np.mean(ssims)


def PSNR(img_true, img_test, data_range=None):
    """
    Peak signal-to-noise ratio.

    ### Args:
    - `img_true`: ground truth image.
    - `img_test`: test image.
    - `data_range`: the dynamic range of the images.

    ### Returns:
    - `psnr`: peak signal-to-noise ratio.
    """
    if data_range == None:
        data_range = img_true.max() - img_true.min()
    if data_range == 0:
        data_range = 1

    mse = np.mean((img_true - img_test) ** 2)
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = skim.peak_signal_noise_ratio(
            image_true=img_true, image_test=img_test, data_range=data_range
        )
    return psnr


def PSNR_tb(img_true, img_test, data_range=None):
    """
    PSNR for a batch of np tensor, the input should be [B, C, [depth], H, W].
    """
    # tensor to numpy array
    img_true = tensor_to_array(img_true)
    img_test = tensor_to_array(img_test)

    psnrs = []
    for i in range(img_true.shape[0]):
        psnrs.append(
            PSNR(img_true=img_true[i], img_test=img_test[i], data_range=data_range)
        )

    # only calculate no inf value.
    psnrs_filtered = [v for v in psnrs if not math.isinf(v)]

    if not psnrs_filtered:  # check whether list is empty
        psnrs_filtered = psnrs

    return np.mean(psnrs_filtered)


def NCC(img_true, img_test):
    """
    Normalized cross-correlation.
    """
    mean_true, mean_test = img_true.mean(), img_test.mean()
    sigma_true, sigma_test = img_true.std(), img_test.std()
    NCC = np.mean(
        (img_true - mean_true) * (img_test - mean_test) / (sigma_true * sigma_test)
    )
    return NCC


def NRMSE(img_true, img_test):
    """
    Normalized roor mean square error.
    """
    xmax, xmin = np.max(img_true), np.min(img_true)
    rmse = np.sqrt(np.mean(np.square(img_test - img_true)))
    nrmse = rmse / (xmax - xmin)
    return nrmse


def ZNCC(img_true, img_test):
    """
    Zero-Normalized Cross-Correlation.

    ### Parameters:
    - `img_true`: ground truth image.
    - `img_test`: test image.

    ### Returns
    - `zncc`: zero-normalized cross-correlation.
    """
    if len(img_true.shape) == 5:  # 3d
        axis = (2, 3, 4)
    elif len(img_true.shape) == 4:
        axis = (2, 3)
    else:
        axis = None

    mu_true = np.mean(img_true, axis=axis, keepdims=True)
    mu_test = np.mean(img_test, axis=axis, keepdims=True)
    sigma_true = np.std(img_true, axis=axis, keepdims=True)
    sigma_test = np.std(img_test, axis=axis, keepdims=True)

    zncc = np.mean(
        (img_true - mu_true) * (img_test - mu_test) / (sigma_true * sigma_test)
    )

    return zncc


def intensity_balance(img_true, img_test, axis=None):
    # tensor to numpy array
    img_true = tensor_to_array(img_true)
    img_test = tensor_to_array(img_test)

    if axis is None:
        keepdims = False
    else:
        keepdims = True

    a = np.sum(img_true, axis=axis, keepdims=keepdims) / np.sum(
        img_test, axis=axis, keepdims=keepdims
    )

    img_test_rescale = img_test * a

    return img_test_rescale


def linear_transform(img_true, img_test, axis=None):
    """
    Linear transform.

    ### Args
    - `img_true`: ground truth image.
    - `img_test`: test image.
    - `axis`: axis to calculate the linear transform.

    ### Returns
    - `img_test_transform`: linear-transformed test image.
    """
    # tensor to numpy array
    img_true = tensor_to_array(img_true).astype(np.float32)
    img_test = tensor_to_array(img_test).astype(np.float32)

    if axis is None:
        keepdims = False
    else:
        keepdims = True

    # calculate mean and std
    mean_true = np.mean(img_true, axis=axis, keepdims=keepdims)
    mean_test = np.mean(img_test, axis=axis, keepdims=keepdims)

    # calculate slope and intercept
    b = np.mean(
        (img_test - mean_test) * (img_true - mean_true),
        axis=axis,
        keepdims=keepdims,
    ) / np.mean(np.square(img_test - mean_test), axis=axis, keepdims=keepdims)

    a = mean_true - b * mean_test

    # linear transform
    img_test_transform = a + b * img_test

    return img_test_transform
