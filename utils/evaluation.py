import numpy as np
from skimage.metrics import (
    structural_similarity,
    peak_signal_noise_ratio,
    normalized_root_mse,
)
from scipy.optimize import linear_sum_assignment
import math, torch
from pytorch_msssim import ms_ssim


def tensor_to_array(img):
    """
    Convert torch Tensor to numpy array.
    ### Parameters:
    - `img`: (torch Tensor/ numpy array) input image.
    ### Returns:
    - `img`: (numpy array) output image.
    """
    if not isinstance(img, np.ndarray):
        img = img.cpu().detach().numpy()
    return img


def MSE(img_true, img_test):
    """
    Mean square error.

    ### Parameters:
    - `img_true`: ground truth image.
    - `img_test`: test image.

    ### Returns:
    - `err`: mean square error.
    """
    img_true = tensor_to_array(img_true)
    img_test = tensor_to_array(img_test)
    err = np.mean((img_test - img_true) ** 2)
    return err


def MAE(img_true, img_test):
    """
    Mean absolute error.
    ### Parameters:
    - `img_true`: ground truth image.
    - `img_test`: test image.
    ### Returns:
    - `err`: mean absolute error.
    """
    img_true = tensor_to_array(img_true)
    img_test = tensor_to_array(img_test)
    err = np.mean(np.abs(img_test - img_true))
    return err


def SSIM(
    img_true,
    img_test,
    data_range=None,
    multichannel=False,
    channle_axis=None,
    version_wang=False,
    convert_to_255: bool = False,
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
    - `convert_to_255`: whether to convert the image to [0,255].

    ### Returns:
    - `ssim`: structural similarity index.
    """
    if data_range == None:
        data_range = img_true.max() - img_true.min()
    if data_range == 0:
        data_range = 1

    if convert_to_255:
        img_true = (img_true * 255).astype(np.uint8)
        img_test = (img_test * 255).astype(np.uint8)
        data_range = 255

    if version_wang == False:
        ssim = structural_similarity(
            im1=img_true,
            im2=img_test,
            multichannel=multichannel,
            data_range=data_range,
            channel_axis=channle_axis,
        )

    if version_wang == True:
        ssim = structural_similarity(
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


def MSSSIM(img_true, img_test, data_range=None):
    """
    Multi-scale structural similarity index.
    ### Parameters:
    - `img_true`: ground truth image. [H, W]
    - `img_test`: test image. [H, W]
    - `data_range`: the dynamic range of the images. default is None.
    ### Returns:
    - `msssim`: (numpy array) multi-scale structural similarity index.
    """
    num_axis = len(img_true.shape)
    if num_axis == 2:
        img_true = img_true[None, None]
        img_test = img_test[None, None]
    elif num_axis == 3:
        img_true = img_true[None]
        img_test = img_test[None]
    elif num_axis == 4:
        pass
    else:
        raise ValueError("Invalid input shape.")

    if data_range == None:
        data_range = img_true.max() - img_true.min()
    if data_range == 0:
        data_range = 1

    # convert to pytorch tensor
    img_true = torch.from_numpy(img_true).float()
    img_test = torch.from_numpy(img_test).float()

    msssim = ms_ssim(img_true, img_test, data_range=data_range)
    return msssim.numpy()


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
    - `ssim`: structural similarity index (mena of batch).
    """
    # tensor to numpy array
    img_true = tensor_to_array(img_true)
    img_test = tensor_to_array(img_test)

    assert (
        img_true.shape == img_test.shape
    ), "The shape of img_true and img_test must be the same."
    assert len(img_true.shape) in [
        4,
        5,
    ], f"The shape of img_true and img_test must be 2D (4 axis) or 3D (5 axis). But got {len(img_true.shape)}."

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


def PSNR(img_true, img_test, data_range=None, convert_to_255=False):
    """
    Peak signal-to-noise ratio.

    ### Parameters:
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

    if convert_to_255:
        img_true = (img_true * 255).astype(np.uint8)
        img_test = (img_test * 255).astype(np.uint8)
        data_range = 255

    mse = np.mean((img_true - img_test) ** 2)
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = peak_signal_noise_ratio(
            image_true=img_true, image_test=img_test, data_range=data_range
        )
    return psnr


def PSNR_tb(img_true, img_test, data_range=None):
    """
    PSNR for a batch of np tensor, the input should be [B, C, [depth], H, W].
    Support 3d and 2d single/multi-channel images.

    ### Parameters:
    - `img_true`: ground truth image. [B, C, [depth], H, W]
    - `img_test`: test image. [B, C, [depth], H, W]
    - `data_range`: the dynamic range of the images. default is None.

    ### Returns:
    - `psnr`: peak signal-to-noise ratio (mean of the batch).
    """
    # tensor to numpy array
    img_true = tensor_to_array(img_true)
    img_test = tensor_to_array(img_test)

    assert (
        img_true.shape == img_test.shape
    ), "The shape of img_true and img_test must be the same."
    assert len(img_true.shape) in [
        4,
        5,
    ], f"The shape of img_true and img_test must be 2D (4 axis) or 3D (5 axis). But got {len(img_true.shape)}."

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


def NRMSE(img_true, img_test, normalization="euclidean"):
    """
    Normalized roor mean square error using scitkit-image.

    ### Parameters:
    - `img_true`: ground truth image.
    - `img_test`: test image.
    - `normalization`: normalization method. default is "euclidean".

    ### Returns:
    - `nrmse`: normalized root mean square error.
    """
    nrmse = normalized_root_mse(
        image_true=img_true, image_test=img_test, normalization=normalization
    )
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
    # covnert to numpy array
    img_true = tensor_to_array(img_true)
    img_test = tensor_to_array(img_test)

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

    a = np.mean(img_true, axis=axis, keepdims=keepdims) / np.mean(
        img_test, axis=axis, keepdims=keepdims
    )

    img_test_rescale = img_test * a

    return img_test_rescale


def linear_transform(img_true, img_test, axis=None):
    """
    Linear transform.
    Do transform for speficify axis. If axis is None, do transform for all axis.

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

    # calculate slope and intercept (checked)
    b = np.mean(
        (img_test - mean_test) * (img_true - mean_true),
        axis=axis,
        keepdims=keepdims,
    ) / np.mean(np.square(img_test - mean_test), axis=axis, keepdims=keepdims)
    a = mean_true - b * mean_test

    # linear transform
    img_test_transform = a + b * img_test

    return img_test_transform


def linear_transform_threshold(img_true, img_test, threshold=0.1, side="right"):
    """
    Linear transform with threshold.
    Cannot specify the axis for the linear transform.

    ### Parameters:
    - `img_true`: ground truth image.
    - `img_test`: test image.
    - `threshold`: linear tranform applied only on value higher than `threshold`. default is 0.1.

    ### Returns:
    - `img_test_transform`: linear-transformed test image.
    """
    # tensor to numpy array
    img_true = tensor_to_array(img_true).astype(np.float32)
    img_test = tensor_to_array(img_test).astype(np.float32)

    # get the value and index in the img_true array, where the value is greater than threshold.
    if side == "right":
        idx = np.where(img_true > threshold)
    elif side == "left":
        idx = np.where(img_true < (1.0 - threshold))
    elif side == "both":
        idx1 = np.where(img_true > threshold)
        idx2 = np.where(img_true < (1.0 - threshold))
        idx = idx1 and idx2
    else:
        raise ValueError("side must be 'right', 'left' or 'both'.")

    img_true_threshold = img_true[idx]
    img_test_threshold = img_test[idx]

    # calculate mean and std
    mean_true = np.mean(img_true_threshold)
    mean_test = np.mean(img_test_threshold)
    # calculate slope and intercept
    b = np.mean(
        (img_test_threshold - mean_test) * (img_true_threshold - mean_true)
    ) / np.mean(np.square(img_test_threshold - mean_test))
    a = mean_true - b * mean_test
    # linear transform
    img_test_transform = a + b * img_test

    return img_test_transform


def average_precision(masks_true, masks_pred, threshold=[0.5, 0.75, 0.9]):
    """
    https://github.com/MouseLand/cellpose/blob/386474831e425abdab789703b5788e749d09960d/cellpose/metrics.py#L82
    Average precision estimation: AP = TP / (TP + FP + FN)

    This function is based heavily on the *fast* stardist matching functions
    (https://github.com/mpicbg-csbd/stardist/blob/master/stardist/matching.py)

    Args:
        masks_true (list of np.ndarrays (int) or np.ndarray (int)):
            where 0=NO masks; 1,2... are mask labels
        masks_pred (list of np.ndarrays (int) or np.ndarray (int)):
            np.ndarray (int) where 0=NO masks; 1,2... are mask labels

    Returns:
        ap (array [len(masks_true) x len(threshold)]):
            average precision at thresholds
        tp (array [len(masks_true) x len(threshold)]):
            number of true positives at thresholds
        fp (array [len(masks_true) x len(threshold)]):
            number of false positives at thresholds
        fn (array [len(masks_true) x len(threshold)]):
            number of false negatives at thresholds
    """
    not_list = False
    if not isinstance(masks_true, list):
        masks_true = [masks_true]
        masks_pred = [masks_pred]
        not_list = True
    if not isinstance(threshold, list) and not isinstance(threshold, np.ndarray):
        threshold = [threshold]

    if len(masks_true) != len(masks_pred):
        raise ValueError(
            "metrics.average_precision requires len(masks_true)==len(masks_pred)"
        )

    ap = np.zeros((len(masks_true), len(threshold)), np.float32)
    tp = np.zeros((len(masks_true), len(threshold)), np.float32)
    fp = np.zeros((len(masks_true), len(threshold)), np.float32)
    fn = np.zeros((len(masks_true), len(threshold)), np.float32)
    n_true = np.array(list(map(np.max, masks_true)))
    n_pred = np.array(list(map(np.max, masks_pred)))

    for n in range(len(masks_true)):
        # _,mt = np.reshape(np.unique(masks_true[n], return_index=True), masks_pred[n].shape)
        if n_pred[n] > 0:
            iou = _intersection_over_union(masks_true[n], masks_pred[n])[1:, 1:]
            for k, th in enumerate(threshold):
                tp[n, k] = _true_positive(iou, th)
        fp[n] = n_pred[n] - tp[n]
        fn[n] = n_true[n] - tp[n]
        ap[n] = tp[n] / (tp[n] + fp[n] + fn[n])

    if not_list:
        ap, tp, fp, fn = ap[0], tp[0], fp[0], fn[0]
    return ap, tp, fp, fn


def _intersection_over_union(masks_true, masks_pred):
    """Calculate the intersection over union of all mask pairs.

    Parameters:
        masks_true (np.ndarray, int): Ground truth masks, where 0=NO masks; 1,2... are mask labels.
        masks_pred (np.ndarray, int): Predicted masks, where 0=NO masks; 1,2... are mask labels.

    Returns:
        iou (np.ndarray, float): Matrix of IOU pairs of size [x.max()+1, y.max()+1].

    How it works:
        The overlap matrix is a lookup table of the area of intersection
        between each set of labels (true and predicted). The true labels
        are taken to be along axis 0, and the predicted labels are taken
        to be along axis 1. The sum of the overlaps along axis 0 is thus
        an array giving the total overlap of the true labels with each of
        the predicted labels, and likewise the sum over axis 1 is the
        total overlap of the predicted labels with each of the true labels.
        Because the label 0 (background) is included, this sum is guaranteed
        to reconstruct the total area of each label. Adding this row and
        column vectors gives a 2D array with the areas of every label pair
        added together. This is equivalent to the union of the label areas
        except for the duplicated overlap area, so the overlap matrix is
        subtracted to find the union matrix.
    """
    overlap = _label_overlap(masks_true, masks_pred)
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0
    return iou


def _label_overlap(x, y):
    """Fast function to get pixel overlaps between masks in x and y.

    Args:
        x (np.ndarray, int): Where 0=NO masks; 1,2... are mask labels.
        y (np.ndarray, int): Where 0=NO masks; 1,2... are mask labels.

    Returns:
        overlap (np.ndarray, int): Matrix of pixel overlaps of size [x.max()+1, y.max()+1].
    """
    # put label arrays into standard form then flatten them
    #     x = (utils.format_labels(x)).ravel()
    #     y = (utils.format_labels(y)).ravel()
    x = x.ravel()
    y = y.ravel()

    # preallocate a "contact map" matrix
    overlap = np.zeros((1 + x.max(), 1 + y.max()), dtype=np.uint)

    # loop over the labels in x and add to the corresponding
    # overlap entry. If label A in x and label B in y share P
    # pixels, then the resulting overlap is P
    # len(x)=len(y), the number of pixels in the whole image
    for i in range(len(x)):
        overlap[x[i], y[i]] += 1
    return overlap


def _true_positive(iou, th):
    """Calculate the true positive at threshold th.

    Args:
        iou (float, np.ndarray): Array of IOU pairs.
        th (float): Threshold on IOU for positive label.

    Returns:
        tp (float): Number of true positives at threshold.

    How it works:
        (1) Find minimum number of masks.
        (2) Define cost matrix; for a given threshold, each element is negative
            the higher the IoU is (perfect IoU is 1, worst is 0). The second term
            gets more negative with higher IoU, but less negative with greater
            n_min (but that's a constant...).
        (3) Solve the linear sum assignment problem. The costs array defines the cost
            of matching a true label with a predicted label, so the problem is to
            find the set of pairings that minimizes this cost. The scipy.optimize
            function gives the ordered lists of corresponding true and predicted labels.
        (4) Extract the IoUs from these pairings and then threshold to get a boolean array
            whose sum is the number of true positives that is returned.
    """
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= th).astype(float) - iou / (2 * n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    match_ok = iou[true_ind, pred_ind] >= th
    tp = match_ok.sum()
    return tp


def IoU(masks_true, masks_pred, threshold=0.5):
    """Calculate the intersection over union.
    ### Parameters:
        masks_true (list of np.ndarray, int): Ground truth masks.
        masks_pred (list of np.ndarray, int): Predicted masks.
        threshold (float): Threshold on masks_pred.
    ### Returns:
        iou (list of float): Intersection over union.
    """
    if not isinstance(masks_true, list):
        masks_true = [masks_true]
    if not isinstance(masks_pred, list):
        masks_pred = [masks_pred]
    if len(masks_true) != len(masks_pred):
        raise ValueError("The number of masks_true and masks_pred must be the same.")
    ious = []
    for i in range(len(masks_true)):
        mask_true_01 = np.where(masks_true[i] >= threshold, 1, 0)
        mask_pred_01 = np.where(masks_pred[i] >= threshold, 1, 0)
        intersection = np.logical_and(mask_true_01, mask_pred_01)
        union = np.logical_or(mask_true_01, mask_pred_01)
        iou = np.sum(intersection) / np.sum(union)
        ious.append(iou)
    return np.array(ious, dtype=np.float32)
