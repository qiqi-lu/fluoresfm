import numpy as np


def align_size(img1, Sx2, Sy2, Sz2, padValue=0):
    Sx1, Sy1, Sz1 = img1.shape
    Sx, Sy, Sz = (
        np.maximum(Sx1, Sx2),
        np.maximum(Sy1, Sy2),
        np.maximum(Sz1, Sz2),
    )
    imgTemp = np.ones(shape=(Sx, Sy, Sz)) * padValue

    Sox, Soy, Soz = (
        int(np.round((Sx - Sx1) / 2)),
        int(np.round((Sy - Sy1) / 2)),
        int(np.round((Sz - Sz1) / 2)),
    )
    imgTemp[Sox : Sox + Sx1, Soy : Soy + Sy1, Soz : Soz + Sz1] = img1

    Sox, Soy, Soz = (
        int(np.round((Sx - Sx2) / 2)),
        int(np.round((Sy - Sy2) / 2)),
        int(np.round((Sz - Sz2) / 2)),
    )
    img2 = imgTemp[Sox : Sox + Sx2, Soy : Soy + Sy2, Soz : Soz + Sz2]
    return img2


def lin_interp(x, y, i, half):
    return x[i] + (x[i + 1] - x[i]) * ((half - y[i]) / (y[i + 1] - y[i]))


def FWHM_1d(y):
    x = np.linspace(start=0, stop=y.shape[0] - 1, num=y.shape[0])
    half = max(y) / 2.0
    signs = np.sign(np.add(y, -half))
    zero_crossings = signs[0:-2] != signs[1:-1]
    zero_crossings_i = np.where(zero_crossings)[0]
    dist = lin_interp(x, y, zero_crossings_i[1], half) - lin_interp(
        x, y, zero_crossings_i[0], half
    )
    return dist


def FWHM_PSF(PSF, pixel_size=1.0, c_flag=0, fit_flag=0):
    dim = len(PSF.shape)
    if dim == 2:
        Sx, Sy = PSF.shape
        if c_flag == 0:
            indx, indy = Sx // 2, Sy // 2
        FWHMx = FWHM_1d(PSF[:, indy]) * pixel_size
        FWHMy = FWHM_1d(PSF[indx]) * pixel_size
        FWHMs = [FWHMx, FWHMy]

    if dim == 3:
        Sx, Sy, Sz = PSF.shape
        if c_flag == 0:
            indx, indy, indz = Sx // 2, Sy // 2, Sz // 2
        FWHMx = FWHM_1d(PSF[:, indy, indz])
        FWHMy = FWHM_1d(PSF[indx, :, indz])
        FWHMz = FWHM_1d(PSF[indx, indy])
        FWHMs = [FWHMx, FWHMy, FWHMz]
    return FWHMs


def sigma2FWHM(sigmas):
    sigmas = np.array(sigmas) * 2.3548
    return sigmas.tolist()


def FWHM2sigma(FWHM):
    FWHM = np.array(FWHM) / 2.3548
    return FWHM.tolist()


def PSF_gaussian(size, sigmas):
    dim = len(size)
    assert (
        len(sigmas) == dim
    ), ">> The length of FWHMs should be same as the length of size."
    if dim == 2:
        Sx, Sy = size
        sigma_x, sigma_y = sigmas
        coef = 1 / (2 * np.pi * sigma_x * sigma_y)

        i, j = np.mgrid[-Sx // 2 + 1 : Sx // 2 + 1, -Sy // 2 + 1 : Sy // 2 + 1]
        if (Sx % 2) == 0:
            i = (i - 0.5).astype(np.float32)
        if (Sy % 2) == 0:
            j = (j - 0.5).astype(np.float32)
        PSF = np.exp(-(i**2 / (2.0 * sigma_x**2) + j**2 / (2.0 * sigma_y**2)))

    if dim == 3:
        Sx, Sy, Sz = size
        sigma_x, sigma_y, sigma_z = sigmas
        coef = 1 / ((2 * np.pi) ** (3 / 2) * sigma_x * sigma_y * sigma_z)

        i, j, k = np.mgrid[
            -Sx // 2 + 1 : Sx // 2 + 1,
            -Sy // 2 + 1 : Sy // 2 + 1,
            -Sz // 2 + 1 : Sz // 2 + 1,
        ]
        if (Sx % 2) == 0:
            i = (i - 0.5).astype(np.float32)
        if (Sy % 2) == 0:
            j = (j - 0.5).astype(np.float32)
        if (Sz % 2) == 0:
            k = (k - 0.5).astype(np.float32)
        PSF = np.exp(
            -(
                i**2 / (2.0 * sigma_x**2)
                + j**2 / (2.0 * sigma_y**2)
                + k**2 / (2.0 * sigma_z**2)
            )
        )

    PSF = coef * PSF
    return PSF


def BackProjector(
    PSF_fp,
    bp_type="traditional",
    alpha=0.001,
    beta=1,
    n=10,
    res_flag=1,
    i_res=(0, 0, 0),
    verbose_flag=0,
):
    """
    Gnerator backrpojector according to the given PSF.

    Args:
    - PSF_fp:   Forward projector.
    - bp_type:  'traditional', 'gaussian', 'butterworth', 'wiener', 'wiener-butterworth'.
    - alpha:    [0.0001, 0.001] or 1 (use OTF value of the PSF_bp at resolution limit).
    - beta:     [0.001, 0.01] or 1 (use OTF value of PSF_bp at resolution limit).
    - n:        [4, 15], order of Butterworth filter.
    - res_flag: 0 (use PSF_fp FWHM/root(2) as resolution limit (for iSIM)),
                1 (use PSF_fp FWHM as resoltuion limit),
                2 (use input values (iRes) as resoltuion limit).
    - i_res:    input resolution limit in 3 dimensions in terms of pixels.
    - verbose_flag: 0 (hide log and intermediate results),
                    1 (show)
    Outpt:
    - PSF_bp:   BP kernel.
    - OTF_bp:   OTF of BP kernel.
    """

    dim = len(PSF_fp.shape)
    # ==========================================================================================
    if verbose_flag:
        print("=" * 100)
        print(f"Back projector type: {bp_type} ({dim}D)")

    assert bp_type in [
        "traditional",
        "gaussian",
        "butterworth",
        "wiener",
        "wiener-butterworth",
    ], "by_type does not match any bakc-projector type"
    if bp_type == "traditional":
        PSF_bp = np.flip(PSF_fp)
        OTF_bp = np.fft.fftn(np.fft.ifftshift(PSF_bp))
        if verbose_flag:
            print("=" * 50)
        return PSF_bp.astype(np.float32), OTF_bp.astype(np.complex64)

    if dim == 2:
        Sx, Sy = PSF_fp.shape
        Scx, Scy = (Sx - 1) / 2, (Sy - 1) / 2
        # Sox, Soy = int(np.round((Sx - 1) / 2)), int(np.round((Sy - 1) / 2))

        # Calculate PSF and OTF size
        FWHMx, FWHMy = FWHM_PSF(PSF_fp)

        if verbose_flag:
            print(f"FWHM  = {FWHMx:>8.4f} x {FWHMy:>8.4f}")
            print(
                f"Sigma = {FWHM2sigma(FWHMx):>8.4f} x {FWHM2sigma(FWHMy):>8.4f}"
            )

        # set resolution cutoff
        assert res_flag in [0, 1, 2], "Please set res_flag as 0, 1, or 2."
        if res_flag == 0:
            resx, resy = FWHMx / (2**0.5), FWHMy / (
                2**0.5
            )  # set resolution as 1/root(2) of PSF_fp FWHM: iSIM case
        if res_flag == 1:
            resx, resy = FWHMx, FWHMy  # set resolution as PSF_fp FWHM
        if res_flag == 2:
            resx, resy = i_res  # set resolution based on input values

        # pixel size in Fourier domain
        px, py = 1 / Sx, 1 / Sy

        # frequency cutoff in terms of pixels
        tx, ty = (1 / resx) / px, (1 / resy) / py

        if verbose_flag:
            print(
                f"Resolution cutoff in spatial domain : {resx:8.4f} x {resy:8.4f}"
            )
            print(
                f"Resolution cutoff in Fourier domain : {tx:8.4f} x {ty:8.4f}"
            )

    if dim == 3:
        Sx, Sy, Sz = PSF_fp.shape
        Scx, Scy, Scz = (Sx - 1) / 2, (Sy - 1) / 2, (Sz - 1) / 2
        # Sox, Soy, Soz = (
        #     int(np.round((Sx - 1) / 2)),
        #     int(np.round((Sy - 1) / 2)),
        #     int(np.round((Sz - 1) / 2)),
        # )
        Soz = int(np.round((Sz - 1) / 2))

        # Calculate PSF and OTF size
        FWHMx, FWHMy, FWHMz = FWHM_PSF(PSF_fp)

        if verbose_flag:
            print(
                f"FWHM  = {FWHMx:>8.4f} x {FWHMy:>8.4f} x {FWHMz:>8.4f} (pixels)"
            )
            print(
                f"Sigma = {FWHM2sigma(FWHMx):>8.4f} x {FWHM2sigma(FWHMy):>8.4f} x {FWHM2sigma(FWHMz):>8.4f} (pixels)"
            )

        # set resolution cutoff
        assert res_flag in [0, 1, 2], "Please set res_flag as 0, 1, or 2."

        if res_flag == 0:
            resx, resy, resz = (
                FWHMx / (2**0.5),
                FWHMy / (2**0.5),
                FWHMz / (2**0.5),
            )
        if res_flag == 1:
            resx, resy, resz = FWHMx, FWHMy, FWHMz
        if res_flag == 2:
            resx, resy, resz = i_res

        # pixel size in Fourier domain
        px, py, pz = 1 / Sx, 1 / Sy, 1 / Sz

        # frequency cutoff in terms of pixels
        tx, ty, tz = (1 / resx) / px, (1 / resy) / py, (1 / resz) / pz

        if verbose_flag:
            print(
                f"Resolution cutoff in spatial domain : {resx:8.4f} x {resy:8.4f} x {resz:8.4f}"
            )
            print(
                f"Resolution cutoff in Fourier domain : {tx:8.4f} x {ty:8.4f} x {tz:8.4f}"
            )

    # normalize flipped PSF: traditional back projector
    PSF_flipped = np.flip(PSF_fp)
    OTF_flip = np.fft.fftn(np.fft.ifftshift(PSF_flipped))
    OTF_abs = np.fft.fftshift(np.abs(OTF_flip))
    OTF_max = np.max(OTF_abs)
    M = OTF_max
    OTF_abs_norm = OTF_abs / M

    # check cutoff gains of traditional back projector
    if dim == 2:
        tline = np.max(OTF_abs_norm, axis=1)
        to1 = int(np.maximum(np.round(Scx - tx), 0))
        to2 = int(np.minimum(np.round(Scx + tx), Sx - 1))
        beta_fpx = (
            tline[to1] + tline[to2]
        ) / 2  # OTF frequency intensity as cutoff: x

        tline = np.max(OTF_abs_norm, axis=0)
        to1 = int(np.maximum(np.round(Scy - ty), 0))
        to2 = int(np.minimum(np.round(Scy + ty), Sy - 1))
        beta_fpy = (
            tline[to1] + tline[to2]
        ) / 2  # OTF frequency intensity as cutoff: y

        beta_fp = (beta_fpx + beta_fpy) / 2
        if verbose_flag:
            print(
                f"Cutoff gain of forward projector : {beta_fpx:>8.4f} x {beta_fpy:>8.4f}, average = {beta_fp:>8.4f}"
            )

    if dim == 3:
        tplane = np.max(OTF_abs_norm, axis=2)
        tline = np.max(tplane, axis=1)
        to1 = int(np.maximum(np.round(Scx - tx), 0))
        to2 = int(np.minimum(np.round(Scx + tx), Sx - 1))
        beta_fpx = (
            tline[to1] + tline[to2]
        ) / 2  # OTF frequency intensity as cutoff: x

        tplane = np.max(OTF_abs_norm, axis=2)
        tline = np.max(tplane, axis=0)
        to1 = int(np.maximum(np.round(Scy - ty), 0))
        to2 = int(np.minimum(np.round(Scy + ty), Sy - 1))
        beta_fpy = (
            tline[to1] + tline[to2]
        ) / 2  # OTF frequency intensity as cutoff: y

        tplane = np.max(OTF_abs_norm, axis=0)
        tline = np.max(tplane, axis=0)
        to1 = int(np.maximum(np.round(Scz - tz), 0))
        to2 = int(np.minimum(np.round(Scz + tz), Sz - 1))
        beta_fpz = (
            tline[to1] + tline[to2]
        ) / 2  # OTF frequency intensity as cutoff: z

        beta_fp = (beta_fpx + beta_fpy + beta_fpz) / 3
        if verbose_flag:
            print(
                f"Cutoff gain of forward projector : {beta_fpx:>8.4f} x {beta_fpy:>8.4f} x {beta_fpz:>8.4f}, average = {beta_fp:>8.4f}"
            )

    if alpha == 1:
        alpha = beta_fp
        if verbose_flag:
            print(
                f"wiener parameter adjusted as traditional BP cutoff gain: alpha = {alpha:>.4f}"
            )
    else:
        if verbose_flag:
            print(f"Wiener parameter set as input: alpha = {alpha:>.4f}")

    if beta == 1:
        beta = beta_fp
        if verbose_flag:
            print(
                f"Cutoff gain adjusted as traditional BP cutoff gain: beta = {beta:>.4f}"
            )
    else:
        if verbose_flag:
            print(f"Cutoff gain set as input: beta = {beta:>.4f}")

    # order of Butterworth filter
    if verbose_flag:
        print(f"Butterworth order (slope parameter) set as: n = {n}")

    if bp_type == "gaussian":
        if dim == 2:
            resx, resy = FWHMx, FWHMy
            PSF_bp = PSF_gaussian(
                size=[Sx, Sy], sigmas=FWHM2sigma([resx, resy])
            )

        if dim == 3:
            resx, resy, resz = FWHMx, FWHMy, FWHMz
            PSF_bp = PSF_gaussian(
                size=[Sx, Sy, Sz], sigmas=FWHM2sigma([resx, resy, resz])
            )
        OTF_bp = np.fft.fftn(np.fft.ifftshift(PSF_bp))

    if bp_type == "butterworth":
        ee = 1 / beta**2 - 1
        # create Butterworth filter (2D)
        if dim == 2:
            kcx, kcy = tx, ty  # width of Butterworth Filter
            i, j = np.mgrid[0 : Sx - 1, 0 : Sy - 1]
            w = ((i - Scx) / kcx) ** 2 + ((j - Scy) / kcy) ** 2
            mask = 1 / np.sqrt(1 + ee * (w**n))

        # create Butterworth filter (3D)
        if dim == 3:
            kcx, kcy, kcz = tx, ty, tz  # width of Butterworth Filter
            i, j, k = np.mgrid[0:Sx, 0:Sy, 0:Sz]
            w = (
                ((i - Scx) / kcx) ** 2
                + ((j - Scy) / kcy) ** 2
                + ((k - Scz) / kcz) ** 2
            )
            mask = 1 / np.sqrt(1 + ee * w**n)  # w^n = (kx/kcx)^pn

        OTF_bp = np.fft.ifftshift(mask)
        PSF_bp = np.fft.fftshift(np.real(np.fft.ifftn(OTF_bp)))

    if bp_type == "wiener":
        OTF_flip_norm = OTF_flip / M  # Normalized OTF_flip
        OTF_bp = OTF_flip_norm / (
            abs(OTF_flip_norm) ** 2 + alpha
        )  # Wiener filter
        PSF_bp = np.fft.fftshift(np.real(np.fft.ifftn(OTF_bp)))

    if bp_type == "wiener-butterworth":
        # create Wiener filter
        OTF_flip_norm = OTF_flip / M
        OTF_Wiener = OTF_flip_norm / (np.abs(OTF_flip_norm) ** 2 + alpha)

        # cut_off gain for wiener filter
        OTF_Wiener_abs = np.fft.fftshift(np.abs(OTF_Wiener))
        if dim == 2:
            tplane = np.abs(OTF_Wiener_abs)
        if dim == 3:
            tplane = np.abs(OTF_Wiener_abs[:, :, Soz])  # central slice

        tline = np.max(tplane, axis=1)
        to1 = int(np.maximum(np.round(Scx - tx), 0))
        to2 = int(np.minimum(np.round(Scx + tx), Sx - 1))
        beta_wienerx = (
            tline[to1] + tline[to2]
        ) / 2  # OTF frequency intensity at cutoff

        if verbose_flag:
            print(f"Wiener cutoff gain: beta_wienerx  = {beta_wienerx}")

        ee = beta_wienerx / beta**2 - 1
        # create Butterworth filter (2D)
        if dim == 2:
            kcx, kcy = tx, ty  # width of Butterworth filter
            i, j = np.mgrid[0 : Sx - 1, 0 : Sy - 1]
            w = ((i - Scx) / kcx) ** 2 + ((j - Scy) / kcy) ** 2
            mask = 1 / np.sqrt(1 + ee * w**n)  # w^n = (kx/kcx)^pn

        # create Butterworth filter (3D)
        if dim == 3:
            kcx, kcy, kcz = tx, ty, tz  # width of Butterworth filter
            i, j, k = np.mgrid[0:Sx, 0:Sy, 0:Sz]
            w = (
                ((i - Scx) / kcx) ** 2
                + ((j - Scy) / kcy) ** 2
                + ((k - Scz) / kcz) ** 2
            )
            mask = 1 / np.sqrt(1 + ee * w**n)  # w^n = (kx/kcx)^pn

        mask = np.fft.ifftshift(mask)
        # create Wiener-Butterworth filter
        OTF_bp = mask * OTF_Wiener
        PSF_bp = np.fft.fftshift(np.real(np.fft.ifftn(OTF_bp)))

    if verbose_flag:
        print("=" * 50)
    return PSF_bp.astype(np.float32), OTF_bp.astype(np.complex64)
