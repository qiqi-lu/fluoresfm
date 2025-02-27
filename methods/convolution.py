import torch


def fftn_conv(signal, kernel, *args, **kwargs):
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


def convolution(x, PSF, padding_mode="reflect", domain="fft"):
    ks = PSF.shape
    dim = len(ks)

    PSF, x = torch.tensor(PSF[None, None]), torch.tensor(x[None, None])
    PSF = torch.round(PSF, decimals=16)

    if dim == 3:
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
        if domain == "direct":
            x_conv = torch.nn.functional.conv3d(input=x_pad, weight=PSF, groups=1)
        if domain == "fft":
            x_conv = fftn_conv(signal=x_pad, kernel=PSF, groups=1)

    if dim == 2:
        x_pad = torch.nn.functional.pad(
            input=x,
            pad=(ks[1] // 2, ks[1] // 2, ks[0] // 2, ks[0] // 2),
            mode=padding_mode,
        )
        if domain == "direct":
            x_conv = torch.nn.functional.conv2d(input=x_pad, weight=PSF, groups=1)
        if domain == "fft":
            x_conv = fftn_conv(signal=x_pad, kernel=PSF, groups=1)

    out = x_conv.numpy()[0, 0]
    return out


if __name__ == "__main__":
    a = torch.ones(128, 128, 128)
    b = torch.ones(127, 127, 127)

    c = convolution(a, b)
    print(c.shape)
