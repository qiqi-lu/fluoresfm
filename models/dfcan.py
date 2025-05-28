import torch, math
import torch.nn as nn
import torchinfo


def fft2d(input, gamma=0.1):
    # input = torch.complex(real=input, imag=torch.zeros_like(input))
    fft = torch.fft.fft2(input)
    absfft = torch.pow(torch.abs(fft) + 1e-8, gamma)
    return absfft


def fftshift2d(input):
    _, _, h, w = input.shape
    fs11 = input[:, :, -h // 2 : h, -w // 2 : w]
    fs12 = input[:, :, -h // 2 : h, 0 : w // 2]
    fs21 = input[:, :, 0 : h // 2, -w // 2 : w]
    fs22 = input[:, :, 0 : h // 2, 0 : w // 2]
    output = torch.cat(
        [torch.cat([fs11, fs21], axis=2), torch.cat([fs12, fs22], axis=2)], axis=3
    )
    return output


class FCALayer(nn.Module):
    def __init__(self, num_features=64, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=num_features,
            out_channels=num_features,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=num_features,
            out_channels=num_features // reduction,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv3 = nn.Conv2d(
            in_channels=num_features // reduction,
            out_channels=num_features,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        absfft1 = fft2d(x, gamma=0.8)
        absfft1 = fftshift2d(absfft1)
        absfft2 = self.act1(self.conv1(absfft1))
        w = torch.mean(absfft2, axis=(-1, -2), keepdims=True)
        w = self.act1(self.conv2(w))
        w = self.act2(self.conv3(w))
        mul = torch.mul(x, w)
        return mul


class FCAB(nn.Module):
    def __init__(self, num_features=64):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=num_features,
            out_channels=num_features,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(
            in_channels=num_features,
            out_channels=num_features,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.fca = FCALayer(num_features=num_features, reduction=16)

    def forward(self, x):
        conv = self.act(self.conv1(x))
        conv = self.act(self.conv2(conv))
        att = self.fca(conv)
        output = torch.add(att, x)
        return output


class ResidualGroup(nn.Module):
    def __init__(self, num_features=64, num_blocks=4):
        super().__init__()
        blocks = []
        for _ in range(num_blocks):
            blocks.append(FCAB(num_features=num_features))
        self.fcabs = nn.Sequential(*blocks)

    def forward(self, x):
        conv = self.fcabs(x)
        conv = torch.add(conv, x)
        return conv


class Upsampler(nn.Module):
    def __init__(self, scale_factor=4, num_features=64, kernel_size=3):
        super().__init__()
        modules = []
        if (scale_factor & (scale_factor - 1)) == 0:  # 2, 4, 8... 2^n
            for _ in range(int(math.log(scale_factor, 2))):
                modules.append(
                    nn.Conv2d(
                        in_channels=num_features,
                        out_channels=4 * num_features,
                        kernel_size=kernel_size,
                        padding=(kernel_size // 2),
                        bias=True,
                    )
                )
                modules.append(nn.GELU())
                modules.append(nn.PixelShuffle(2))
        elif scale_factor == 3:
            modules.append(
                nn.Conv2d(
                    in_channels=num_features,
                    out_channels=9 * num_features,
                    kernel_size=kernel_size,
                    padding=(kernel_size // 2),
                    bias=True,
                )
            )
            modules.append(nn.GELU())
            modules.append(nn.PixelShuffle(3))
        else:
            raise NotImplementedError
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        return self.body(x)


class DFCAN(nn.Module):
    def __init__(
        self, in_channels=1, scale_factor=4, num_features=64, num_groups=4
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_features,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.act = nn.GELU()

        groups = []
        for _ in range(num_groups):
            groups.append(ResidualGroup(num_features=num_features, num_blocks=4))
        self.residual_groups = nn.Sequential(*groups)

        self.upsampler = Upsampler(
            scale_factor=scale_factor, num_features=num_features, kernel_size=3
        )

        self.conv_tail = nn.Conv2d(
            in_channels=num_features,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.act_tail = nn.Sigmoid()

    def forward(self, x):
        conv = self.act(self.conv1(x))
        conv = self.residual_groups(conv)
        conv_up = self.upsampler(conv)
        out = self.act_tail(self.conv_tail(conv_up))
        return out


if __name__ == "__main__":
    in_channels = 1
    input_size = (2, in_channels, 128, 128)
    x = torch.ones(size=input_size)
    bs, ch, h, w = x.shape
    model = DFCAN(
        in_channels=in_channels, scale_factor=8, num_features=64, num_groups=4
    )
    o = model(x)
    torchinfo.summary(model=model, input_size=input_size)
    print(o.shape)
