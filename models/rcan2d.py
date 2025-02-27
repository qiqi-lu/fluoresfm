# Residual Channel Attention Network (RCAN)

# Y. Zhang, K. Li, K. Li, L. Wang, B. Zhong, and Y. Fu, “Image Super-Resolution Using
# Very Deep Residual Channel Attention Networks,” in Computer Vision-ECCV 2018, vol.
# 11211, V. Ferrari, M. Hebert, C. Sminchisescu, and Y. Weiss, Eds. Cham: Springer
# International Publishing, 2018, pp. 294-310.

from torch import nn
import math
import torch


class CALayer(nn.Module):
    """
    Channel Attention (CA) layer.
    """

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv_du = nn.Sequential(
            nn.Conv2d(
                in_channels=channel,
                out_channels=channel // reduction,
                kernel_size=1,
                padding=0,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=channel // reduction,
                out_channels=channel,
                kernel_size=1,
                padding=0,
                bias=True,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.avg_pool(x)
        w = self.conv_du(w)
        out = x * w
        return out


class RCAB(nn.Module):
    """
    Residual Channel Attention Block (RCAB)
    """

    def __init__(
        self, n_features, kernel_size, reduction, bias=True, bn=False, res_scale=1.0
    ):
        super().__init__()
        self.res_scale = res_scale

        modules_body = []
        for i in range(2):
            modules_body.append(
                nn.Conv2d(
                    in_channels=n_features,
                    out_channels=n_features,
                    kernel_size=kernel_size,
                    padding=(kernel_size // 2),
                    bias=bias,
                )
            )
            if bn == True:
                modules_body.append(nn.BatchNorm2d(num_features=n_features))
            if i == 0:
                modules_body.append(nn.ReLU(inplace=True))
        modules_body.append(CALayer(channel=n_features, reduction=reduction))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        out = res + x
        return out


class ResidualGroup(nn.Module):
    """
    Residual Group (GP)
    """

    def __init__(
        self, n_features, n_resblocks, kernel_size=3, reduction=16, res_scale=1.0
    ):
        super().__init__()
        modules_body = [
            RCAB(
                n_features=n_features,
                kernel_size=kernel_size,
                reduction=reduction,
                bias=True,
                bn=False,
                res_scale=res_scale,
            )
            for _ in range(n_resblocks)
        ]
        modules_body.append(
            nn.Conv2d(
                in_channels=n_features,
                out_channels=n_features,
                kernel_size=kernel_size,
                padding=(kernel_size // 2),
                bias=True,
            )
        )
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        out = res + x
        return out


class backbone(nn.Module):
    def __init__(
        self,
        n_colors=3,
        n_features=64,
        n_resblocks=20,
        n_resgroups=10,
        kernel_size=3,
        reduction=16,
        res_scale=1.0,
    ):
        super().__init__()
        # Shallow feature extraction
        self.head = nn.Conv2d(
            in_channels=n_colors,
            out_channels=n_features,
            kernel_size=kernel_size,
            padding=(kernel_size // 2),
            stride=1,
            bias=True,
        )

        # Residual in residaul (RIR) deep feature extraction
        modules_body = [
            ResidualGroup(
                n_features=n_features,
                kernel_size=kernel_size,
                reduction=reduction,
                res_scale=res_scale,
                n_resblocks=n_resblocks,
            )
            for _ in range(n_resgroups)
        ]
        modules_body.append(
            nn.Conv2d(
                in_channels=n_features,
                out_channels=n_features,
                kernel_size=kernel_size,
                padding=(kernel_size // 2),
                stride=1,
                bias=True,
            )
        )
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        fes = self.head(x)
        res = self.body(fes)
        out = res + fes
        return out


class RCAN(nn.Module):
    """
    Residaul Channel Attention Network (RCAN)
    """

    def __init__(
        self,
        scale=4,
        n_colors=3,
        n_resgroups=10,
        n_resblocks=20,
        n_features=64,
        kernel_size=3,
        reduction=16,
        res_scale=1.0,
    ):
        super().__init__()

        # feature extraction
        self.extractor = backbone(
            n_colors=n_colors,
            n_features=n_features,
            n_resblocks=n_resblocks,
            n_resgroups=n_resgroups,
            kernel_size=kernel_size,
            reduction=reduction,
            res_scale=res_scale,
        )

        # Upscale and reconstruction
        self.tail = nn.Sequential(
            Upsampler(
                scale=scale,
                n_features=n_features,
                kernel_size=kernel_size,
                bn=False,
                act=False,
                bias=True,
            ),
            nn.Conv2d(
                in_channels=n_features,
                out_channels=n_colors,
                kernel_size=kernel_size,
                padding=(kernel_size // 2),
                stride=1,
                bias=True,
            ),
        )

    def forward(self, x):
        fes = self.extractor(x)
        out = self.tail(fes)
        return out


class Upsampler(nn.Module):
    def __init__(
        self, scale, n_features, kernel_size=3, bn=False, act=False, bias=True
    ):
        super().__init__()

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(
                    nn.Conv2d(
                        in_channels=n_features,
                        out_channels=4 * n_features,
                        kernel_size=kernel_size,
                        padding=(kernel_size // 2),
                        bias=bias,
                    )
                )
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_features))
                if act:
                    m.append(nn.ReLU(inplace=True))
        elif scale == 3:
            m.append(
                nn.Conv2d(
                    in_channels=n_features,
                    out_channels=9 * n_features,
                    kernel_size=kernel_size,
                    bias=bias,
                    padding=(kernel_size // 2),
                )
            )
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_features))
            if act:
                m.append(nn.ReLU(inplace=True))
        else:
            raise NotImplementedError

        self.block = nn.Sequential(*m)

    def forward(self, x):
        x = self.block(x)
        return x


if __name__ == "__main__":
    import torchinfo

    model = RCAN(
        scale=2,
        n_colors=1,
        n_resgroups=10,
        n_resblocks=20,
        n_features=64,
        kernel_size=3,
        reduction=16,
        res_scale=1,
    )

    input = torch.ones(size=(2, 1, 64, 64))
    out = model(input)
    print(out.shape)
    torchinfo.summary(model=model, input_size=(1, 1, 64, 64))
