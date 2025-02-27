"""
Original UNet.
@ https://github.com/milesial/Pytorch-UNet/tree/master
"""

import torch.nn as nn
import torch


class DoubleConv(nn.Module):
    """
    (convolution => [BN] => ReLU) * 2.
    """

    def __init__(self, in_channels, out_channels, mid_channels=None, use_bn=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        dc = []
        dc.append(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        )
        if use_bn == True:
            dc.append(nn.BatchNorm2d(mid_channels))
        dc.append(nn.ReLU(inplace=True))
        dc.append(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        )
        if use_bn == True:
            dc.append(nn.BatchNorm2d(out_channels))
        dc.append(nn.ReLU(inplace=True))

        self.double_conv = nn.Sequential(*dc)

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling with maxpool then double conv.
    """

    def __init__(self, in_channels, out_channels, use_bn=True):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, use_bn=use_bn),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling then double conv.
    """

    def __init__(self, in_channels, out_channels, bilinear=True, use_bn=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(
                in_channels, out_channels, in_channels // 2, use_bn=use_bn
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels, use_bn=use_bn)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(
            x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
        )
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self, in_channels, out_channels, bilinear=False, residual=False, pos_out=False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.residual = residual
        self.pos_out = pos_out

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, out_channels)

    def forward(self, x):
        inpt = x
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        res = self.outc(x)
        if self.residual:
            out = res + inpt
        if self.pos_out:
            out = torch.abs(out)
        return out


class backbone(nn.Module):
    """
    Light-weight UNet backbone.
    """

    def __init__(self, in_channels, bilinear=False, use_bn=False):
        super().__init__()
        self.inc = DoubleConv(in_channels=in_channels, out_channels=8, use_bn=use_bn)
        self.down1 = Down(in_channels=8, out_channels=16, use_bn=use_bn)
        self.down2 = Down(in_channels=16, out_channels=32, use_bn=use_bn)
        self.up1 = Up(in_channels=32, out_channels=16, bilinear=bilinear, use_bn=use_bn)
        self.up2 = Up(in_channels=16, out_channels=8, bilinear=bilinear, use_bn=use_bn)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        return x


class UNet_lw(nn.Module):
    """
    Light-weight UNet backbone.
    """

    def __init__(self, in_channels, out_channels, bilinear=False, use_bn=False):
        super().__init__()
        self.backbone = backbone(
            in_channels=in_channels, bilinear=bilinear, use_bn=use_bn
        )
        self.convout = OutConv(in_channels=8, out_channels=out_channels)

    def forward(self, x):
        x = self.backbone(x)
        out = self.convout(x)
        return out


if __name__ == "__main__":
    from torchinfo import summary

    x = torch.ones(size=(2, 1, 128, 128))
    # model = UNet_lw(in_channels=1, out_channels=3, bilinear=False, use_bn=False)
    model = UNet(
        in_channels=1, out_channels=1, bilinear=False, residual=False, pos_out=False
    )
    o = model(x)
    print(o.shape)
    summary(model=model, input_size=(1, 1, 128, 128))
