"""
Content-aware imgae restoration (CARE) based on deep learning.
** Network architecture used for isotropic resonctrution.

https://github.com/CSBDeep/CSBDeep/blob/main/csbdeep/internals/nets.py

Ref:
Weigert, M., Schmidt, U., Boothe, T., Müller, A., Dibrov, A., Jain, A., 
Wilhelm, B., Schmidt, D., Broaddus, C., Culley, S., et al. (2018). 
Content-aware image restoration: pushing the limits of fluorescence microscopy. 
Nat Methods 15, 1090–1097. https://doi.org/10.1038/s41592-018-0216-7.

"""

import torch.nn as nn
import torch, torchinfo


class ConvBlock2D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dropout=0.0,
        batch_norm=False,
        **kwargs
    ):
        super().__init__()
        blocks = []
        blocks.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                **kwargs
            )
        )
        if batch_norm:
            blocks.append(nn.BatchNorm2d(num_features=out_channels)),
            blocks.append(nn.ReLU(inplace=True))
        if dropout is not None and dropout > 0:
            blocks.append(nn.Dropout2d(p=dropout))

        self.base = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.base(x)
        return x


class ConvBlock3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dropout=0.0,
        batch_norm=False,
        **kwargs
    ):
        super().__init__()
        blocks = []
        blocks.append(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                **kwargs
            )
        )
        if batch_norm:
            blocks.append(nn.BatchNorm3d(num_features=out_channels)),
            blocks.append(nn.ReLU(inplace=True))
        if dropout is not None and dropout > 0:
            blocks.append(nn.Dropout3d(p=dropout))

        self.base = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.base(x)
        return x


class UNetBlock(nn.Module):
    def __init__(
        self,
        in_channels=1,
        n_filter_base=16,
        kernel_size=3,
        batch_norm=False,
        dropout=0.0,
        expansion=2,
    ):
        super().__init__()

        self.conv1 = ConvBlock2D(
            in_channels, n_filter_base * expansion, kernel_size, dropout, batch_norm
        )  # 32

        self.conv2 = ConvBlock2D(
            n_filter_base * expansion,
            n_filter_base * expansion,
            kernel_size,
            dropout,
            batch_norm,
        )  # 32

        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)  # 32

        self.conv3 = ConvBlock2D(
            n_filter_base * expansion,
            n_filter_base * expansion * 2,
            kernel_size,
            dropout,
            batch_norm,
        )  # 64

        self.conv4 = ConvBlock2D(
            n_filter_base * expansion * 2,
            n_filter_base * expansion * 2,
            kernel_size,
            dropout,
            batch_norm,
        )  # 64

        self.conv5 = ConvBlock2D(
            n_filter_base * expansion * 2,
            n_filter_base * expansion * 4,
            kernel_size,
            dropout,
            batch_norm,
        )  # 128

        self.conv6 = ConvBlock2D(
            n_filter_base * expansion * 4,
            n_filter_base * expansion * 2,
            kernel_size,
            dropout,
            batch_norm,
        )  # 64

        self.unsample1 = nn.UpsamplingNearest2d(scale_factor=2)  # 64
        self.unsample2 = nn.UpsamplingNearest2d(scale_factor=2)  # 64

        self.conv7 = ConvBlock2D(
            n_filter_base * expansion * 4,
            n_filter_base * expansion * 2,
            kernel_size,
            dropout,
            batch_norm,
        )  # 64

        self.conv8 = ConvBlock2D(
            n_filter_base * expansion * 2,
            n_filter_base * expansion,
            kernel_size,
            dropout,
            batch_norm,
        )  # 32

        self.conv9 = ConvBlock2D(
            n_filter_base * expansion * 2,
            n_filter_base * expansion,
            kernel_size,
            dropout,
            batch_norm,
        )  # 32

        self.conv10 = ConvBlock2D(
            n_filter_base * expansion,
            n_filter_base * expansion,
            kernel_size,
            dropout,
            batch_norm,
        )  # 32

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.max_pooling(x2)
        x3 = self.conv3(x3)
        x4 = self.conv4(x3)
        x5 = self.max_pooling(x4)
        x5 = self.conv5(x5)
        x6 = self.conv6(x5)
        x6 = self.unsample1(x6)
        x7 = self.conv7(torch.cat([x4, x6], dim=1))
        x8 = self.conv8(x7)
        x8 = self.unsample2(x8)
        x9 = self.conv9(torch.cat([x2, x8], dim=1))
        x10 = self.conv10(x9)
        return x10


class CARE(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        n_filter_base=16,
        kernel_size=3,
        batch_norm=False,
        dropout=0.0,
        residual=False,
        expansion=2,
        pos_out=False,
    ):
        self.residual = residual
        self.pos_out = pos_out
        super().__init__()
        self.unet = UNetBlock(
            in_channels=in_channels,
            n_filter_base=n_filter_base,
            kernel_size=kernel_size,
            batch_norm=batch_norm,
            dropout=dropout,
        )

        self.conv = nn.Conv2d(
            in_channels=n_filter_base * expansion,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            stride=1,
        )
        if self.pos_out:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.unet(x)
        out = self.conv(out)
        if self.residual:
            out = out + x
        if self.pos_out:
            out = self.act(out)
        return out


if __name__ == "__main__":
    x = torch.ones((2, 1, 64, 64))
    care = CARE(
        in_channels=1,
        out_channels=1,
        n_filter_base=16,
        kernel_size=5,
        batch_norm=False,
        dropout=0.0,
        residual=True,
        expansion=2,
    )
    o = care(x)
    print(o)
    torchinfo.summary(care, input_size=(2, 1, 64, 64))
