# RCAN 3D
import torch
import torch.nn as nn


class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_dims=3, kernel_size=3):
        super().__init__()

        if num_dims == 3:
            conv = nn.Conv3d
        if num_dims == 2:
            conv = nn.Conv2d

        self.conv = conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class CAB(nn.Module):
    """Channel attention block."""

    def __init__(self, in_channels, num_dims=3, reduction=1):
        super().__init__()

        self.num_dims = num_dims
        self.block = nn.Sequential(
            SingleConv(
                in_channels=in_channels,
                out_channels=in_channels // reduction,
                num_dims=num_dims,
                kernel_size=1,
            ),
            nn.ReLU(inplace=True),
            SingleConv(
                in_channels=in_channels // reduction,
                out_channels=in_channels,
                num_dims=num_dims,
                kernel_size=1,
            ),
        )

    def forward(self, x):
        if self.num_dims == 3:
            att = torch.mean(x, dim=(2, 3, 4), keepdim=True)
        if self.num_dims == 2:
            att = torch.mean(x, dim=(2, 3), keepdim=True)
        att = self.block(att)
        att = torch.sigmoid(att)
        x = x * att
        return x


class RCAB(nn.Module):
    def __init__(
        self,
        in_channels,
        num_dims=3,
        channel_reduction=8,
        residual_scaling=1.0,
    ):
        super().__init__()
        self.residual_scaling = residual_scaling

        self.conv1 = SingleConv(
            in_channels=in_channels,
            out_channels=in_channels,
            num_dims=num_dims,
            kernel_size=3,
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = SingleConv(
            in_channels=in_channels,
            out_channels=in_channels,
            num_dims=num_dims,
            kernel_size=3,
        )
        self.cab = CAB(
            in_channels=in_channels, num_dims=num_dims, reduction=channel_reduction
        )

    def forward(self, x):
        skip = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        x = self.cab(x)
        if self.residual_scaling != 1.0:
            x = x * self.residual_scaling
        x = x + skip
        return x


class ResidualGroup(nn.Module):
    def __init__(
        self,
        in_channels,
        num_residual_block=3,
        num_dims=3,
        channel_reduction=8,
        residual_scaling=1.0,
    ):
        super().__init__()
        blocks = []
        for _ in range(num_residual_block):
            blocks.append(
                RCAB(
                    in_channels=in_channels,
                    num_dims=num_dims,
                    channel_reduction=channel_reduction,
                    residual_scaling=residual_scaling,
                )
            )
        blocks.append(
            SingleConv(
                in_channels=in_channels,
                out_channels=in_channels,
                num_dims=num_dims,
                kernel_size=3,
            )
        )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        short_skip = x
        x = self.blocks(x)
        x = x + short_skip
        return x


class RCAN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_dims=3,
        fea_channels=32,
        num_residual_blocks=3,
        num_residual_groups=5,
        channel_resuction=8,
        residual_scaling=1.0,
        enable_standardize=True,
        pos_out=False,
    ):
        super().__init__()
        self.num_residual_groups = num_residual_groups
        self.enable_standardize = enable_standardize
        self.pos_out = pos_out

        self.conv_head = SingleConv(
            in_channels=in_channels,
            out_channels=fea_channels,
            num_dims=num_dims,
            kernel_size=3,
        )

        self.res_groups = nn.ModuleList()
        for _ in range(num_residual_groups):
            self.res_groups.append(
                ResidualGroup(
                    in_channels=fea_channels,
                    num_residual_block=num_residual_blocks,
                    num_dims=num_dims,
                    channel_reduction=channel_resuction,
                    residual_scaling=residual_scaling,
                )
            )

        self.conv_rr = SingleConv(
            in_channels=fea_channels,
            out_channels=fea_channels,
            num_dims=num_dims,
            kernel_size=3,
        )

        self.conv_tail = SingleConv(
            in_channels=fea_channels,
            out_channels=out_channels,
            num_dims=num_dims,
            kernel_size=3,
        )

    def forward(self, x):
        if self.enable_standardize:
            x = self.standardize(x)

        x = self.conv_head(x)

        long_skip = x
        for i in range(self.num_residual_groups):
            x = self.res_groups[i](x)
        x = self.conv_rr(x)
        x = x + long_skip

        x = self.conv_tail(x)

        if self.enable_standardize:
            x = self.destandardize(x)

        if self.pos_out:
            x = torch.abs(x)

        return x

    def standardize(self, x):
        """Assuming the original range is [0,1]"""
        return 2.0 * x - 1

    def destandardize(self, x):
        """Undo standardize."""
        return 0.5 * x + 0.5

    def gauss(self, x):
        return torch.exp(-(x**2))


if __name__ == "__main__":
    from torchinfo import summary

    device = torch.device("cuda:0")
    x = torch.ones(size=(1, 1, 64, 64, 64)).to(device)
    model = RCAN(
        in_channels=1,
        out_channels=1,
        num_dims=3,
        fea_channels=32,
        num_residual_blocks=5,
        num_residual_groups=5,
        channel_resuction=8,
        residual_scaling=1.0,
        enable_standardize=True,
        pos_out=False,
    )

    model = model.to(device)
    o = model(x)
    print(o.shape)
    summary(model, input_size=(1, 1, 64, 64, 64))
