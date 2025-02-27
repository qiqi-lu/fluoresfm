# RCAN 3D
import torch
import torch.nn as nn
import sys

sys.path.insert(1, "E:\qiqilu\Project\\2024 Foundation model\code")
import models.PSFmodels as PSFModel


class SqueezeBlock(nn.Module):
    def __init__(self, in_channels, num_dims=3, kernel_size=3, groups=1):
        super().__init__()

        if num_dims == 3:
            conv = nn.Conv3d
            self.avg_dims = (2, 3, 4)

        if num_dims == 2:
            conv = nn.Conv2d
            self.avg_dims = (2, 3)

        self.block = nn.Sequential(
            conv(
                in_channels=in_channels,
                out_channels=in_channels,
                groups=groups,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.ReLU(inplace=True),
            conv(
                in_channels=in_channels,
                out_channels=in_channels,
                groups=groups,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        x = torch.mean(input=x, dim=self.avg_dims)
        return x


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
        if num_dims == 3:
            self.avg_dims = (2, 3, 4)
        if num_dims == 2:
            self.avg_dims = (2, 3)

    def forward(self, x):
        skip = x
        x = torch.mean(x, dim=self.avg_dims, keepdim=True)
        x = self.block(x)
        x = torch.sigmoid(x)
        x = x * skip
        return x


class RCAB(nn.Module):
    """Residual Channel Attention Block"""

    def __init__(
        self, in_channels, num_dims=3, channel_reduction=8, residual_scaling=1.0
    ):
        super().__init__()
        self.residual_scaling = residual_scaling

        self.block = nn.Sequential(
            SingleConv(
                in_channels=in_channels,
                out_channels=in_channels,
                num_dims=num_dims,
                kernel_size=3,
            ),
            nn.ReLU(inplace=True),
            SingleConv(
                in_channels=in_channels,
                out_channels=in_channels,
                num_dims=num_dims,
                kernel_size=3,
            ),
            CAB(
                in_channels=in_channels, num_dims=num_dims, reduction=channel_reduction
            ),
        )

    def forward(self, x):
        skip = x
        x = self.block(x)
        if self.residual_scaling != 1.0:
            x = x * self.residual_scaling
        x = x + skip
        return x


class BlockGroup(nn.Module):
    """Group of blocks"""

    def __init__(
        self,
        in_channels,
        num_blocks=3,
        num_dims=3,
        block_type="RCAB",
        channel_reduction=8,
        residual_scaling=1.0,
    ):
        super().__init__()

        blocks = []
        for _ in range(num_blocks):
            if block_type == "RCAB":
                blocks.append(
                    RCAB(
                        in_channels=in_channels,
                        num_dims=num_dims,
                        channel_reduction=channel_reduction,
                        residual_scaling=residual_scaling,
                    )
                )
            if block_type == "Swin":
                pass
            if block_type == "Mamba":
                pass

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
        skip = x
        x = self.blocks(x)
        x = x + skip
        return x


class TeeResNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_dims=3,
        num_features=32,
        num_blocks=3,
        num_groups=5,
        block_type="RCAB",
        channel_reduction=8,
        residual_scaling=1.0,
        enable_standardize=True,
        pos_out=False,
        psf_model="bw",
        enable_groups=False,
        psf_size=(127, 127, 127),
        kernel_norm=True,
        over_sampling=2,
        center_one=True,
        psf_part_freeze=False,
        img_part_freeze=False,
        head_tail_freeze=False,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.enable_standardize = enable_standardize
        self.pos_out = pos_out

        # ----------------------------------------------------------------------
        # Head
        # ----------------------------------------------------------------------
        self.head = SingleConv(
            in_channels=in_channels,
            out_channels=num_features,
            num_dims=num_dims,
            kernel_size=3,
        )

        # ----------------------------------------------------------------------
        # Base
        # ----------------------------------------------------------------------
        self.base = nn.ModuleList()
        for _ in range(num_groups):
            self.base.append(
                BlockGroup(
                    block_type=block_type,
                    in_channels=num_features,
                    num_blocks=num_blocks,
                    num_dims=num_dims,
                    channel_reduction=channel_reduction,
                    residual_scaling=residual_scaling,
                )
            )

        self.base.append(
            SingleConv(
                in_channels=num_features,
                out_channels=num_features,
                num_dims=num_dims,
                kernel_size=3,
            )
        )

        # ----------------------------------------------------------------------
        # Tail
        # ----------------------------------------------------------------------
        self.tail = SingleConv(
            in_channels=num_features,
            out_channels=out_channels,
            num_dims=num_dims,
            kernel_size=3,
        )

        # ----------------------------------------------------------------------
        # PSF
        # ----------------------------------------------------------------------
        # squeezers
        base_block_sq = SqueezeBlock

        self.squeezers = nn.ModuleList([])
        for _ in range(num_groups):
            groups = num_features if enable_groups else 1
            self.squeezers.append(
                base_block_sq(in_channels=num_features, groups=groups)
            )

        if psf_model == "half":
            self.psf_generator = PSFModel.HalfPlane(
                kernel_size=psf_size,
                kernel_norm=kernel_norm,
                over_sampling=over_sampling,
                center_one=center_one,
            )

        if psf_model == "bw":
            self.psf_generator = PSFModel.BWModel(
                kernel_size=psf_size,
                kernel_norm=kernel_norm,
                num_integral=100,
                over_sampling=over_sampling,
                pixel_size_z=1,
            )

        self.fc = nn.Sequential(
            nn.Linear(in_features=num_features * num_groups, out_features=64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=self.psf_generator.get_num_params()),
        )

        # ----------------------------------------------------------------------
        if psf_part_freeze == True:
            for params in self.get_psf_part_params():
                params.requires_grad = False

        if img_part_freeze == True:
            for params in self.get_base_params():
                params.requires_grad = False

        if head_tail_freeze == True:
            for params in self.get_head_tail_params():
                params.requires_grad = False

    def get_psf_part_params(self):
        params = []
        params.extend(self.squeezers.parameters())
        params.extend(self.fc.parameters())
        return params

    def get_base_params(self):
        params = []
        params.extend(self.base.parameters())
        return params

    def get_head_tail_params(self):
        params = []
        params.extend(self.head.parameters())
        params.extend(self.tail.parameters())
        return params

    def forward(self, x):
        if self.enable_standardize:
            x = self.standardize(x)

        x = self.head(x)

        long_skip = x
        fea_sq = []

        for i in range(self.num_groups):
            x = self.base[i](x)
            fea_sq.append(self.squeezers[i](x))
        x = self.base[-1](x)

        x = x + long_skip

        x = self.tail(x)

        if self.enable_standardize:
            x = self.destandardize(x)

        if self.pos_out:
            x = torch.abs(x)

        # ----------------------------------------------------------------------
        # PSF
        fea_cat = torch.concat(fea_sq, dim=1)
        params = self.fc(fea_cat)
        params = torch.reshape(params, shape=(fea_cat.shape[0], 1, -1))
        psf = self.psf_generator(params)

        return x, psf

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
    model = TeeResNet(
        in_channels=1,
        out_channels=1,
        num_dims=3,
        num_features=32,
        num_blocks=3,
        num_groups=5,
        block_type="RCAB",
        channel_reduction=8,
        residual_scaling=1.0,
        enable_standardize=True,
        pos_out=False,
        psf_model="bw",
        enable_groups=False,
        psf_size=(127, 127, 127),
        kernel_norm=True,
        over_sampling=2,
        center_one=True,
        psf_part_freeze=False,
        img_part_freeze=False,
        head_tail_freeze=False,
    )

    model = model.to(device)
    o, psf = model(x)
    print(o.shape, psf.shape)
    summary(model, input_size=(1, 1, 64, 64, 64))
