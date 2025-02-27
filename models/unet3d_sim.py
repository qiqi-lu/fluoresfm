import torch.nn as nn
import torch


class BaseBlock(nn.Module):
    """
    Conv-BN-ReLU-Conv-BN-ReLU.
    """

    def __init__(self, in_channels, out_channels, bias=True, use_bn=True):
        super().__init__()

        if len(out_channels) == 1:
            out_channels_conv1, out_channels_conv2 = out_channels, out_channels
        else:
            out_channels_conv1, out_channels_conv2 = out_channels

        modules = []

        # conv1
        modules.append(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels_conv1,
                kernel_size=3,
                padding=1,
                bias=bias,
            )
        )
        if use_bn:
            modules.append(nn.BatchNorm3d(num_features=out_channels_conv1))
        modules.append(nn.ReLU(inplace=True))

        # conv2
        modules.append(
            nn.Conv3d(
                in_channels=out_channels_conv1,
                out_channels=out_channels_conv2,
                kernel_size=3,
                padding=1,
                bias=bias,
            )
        )
        if use_bn:
            modules.append(nn.BatchNorm3d(num_features=out_channels_conv2))
        modules.append(nn.ReLU(inplace=True))

        self.module = nn.Sequential(*modules)

    def forward(self, x):
        x = self.module(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, mode="convtrans"):
        super().__init__()

        if mode == "convtrans":
            self.up = nn.ConvTranspose3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2,
            )
        if mode in ["trilinear", "nearest"]:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode=mode),
            )

    def forward(self, x):
        x = self.up(x)
        return x


class UNet3D_SIM(nn.Module):
    """
    "Deep learning enables structured illumination microscopy with low light levels and enhanced speed"
    replace the 2D convolution with 3D convolution.
    """

    def __init__(
        self, in_channels, out_channels, use_bn=True, bias=True, pos_out=False
    ):
        super().__init__()

        # ----------------------------------------------------------------------
        # parameters
        self.pos_out = pos_out
        in_channels_encoders = (in_channels, 64, 128, 256, 512)
        out_channels_encoders = (
            (64, 64),
            (128, 128),
            (256, 256),
            (512, 512),
            (1024, 1024),
        )

        in_channels_decoders = (512 + 512, 256 + 256, 128 + 128, 64 + 64)
        out_channels_decoders = ((512, 512), (256, 256), (128, 128), (64, 64))

        in_channels_ups = (1024, 512, 256, 128)
        out_channels_ups = (512, 256, 128, 64)

        # ----------------------------------------------------------------------
        # encoders
        self.encoders = nn.ModuleList([])
        for i in range(len(in_channels_encoders)):
            self.encoders.append(
                BaseBlock(
                    in_channels=in_channels_encoders[i],
                    out_channels=out_channels_encoders[i],
                    bias=bias,
                    use_bn=use_bn,
                )
            )

        # ----------------------------------------------------------------------
        # decoders
        self.decoders = nn.ModuleList([])
        for i in range(len(in_channels_decoders)):
            self.decoders.append(
                BaseBlock(
                    in_channels=in_channels_decoders[i],
                    out_channels=out_channels_decoders[i],
                    bias=bias,
                    use_bn=use_bn,
                )
            )

        # ----------------------------------------------------------------------
        # downsampling and upsampling
        self.down_sampling = nn.MaxPool3d(kernel_size=2)

        self.up_samplings = nn.ModuleList([])
        for i in range(len(in_channels_ups)):
            self.up_samplings.append(
                Up(
                    in_channels=in_channels_ups[i],
                    out_channels=out_channels_ups[i],
                    mode="convtrans",
                )
            )

        # ----------------------------------------------------------------------
        # last conv
        self.last_conv = nn.Conv3d(
            in_channels=out_channels_decoders[-1][-1],
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            bias=bias,
        )

    def forward(self, x):
        feature_down_1 = self.encoders[0](x)

        feature_down_2 = self.down_sampling(feature_down_1)
        feature_down_2 = self.encoders[1](feature_down_2)

        feature_down_3 = self.down_sampling(feature_down_2)
        feature_down_3 = self.encoders[2](feature_down_3)

        feature_down_4 = self.down_sampling(feature_down_3)
        feature_down_4 = self.encoders[3](feature_down_4)

        feature_down_5 = self.down_sampling(feature_down_4)
        feature_down_5 = self.encoders[4](feature_down_5)

        feature_up_4 = self.up_samplings[0](feature_down_5)
        feature_up_4 = torch.cat([feature_down_4, feature_up_4], dim=1)
        feature_up_4 = self.decoders[0](feature_up_4)

        feature_up_3 = self.up_samplings[1](feature_up_4)
        feature_up_3 = torch.cat([feature_down_3, feature_up_3], dim=1)
        feature_up_3 = self.decoders[1](feature_up_3)

        feature_up_2 = self.up_samplings[2](feature_up_3)
        feature_up_2 = torch.cat([feature_down_2, feature_up_2], dim=1)
        feature_up_2 = self.decoders[2](feature_up_2)

        feature_up_1 = self.up_samplings[3](feature_up_2)
        feature_up_1 = torch.cat([feature_down_1, feature_up_1], dim=1)
        feature_up_1 = self.decoders[3](feature_up_1)

        out = self.last_conv(feature_up_1)
        if self.pos_out == True:
            out = torch.abs(out)
        return out


if __name__ == "__main__":
    from torchinfo import summary

    device = torch.device("cuda:0")

    x = torch.ones(size=(1, 1, 128, 128, 128)).to(device)
    model = UNet3D_SIM(
        in_channels=1,
        out_channels=1,
        use_bn=True,
        bias=True,
    )

    model = model.to(device)
    o = model(x)
    print(o.shape)
    summary(model, input_size=(1, 1, 128, 128, 128))
