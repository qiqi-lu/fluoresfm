import torch.nn as nn
import torch
import sys

sys.path.insert(1, "E:\qiqilu\Project\\2024 Foundation model\code")
import models.PSFmodels as PSFModel


def gauss(x):
    x = torch.exp(-torch.square(x))
    return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        return self.conv(x)


class Feature2Kernel(nn.Module):
    def __init__(self, in_channels, out_channels, out_size):
        super().__init__()
        self.out_size = out_size
        self.conv = nn.Conv3d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1
        )
        self.pooling = nn.AvgPool3d(kernel_size=2)

    def forward(self, x):
        in_size = x.shape[2:]
        crop_out = (
            in_size[0] - self.out_size[0],
            in_size[1] - self.out_size[1],
            in_size[2] - self.out_size[2],
        )

        x = self.conv(x)
        x = nn.functional.interpolate(input=x, scale_factor=2, mode="nearest")
        x = x[
            :,
            :,
            crop_out[0] : crop_out[0] + self.out_size[0] * 2,
            crop_out[1] : crop_out[1] + self.out_size[1] * 2,
            crop_out[2] : crop_out[2] + self.out_size[2] * 2,
        ]
        x = self.pooling(x)
        x = torch.abs(x)
        x = torch.div(x, torch.sum(x, dim=(2, 3, 4), keepdim=True))
        return x


class DoubleConvBlock(nn.Module):
    """
    Conv-BN-ReLU-Conv-BN-ReLU.
    """

    def __init__(self, in_channels, out_channels, bias=True, use_bn=True):
        super().__init__()

        if isinstance(out_channels, int):
            out_channels_conv1 = out_channels
            out_channels_conv2 = out_channels
        elif len(out_channels) == 1:
            out_channels_conv1 = out_channels[0]
            out_channels_conv2 = out_channels[0]
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


class ResConvBlock(nn.Module):
    """
    From `Uformer`.
    https://github.com/ZhendongWang6/Uformer/blob/main/model.py#L781
    """

    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()

        if len(out_channels) == 1 or isinstance(out_channels, int):
            out_channels_conv1 = out_channels[0]
            out_channels_conv2 = out_channels[0]
        else:
            out_channels_conv1, out_channels_conv2 = out_channels

        self.module = nn.Sequential(
            nn.Conv3d(
                in_channels, out_channels_conv1, kernel_size=3, padding=1, bias=bias
            ),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(
                out_channels_conv1,
                out_channels_conv2,
                kernel_size=3,
                padding=1,
                bias=bias,
            ),
            nn.LeakyReLU(inplace=True),
        )

        self.shortcut = nn.Conv2d(
            in_channels, out_channels_conv2, kernel_size=1, bias=bias
        )

    def forward(self, x):
        out = self.module(x)
        sc = self.shortcut(x)
        out = out + sc
        return out


class SingleConvBlock(nn.Module):
    """
    Single convolution with kernel size = 1.
    """

    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            bias=bias,
        )

    def forward(self, x):
        x = self.conv(x)
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


class Down(nn.Module):
    def __init__(self):
        super().__init__()
        self.pooling = nn.MaxPool3d(kernel_size=2)

    def forward(self, x):
        x = self.pooling(x)
        return x


class SqueezeBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=3, groups=1, bias=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=in_channels,
                groups=groups,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=bias,
            ),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=in_channels,
                groups=groups,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=bias,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        x_squeeze = torch.mean(input=x, dim=(2, 3, 4))
        return x_squeeze


class ExtractBlock(nn.Module):
    def __init__(self, in_channels, bias=True):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=1,
                bias=bias,
            ),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=1,
                bias=bias,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class RCABlock(nn.Module):
    """
    Residual Channel Attention Block
    """

    def __init__(self, in_channels, bias=True, return_gap=False):
        super().__init__()
        self.return_gap = return_gap

        self.conv_conv = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                padding=1,
                bias=bias,
            ),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                padding=1,
                bias=bias,
            ),
        )

        self.conv_conv_sigmoid = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=1,
                bias=bias,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=1,
                bias=bias,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_conv = self.conv_conv(x)
        x_gap = torch.mean(input=x_conv, dim=(2, 3, 4), keepdim=True)
        x_gap_conv = self.conv_conv_sigmoid(x_gap)
        x_mul = x_conv * x_gap_conv
        x_add = x_mul + x

        if self.return_gap:
            return x_add, x_gap
        else:
            return x_add


class DoubleConvBlock1x1(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()

        if not isinstance(out_channels, list):
            out_channels = [in_channels, out_channels]

        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels[0],
                kernel_size=1,
                bias=bias,
            ),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                in_channels=out_channels[0],
                out_channels=out_channels[1],
                kernel_size=1,
                bias=bias,
            ),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class TeeNet3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        use_bn=True,
        bias=True,
        encoder_type="doubleconv",
        decoder_type="doubleconv",
        psf_size=(127, 127, 127),
    ):
        super().__init__()

        # ----------------------------------------------------------------------
        # parameters
        out_channels_head = 32

        in_channels_encoders = (out_channels_head, 64, 128, 256, 512)
        out_channels_encoders = (64, 128, 256, 512, 1024)

        in_channels_ups_psf = (1024, 512, 256, 128)
        out_channels_ups_psf = (512, 256, 128, 64)

        in_channels_decoders_psf = (512 + 512, 256 + 256, 128 + 128, 64 + 64)
        out_channels_decoders_psf = (512, 256, 128, 64)

        in_channels_ups_img = (1024, 512, 256, 128)
        out_channels_ups_img = (512, 256, 128, 64)

        in_channels_decoders_img = (
            512 + 512 + 512,
            256 + 256 + 256,
            128 + 128 + 128,
            64 + 64 + 64,
        )
        out_channels_decoders_img = (512, 256, 128, 64)

        if encoder_type == "doubleconv":
            baseblock_encoder = DoubleConvBlock

        if decoder_type == "doubleconv":
            baseblock_decoder_img = DoubleConvBlock
            baseblock_decoder_psf = DoubleConvBlock

        # ----------------------------------------------------------------------
        # head
        self.head = ConvBlock(in_channels=in_channels, out_channels=out_channels_head)

        # ----------------------------------------------------------------------
        # encoders
        self.encoders = nn.ModuleList([])
        for i in range(len(in_channels_encoders)):
            self.encoders.append(
                baseblock_encoder(
                    in_channels=in_channels_encoders[i],
                    out_channels=(out_channels_encoders[i],),
                    bias=bias,
                    use_bn=use_bn,
                )
            )

        # ----------------------------------------------------------------------
        # decoders (psf)
        self.decoders_psf = nn.ModuleList([])
        for i in range(len(in_channels_decoders_psf)):
            self.decoders_psf.append(
                baseblock_decoder_psf(
                    in_channels=in_channels_decoders_psf[i],
                    out_channels=(out_channels_decoders_psf[i],),
                    bias=bias,
                    use_bn=use_bn,
                )
            )

        # ----------------------------------------------------------------------
        # decoders (img)
        self.decoders_img = nn.ModuleList([])
        for i in range(len(in_channels_decoders_img)):
            self.decoders_img.append(
                baseblock_decoder_img(
                    in_channels=in_channels_decoders_img[i],
                    out_channels=(out_channels_decoders_img[i],),
                    bias=bias,
                    use_bn=use_bn,
                )
            )

        # ----------------------------------------------------------------------
        # downsampling
        self.down_sampling = Down()

        # ----------------------------------------------------------------------
        # upsampling (psf)
        self.up_samplings_psf = nn.ModuleList([])
        for i in range(len(in_channels_decoders_psf)):
            self.up_samplings_psf.append(
                Up(
                    in_channels=in_channels_ups_psf[i],
                    out_channels=out_channels_ups_psf[i],
                    mode="convtrans",
                )
            )

        # ----------------------------------------------------------------------
        # upsampling (img)
        self.up_samplings_img = nn.ModuleList([])
        for i in range(len(in_channels_decoders_img)):
            self.up_samplings_img.append(
                Up(
                    in_channels=in_channels_ups_img[i],
                    out_channels=out_channels_ups_img[i],
                    mode="convtrans",
                )
            )

        # ----------------------------------------------------------------------
        # tail (psf)
        self.tail_psf = Feature2Kernel(
            in_channels=out_channels_decoders_psf[-1],
            out_channels=out_channels,
            out_size=psf_size,
        )

        # ----------------------------------------------------------------------
        # tail (img)
        self.tail_img = SingleConvBlock(
            in_channels=out_channels_decoders_img[-1],
            out_channels=out_channels,
            bias=bias,
        )

    def forward(self, x):
        # head
        fea_head = self.head(x)  # (N, 32, 128, 128, 128)

        # encode
        fea_down_1 = self.encoders[0](fea_head)  # (N, 64, 128, 128, 128)

        fea_down_2 = self.down_sampling(fea_down_1)
        fea_down_2 = self.encoders[1](fea_down_2)  # (N, 128, 64, 64, 64)

        fea_down_3 = self.down_sampling(fea_down_2)
        fea_down_3 = self.encoders[2](fea_down_3)  # (N, 256, 32, 32, 32)

        fea_down_4 = self.down_sampling(fea_down_3)
        fea_down_4 = self.encoders[3](fea_down_4)  # (N, 512, 16, 16, 16)

        fea_down_5 = self.down_sampling(fea_down_4)
        fea_down_5 = self.encoders[4](fea_down_5)  # (N, 1024, 8, 8, 8)

        # decoders (psf)
        fea_up_4_psf = self.up_samplings_psf[0](fea_down_5)  # (N, 512, 16, 16, 16)
        fea_up_4_psf = torch.cat([fea_down_4, fea_up_4_psf], dim=1)
        fea_up_4_psf = self.decoders_psf[0](fea_up_4_psf)  # (N, 512, 16, 16, 16)

        fea_up_3_psf = self.up_samplings_psf[1](fea_up_4_psf)  # (N, 256, 32, 32, 32)
        fea_up_3_psf = torch.cat([fea_down_3, fea_up_3_psf], dim=1)
        fea_up_3_psf = self.decoders_psf[1](fea_up_3_psf)  # (N, 256, 32, 32, 32)

        fea_up_2_psf = self.up_samplings_psf[2](fea_up_3_psf)  # (N, 128, 64, 64, 64)
        fea_up_2_psf = torch.cat([fea_down_2, fea_up_2_psf], dim=1)
        fea_up_2_psf = self.decoders_psf[2](fea_up_2_psf)  # (N, 128, 64, 64, 64)

        fea_up_1_psf = self.up_samplings_psf[3](fea_up_2_psf)  # (N, 64, 128, 128, 128)
        fea_up_1_psf = torch.cat([fea_down_1, fea_up_1_psf], dim=1)
        fea_up_1_psf = self.decoders_psf[3](fea_up_1_psf)  # (N, 64, 128, 128, 128)

        out_psf = self.tail_psf(fea_up_1_psf)

        # decoders (img)
        fea_up_4_img = self.up_samplings_img[0](fea_down_5)  # (N, 512, 16, 16, 16)
        fea_up_4_img = torch.cat([fea_down_4, fea_up_4_img, fea_up_4_psf], dim=1)
        fea_up_4_img = self.decoders_img[0](fea_up_4_img)  # (N, 512, 16, 16, 16)

        fea_up_3_img = self.up_samplings_img[1](fea_up_4_img)  # (N, 256, 32, 32, 32)
        fea_up_3_img = torch.cat([fea_down_3, fea_up_3_img, fea_up_3_psf], dim=1)
        fea_up_3_img = self.decoders_img[1](fea_up_3_img)  # (N, 256, 32, 32, 32)

        fea_up_2_img = self.up_samplings_img[2](fea_up_3_img)  # (N, 128, 64, 64, 64)
        fea_up_2_img = torch.cat([fea_down_2, fea_up_2_img, fea_up_2_psf], dim=1)
        fea_up_2_img = self.decoders_img[2](fea_up_2_img)  # (N, 128, 64, 64, 64)

        fea_up_1_img = self.up_samplings_img[3](fea_up_2_img)  # (N, 64, 128, 128, 128)
        fea_up_1_img = torch.cat([fea_down_1, fea_up_1_img, fea_up_1_psf], dim=1)
        fea_up_1_img = self.decoders_img[3](fea_up_1_img)  # (N, 64, 128, 128, 128)

        out_img = self.tail_img(fea_up_1_img)
        out_img = torch.abs(out_img + x)

        return out_img, out_psf


class TeeNet3D_Att(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        use_bn=True,
        bias=True,
        encoder_type="doubleconv",
        decoder_type="doubleconv",
        psf_size=(127, 127, 127),
        use_att=True,
    ):
        super().__init__()
        self.use_att = use_att

        # ----------------------------------------------------------------------
        # parameters
        out_channels_head = 32

        in_channels_encoders = (32, 64, 128, 256, 512)
        out_channels_encoders = (64, 128, 256, 512, 1024)

        in_channels_ups_img = (1024, 512, 256, 128)
        out_channels_ups_img = (512, 256, 128, 64)

        in_channels_decoders_img = (512 + 512, 256 + 256, 128 + 128, 64 + 64)
        out_channels_decoders_img = (512, 256, 128, 64)

        if encoder_type == "doubleconv":
            baseblock_encoder = DoubleConvBlock

        if decoder_type == "doubleconv":
            baseblock_decoder_img = DoubleConvBlock

        # ----------------------------------------------------------------------
        # head
        self.head = ConvBlock(in_channels=in_channels, out_channels=out_channels_head)

        # ----------------------------------------------------------------------
        # encoders
        self.encoders = nn.ModuleList([])
        for i in range(len(in_channels_encoders)):
            self.encoders.append(
                baseblock_encoder(
                    in_channels=in_channels_encoders[i],
                    out_channels=out_channels_encoders[i],
                    bias=bias,
                    use_bn=use_bn,
                )
            )

        self.down_sampling = Down()

        # ----------------------------------------------------------------------
        # decoders
        self.decoders_img = nn.ModuleList([])
        for i in range(len(in_channels_decoders_img)):
            self.decoders_img.append(
                baseblock_decoder_img(
                    in_channels=in_channels_decoders_img[i],
                    out_channels=out_channels_decoders_img[i],
                    bias=bias,
                    use_bn=use_bn,
                )
            )

        self.up_samplings_img = nn.ModuleList([])
        for i in range(len(in_channels_ups_img)):
            self.up_samplings_img.append(
                Up(
                    in_channels=in_channels_ups_img[i],
                    out_channels=out_channels_ups_img[i],
                    mode="convtrans",
                )
            )

        # ----------------------------------------------------------------------
        # tail (img)
        self.tail_img = SingleConvBlock(
            in_channels=out_channels_decoders_img[-1],
            out_channels=out_channels,
            bias=bias,
        )

        # ----------------------------------------------------------------------
        # squeezers
        self.squeezers = nn.ModuleList([])
        for i in range(len(out_channels_encoders)):
            self.squeezers.append(
                SqueezeBlock(in_channels=out_channels_encoders[i], bias=bias)
            )

        if self.use_att:
            self.extracters = nn.ModuleList([])
            for i in range(len(out_channels_encoders)):
                self.extracters.append(
                    ExtractBlock(in_channels=out_channels_encoders[i], bias=bias)
                )

        if not self.use_att:
            self.register_buffer("one_tensor", torch.tensor(1))

        # ----------------------------------------------------------------------
        # tail (PSF)

        self.psf_generator = PSFModel.HalfPlane(
            kernel_size=psf_size, kernel_norm=True, over_sampling=2, center_one=True
        )

        num_features = sum(out_channels_encoders)
        self.psf_plane_size = self.psf_generator.get_plane_size()

        self.conv1x1 = DoubleConvBlock1x1(
            in_channels=num_features,
            out_channels=self.psf_plane_size[0] * self.psf_plane_size[1],
            bias=True,
        )

    def forward(self, x):
        # head
        fea_head = self.head(x)  # (N, 32, 128, 128, 128)

        # encode
        fea_down_1 = self.encoders[0](fea_head)  # (N, 64, 128, 128, 128)

        fea_down_2 = self.down_sampling(fea_down_1)
        fea_down_2 = self.encoders[1](fea_down_2)  # (N, 128, 64, 64, 64)

        fea_down_3 = self.down_sampling(fea_down_2)
        fea_down_3 = self.encoders[2](fea_down_3)  # (N, 256, 32, 32, 32)

        fea_down_4 = self.down_sampling(fea_down_3)
        fea_down_4 = self.encoders[3](fea_down_4)  # (N, 512, 16, 16, 16)

        fea_down_5 = self.down_sampling(fea_down_4)
        fea_down_5 = self.encoders[4](fea_down_5)  # (N, 1024, 8, 8, 8)

        # decoders (psf)
        fea_squee_1 = self.squeezers[0](fea_down_1)  # (N, 64, 1, 1, 1)
        fea_squee_2 = self.squeezers[1](fea_down_2)  # (N, 128, 1, 1, 1)
        fea_squee_3 = self.squeezers[2](fea_down_3)  # (N, 256, 1, 1, 1)
        fea_squee_4 = self.squeezers[3](fea_down_4)  # (N, 512, 1, 1, 1)
        fea_squee_5 = self.squeezers[4](fea_down_5)  # (N, 1024, 1, 1, 1)

        if self.use_att:
            att_5 = self.extracters[4](fea_squee_5)  # (N, 1024, 1, 1, 1)
            att_4 = self.extracters[3](fea_squee_4)  # (N, 512, 1, 1, 1)
            att_3 = self.extracters[2](fea_squee_3)  # (N, 256, 1, 1, 1)
            att_2 = self.extracters[1](fea_squee_2)  # (N, 128, 1, 1, 1)
            att_1 = self.extracters[0](fea_squee_1)  # (N, 64, 1, 1, 1)
        else:
            att_5, att_4, att_3, att_2, att_1 = (
                self.one_tensor,
                self.one_tensor,
                self.one_tensor,
                self.one_tensor,
                self.one_tensor,
            )

        # decoders (img)
        fea_up_4_img = self.up_samplings_img[0](
            fea_down_5 * att_5
        )  # (N, 512, 16, 16, 16)
        fea_up_4_img = torch.cat([fea_down_4 * att_4, fea_up_4_img], dim=1)
        fea_up_4_img = self.decoders_img[0](fea_up_4_img)  # (N, 512, 16, 16, 16)

        fea_up_3_img = self.up_samplings_img[1](fea_up_4_img)  # (N, 256, 32, 32, 32)
        fea_up_3_img = torch.cat([fea_down_3 * att_3, fea_up_3_img], dim=1)
        fea_up_3_img = self.decoders_img[1](fea_up_3_img)  # (N, 256, 32, 32, 32)

        fea_up_2_img = self.up_samplings_img[2](fea_up_3_img)  # (N, 128, 64, 64, 64)
        fea_up_2_img = torch.cat([fea_down_2 * att_2, fea_up_2_img], dim=1)
        fea_up_2_img = self.decoders_img[2](fea_up_2_img)  # (N, 128, 64, 64, 64)

        fea_up_1_img = self.up_samplings_img[3](fea_up_2_img)  # (N, 64, 128, 128, 128)
        fea_up_1_img = torch.cat([fea_down_1 * att_1, fea_up_1_img], dim=1)
        fea_up_1_img = self.decoders_img[3](fea_up_1_img)  # (N, 64, 128, 128, 128)

        out_img = self.tail_img(fea_up_1_img)
        out_img = torch.abs(out_img + x)

        # generator psf
        fea_cat = torch.concat(
            [fea_squee_1, fea_squee_2, fea_squee_3, fea_squee_4, fea_squee_5], dim=1
        )
        fea_cat = torch.abs(self.conv1x1(fea_cat))  # (B, C, 1, 1, 1)
        psf_planes = torch.reshape(
            fea_cat,
            shape=(fea_cat.shape[0], 1, self.psf_plane_size[0], self.psf_plane_size[1]),
        )  # (B, 1, Nz, num_anchor)
        out_psf = self.psf_generator(psf_planes)  # (B, 1) + psf_size

        return out_img, out_psf


class TeeNet3D_sq(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        use_bn=True,
        bias=True,
        encoder_type="doubleconv",
        decoder_type="doubleconv",
        psf_size=(127, 127, 127),
        psf_model="half",
        kernel_norm=True,
        residual=True,
        groups=True,
        over_sampling=2,
        num_gauss_model=2,
        enable_constaints=False,
        center_one=True,
        num_integral=100,
        pixel_size_z=1,
        psf_part_freeze=False,
        img_part_freeze=False,
        head_tail_freeze=False,
    ):
        super().__init__()

        self.residual = residual
        # ----------------------------------------------------------------------
        # parameters
        out_channels_head = 32

        in_channels_encoders = (32, 64, 128, 256, 512)
        out_channels_encoders = (64, 128, 256, 512, 1024)

        in_channels_ups = (1024, 512, 256, 128)
        out_channels_ups = (512, 256, 128, 64)

        in_channels_decoders = (512 + 512, 256 + 256, 128 + 128, 64 + 64)
        out_channels_decoders = (512, 256, 128, 64)

        if encoder_type == "doubleconv":
            baseblock_encoder = DoubleConvBlock

        if decoder_type == "doubleconv":
            baseblock_decoder = DoubleConvBlock

        baseblock_squeeze = SqueezeBlock

        # ----------------------------------------------------------------------
        # head
        self.head = ConvBlock(in_channels=in_channels, out_channels=out_channels_head)

        # ----------------------------------------------------------------------
        # encoders
        self.encoders = nn.ModuleList([])
        for i in range(len(in_channels_encoders)):
            self.encoders.append(
                baseblock_encoder(
                    in_channels=in_channels_encoders[i],
                    out_channels=out_channels_encoders[i],
                    bias=bias,
                    use_bn=use_bn,
                )
            )

        self.down_sampling = Down()

        # ----------------------------------------------------------------------
        # decoders
        self.decoders_img = nn.ModuleList([])

        for i in range(len(in_channels_decoders)):
            self.decoders_img.append(
                baseblock_decoder(
                    in_channels=in_channels_decoders[i],
                    out_channels=out_channels_decoders[i],
                    bias=bias,
                    use_bn=use_bn,
                )
            )

        self.up_samplings_img = nn.ModuleList([])

        for i in range(len(in_channels_ups)):
            self.up_samplings_img.append(
                Up(
                    in_channels=in_channels_ups[i],
                    out_channels=out_channels_ups[i],
                    mode="convtrans",
                )
            )

        # ----------------------------------------------------------------------
        # tail (img)
        self.tail_img = SingleConvBlock(
            in_channels=out_channels_decoders[-1],
            out_channels=out_channels,
            bias=bias,
        )

        # ----------------------------------------------------------------------
        # squeezers
        in_channels_squeezers = out_channels_encoders + out_channels_decoders
        self.squeezers = nn.ModuleList([])

        for i in range(len(in_channels_squeezers)):
            num_groups = in_channels_squeezers[i] if groups else 1
            self.squeezers.append(
                baseblock_squeeze(
                    in_channels=in_channels_squeezers[i],
                    groups=num_groups,
                    bias=bias,
                )
            )

        # ----------------------------------------------------------------------
        # tail (PSF)
        if psf_model == "half":
            self.psf_generator = PSFModel.HalfPlane(
                kernel_size=psf_size,
                kernel_norm=kernel_norm,
                over_sampling=over_sampling,
                center_one=center_one,
            )

        if psf_model == "gmm":
            self.psf_generator = PSFModel.GaussianMixtureModel(
                kernel_size=psf_size,
                kernel_norm=kernel_norm,
                num_gauss_model=num_gauss_model,
                enable_constraint=enable_constaints,
            )

        if psf_model == "gm":
            self.psf_generator = PSFModel.GaussianModel(
                kernel_size=psf_size,
                kernel_norm=kernel_norm,
                num_params=2,
            )

        if psf_model == "bw":
            self.psf_generator = PSFModel.BWModel(
                kernel_size=psf_size,
                kernel_norm=kernel_norm,
                num_integral=num_integral,
                over_sampling=over_sampling,
                pixel_size_z=pixel_size_z,
            )

        self.fc = nn.Sequential(
            nn.Linear(
                in_features=sum(in_channels_squeezers),
                out_features=512,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=512,
                out_features=self.psf_generator.get_num_params(),
                bias=True,
            ),
        )

        if psf_part_freeze == True:
            for params in self.get_psf_part_params():
                params.requires_grad = False

        if img_part_freeze == True:
            for params in self.get_img_part_params():
                params.requires_grad = False

        if head_tail_freeze == True:
            for params in self.get_head_tail_params():
                params.requires_grad = False

    def get_psf_part_params(self):
        params = []
        params.extend(self.squeezers.parameters())
        params.extend(self.fc.parameters())
        return params

    def get_img_part_params(self):
        params = []
        params.extend(self.encoders.parameters())
        params.extend(self.decoders_img.parameters())
        params.extend(self.up_samplings_img.parameters())
        return params

    def get_head_tail_params(self):
        params = []
        params.extend(self.head.parameters())
        params.extend(self.tail_img.parameters())
        return params

    def forward(self, x):
        # head
        fea_head = self.head(x)  # (N, 32, 128, 128, 128)

        # ----------------------------------------------------------------------
        # encode
        fea_down_1 = self.encoders[0](fea_head)  # (N, 64, 128, 128, 128)

        fea_down_2 = self.down_sampling(fea_down_1)
        fea_down_2 = self.encoders[1](fea_down_2)  # (N, 128, 64, 64, 64)

        fea_down_3 = self.down_sampling(fea_down_2)
        fea_down_3 = self.encoders[2](fea_down_3)  # (N, 256, 32, 32, 32)

        fea_down_4 = self.down_sampling(fea_down_3)
        fea_down_4 = self.encoders[3](fea_down_4)  # (N, 512, 16, 16, 16)

        fea_down_5 = self.down_sampling(fea_down_4)
        fea_down_5 = self.encoders[4](fea_down_5)  # (N, 1024, 8, 8, 8)

        # ----------------------------------------------------------------------
        # decoders
        fea_up_4 = self.up_samplings_img[0](fea_down_5)  # (N, 512, 16, 16, 16)
        fea_up_4 = torch.cat([fea_down_4, fea_up_4], dim=1)
        fea_up_4 = self.decoders_img[0](fea_up_4)  # (N, 512, 16, 16, 16)

        fea_up_3 = self.up_samplings_img[1](fea_up_4)  # (N, 256, 32, 32, 32)
        fea_up_3 = torch.cat([fea_down_3, fea_up_3], dim=1)
        fea_up_3 = self.decoders_img[1](fea_up_3)  # (N, 256, 32, 32, 32)

        fea_up_2 = self.up_samplings_img[2](fea_up_3)  # (N, 128, 64, 64, 64)
        fea_up_2 = torch.cat([fea_down_2, fea_up_2], dim=1)
        fea_up_2 = self.decoders_img[2](fea_up_2)  # (N, 128, 64, 64, 64)

        fea_up_1 = self.up_samplings_img[3](fea_up_2)  # (N, 64, 128, 128, 128)
        fea_up_1 = torch.cat([fea_down_1, fea_up_1], dim=1)
        fea_up_1 = self.decoders_img[3](fea_up_1)  # (N, 64, 128, 128, 128)

        out_img = self.tail_img(fea_up_1)

        if self.residual == True:
            out_img = torch.abs(out_img + x)
        else:
            out_img = torch.abs(out_img)

        # ----------------------------------------------------------------------
        # generator psf
        fea_cat = torch.concat(
            [
                self.squeezers[0](fea_down_1),  # (N, 64, 1, 1, 1)
                self.squeezers[1](fea_down_2),  # (N, 128, 1, 1, 1)
                self.squeezers[2](fea_down_3),  # (N, 256, 1, 1, 1)
                self.squeezers[3](fea_down_4),  # (N, 512, 1, 1, 1)
                self.squeezers[4](fea_down_5),  # (N, 1024, 1, 1, 1)
                self.squeezers[5](fea_up_4),  # (N, 512, 1, 1, 1)
                self.squeezers[6](fea_up_3),  # (N, 256, 1, 1, 1)
                self.squeezers[7](fea_up_2),  # (N, 128, 1, 1, 1)
                self.squeezers[8](fea_up_1),  # (N, 64, 1, 1, 1)
            ],
            dim=1,
        )

        params = self.fc(fea_cat)
        params = torch.reshape(params, shape=(fea_cat.shape[0], 1, -1))
        out_psf = self.psf_generator(params)

        return out_img, out_psf


if __name__ == "__main__":
    from torchinfo import summary

    device = torch.device("cuda:0")
    x = torch.ones(size=(1, 1, 128, 128, 128)).to(device)

    # model = TeeNet3D(
    #     in_channels=1,
    #     out_channels=1,
    #     use_bn=True,
    #     bias=True,
    #     encoder_type="doubleconv",
    #     decoder_type="doubleconv",
    #     psf_size=(127, 127, 127),
    # )

    # model = TeeNet3D_Att(
    #     in_channels=1,
    #     out_channels=1,
    #     use_bn=True,
    #     bias=True,
    #     encoder_type="doubleconv",
    #     decoder_type="doubleconv",
    #     psf_size=(127, 127, 127),
    #     use_att=True,
    # )

    model = TeeNet3D_sq(
        in_channels=1,
        out_channels=1,
        use_bn=True,
        bias=True,
        encoder_type="doubleconv",
        decoder_type="doubleconv",
        psf_size=(127, 127, 127),
        psf_model="half",
        residual=False,
        kernel_norm=True,
        groups=True,
        over_sampling=2,
        num_gauss_model=2,
        enable_constaints=True,
        center_one=1,
        num_integral=100,
        pixel_size_z=1,
        psf_part_freeze=True,
        img_part_freeze=True,
        head_tail_freeze=False,
    )

    model = model.to(device)
    o = model(x)
    print(o[0].shape, o[1].shape)

    summary(model, input_size=(2, 1, 128, 128, 128))
