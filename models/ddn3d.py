import torch.nn as nn
import torch
import torchinfo


def init_weights(module):
    if isinstance(module, nn.Conv3d):
        std = 0.1
        torch.nn.init.trunc_normal_(
            module.weight, mean=0.0, std=0.1, a=-2 * std, b=2 * std
        )


class BaseBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bias=False,
        kernel_size=3,
        padding=1,
        stride=1,
        act="relu",
    ):
        super().__init__()
        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        if act == "leaky_relu":
            self.act = nn.LeakyReLU(inplace=True, negative_slope=0.2)

        self.module = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.BatchNorm3d(num_features=out_channels),
            self.act,
        )

        self.module.apply(init_weights)

    def forward(self, x):
        x = self.module(x)
        return x


class backbone(nn.Module):
    def __init__(self, in_channels, bias=False):
        super().__init__()

        self.conv1 = BaseBlock(in_channels=in_channels, out_channels=4, bias=bias)
        self.conv2 = BaseBlock(in_channels=4, out_channels=8, bias=bias)
        self.conv3 = BaseBlock(in_channels=12, out_channels=4, bias=bias)
        self.conv4 = BaseBlock(in_channels=16, out_channels=4, bias=bias)

        self.conv5 = BaseBlock(
            in_channels=4,
            out_channels=8,
            bias=bias,
            kernel_size=(2, 2, 2),
            stride=(2, 2, 2),
            padding=0,
        )
        self.conv6 = BaseBlock(in_channels=8, out_channels=4, bias=bias)
        self.conv7 = BaseBlock(in_channels=12, out_channels=4, bias=bias)
        self.conv8 = BaseBlock(in_channels=16, out_channels=8, bias=bias)

        self.conv9 = BaseBlock(
            in_channels=8, out_channels=4, bias=bias, kernel_size=(1, 1, 1), padding=0
        )
        self.conv10 = BaseBlock(in_channels=4, out_channels=8, bias=bias)
        self.conv11 = BaseBlock(in_channels=12, out_channels=4, bias=bias)
        self.conv12 = BaseBlock(in_channels=16, out_channels=8, bias=bias)

        self.conv_up = nn.ConvTranspose3d(
            in_channels=8, out_channels=4, kernel_size=2, stride=2, bias=bias
        )
        self.bn_up = nn.BatchNorm3d(num_features=4)
        self.act_up = nn.ReLU(inplace=True)

    def forward(self, x):

        o11 = self.conv1(x)
        o12 = self.conv2(o11)
        o13 = self.conv3(torch.cat((o11, o12), dim=1))
        o14 = self.conv4(torch.cat((o11, o12, o13), dim=1))

        # down-sampling
        o21 = self.conv5(o14)
        o22 = self.conv6(o21)
        o23 = self.conv7(torch.cat((o21, o22), dim=1))
        o24 = self.conv8(torch.cat((o21, o22, o23), dim=1))

        o31 = self.conv9(o24)
        o32 = self.conv10(o31)
        o33 = self.conv11(torch.cat((o31, o32), dim=1))
        o34 = self.conv12(torch.cat((o31, o32, o33), dim=1))

        # up-sampling
        out = self.bn_up(self.conv_up(o34))
        out = self.act_up(torch.add(out, o11))
        return out


class DenseDeconNet(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, scale_factor=1):
        super().__init__()
        self.scale_facotr = scale_factor

        self.backbone = backbone(in_channels=in_channels, bias=bias)
        self.conv_out = BaseBlock(
            in_channels=4, out_channels=out_channels, bias=bias, act="leaky_relu"
        )

    def forward(self, x):
        if self.scale_facotr > 1:
            x = nn.functional.interpolate(
                input=x, scale_factor=self.scale_facotr, mode="tricubic"
            )

        fea = self.backbone(x)
        out = self.conv_out(fea)
        return out


if __name__ == "__main__":
    x = torch.ones(size=(2, 2, 128, 128, 128))
    model = DenseDeconNet(in_channels=2, out_channels=1, scale_factor=1, bias=True)
    o = model(x)
    torchinfo.summary(model=model, input_size=x.shape)
    print(o.shape)
