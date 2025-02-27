import torch.nn as nn
import torch
import torchinfo


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        std = 0.1
        torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=0.1, a=-2 * std, b=2 * std)


class backbone(nn.Module):
    def __init__(self, in_channels, bias=False):
        super().__init__()
        self.act = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=3 // 2,
            bias=bias,
        )
        self.bn1 = nn.BatchNorm2d(num_features=4)

        self.conv2 = nn.Conv2d(
            in_channels=4,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=3 // 2,
            bias=bias,
        )
        self.bn2 = nn.BatchNorm2d(num_features=8)

        self.conv3 = nn.Conv2d(
            in_channels=12,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=3 // 2,
            bias=bias,
        )
        self.bn3 = nn.BatchNorm2d(num_features=4)

        self.conv4 = nn.Conv2d(
            in_channels=16,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=3 // 2,
            bias=bias,
        )
        self.bn4 = nn.BatchNorm2d(num_features=4)

        self.conv5 = nn.Conv2d(
            in_channels=4, out_channels=8, kernel_size=2, stride=2, bias=bias
        )
        self.bn5 = nn.BatchNorm2d(num_features=8)

        self.conv6 = nn.Conv2d(
            in_channels=8,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=3 // 2,
            bias=bias,
        )
        self.bn6 = nn.BatchNorm2d(num_features=4)

        self.conv7 = nn.Conv2d(
            in_channels=12,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=3 // 2,
            bias=bias,
        )
        self.bn7 = nn.BatchNorm2d(num_features=4)

        self.conv8 = nn.Conv2d(
            in_channels=16,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=3 // 2,
            bias=bias,
        )
        self.bn8 = nn.BatchNorm2d(num_features=8)

        self.conv9 = nn.Conv2d(
            in_channels=8,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=3 // 2,
            bias=bias,
        )
        # self.conv9 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=1, stride=1, padding=0,bias=bias)
        self.bn9 = nn.BatchNorm2d(num_features=4)

        self.conv10 = nn.Conv2d(
            in_channels=4,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=3 // 2,
            bias=bias,
        )
        self.bn10 = nn.BatchNorm2d(num_features=8)

        self.conv11 = nn.Conv2d(
            in_channels=12,
            out_channels=4,
            kernel_size=3,
            stride=1,
            padding=3 // 2,
            bias=bias,
        )
        self.bn11 = nn.BatchNorm2d(num_features=4)

        self.conv12 = nn.Conv2d(
            in_channels=16,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=3 // 2,
            bias=bias,
        )
        self.bn12 = nn.BatchNorm2d(num_features=8)

        self.conv_up = nn.ConvTranspose2d(
            in_channels=8, out_channels=4, kernel_size=2, stride=2, bias=bias
        )
        self.bn_up = nn.BatchNorm2d(num_features=4)

    def forward(self, x):

        o11 = self.act(self.bn1(self.conv1(x)))
        o12 = self.act(self.bn2(self.conv2(o11)))
        o13 = self.act(self.bn3(self.conv3(torch.cat((o11, o12), dim=1))))
        o14 = self.act(self.bn4(self.conv4(torch.cat((o11, o12, o13), dim=1))))

        o21 = self.act(self.bn5(self.conv5(o14)))
        o22 = self.act(self.bn6(self.conv6(o21)))
        o23 = self.act(self.bn7(self.conv7(torch.cat((o21, o22), dim=1))))
        o24 = self.act(self.bn8(self.conv8(torch.cat((o21, o22, o23), dim=1))))

        o31 = self.act(self.bn9(self.conv9(o24)))
        o32 = self.act(self.bn10(self.conv10(o31)))
        o33 = self.act(self.bn11(self.conv11(torch.cat((o31, o32), dim=1))))
        o34 = self.act(self.bn12(self.conv12(torch.cat((o31, o32, o33), dim=1))))

        out = self.bn_up(self.conv_up(o34))
        out = self.act(torch.add(out, o11))
        return out


class DenseDeconNet(nn.Module):
    def __init__(self, in_channels, scale_factor=1) -> None:
        super().__init__()
        self.scale_facotr = scale_factor

        self.backbone = backbone(in_channels=in_channels, bias=False)
        self.conv_out = nn.Conv2d(
            in_channels=4,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=3 // 2,
            bias=False,
        )
        self.bn_out = nn.BatchNorm2d(num_features=in_channels)
        self.act_out = nn.LeakyReLU(inplace=True, negative_slope=0.2)

        self.backbone.apply(init_weights)
        init_weights(self.conv_out)

    def forward(self, x):
        if self.scale_facotr > 1:
            x = nn.functional.interpolate(
                input=x, scale_factor=self.scale_facotr, mode="bicubic"
            )

        fea = self.backbone(x)
        out = self.act_out(self.bn_out(self.conv_out(fea)))
        return out


if __name__ == "__main__":
    x = torch.ones(size=(1, 128, 128, 128))
    model = DenseDeconNet(in_channels=3, scale_factor=4)
    o = model(x)
    print(o.shape)
