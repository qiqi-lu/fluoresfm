import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        use_bn=True,
        bias=True,
        downsample=None,
    ):
        super().__init__()

        blocks = []
        blocks.append(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=bias,
            )
        )
        if use_bn == True:
            blocks.append(nn.BatchNorm3d(num_features=out_channels))
        blocks.append(nn.ReLU(inplace=True))
        blocks.append(
            nn.Conv3d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
            )
        )
        if use_bn == True:
            blocks.append(nn.BatchNorm3d(num_features=out_channels))
        self.block = nn.Sequential(*blocks)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.block(x)
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, layers, in_channels, out_channels, use_bn=True, bias=True):
        super().__init__()

        self.last_channels = 64
        self.use_bn = use_bn
        self.bias = bias

        # first layer
        conv1 = []
        conv1.append(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=self.last_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=self.bias,
            )
        )
        if use_bn:
            conv1.append(nn.BatchNorm3d(self.last_channels))
        conv1.append(nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(*conv1)

        # pooling
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer0 = self._make_layers(64, layers[0], stride=1)
        self.layer1 = self._make_layers(128, layers[1], stride=2)
        self.layer2 = self._make_layers(256, layers[2], stride=2)
        self.layer3 = self._make_layers(512, layers[3], stride=2)

        self.fc = nn.Linear(in_features=512, out_features=out_channels)

    def _make_layers(self, out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or out_channels != self.last_channels:
            downsample = []
            downsample.append(
                nn.Conv3d(
                    self.last_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                )
            )
            if self.use_bn:
                downsample.append(nn.BatchNorm3d(out_channels))
            downsample = nn.Sequential(*downsample)

        blocks = []
        blocks.append(
            ResidualBlock(
                in_channels=self.last_channels,
                out_channels=out_channels,
                stride=stride,
                downsample=downsample,
                use_bn=self.use_bn,
                bias=self.bias,
            )
        )

        self.last_channels = out_channels

        for _ in range(1, num_blocks):
            blocks.append(
                ResidualBlock(
                    in_channels=self.last_channels,
                    out_channels=out_channels,
                    stride=1,
                    downsample=None,
                    use_bn=self.use_bn,
                    bias=self.bias,
                )
            )
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = torch.mean(x, dim=(2, 3, 4), keepdim=True)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    from torchinfo import summary

    device = torch.device("cuda:0")
    x = torch.ones(size=(2, 1, 128, 128, 128)).to(device)

    model = ResNet(
        layers=[3, 4, 6, 3], in_channels=1, out_channels=186, use_bn=True, bias=True
    )

    model = model.to(device)
    o = model(x)
    print(o.shape)
    summary(model, input_size=(2, 1, 128, 128, 128))
