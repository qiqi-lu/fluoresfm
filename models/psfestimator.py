import torch
from torch import nn
import sys

sys.path.insert(1, "E:\qiqilu\Project\\2024 Foundation model\code")

from models.resnet import ResNet
from models.PSFmodels import GaussianMixtureModel, GaussianModel, HalfPlane, BWModel


class CNNFCN(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
            ),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(in_features=256, out_features=out_channels)

    def forward(self, x):
        x = self.block(x)
        x = torch.mean(x, dim=(2, 3, 4))
        out = self.fc(x)
        return out


class PSFEstimator(nn.Module):
    def __init__(
        self,
        in_channels=1,
        psf_model="gmm",
        kernel_size=[25, 25, 25],
        kernel_norm=True,
        num_gauss_model=2,
        enable_constraint=False,
        over_sampling=2,
        center_one=True,
        use_bn=True,
        bias=True,
        pixel_size_z=1,
    ):
        super().__init__()
        self.in_channels = in_channels

        if psf_model == "gmm":
            self.psf_generator = GaussianMixtureModel(
                kernel_size=kernel_size,
                kernel_norm=kernel_norm,
                num_gauss_model=num_gauss_model,
                enable_constraint=enable_constraint,
            )
        if psf_model == "half":
            self.psf_generator = HalfPlane(
                kernel_size=kernel_size,
                kernel_norm=kernel_norm,
                over_sampling=over_sampling,
                center_one=center_one,
            )

        if psf_model == "gm":
            self.psf_generator = GaussianModel(
                kernel_size=kernel_size,
                kernel_norm=kernel_norm,
                num_params=2,
            )

        if psf_model == "bw":
            self.psf_generator = BWModel(
                kernel_size=kernel_size,
                kernel_norm=kernel_norm,
                num_integral=100,
                over_sampling=over_sampling,
                pixel_size_z=pixel_size_z,
            )

        self.num_params = self.psf_generator.get_num_params()

        self.params_estimator = ResNet(
            layers=[3, 4, 6, 3],
            in_channels=in_channels,
            out_channels=self.num_params,
            use_bn=use_bn,
            bias=bias,
        )

    def forward(self, x):
        params = self.params_estimator(x)
        params = torch.reshape(params, shape=(params.shape[0], 1, self.num_params))
        psf = self.psf_generator(params)
        return psf


if __name__ == "__main__":
    from torchinfo import summary

    device = torch.device("cuda:0")
    x = torch.ones(size=(2, 1, 128, 128, 128)).to(device)

    model = PSFEstimator(
        in_channels=1,
        # psf_model="gmm",
        psf_model="half",
        # psf_model="gm",
        kernel_norm=True,
        kernel_size=(127, 127, 127),
        num_gauss_model=2,
        enable_constraint=True,
        over_sampling=2,
        center_one=True,
        use_bn=True,
        bias=True,
    )

    model = model.to(device)
    o = model(x)
    print(o.shape)
    summary(model, input_size=(2, 1, 128, 128, 128))
