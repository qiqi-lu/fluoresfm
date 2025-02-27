import torch
from torch import nn


class GaussianModel(nn.Module):
    def __init__(self, kernel_size=(25, 25, 25), kernel_norm=True, num_params=2):
        super().__init__()

        self.kernel_size = torch.tensor(kernel_size)
        self.kernel_norm = kernel_norm

        self.Nz, self.Ny, self.Nx = self.kernel_size

        zp, yp, xp = (self.Nz - 1) / 2, (self.Ny - 1) / 2, (self.Nx - 1) / 2
        gridz = torch.linspace(start=0, end=self.Nz - 1, steps=self.Nz)
        gridy = torch.linspace(start=0, end=self.Ny - 1, steps=self.Ny)
        gridx = torch.linspace(start=0, end=self.Nx - 1, steps=self.Nx)
        Z, Y, X = torch.meshgrid(gridz, gridy, gridx)

        rz, ry, rx = Z - zp, Y - yp, X - xp
        self.register_buffer("rz_flatten", rz.reshape(-1))
        self.register_buffer("ry_flatten", ry.reshape(-1))
        self.register_buffer("rx_flatten", rx.reshape(-1))

        self.num_params = num_params

    def get_num_params(self):
        return self.num_params

    def gauss3d_3p(self, params):
        sigmax = torch.reshape(params[..., 0][..., None], shape=(-1, 1))
        sigmay = torch.reshape(params[..., 1][..., None], shape=(-1, 1))
        sigmaz = torch.reshape(params[..., 2][..., None], shape=(-1, 1))

        out = torch.exp(
            -(
                self.rx_flatten**2 / (2 * (sigmax**2))
                + self.ry_flatten**2 / (2 * (sigmay**2))
                + self.rz_flatten**2 / (2 * (sigmaz**2))
            )
        )
        return out

    def gauss3d_2p(self, params):
        sigmaxy = torch.reshape(params[..., 0][..., None], shape=(-1, 1))
        sigmaz = torch.reshape(params[..., 1][..., None], shape=(-1, 1))

        out = torch.exp(
            -(
                (self.rx_flatten**2 + self.ry_flatten**2) / (2 * (sigmaxy**2))
                + self.rz_flatten**2 / (2 * (sigmaz**2))
            )
        )
        return out

    def forward(self, params):
        num_batch, num_channel, _ = params.shape
        if self.num_params == 3:
            PSF = self.gauss3d_3p(params=params)
        if self.num_params == 2:
            PSF = self.gauss3d_2p(params=params)

        PSF = torch.reshape(
            PSF, shape=(num_batch, num_channel, self.Nz, self.Ny, self.Nx)
        )

        if self.kernel_norm == True:
            PSF = torch.div(PSF, torch.sum(PSF, dim=(2, 3, 4), keepdim=True))

        return PSF


class GaussianMixtureModel(nn.Module):
    def __init__(
        self,
        kernel_size=[25, 25, 25],
        kernel_norm=True,
        num_gauss_model=2,
        enable_constraint=False,
    ):
        super().__init__()
        self.kernel_size = torch.tensor(kernel_size)
        self.kernel_norm = kernel_norm
        self.num_gauss_model = num_gauss_model
        self.enable_constraint = enable_constraint

        self.Nz, self.Ny, self.Nx = self.kernel_size
        self.num_params = self.Nz * num_gauss_model * 3

        yp, xp = (self.Ny - 1) / 2, (self.Nx - 1) / 2
        gridy = torch.linspace(start=0, end=self.Ny - 1, steps=self.Ny)
        gridx = torch.linspace(start=0, end=self.Nx - 1, steps=self.Nx)
        Y, X = torch.meshgrid(gridy, gridx)

        rPixel = torch.sqrt((X - xp) ** 2 + (Y - yp) ** 2)
        rPixel_flatten = rPixel.reshape(-1)
        self.register_buffer("rPixel_flatten", rPixel_flatten)

    def get_num_params(self):
        return self.num_params

    def gauss(self, x):
        x = torch.exp(-torch.square(x))
        return x

    def constraints_a(self, params):
        params = self.gauss(params)

        # make the center to be 1
        if self.Nz % 2 != 0:
            params_center = (
                params[:, :, self.Nz // 2, 0, 0][..., None, None, None] - 1.0
            )
            params_center = torch.nn.functional.pad(
                params_center,
                pad=(
                    0,
                    0,
                    0,
                    self.num_gauss_model - 1,
                    self.Nz // 2,
                    self.Nz // 2,
                ),
                mode="constant",
                value=0,
            )

            params = params - params_center
        return params

    def constraints_mu(self, params):
        # make the first kernel position at 0
        params_pos = params[:, :, :, 0, 0][..., None, None]
        params_pos = torch.nn.functional.pad(
            params_pos,
            pad=(
                0,
                0,
                0,
                self.num_gauss_model - 1,
            ),
            mode="constant",
            value=0,
        )
        params = params - params_pos
        return params

    def forward(self, params):
        params = torch.abs(params)
        num_batch, num_channel, _ = params.shape
        params = torch.reshape(
            params, shape=(num_batch, num_channel, self.Nz, self.num_gauss_model, 3)
        )
        params_a = params[..., 0][..., None]
        params_mu = params[..., 1][..., None]
        params_std = params[..., 2][..., None]

        if self.enable_constraint:
            # params_a = self.constraints_a(params_a)
            params_a = self.constraints_a(params_a) + 0.000001
            params_mu = self.constraints_mu(params_mu) * 10
            # params_std = params_std + 0.0001
            params_std = params_std + 0.5
        else:
            params_a = params_a + 0.000001
            params_mu = params_mu * 10.0
            params_std = params_std + 0.5

        params_a = torch.reshape(params_a, shape=(-1, 1))
        params_mu = torch.reshape(params_mu, shape=(-1, 1))
        params_std = torch.reshape(params_std, shape=(-1, 1))

        kernels = params_a * torch.exp(
            -((self.rPixel_flatten - params_mu) ** 2) / (2 * params_std**2)
        )

        kernels = torch.reshape(
            kernels,
            shape=(
                num_batch,
                num_channel,
                self.Nz,
                self.num_gauss_model,
                self.Ny,
                self.Nx,
            ),
        )
        PSF = torch.sum(kernels, dim=3)

        if self.kernel_norm == True:
            PSF = torch.div(PSF, torch.sum(PSF, dim=(2, 3, 4), keepdim=True))

        return PSF


class HalfPlane(nn.Module):
    def __init__(
        self,
        kernel_size=[25, 25, 25],
        kernel_norm=True,
        over_sampling=2,
        center_one=True,
    ):
        super().__init__()
        self.kernel_size = torch.tensor(kernel_size)
        self.kernel_norm = kernel_norm
        Nz, Ny, Nx = self.kernel_size
        self.center_one = center_one

        # xy plane
        # center point position
        yp, xp = (Ny - 1) / 2, (Nx - 1) / 2
        max_anchor = torch.ceil(torch.sqrt(((Nx - 1) - xp) ** 2 + ((Ny - 1) - yp) ** 2))

        # additional one anchor for poistion 0.
        num_anchor = int(max_anchor * over_sampling + 1)

        R = (
            torch.linspace(start=0, end=max_anchor * over_sampling, steps=num_anchor)
            / over_sampling
        )

        gridy = torch.linspace(start=0, end=Ny - 1, steps=Ny)
        gridx = torch.linspace(start=0, end=Nx - 1, steps=Nx)
        Y, X = torch.meshgrid(gridy, gridx)

        rPixel = torch.sqrt((X - xp) ** 2 + (Y - yp) ** 2)
        index = torch.floor(rPixel * over_sampling).type(torch.int)

        # z direction
        index = index[None].repeat(Nz, 1, 1)
        index_slice = torch.linspace(start=0, end=Nz - 1, steps=Nz)[..., None, None]
        index_slice = index_slice.repeat(1, Ny, Nx).type(torch.int)

        self.register_buffer("index_slice", index_slice)
        self.register_buffer("index1", index)
        disR = (rPixel - R[index]) * over_sampling
        self.register_buffer("disR_1", disR)
        self.register_buffer("disR_2", 1.0 - disR)
        self.register_buffer("index2", index + 1)

        self.psf_plane_size = (Nz, torch.tensor(num_anchor))
        self.num_params = Nz * torch.tensor(num_anchor)

    def get_num_params(self):
        return self.num_params

    def gauss(self, x):
        x = torch.exp(-torch.square(x))
        return x

    def forward(self, params):
        # params = self.gauss(params)
        params = torch.abs(params)
        num_batch, num_channel, _ = params.shape
        psf_plane = torch.reshape(
            params,
            shape=(
                num_batch,
                num_channel,
                self.psf_plane_size[0],
                self.psf_plane_size[1],
            ),
        )
        # psf_plane : (B, C, Nz, num_archor)
        # ----------------------------------------------------------------------
        # make center to 1
        if self.center_one:
            plane_center_only = (
                psf_plane[:, :, self.psf_plane_size[0] // 2, 0][:, :, None, None] - 1
            )
            plane_center_only = torch.nn.functional.pad(
                plane_center_only,
                pad=(
                    0,
                    self.psf_plane_size[1] - 1,
                    self.psf_plane_size[0] // 2,
                    self.psf_plane_size[0] // 2,
                ),
                mode="constant",
                value=0,
            )
            psf_plane = psf_plane - plane_center_only
        # ----------------------------------------------------------------------

        # linear interpolation
        PSF = (
            psf_plane[:, :, self.index_slice, self.index2] * self.disR_1
            + psf_plane[:, :, self.index_slice, self.index1] * self.disR_2
        )

        # ----------------------------------------------------------------------
        # normalization
        if self.kernel_norm == True:
            PSF = torch.div(PSF, torch.sum(PSF, dim=(2, 3, 4), keepdim=True))

        return PSF


class BWModel(nn.Module):
    def __init__(
        self,
        kernel_size=[25, 25, 25],
        kernel_norm=True,
        num_integral=100,
        over_sampling=2,
        pixel_size_z=1,  # * pixel_size_xy
    ):
        super().__init__()
        integral = torch.linspace(start=0, end=1, steps=num_integral + 1)
        self.register_buffer("integral", integral)

        self.dx = 1 / num_integral
        self.kernel_size = torch.tensor(kernel_size)
        self.kernel_norm = kernel_norm
        self.Nz, Ny, Nx = self.kernel_size
        self.pixel_size_z = torch.tensor(pixel_size_z)

        # xy plane
        # center point position
        yp, xp = (Ny - 1) / 2, (Nx - 1) / 2
        max_anchor = torch.ceil(torch.sqrt(((Nx - 1) - xp) ** 2 + ((Ny - 1) - yp) ** 2))

        # additional one anchor for poistion 0.
        self.num_anchor = int(max_anchor * over_sampling + 1)

        # ----------------------------------------------------------------------
        rAnchor = (
            torch.linspace(
                start=0, end=max_anchor * over_sampling, steps=self.num_anchor
            )
            / over_sampling
        )
        rAnchor = rAnchor[None].repeat(self.Nz // 2 + 1, 1)
        index_slice_half = torch.linspace(
            start=0, end=self.Nz // 2, steps=self.Nz // 2 + 1
        )[..., None]
        index_slice_half = index_slice_half.repeat(1, self.num_anchor).type(torch.int)

        rAnchor_flat = torch.reshape(rAnchor, shape=(-1, 1))
        index_slice_flat = torch.reshape(index_slice_half, shape=(-1, 1))

        self.register_buffer("rAnchor_flat", rAnchor_flat)
        self.register_buffer("index_slice_flat", index_slice_flat)

        # ----------------------------------------------------------------------
        # rotation
        R = (
            torch.linspace(
                start=0, end=max_anchor * over_sampling, steps=self.num_anchor
            )
            / over_sampling
        )
        gridy = torch.linspace(start=0, end=Ny - 1, steps=Ny)
        gridx = torch.linspace(start=0, end=Nx - 1, steps=Nx)
        Y, X = torch.meshgrid(gridy, gridx)

        rPixel = torch.sqrt((X - xp) ** 2 + (Y - yp) ** 2)
        index = torch.floor(rPixel * over_sampling).type(torch.int)
        index = index[None].repeat(self.Nz, 1, 1)
        index_slice = torch.linspace(start=0, end=self.Nz - 1, steps=self.Nz)[
            ..., None, None
        ]
        index_slice = index_slice.repeat(1, Ny, Nx).type(torch.int)

        self.register_buffer("index_slice", index_slice)
        self.register_buffer("index1", index)
        disR = (rPixel - R[index]) * over_sampling
        self.register_buffer("disR_1", disR)
        self.register_buffer("disR_2", 1.0 - disR)
        self.register_buffer("index2", index + 1)

    def jep(self, lam, n, r, z, rho):
        # n = NA / ni
        z = z * self.pixel_size_z
        k = 2 * torch.pi / lam
        j0 = torch.special.bessel_j0(k * n * r * rho)
        comp = torch.complex(torch.tensor(0.0), torch.tensor(1.0))
        w = 0.5 * (k * (rho**2) * z * (n**2))
        exp = torch.exp(-comp * w)
        return j0 * exp * rho

    def get_num_params(self):
        return 2

    def forward(self, params):
        params = torch.abs(params)
        num_batch, num_channel, _ = params.shape
        lam = params[..., 0][..., None, None]
        n = params[..., 1][..., None, None]

        sample = self.jep(
            lam=lam,
            n=n,
            r=self.rAnchor_flat,
            z=self.index_slice_flat,
            rho=self.integral,
        )

        plane = torch.trapezoid(sample, dx=self.dx, dim=-1)
        plane = torch.square(torch.abs(plane))
        plane = torch.reshape(
            plane,
            shape=(num_batch, num_channel, self.Nz // 2 + 1, self.num_anchor),
        )

        # ----------------------------------------------------------------------
        plane = torch.nn.functional.pad(
            plane, pad=[0, 0, self.Nz // 2, 0], mode="reflect"
        )

        # linear interpolation
        PSF = (
            plane[:, :, self.index_slice, self.index2] * self.disR_1
            + plane[:, :, self.index_slice, self.index1] * self.disR_2
        )

        # normalization
        if self.kernel_norm == True:
            PSF = torch.div(PSF, torch.sum(PSF, dim=(2, 3, 4), keepdim=True))

        return PSF


if __name__ == "__main__":
    gen = BWModel(
        kernel_norm=True, kernel_size=(127, 127, 127), num_integral=100, over_sampling=2
    )

    params = torch.tensor([457, 1.4 / 1.5])[None, None]

    plane = gen(params)
    print(plane.shape)
