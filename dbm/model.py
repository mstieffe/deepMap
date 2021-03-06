import torch
import torch.nn as nn
import torch.nn.init as init

from dbm.util import compute_same_padding


def _facify(n, fac):
    return int(n // fac)


def _sn_to_specnorm(sn: int):
    if sn > 0:

        def specnorm(module):
            return nn.utils.spectral_norm(module, n_power_iterations=sn)

    else:

        def specnorm(module, **kw):
            return module

    return specnorm

class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, kernel_dims, mean=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims
        self.mean = mean
        self.T = nn.Parameter(torch.Tensor(in_features, out_features, kernel_dims))
        init.normal(self.T, 0, 1)

    def forward(self, x):
        # x is NxA
        # T is AxBxC
        matrices = x.mm(self.T.view(self.in_features, -1))
        matrices = matrices.view(-1, self.out_features, self.kernel_dims)

        M = matrices.unsqueeze(0)  # 1xNxBxC
        M_T = M.permute(1, 0, 2, 3)  # Nx1xBxC
        norm = torch.abs(M - M_T).sum(3)  # NxNxB
        expnorm = torch.exp(-norm)
        o_b = (expnorm.sum(0) - 1)   # NxB, subtract self distance
        if self.mean:
            o_b /= x.size(0) - 1

        x = torch.cat([x, o_b], 1)
        return x

class EmbedNoise(nn.Module):
    def __init__(self, z_dim, channels, sn=0):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        self.pad = nn.Linear(z_dim, channels * 4 * 4 * 4)
        self.pad = specnorm(self.pad)
        # self.pad = nn.ConstantPad3d(padding=(3, 3, 3, 3, 3, 3), value=0.)  # -> (B, z_dim, 7, 7, 7)
        # self.conv = nn.Conv3d(z_dim, channels, kernel_size=4, stride=1, padding=0)  # -> (B, channels, 4, 4, 4)
        self.nonlin = nn.LeakyReLU()
        self.z_dim = z_dim
        self.channels = channels

    def forward(self, z):
        # batch_size = z.shape[0]
        out = self.pad(z)
        # out = self.conv(out.view((-1, self.z_dim, 7, 7, 7)))
        out = self.nonlin(out)
        out = out.view((-1, self.channels, 4, 4, 4))
        return out


class G_tiny_with_noise(nn.Module):
    def __init__(
        self,
        z_dim,
        n_input,
        n_output,
        start_channels,
        fac=1,
        sn: int = 0,
        device=None,
    ):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        embed_condition_blocks = [
            specnorm(
                nn.Conv3d(
                    in_channels=n_input,
                    out_channels=_facify(start_channels, fac),
                    kernel_size=5,
                    stride=1,
                    padding=2,
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(start_channels, fac)),
            nn.LeakyReLU(),
            specnorm(
                nn.Conv3d(
                    in_channels=start_channels,
                    out_channels=_facify(start_channels, fac),
                    kernel_size=3,
                    stride=1,
                    padding=compute_same_padding(3, 1, 1)
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(start_channels, fac)),
            nn.LeakyReLU(),

        ]
        self.embed_condition = nn.Sequential(*tuple(embed_condition_blocks)).to(device=device)

        downsample_cond_block = [
            specnorm(
                nn.Conv3d(
                    in_channels=start_channels,
                    out_channels=_facify(start_channels*2, fac),
                    kernel_size=3,
                    stride=2,
                    padding=compute_same_padding(3, 2, 1)
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(start_channels*2, fac)),
            nn.LeakyReLU(),

        ]
        self.downsample_cond = nn.Sequential(*tuple(downsample_cond_block)).to(device=device)

        self.embed_noise_label = EmbedNoise(z_dim, _facify(start_channels*2, fac), sn=sn)

        combined_block = [
            specnorm(
                nn.Conv3d(
                    in_channels=start_channels*4,
                    out_channels=_facify(start_channels*2, fac),
                    kernel_size=3,
                    stride=1,
                    padding=compute_same_padding(3, 1, 1),
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(start_channels*2, fac)),
            nn.LeakyReLU(),

        ]
        self.combined = nn.Sequential(*tuple(combined_block)).to(device=device)

        deconv_block = [
            specnorm(
                nn.Conv3d(
                    in_channels=start_channels*2,
                    out_channels=_facify(start_channels, fac),
                    kernel_size=3,
                    stride=1,
                    padding=compute_same_padding(3, 1, 1),
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(start_channels, fac)),
            nn.LeakyReLU(),

        ]
        self.deconv = nn.Sequential(*tuple(deconv_block)).to(device=device)

        to_image_blocks = [
            specnorm(
                nn.Conv3d(
                    in_channels=start_channels*2,
                    out_channels=_facify(start_channels, fac),
                    kernel_size=3,
                    stride=1,
                    padding=compute_same_padding(3, 1, 1),
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(start_channels, fac)),
            nn.LeakyReLU(),
            specnorm(
                nn.Conv3d(
                    in_channels=start_channels,
                    out_channels=_facify(start_channels / 2, fac),
                    kernel_size=3,
                    stride=1,
                    padding=compute_same_padding(3, 1, 1),
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(start_channels/2, fac)),
            nn.LeakyReLU(),
            specnorm(nn.Conv3d(_facify(start_channels/2, fac), n_output, kernel_size=1, stride=1)),
            nn.Sigmoid(),
        ]
        self.to_image = nn.Sequential(*tuple(to_image_blocks)).to(device=device)

    def forward(self, z, c):
        #z_l = torch.cat((z, l), dim=1)
        embedded_c = self.embed_condition(c)
        down = self.downsample_cond(embedded_c)
        embedded_z_l = self.embed_noise_label(z)
        out = torch.cat((embedded_z_l, down), dim=1)
        out = self.combined(out)
        out = out.repeat(1, 1, 2, 2, 2)
        out = self.deconv(out)
        out = torch.cat((out, embedded_c), dim=1)
        out = self.to_image(out)

        return out


class G_tiny(nn.Module):
    def __init__(
        self,
        n_input,
        n_output,
        start_channels,
        fac=1,
        sn: int = 0,
        device=None,
    ):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        conv_blocks = [
            specnorm(
                nn.Conv3d(
                    in_channels=n_input,
                    out_channels=_facify(start_channels, fac),
                    kernel_size=5,
                    stride=1,
                    padding=2,
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(start_channels, fac)),
            nn.LeakyReLU(),
            specnorm(
                nn.Conv3d(
                    in_channels=start_channels,
                    out_channels=_facify(start_channels/2, fac),
                    kernel_size=3,
                    stride=1,
                    padding=compute_same_padding(3, 1, 1)
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(start_channels/2, fac)),
            nn.LeakyReLU(),

            specnorm(nn.Conv3d(_facify(start_channels / 2, fac), n_output, kernel_size=1, stride=1)),
            nn.Sigmoid(),


        ]
        self.conv = nn.Sequential(*tuple(conv_blocks)).to(device=device)

    def forward(self, inputs):
        #z_l = torch.cat((z, l), dim=1)
        out = self.conv(inputs)

        return out


class G_mid(nn.Module):
    def __init__(
        self,
        n_input,
        n_output,
        start_channels,
        fac=1,
        sn: int = 0,
        device=None,
    ):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        conv_blocks = [
            specnorm(
                nn.Conv3d(
                    in_channels=n_input,
                    out_channels=_facify(start_channels, fac),
                    kernel_size=5,
                    stride=1,
                    padding=2,
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(start_channels, fac)),
            nn.LeakyReLU(),
            specnorm(
                nn.Conv3d(
                    in_channels=_facify(start_channels, fac),
                    out_channels=_facify(start_channels/2, fac),
                    kernel_size=3,
                    stride=1,
                    padding=compute_same_padding(3, 1, 1)
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(start_channels/2, fac)),
            nn.LeakyReLU(),

            specnorm(
                nn.Conv3d(
                    in_channels=_facify(start_channels/2, fac),
                    out_channels=_facify(start_channels / 2, fac),
                    kernel_size=3,
                    stride=1,
                    padding=compute_same_padding(3, 1, 1)
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(start_channels / 2, fac)),
            nn.LeakyReLU(),

            specnorm(
                nn.Conv3d(
                    in_channels=_facify(start_channels/2, fac),
                    out_channels=_facify(start_channels / 2, fac),
                    kernel_size=3,
                    stride=1,
                    padding=compute_same_padding(3, 1, 1)
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(start_channels / 2, fac)),
            nn.LeakyReLU(),

            specnorm(
                nn.Conv3d(
                    in_channels=_facify(start_channels/2, fac),
                    out_channels=_facify(start_channels / 2, fac),
                    kernel_size=3,
                    stride=1,
                    padding=compute_same_padding(3, 1, 1)
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(start_channels / 2, fac)),
            nn.LeakyReLU(),

            specnorm(nn.Conv3d(_facify(start_channels / 2, fac), n_output, kernel_size=1, stride=1)),
            nn.Sigmoid(),


        ]
        self.conv = nn.Sequential(*tuple(conv_blocks)).to(device=device)

    def forward(self, inputs):
        #z_l = torch.cat((z, l), dim=1)
        out = self.conv(inputs)

        return out

class C_tiny(nn.Module):
    def __init__(
        self, in_channels, start_channels, fac=1, sn: int = 0, device=None
    ):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        self.step1 = specnorm(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=_facify(start_channels, fac),
                kernel_size=5,
                stride=1,
                padding=compute_same_padding(5, 1, 1),
            )
        )
        self.step2 = nn.LeakyReLU()

        self.step3 = specnorm(
            nn.Conv3d(
                in_channels=_facify(start_channels, fac),
                out_channels=_facify(start_channels*2, fac),
                kernel_size=3,
                stride=2,
                padding=compute_same_padding(3, 1, 1),
            )
        )
        self.step4 = nn.GroupNorm(1, _facify(start_channels*2, fac))
        self.step5 = nn.LeakyReLU()

        self.step6 = specnorm(
            nn.Conv3d(
                in_channels=_facify(start_channels*2, fac),
                out_channels=_facify(start_channels*2, fac),
                kernel_size=3,
                stride=1,
                padding=compute_same_padding(3, 1, 1),
            )
        )
        self.step7 = nn.GroupNorm(1, _facify(start_channels*2, fac))
        self.step8 = nn.LeakyReLU()

        self.step9 = specnorm(
            nn.Conv3d(
                in_channels=_facify(start_channels*2, fac),
                out_channels=_facify(start_channels, fac),
                kernel_size=4,
                stride=1,
                padding=0,
            )
        )
        self.step10 = nn.GroupNorm(1, _facify(start_channels, fac))
        self.step11 = nn.LeakyReLU()

        self.to_critic_value = specnorm(
            nn.Linear(
                in_features=_facify(start_channels, fac), out_features=1
            )
        )

    def forward(self, inputs):
        out = self.step1(inputs)
        out = self.step2(out)
        out = self.step3(out)
        out = self.step4(out)
        out = self.step5(out)
        out = self.step6(out)
        out = self.step7(out)
        out = self.step8(out)
        out = self.step9(out)
        out = self.step10(out)
        out = self.step11(out)
        out = torch.flatten(out, start_dim=1, end_dim=-1)
        out = self.to_critic_value(out)
        return out


class C_tiny_mbd(nn.Module):
    def __init__(
        self, in_channels, start_channels, fac=1, sn: int = 0, device=None
    ):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        self.step1 = specnorm(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=_facify(start_channels, fac),
                kernel_size=5,
                stride=1,
                padding=compute_same_padding(5, 1, 1),
            )
        )
        self.step2 = nn.LeakyReLU()

        self.step3 = specnorm(
            nn.Conv3d(
                in_channels=_facify(start_channels, fac),
                out_channels=_facify(start_channels*2, fac),
                kernel_size=3,
                stride=2,
                padding=compute_same_padding(3, 1, 1),
            )
        )
        self.step4 = nn.GroupNorm(1, _facify(start_channels*2, fac))
        self.step5 = nn.LeakyReLU()

        self.step6 = specnorm(
            nn.Conv3d(
                in_channels=_facify(start_channels*2, fac),
                out_channels=_facify(start_channels*2, fac),
                kernel_size=3,
                stride=1,
                padding=compute_same_padding(3, 1, 1),
            )
        )
        self.step7 = nn.GroupNorm(1, _facify(start_channels*2, fac))
        self.step8 = nn.LeakyReLU()

        self.step9 = specnorm(
            nn.Conv3d(
                in_channels=_facify(start_channels*2, fac),
                out_channels=_facify(start_channels, fac),
                kernel_size=4,
                stride=1,
                padding=0,
            )
        )
        self.step10 = nn.GroupNorm(1, _facify(start_channels, fac))
        self.step11 = nn.LeakyReLU()

        self.mbd = MinibatchDiscrimination(_facify(start_channels, fac), _facify(start_channels/2, fac), 50)

        self.to_critic_value = specnorm(
            nn.Linear(
                in_features=_facify(start_channels, fac)+_facify(start_channels/2, fac), out_features=1
            )
        )

    def forward(self, inputs):
        out = self.step1(inputs)
        out = self.step2(out)
        out = self.step3(out)
        out = self.step4(out)
        out = self.step5(out)
        out = self.step6(out)
        out = self.step7(out)
        out = self.step8(out)
        out = self.step9(out)
        out = self.step10(out)
        out = self.step11(out)
        out = torch.flatten(out, start_dim=1, end_dim=-1)
        out = self.mbd(out)
        out = self.to_critic_value(out)
        return out


class G_dense(nn.Module):
    def __init__(
        self,
        n_input,
        n_output,
        start_channels,
        fac=1,
        sn: int = 0,
        device=None,
    ):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)

        dense_block1 = [
            specnorm(
                nn.Linear(in_features=n_input, out_features=start_channels)),
            nn.LeakyReLU()]
        self.dense1 = nn.Sequential(*tuple(dense_block1)).to(device=device)

        dense_block2 = [
            specnorm(
                nn.Linear(in_features=start_channels, out_features=int(start_channels * 2))),
            nn.LeakyReLU()]
        self.dense2 = nn.Sequential(*tuple(dense_block2)).to(device=device)

        dense_block3 = [
            specnorm(
                nn.Linear(in_features=int(start_channels * 2), out_features=int(start_channels * 2))),
            nn.LeakyReLU()]
        self.dense3 = nn.Sequential(*tuple(dense_block3)).to(device=device)

        self.mbd = MinibatchDiscrimination(int(start_channels * 2), int(start_channels), 20)

        self.to_critic_value = specnorm(
            nn.Linear(
                in_features=int(start_channels * 2 + start_channels), out_features=n_output
            )
        )

    def forward(self, inputs):
        out = self.dense1(inputs)
        out = self.dense2(out)
        out = self.dense3(out)
        out = self.mbd(out)
        out = self.to_critic_value(out)
        return out

class C_dense(nn.Module):
    def __init__(
        self, in_channels, start_channels, fac=1, sn: int = 0, device=None
    ):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)

        dense_block1 = [
            specnorm(
            nn.Linear(in_features=in_channels, out_features=start_channels)),
            nn.LeakyReLU()]
        self.dense1 = nn.Sequential(*tuple(dense_block1)).to(device=device)

        dense_block2 = [
            specnorm(
            nn.Linear(in_features=start_channels, out_features=int(start_channels/2))),
            nn.LeakyReLU()]
        self.dense2 = nn.Sequential(*tuple(dense_block2)).to(device=device)

        dense_block3 = [
            specnorm(
            nn.Linear(in_features=int(start_channels/2), out_features=int(start_channels/2))),
            nn.LeakyReLU()]
        self.dense3 = nn.Sequential(*tuple(dense_block3)).to(device=device)

        self.mbd = MinibatchDiscrimination(int(start_channels/2), int(start_channels/4), 20)

        self.to_critic_value = specnorm(
            nn.Linear(
                in_features=int(start_channels/2+start_channels/4), out_features=1
            )
        )

    def forward(self, inputs):
        out = self.dense1(inputs)
        out = self.dense2(out)
        out = self.dense3(out)
        out = self.mbd(out)
        out = self.to_critic_value(out)
        return out


class C_tiny_mbd_dstr(nn.Module):
    def __init__(
        self, in_channels, start_channels, dstr_chns=64, fac=1, sn: int = 0, device=None
    ):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        self.step1 = specnorm(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=_facify(start_channels, fac),
                kernel_size=5,
                stride=1,
                padding=compute_same_padding(5, 1, 1),
            )
        )
        self.step2 = nn.LeakyReLU()

        self.step3 = specnorm(
            nn.Conv3d(
                in_channels=_facify(start_channels, fac),
                out_channels=_facify(start_channels*2, fac),
                kernel_size=3,
                stride=2,
                padding=compute_same_padding(3, 1, 1),
            )
        )
        self.step4 = nn.GroupNorm(1, _facify(start_channels*2, fac))
        self.step5 = nn.LeakyReLU()

        self.step6 = specnorm(
            nn.Conv3d(
                in_channels=_facify(start_channels*2, fac),
                out_channels=_facify(start_channels*2, fac),
                kernel_size=3,
                stride=1,
                padding=compute_same_padding(3, 1, 1),
            )
        )
        self.step7 = nn.GroupNorm(1, _facify(start_channels*2, fac))
        self.step8 = nn.LeakyReLU()

        self.step9 = specnorm(
            nn.Conv3d(
                in_channels=_facify(start_channels*2, fac),
                out_channels=_facify(start_channels, fac),
                kernel_size=4,
                stride=1,
                padding=0,
            )
        )
        self.step10 = nn.GroupNorm(1, _facify(start_channels, fac))
        self.step11 = nn.LeakyReLU()

        self.mbd = MinibatchDiscrimination(_facify(start_channels, fac), _facify(start_channels/2, fac), 50)

        dense_blocks = [
            specnorm(
            nn.Linear(in_features=_facify(start_channels, fac) + _facify(start_channels / 2, fac) + dstr_chns, out_features=_facify(start_channels, fac))),
            nn.LeakyReLU()]
        self.dense = nn.Sequential(*tuple(dense_blocks)).to(device=device)

        self.to_critic_value = specnorm(
            nn.Linear(
                in_features=_facify(start_channels, fac), out_features=1
            )
        )

    def forward(self, inputs, dstr):
        out = self.step1(inputs)
        out = self.step2(out)
        out = self.step3(out)
        out = self.step4(out)
        out = self.step5(out)
        out = self.step6(out)
        out = self.step7(out)
        out = self.step8(out)
        out = self.step9(out)
        out = self.step10(out)
        out = self.step11(out)
        out = torch.flatten(out, start_dim=1, end_dim=-1)
        out = self.mbd(out)
        out = torch.cat((out, dstr), dim=1)
        out = self.dense(out)
        out = self.to_critic_value(out)
        return out


class C_tiny16(nn.Module):
    def __init__(
        self, in_channels, start_channels, fac=1, sn: int = 0, device=None
    ):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        self.step1 = specnorm(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=_facify(start_channels, fac),
                kernel_size=5,
                stride=1,
                padding=compute_same_padding(5, 1, 1),
            )
        )
        self.step2 = nn.LeakyReLU()

        self.step3 = specnorm(
            nn.Conv3d(
                in_channels=_facify(start_channels, fac),
                out_channels=_facify(start_channels*2, fac),
                kernel_size=3,
                stride=2,
                padding=compute_same_padding(3, 1, 1),
            )
        )
        self.step4 = nn.GroupNorm(1, _facify(start_channels*2, fac))
        self.step5 = nn.LeakyReLU()

        self.step6 = specnorm(
            nn.Conv3d(
                in_channels=_facify(start_channels*2, fac),
                out_channels=_facify(start_channels*2, fac),
                kernel_size=3,
                stride=1,
                padding=compute_same_padding(3, 1, 1),
            )
        )
        self.step7 = nn.GroupNorm(1, _facify(start_channels*2, fac))
        self.step8 = nn.LeakyReLU()

        self.step9 = specnorm(
            nn.Conv3d(
                in_channels=_facify(start_channels*2, fac),
                out_channels=_facify(start_channels*4, fac),
                kernel_size=3,
                stride=2,
                padding=compute_same_padding(3, 1, 1),
            )
        )
        self.step10 = nn.GroupNorm(1, _facify(start_channels*4, fac))
        self.step11 = nn.LeakyReLU()

        self.step12 = specnorm(
            nn.Conv3d(
                in_channels=_facify(start_channels*4, fac),
                out_channels=_facify(start_channels, fac),
                kernel_size=4,
                stride=1,
                padding=0,
            )
        )
        self.step13 = nn.GroupNorm(1, _facify(start_channels, fac))
        self.step14 = nn.LeakyReLU()

        self.to_critic_value = specnorm(
            nn.Linear(
                in_features=_facify(start_channels, fac), out_features=1
            )
        )

    def forward(self, inputs):
        out = self.step1(inputs)
        out = self.step2(out)
        out = self.step3(out)
        out = self.step4(out)
        out = self.step5(out)
        out = self.step6(out)
        out = self.step7(out)
        out = self.step8(out)
        out = self.step9(out)
        out = self.step10(out)
        out = self.step11(out)
        out = self.step12(out)
        out = self.step13(out)
        out = self.step14(out)
        out = torch.flatten(out, start_dim=1, end_dim=-1)
        out = self.to_critic_value(out)
        return out