import torch
import torch.nn as nn

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


class AtomGen_noise(nn.Module):
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

        self.embed_noise = EmbedNoise(n_input, _facify(start_channels * 2, fac), sn=sn)

        conv_blocks = [
            specnorm(
                nn.Conv3d(
                    in_channels=start_channels * 2,
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
        out = self.embed_noise(inputs)
        out = out.repeat(1, 1, 2, 2, 2)
        out = self.conv(out)

        return out



class AtomGen_tiny2(nn.Module):
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


class AtomGen_tiny(nn.Module):
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


class AtomGen_mid(nn.Module):
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

class AtomCrit_tiny(nn.Module):
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


class AtomCrit_tiny16(nn.Module):
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