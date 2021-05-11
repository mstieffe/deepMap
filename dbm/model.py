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
                    in_channels=start_channels,
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
                    in_channels=start_channels,
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
                    in_channels=start_channels,
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
                    in_channels=start_channels,
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