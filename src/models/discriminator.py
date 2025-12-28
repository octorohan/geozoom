import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    PatchGAN Discriminator
    Input:  (B, 7, 256, 256)  -> concat(LR(4), HR(3))
    Output: (B, 1, H', W')    -> patch-wise realism
    """

    def __init__(self):
        super().__init__()

        def block(in_c, out_c, stride=2, norm=True):
            layers = [nn.Conv2d(in_c, out_c, 4, stride, 1)]
            if norm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *block(7, 64, norm=False),     # (256 -> 128)
            *block(64, 128),               # (128 -> 64)
            *block(128, 256),              # (64 -> 32)
            *block(256, 512, stride=1),    # (32 -> 31)
            nn.Conv2d(512, 1, 4, 1, 1)      # Patch output
        )

    def forward(self, lr, hr):
        x = torch.cat([lr, hr], dim=1)
        return self.net(x)
