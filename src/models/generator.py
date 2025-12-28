import torch
import torch.nn as nn


# -------------------------
# Basic blocks
# -------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, norm=True):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)]
        if norm:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


# -------------------------
# Generator
# -------------------------
class Generator(nn.Module):
    """
    Residual U-Net Generator
    Input : (B, 4, 256, 256)  -> RGB + NIR
    Output: (B, 3, 256, 256)  -> RGB
    """

    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(4, 64, norm=False)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.res_blocks = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
        )

        # Decoder
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        # Output
        self.out = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)              # (B, 64, 256, 256)
        e2 = self.enc2(self.pool(e1))  # (B, 128, 128, 128)
        e3 = self.enc3(self.pool(e2))  # (B, 256, 64, 64)

        # Bottleneck
        b = self.res_blocks(e3)

        # Decoder
        d2 = self.up2(b)               # (B, 128, 128, 128)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)              # (B, 64, 256, 256)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)
