import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralNDVILoss(nn.Module):
    """
    Strict NDVI consistency loss.
    Enforces NDVI consistency after downsampling HR outputs.
    """

    def __init__(self, scale=4):
        super().__init__()
        self.scale = scale
        self.criterion = nn.MSELoss()

    def ndvi(self, x):
        """
        x: (B, 4, H, W) -> RGB + NIR
        NDVI = (NIR - R) / (NIR + R)
        """
        red = x[:, 0:1]
        nir = x[:, 3:4]
        return (nir - red) / (nir + red + 1e-6)

    def forward(self, fake_hr, real_hr):
        """
        fake_hr, real_hr: (B, 4, H, W)
        """
        # Downsample to LR scale
        fake_lr = F.avg_pool2d(fake_hr, self.scale)
        real_lr = F.avg_pool2d(real_hr, self.scale)

        ndvi_fake = self.ndvi(fake_lr)
        ndvi_real = self.ndvi(real_lr)

        return self.criterion(ndvi_fake, ndvi_real)
