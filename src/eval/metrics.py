import torch
import torch.nn.functional as F
import numpy as np
import lpips
from skimage.metrics import structural_similarity as ssim_fn


# =========================
# PSNR
# =========================
def psnr(pred, target, max_val=1.0):
    """
    pred, target: (C, H, W) torch tensors in [0,1]
    """
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * torch.log10(max_val / torch.sqrt(mse)).item()


# =========================
# SSIM
# =========================
def ssim(pred, target):
    """
    pred, target: (C, H, W) torch tensors in [0,1]
    Uses skimage (expects HWC, numpy)
    """
    pred_np = pred.permute(1, 2, 0).cpu().numpy()
    tgt_np = target.permute(1, 2, 0).cpu().numpy()

    return ssim_fn(
        pred_np,
        tgt_np,
        data_range=1.0,
        channel_axis=2
    )


# =========================
# LPIPS (VGG)
# =========================
class LPIPSVGG:
    def __init__(self, device="cpu"):
        self.model = lpips.LPIPS(net="vgg").to(device)
        self.model.eval()
        self.device = device

    def __call__(self, pred, target):
        """
        pred, target: (C, H, W) torch tensors in [0,1]
        """
        # LPIPS expects [-1, 1]
        pred = pred.unsqueeze(0).to(self.device) * 2 - 1
        target = target.unsqueeze(0).to(self.device) * 2 - 1

        with torch.no_grad():
            d = self.model(pred, target)

        return d.item()


# =========================
# NDVI RMSE
# =========================
def ndvi(x):
    """
    x: (4, H, W) tensor -> RGB + NIR
    """
    red = x[0]
    nir = x[3]
    return (nir - red) / (nir + red + 1e-6)


def ndvi_rmse(fake_hr_4, real_hr_4, scale=4):
    """
    Strict NDVI RMSE:
    1) Downsample HR to LR scale
    2) Compute NDVI
    3) RMSE
    """
    fake_lr = F.avg_pool2d(fake_hr_4.unsqueeze(0), scale).squeeze(0)
    real_lr = F.avg_pool2d(real_hr_4.unsqueeze(0), scale).squeeze(0)

    ndvi_fake = ndvi(fake_lr)
    ndvi_real = ndvi(real_lr)

    return torch.sqrt(torch.mean((ndvi_fake - ndvi_real) ** 2)).item()
