import torch
import numpy as np
import rasterio
from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn.functional as F

from src.models.generator import Generator
from src.baselines.srcnn_model import SRCNN


# =========================
# CONFIG
# =========================
LR_DIR = Path("data/paired/satellite")
HR_DIR = Path("data/paired/drone")
OUT_DIR = Path("outputs/visuals")

NUM_SAMPLES = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# UTILS
# =========================
def load_pair(lr_path, hr_path):
    with rasterio.open(lr_path) as ds:
        lr = torch.from_numpy(ds.read().astype(np.float32)) / ds.read().max()

    with rasterio.open(hr_path) as ds:
        hr = torch.from_numpy(ds.read().astype(np.float32)) / ds.read().max()

    return lr, hr


def bicubic_upsample(lr, target_hw):
    lr_rgb = lr[:3].unsqueeze(0)
    return F.interpolate(
        lr_rgb, size=target_hw, mode="bicubic", align_corners=False
    ).squeeze(0)


def ndvi(x):
    red = x[0]
    nir = x[3]
    return (nir - red) / (nir + red + 1e-6)


def to_numpy(img):
    return img.permute(1, 2, 0).cpu().numpy().clip(0, 1)


# =========================
# MAIN
# =========================
def main():
    lr_files = sorted(LR_DIR.glob("*.tif"))[:NUM_SAMPLES]
    hr_files = sorted(HR_DIR.glob("*.tif"))[:NUM_SAMPLES]

    # Load models
    G = Generator().to(DEVICE)
    G.load_state_dict(torch.load("models/gan_generator.pt", map_location=DEVICE))
    G.eval()

    srcnn = SRCNN().to(DEVICE)
    srcnn.load_state_dict(torch.load("models/gan_generator.pt", map_location=DEVICE), strict=False)
    srcnn.eval()

    for idx, (lr_path, hr_path) in enumerate(zip(lr_files, hr_files), 1):
        lr, hr = load_pair(lr_path, hr_path)
        lr = lr.to(DEVICE)
        hr = hr.to(DEVICE)

        # Predictions
        bic = bicubic_upsample(lr, hr.shape[1:])
        with torch.no_grad():
            src_out = srcnn(lr.unsqueeze(0)).squeeze(0)
            gan_out = G(lr.unsqueeze(0)).squeeze(0)

        # =====================
        # RGB COMPARISON
        # =====================
        fig, axes = plt.subplots(1, 5, figsize=(18, 4))

        images = [
            ("LR", bicubic_upsample(lr, hr.shape[1:])),
            ("Bicubic", bic),
            ("SRCNN", src_out),
            ("GAN", gan_out),
            ("HR", hr),
        ]

        for ax, (title, img) in zip(axes, images):
            ax.imshow(to_numpy(img))
            ax.set_title(title)
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(OUT_DIR / f"sample_{idx:02d}.png", dpi=200)
        plt.close()

        # =====================
        # NDVI COMPARISON
        # =====================
        nir = lr[3:4]
        nir_up = F.interpolate(
            nir.unsqueeze(0), size=hr.shape[1:], mode="bilinear", align_corners=False
        ).squeeze(0)

        gan_4 = torch.cat([gan_out, nir_up], dim=0)
        hr_4 = torch.cat([hr, nir_up], dim=0)

        ndvi_gan = ndvi(gan_4)
        ndvi_hr = ndvi(hr_4)
        diff = torch.abs(ndvi_gan - ndvi_hr)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        for ax, data, title in zip(
            axes,
            [ndvi_gan, ndvi_hr, diff],
            ["NDVI (GAN)", "NDVI (HR)", "|Difference|"]
        ):
            im = ax.imshow(data.cpu(), cmap="RdYlGn")
            ax.set_title(title)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046)

        plt.tight_layout()
        plt.savefig(OUT_DIR / f"sample_{idx:02d}_ndvi.png", dpi=200)
        plt.close()

    print("✅ Visualization complete. Images saved to outputs/visuals/")


if __name__ == "__main__":
    main()
