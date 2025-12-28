import torch
import json
from pathlib import Path
import rasterio
import numpy as np
import torch.nn.functional as F

from src.models.generator import Generator
from src.baselines.srcnn_model import SRCNN
from src.eval.metrics import psnr, ssim, ndvi_rmse, LPIPSVGG


# =========================
# CONFIG
# =========================
LR_DIR = Path("data/paired/satellite")
HR_DIR = Path("data/paired/drone")
OUT_FILE = Path("outputs/metrics/metrics.json")

NUM_SAMPLES = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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

    lpips_fn = LPIPSVGG(device=DEVICE)

    results = {
        "bicubic": [],
        "srcnn": [],
        "gan": []
    }

    for lr_path, hr_path in zip(lr_files, hr_files):
        lr, hr = load_pair(lr_path, hr_path)
        lr = lr.to(DEVICE)
        hr = hr.to(DEVICE)

        # Bicubic
        bic = bicubic_upsample(lr, hr.shape[1:])

        # SRCNN
        with torch.no_grad():
            src_out = srcnn(lr.unsqueeze(0)).squeeze(0)

        # GAN
        with torch.no_grad():
            gan_out = G(lr.unsqueeze(0)).squeeze(0)

        # Metrics
        for name, pred in [
            ("bicubic", bic),
            ("srcnn", src_out),
            ("gan", gan_out)
        ]:
            metrics = {
                "psnr": float(psnr(pred, hr)),
                "ssim": float(ssim(pred, hr)),
                "lpips": float(lpips_fn(pred, hr)),
            }

            # NDVI RMSE (append NIR)
            nir = lr[3:4]
            nir_up = F.interpolate(
                nir.unsqueeze(0), size=pred.shape[1:], mode="bilinear", align_corners=False
            ).squeeze(0)

            pred_4 = torch.cat([pred, nir_up], dim=0)
            hr_4 = torch.cat([hr, nir_up], dim=0)

            metrics["ndvi_rmse"] = float(ndvi_rmse(pred_4, hr_4))

            results[name].append(metrics)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print("✅ Evaluation complete.")
    print(f"Metrics saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
