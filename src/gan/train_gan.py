import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import rasterio
import numpy as np

from src.models.generator import Generator
from src.models.discriminator import Discriminator
from src.losses.perceptual import PerceptualLoss
from src.losses.spectral import SpectralNDVILoss


# =========================
# CONFIG (OVERFIT MODE)
# =========================
LR_DIR = Path("data/paired/satellite")
HR_DIR = Path("data/paired/drone")

EPOCHS = 30
BATCH_SIZE = 1
LR = 2e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LAMBDA_L1 = 100.0
LAMBDA_PERC = 10.0
LAMBDA_SPEC = 5.0


# =========================
# DATASET
# =========================
class PairedDataset(Dataset):
    def __init__(self, lr_dir, hr_dir):
        self.lr_files = sorted(lr_dir.glob("*.tif"))
        self.hr_files = sorted(hr_dir.glob("*.tif"))
        assert len(self.lr_files) == len(self.hr_files)

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, idx):
        with rasterio.open(self.lr_files[idx]) as ds:
            lr = ds.read().astype(np.float32)

        with rasterio.open(self.hr_files[idx]) as ds:
            hr = ds.read().astype(np.float32)

        lr = lr / lr.max()
        hr = hr / hr.max()

        return torch.from_numpy(lr), torch.from_numpy(hr)


# =========================
# TRAINING
# =========================
def main():
    dataset = PairedDataset(LR_DIR, HR_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)

    opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

    gan_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    perc_loss = PerceptualLoss().to(DEVICE)
    spec_loss = SpectralNDVILoss(scale=4).to(DEVICE)

    print(f"Training on device: {DEVICE}")
    print(f"Total samples: {len(dataset)}")

    for epoch in range(1, EPOCHS + 1):
        G.train()
        D.train()

        g_loss_epoch = 0.0
        d_loss_epoch = 0.0

        for lr_img, hr_img in loader:
            lr_img = lr_img.to(DEVICE)
            hr_img = hr_img.to(DEVICE)

            # =====================
            # Train Discriminator
            # =====================
            with torch.no_grad():
                fake_hr = G(lr_img)

            real_pred = D(lr_img, hr_img)
            fake_pred = D(lr_img, fake_hr)

            real_labels = torch.ones_like(real_pred)
            fake_labels = torch.zeros_like(fake_pred)

            d_loss = (
                gan_loss(real_pred, real_labels)
                + gan_loss(fake_pred, fake_labels)
            ) * 0.5

            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

            # =====================
            # Train Generator
            # =====================
            fake_hr = G(lr_img)
            fake_pred = D(lr_img, fake_hr)

            g_gan = gan_loss(fake_pred, real_labels)
            g_l1 = l1_loss(fake_hr, hr_img)
            g_perc = perc_loss(fake_hr, hr_img)

            # For spectral loss, append NIR from LR (upsampled)
            nir = lr_img[:, 3:4]
            nir_up = torch.nn.functional.interpolate(
                nir, size=fake_hr.shape[2:], mode="bilinear", align_corners=False
            )

            fake_hr_4 = torch.cat([fake_hr, nir_up], dim=1)
            real_hr_4 = torch.cat([hr_img, nir_up], dim=1)

            g_spec = spec_loss(fake_hr_4, real_hr_4)

            g_loss = (
                g_gan
                + LAMBDA_L1 * g_l1
                + LAMBDA_PERC * g_perc
                + LAMBDA_SPEC * g_spec
            )

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

            g_loss_epoch += g_loss.item()
            d_loss_epoch += d_loss.item()

        print(
            f"Epoch [{epoch}/{EPOCHS}] | "
            f"G Loss: {g_loss_epoch/len(loader):.4f} | "
            f"D Loss: {d_loss_epoch/len(loader):.4f}"
        )

    # =====================
    # SAVE FINAL MODELS
    # =====================
    Path("models").mkdir(exist_ok=True)

    torch.save(G.state_dict(), "models/gan_generator.pt")
    torch.save(D.state_dict(), "models/gan_discriminator.pt")

    print("✅ GAN training complete. Models saved.")


if __name__ == "__main__":
    main()
