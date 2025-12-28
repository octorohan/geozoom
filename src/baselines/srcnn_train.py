import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import rasterio
from pathlib import Path
import numpy as np

from src.baselines.srcnn_model import SRCNN


# =========================
# CONFIG (SANITY MODE)
# =========================
LR_DIR = Path("data/paired/satellite")
HR_DIR = Path("data/paired/drone")

EPOCHS = 20
BATCH_SIZE = 2
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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

        # Normalize to [0,1]
        lr = lr / lr.max()
        hr = hr / hr.max()

        return torch.from_numpy(lr), torch.from_numpy(hr)


# =========================
# TRAINING
# =========================
def main():
    dataset = PairedDataset(LR_DIR, HR_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SRCNN().to(DEVICE)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"Training on device: {DEVICE}")
    print(f"Total samples: {len(dataset)}")

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0

        for lr_img, hr_img in loader:
            lr_img = lr_img.to(DEVICE)
            hr_img = hr_img.to(DEVICE)

            pred = model(lr_img)
            loss = criterion(pred, hr_img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch [{epoch}/{EPOCHS}] - L1 Loss: {avg_loss:.6f}")

    print("✅ SRCNN sanity training complete.")


if __name__ == "__main__":
    main()
