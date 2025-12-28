import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from pathlib import Path
import numpy as np
import cv2

# =========================
# CONFIG (LOCKED)
# =========================
PATCH_SIZE = 256
SCALE = 4  # LR -> HR
USE_SIFT_FALLBACK = True

LR_DIR = Path("data/satellite")
HR_DIR = Path("data/drone")

OUT_LR = Path("data/paired/satellite")
OUT_HR = Path("data/paired/drone")

OUT_LR.mkdir(parents=True, exist_ok=True)
OUT_HR.mkdir(parents=True, exist_ok=True)


def read_rgb_nir(ds):
    """
    Assumes Sentinel-2 style ordering.
    Keeps first 4 bands: R,G,B,NIR
    """
    arr = ds.read()
    if arr.shape[0] < 4:
        raise ValueError("LR image has <4 bands; RGB+NIR required")
    return arr[:4]


def read_rgb(ds):
    arr = ds.read()
    return arr[:3]

def sifts_align(src, dst):
    """Align src to dst using SIFT homography (robust fallback)."""

    src_gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    dst_gray = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(src_gray, None)
    kp2, des2 = sift.detectAndCompute(dst_gray, None)

    # If descriptors missing → skip
    if des1 is None or des2 is None:
        return src

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for pair in matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)

    # Need at least 4 matches for homography
    if len(good) < 4:
        return src

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    if H is None:
        return src

    h, w = dst.shape[:2]
    aligned = cv2.warpPerspective(src, H, (w, h))
    return aligned


def main():
    lr_files = sorted(LR_DIR.glob("*.tif"))

    patch_count = 0

    for lr_path in lr_files:
        hr_path = HR_DIR / lr_path.name.replace("_lr", "_hr")
        if not hr_path.exists():
            print(f"Missing HR for {lr_path.name}, skipping")
            continue

        print(f"Processing {lr_path.name}")

        with rasterio.open(lr_path) as lr_ds, rasterio.open(hr_path) as hr_ds:
            lr = read_rgb_nir(lr_ds)
            hr = read_rgb(hr_ds)

            # Resample LR to HR grid using georeferencing
            lr_resampled = np.zeros(
                (lr.shape[0], hr.shape[1], hr.shape[2]), dtype=np.float32
            )

            for b in range(lr.shape[0]):
                reproject(
                    source=lr[b],
                    destination=lr_resampled[b],
                    src_transform=lr_ds.transform,
                    src_crs=lr_ds.crs,
                    dst_transform=hr_ds.transform,
                    dst_crs=hr_ds.crs,
                    resampling=Resampling.bilinear,
                )

            # Optional SIFT fallback (RGB only)
            if USE_SIFT_FALLBACK:
                lr_rgb = np.transpose(lr_resampled[:3], (1, 2, 0)).astype(np.uint8)
                hr_rgb = np.transpose(hr, (1, 2, 0)).astype(np.uint8)
                lr_rgb_aligned = sifts_align(lr_rgb, hr_rgb)
                lr_resampled[:3] = np.transpose(lr_rgb_aligned, (2, 0, 1))

            H, W = hr.shape[1], hr.shape[2]

            for y in range(0, H - PATCH_SIZE + 1, PATCH_SIZE):
                for x in range(0, W - PATCH_SIZE + 1, PATCH_SIZE):
                    lr_patch = lr_resampled[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                    hr_patch = hr[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]

                    if lr_patch.shape[1:] != (PATCH_SIZE, PATCH_SIZE):
                        continue

                    out_lr = OUT_LR / f"{lr_path.stem}_p{patch_count}.tif"
                    out_hr = OUT_HR / f"{hr_path.stem}_p{patch_count}.tif"

                    with rasterio.open(
                        out_lr,
                        "w",
                        driver="GTiff",
                        height=PATCH_SIZE,
                        width=PATCH_SIZE,
                        count=lr_patch.shape[0],
                        dtype=lr_patch.dtype,
                    ) as dst:
                        dst.write(lr_patch)

                    with rasterio.open(
                        out_hr,
                        "w",
                        driver="GTiff",
                        height=PATCH_SIZE,
                        width=PATCH_SIZE,
                        count=hr_patch.shape[0],
                        dtype=hr_patch.dtype,
                    ) as dst:
                        dst.write(hr_patch)

                    patch_count += 1

    print(f"✅ Patch extraction complete. Total patches: {patch_count}")


if __name__ == "__main__":
    main()
