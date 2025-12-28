from huggingface_hub import hf_hub_download
from pathlib import Path
import zipfile
import shutil

# =========================
# CONFIG
# =========================
REPO_ID = "isp-uv-es/SEN2NAIP"
ZIP_PATH = "cross-sensor/cross-sensor.zip"
MAX_PAIRS = 200

ROOT = Path(".")
TMP_DIR = ROOT / "tmp_sen2naip"
SAT_OUT = ROOT / "data" / "satellite"
DRONE_OUT = ROOT / "data" / "drone"

SAT_OUT.mkdir(parents=True, exist_ok=True)
DRONE_OUT.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# DOWNLOAD ZIP
# =========================
print("Downloading SEN2NAIP cross-sensor.zip ...")

zip_file = hf_hub_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    filename=ZIP_PATH,
    local_dir=TMP_DIR,
    local_dir_use_symlinks=False
)

print("Download complete.")

# =========================
# EXTRACT ZIP
# =========================
print("Extracting zip...")
with zipfile.ZipFile(zip_file, "r") as zf:
    zf.extractall(TMP_DIR)

cross_sensor_root = TMP_DIR / "cross-sensor"
roi_dirs = sorted([d for d in cross_sensor_root.iterdir() if d.is_dir()])

print(f"Found {len(roi_dirs)} ROI folders")

# =========================
# COPY FIRST N PAIRS
# =========================
count = 0
for roi in roi_dirs:
    lr = roi / "lr.tif"
    hr = roi / "hr.tif"

    if lr.exists() and hr.exists():
        shutil.copy(lr, SAT_OUT / f"{roi.name}_lr.tif")
        shutil.copy(hr, DRONE_OUT / f"{roi.name}_hr.tif")
        count += 1
        print(f"Copied pair {count}: {roi.name}")

    if count >= MAX_PAIRS:
        break

print(f"✅ Finished. Total paired samples copied: {count}")
