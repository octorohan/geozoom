import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import streamlit as st
import torch
import rasterio
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.models.generator import Generator


# =========================
# CONFIG
# =========================
MODEL_PATH = "models/gan_generator.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# UTILS
# =========================
def pad_to_multiple(x, multiple=16):
    """
    Pads tensor so H and W are divisible by `multiple`
    """
    _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    return F.pad(x, (0, pad_w, 0, pad_h)), h, w


def load_tif(file):
    with rasterio.open(file) as ds:
        img = ds.read().astype(np.float32)
    img = img / img.max()
    return torch.from_numpy(img)


def ndvi(x):
    red = x[0]
    nir = x[3]
    return (nir - red) / (nir + red + 1e-6)


def to_numpy(img):
    return img.permute(1, 2, 0).cpu().numpy().clip(0, 1)


# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    model = Generator().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="GeoZoom Demo", layout="wide")
st.title("🌍 GeoZoom — Satellite → Aerial Super-Resolution")

st.markdown(
    """
Upload a **4-channel satellite GeoTIFF (RGB + NIR)**  
The model generates a **drone-like high-resolution image**  
while preserving **vegetation (NDVI) consistency**.
"""
)

uploaded = st.file_uploader("Upload satellite tile (.tif)", type=["tif", "tiff"])

if uploaded:
    lr = load_tif(uploaded).to(DEVICE)

    if lr.shape[0] != 4:
        st.error("Uploaded image must have 4 channels (RGB + NIR).")
        st.stop()

    model = load_model()

    # Pad input to safe size
    lr_padded, orig_h, orig_w = pad_to_multiple(lr)
    with torch.no_grad():
        gan_padded = model(lr_padded.unsqueeze(0)).squeeze(0)

    # Crop back to original size
    gan_out = gan_padded[:, :orig_h, :orig_w]
    
    # NDVI
    nir_up = F.interpolate(
        lr[3:4].unsqueeze(0),
        size=gan_out.shape[1:],
        mode="bilinear",
        align_corners=False
    ).squeeze(0)

    gan_4 = torch.cat([gan_out, nir_up], dim=0)
    ndvi_map = ndvi(gan_4)

    # =====================
    # DISPLAY
    # =====================
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Input (Satellite RGB)")
        st.image(to_numpy(lr[:3]), width="stretch")

    with col2:
        st.subheader("Output (GAN High-Res)")
        st.image(to_numpy(gan_out), width="stretch")

    with col3:
        st.subheader("NDVI (GAN)")
        fig, ax = plt.subplots()
        im = ax.imshow(ndvi_map.cpu(), cmap="RdYlGn")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
        st.pyplot(fig)
