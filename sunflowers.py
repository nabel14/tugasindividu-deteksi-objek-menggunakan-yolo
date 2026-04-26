import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image

# ===============================
# LOAD MODEL
# ===============================
model = YOLO("model/best.pt")  # pastikan ini model sunflowers kamu

# ===============================
# UI
# ===============================
st.set_page_config(page_title="Deteksi Bunga Matahari", layout="centered")

st.title("Deteksi Bunga Matahari dengan YOLO")
st.write("Upload gambar untuk mendeteksi Bunga Matahari")

# Upload file
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "png", "jpeg"])

# Slider confidence
conf = st.slider("Confidence Threshold", 0.1, 1.0, 0.5)

if uploaded_file is not None:
    # Baca gambar
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    st.subheader("📷 Gambar Asli")
    st.image(image, use_column_width=True)

    # ===============================
    # DETEKSI
    # ===============================
    results = model(image_np, conf=conf)

    # Gambar hasil
    annotated_frame = results[0].plot()

    st.subheader("🎯 Hasil Deteksi")
    st.image(annotated_frame, use_column_width=True)

    # ===============================
    # INFO
    # ===============================
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        st.success(f"Jumlah Bunga Matahari terdeteksi: {len(boxes)}")
    else:
        st.warning("Tidak ada Bunga Matahari terdeteksi")