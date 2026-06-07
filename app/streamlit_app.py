"""
streamlit_app.py — Interfaz gráfica del clasificador de calidad de frutas.

Permite cargar o capturar la imagen de una fruta (sobre fondo simple), elegir el
tipo de fruta y el modelo (SVM o CNN), y muestra:
    - la clase de calidad predicha y las probabilidades por clase,
    - la estimación de tamaño (pequeño/mediano/grande),
    - el overlay de la segmentación.

Ejecutar con:
    .venv/bin/streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# Exponer la raíz del proyecto para importar `src`.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import FRUITS, QUALITY_ORDER  # noqa: E402
from src.deployment.predict import estimate_size, predict_quality  # noqa: E402

st.set_page_config(page_title="Calidad de Frutas", page_icon="🍎", layout="centered")

st.title("🍎 Clasificación automática de calidad de frutas")
st.caption("Proyecto Final · Algoritmos y Programación III · Universidad ICESI · 2026-1")

# --- Controles ---
st.sidebar.header("Configuración")
fruit = st.sidebar.selectbox("Tipo de fruta", FRUITS)
model_name = st.sidebar.radio("Modelo de calidad", ["SVM (RBF)", "CNN"])
kind = "svm" if model_name.startswith("SVM") else "cnn"
source = st.sidebar.radio("Fuente de imagen", ["Subir archivo", "Cámara"])
st.sidebar.info("Usa una imagen de **una sola fruta** sobre **fondo simple y "
                "uniforme** para mejores resultados.")

# --- Entrada de imagen ---
img = None
if source == "Subir archivo":
    up = st.file_uploader("Imagen de la fruta", type=["jpg", "jpeg", "png"])
    if up is not None:
        img = Image.open(up)
else:
    cam = st.camera_input("Captura la fruta")
    if cam is not None:
        img = Image.open(cam)

# --- Inferencia y resultados ---
if img is not None:
    rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)

    with st.spinner("Analizando imagen..."):
        proba = predict_quality(rgb, kind=kind)
        size = estimate_size(rgb, fruit)
    pred = max(proba, key=proba.get)

    col1, col2 = st.columns(2)
    col1.image(img, caption="Imagen de entrada", use_column_width=True)
    with col2:
        st.metric("Calidad predicha", pred)
        st.metric("Tamaño estimado", size["size"].capitalize())
        st.caption(f"Fruta: {fruit} · Modelo: {model_name}")
        st.caption(f"Área relativa segmentada: {size['area_fraction'] * 100:.1f}%")

    st.subheader("Probabilidades por clase de calidad")
    st.bar_chart({"probabilidad": {q: proba[q] for q in QUALITY_ORDER}})

    st.subheader("Segmentación de la fruta")
    mask = size["mask"]
    base = np.asarray(img.convert("RGB").resize((mask.shape[1], mask.shape[0])))
    overlay = base.copy()
    overlay[~mask] = (overlay[~mask] * 0.3).astype(np.uint8)  # atenuar el fondo
    st.image(overlay, caption="Fruta segmentada (fondo atenuado)",
             use_column_width=True)
else:
    st.info("⬅️ Sube o captura una imagen para comenzar el análisis.")
