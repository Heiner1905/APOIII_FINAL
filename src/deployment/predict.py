"""
predict.py — Funciones de inferencia de alto nivel para el despliegue (Fase 6).

Envuelven la predicción de **calidad** (con SVM o CNN) y la estimación de
**tamaño** (segmentación + terciles por fruta), partiendo de una imagen RGB en
memoria (np.ndarray uint8). Reutilizan el preprocesamiento de la Fase 3, de modo
que la inferencia es idéntica al entrenamiento.

Estas funciones son usadas por la app de Streamlit (`app/streamlit_app.py`) y son
testeables de forma aislada.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.dataset import PROJECT_ROOT, QUALITY_ORDER
from src.data import preprocess as pp
from src.utils.segmentation import classify_size, segment_fruit

ARTIFACTS = PROJECT_ROOT / "artifacts"


# --------------------------------------------------------------------------- #
# Carga perezosa de modelos y umbrales (cacheada)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def load_svm():
    """Carga el SVM de despliegue (con probabilidades)."""
    import joblib
    path = ARTIFACTS / "svm_deploy.pkl"
    if not path.exists():
        path = ARTIFACTS / "svm.pkl"
    return joblib.load(path)


@lru_cache(maxsize=1)
def load_cnn():
    """Carga la CNN entrenada (Keras)."""
    import tensorflow as tf
    return tf.keras.models.load_model(ARTIFACTS / "cnn.keras")


@lru_cache(maxsize=1)
def load_size_thresholds() -> dict:
    """Carga los terciles de tamaño por fruta."""
    with open(ARTIFACTS / "size_thresholds.json", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Predicción de calidad
# --------------------------------------------------------------------------- #
def _to_array(rgb: np.ndarray) -> np.ndarray:
    """Redimensiona a 128x128 y normaliza a [0,1] (RGB)."""
    im = Image.fromarray(rgb).convert("RGB").resize(pp.IMG_SIZE)
    return np.asarray(im, dtype=np.float32) / 255.0


def predict_quality(rgb: np.ndarray, kind: str = "svm") -> dict[str, float]:
    """Predice la calidad y devuelve probabilidades por clase.

    Args:
        rgb: imagen RGB uint8 (H, W, 3).
        kind: 'svm' o 'cnn'.

    Returns:
        dict {clase: probabilidad} ordenado como QUALITY_ORDER.
    """
    if kind == "svm":
        arr = _to_array(rgb)
        feats = np.concatenate([pp.color_hist_hsv(arr), pp.hog_features(arr)])
        model = load_svm()
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([feats])[0]
            order = list(model.classes_)
            return {q: float(proba[order.index(q)]) for q in QUALITY_ORDER}
        # Sin probabilidades: one-hot de la clase predicha.
        pred = model.predict([feats])[0]
        return {q: float(q == pred) for q in QUALITY_ORDER}

    if kind == "cnn":
        im = Image.fromarray(rgb).convert("RGB").resize(pp.IMG_SIZE)
        x = np.asarray(im, dtype=np.uint8)[None]
        proba = load_cnn().predict(x, verbose=0)[0]
        return {q: float(proba[i]) for i, q in enumerate(QUALITY_ORDER)}

    raise ValueError(f"kind desconocido: {kind!r} (usa 'svm' o 'cnn')")


# --------------------------------------------------------------------------- #
# Estimación de tamaño
# --------------------------------------------------------------------------- #
def estimate_size(rgb: np.ndarray, fruit: str) -> dict:
    """Segmenta la fruta y estima su tamaño relativo.

    Returns:
        dict con: size ('pequeño'/'mediano'/'grande'), area_fraction (float) y
        mask (np.ndarray booleano, al tamaño reducido de la segmentación).
    """
    mask, area = segment_fruit(rgb)
    size = classify_size(area, fruit, load_size_thresholds())
    return {"size": size, "area_fraction": area, "mask": mask}
