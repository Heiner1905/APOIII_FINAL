"""
segmentation.py — Segmentación de la fruta y estimación de tamaño (Fase 6).

Sobre fondo simple/uniforme (como exige el problema y como ocurre en la GUI), se
segmenta la fruta del fondo con umbral de Otsu y limpieza morfológica, y se mide
su **área relativa** (fracción de la imagen). El tamaño (pequeño/mediano/grande)
se decide comparando esa área con los **terciles por tipo de fruta** calibrados
sobre el dataset (ver `artifacts/size_thresholds.json`).

Solo skimage/numpy/PIL (sin dependencias nuevas).
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.morphology import (
    binary_closing,
    disk,
    remove_small_holes,
    remove_small_objects,
)

SIZE_LABELS = ["pequeño", "mediano", "grande"]


def segment_fruit(rgb: np.ndarray, max_side: int = 256) -> tuple[np.ndarray, float]:
    """Segmenta la fruta del fondo y devuelve (máscara, fracción de área).

    Estrategia: Otsu sobre la imagen en gris; se identifica el fondo como la
    clase que predomina en el **borde** de la imagen, y la fruta como la otra.
    Se limpia con morfología y se conserva el componente conexo más grande.

    Args:
        rgb: imagen RGB uint8 (H, W, 3).
        max_side: lado máximo al que se reduce para acelerar.

    Returns:
        (mask, area_fraction): máscara booleana (al tamaño reducido) y fracción
        del área de la imagen ocupada por la fruta en [0, 1].
    """
    im = Image.fromarray(rgb).convert("RGB")
    # Reducir para acelerar manteniendo proporción.
    w, h = im.size
    scale = max_side / max(w, h)
    if scale < 1:
        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    arr = np.asarray(im)
    gray = rgb2gray(arr)

    try:
        t = threshold_otsu(gray)
    except ValueError:
        return np.ones(gray.shape, dtype=bool), 1.0

    dark = gray < t
    light = ~dark
    # Máscara del borde para decidir cuál clase es fondo.
    border = np.zeros(gray.shape, dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    # El fondo domina el borde; la fruta es la clase contraria.
    mask = light if dark[border].mean() >= light[border].mean() else dark

    # Limpieza morfológica.
    min_size = max(16, int(0.001 * mask.size))
    mask = binary_closing(mask, disk(3))
    mask = remove_small_holes(mask, area_threshold=min_size)
    mask = remove_small_objects(mask, min_size=min_size)

    # Conservar el componente conexo más grande (la fruta central).
    lbl = label(mask)
    if lbl.max() == 0:
        return mask, float(mask.mean())
    counts = np.bincount(lbl.ravel())
    counts[0] = 0  # ignorar fondo
    mask = lbl == counts.argmax()
    return mask, float(mask.mean())


def area_fraction(rgb: np.ndarray) -> float:
    """Atajo: devuelve solo la fracción de área de la fruta segmentada."""
    return segment_fruit(rgb)[1]


def classify_size(area_frac: float, fruit: str, thresholds: dict) -> str:
    """Clasifica el tamaño según los terciles por fruta.

    Args:
        area_frac: fracción de área de la fruta (salida de `segment_fruit`).
        fruit: nombre de la fruta (clave en `thresholds`).
        thresholds: dict {fruta: [p33, p66]}.

    Returns:
        'pequeño' | 'mediano' | 'grande'. Si la fruta no está calibrada, usa
        los percentiles globales bajo la clave '_global'.
    """
    p33, p66 = thresholds.get(fruit, thresholds.get("_global", [0.0, 1.0]))
    if area_frac < p33:
        return "pequeño"
    if area_frac < p66:
        return "mediano"
    return "grande"
