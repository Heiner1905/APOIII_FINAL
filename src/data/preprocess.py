"""
preprocess.py — Preparación de los datos (Fase 3 de CRISP-DM).

Funciones reutilizables (sin gráficas ni guardado a disco) que los notebooks
importan. Cubren:

    - Limpieza del índice: descartar imágenes corruptas y diminutas (mitiga el
      sesgo de procedencia detectado en el EDA).
    - Splits estratificados y REPRODUCIBLES (mismo `seed` => mismos splits en
      todas las fases, sin necesidad de persistir archivos pesados).
    - Carga y normalización de imágenes a tamaño fijo (RGB, [0, 1]).
    - Data augmentation ligera (para la CNN).
    - Pesos de clase para manejar el desbalanceo.
    - Extracción de características para los modelos de ML tradicional:
      histogramas de color en HSV + HOG.

Decisiones de diseño (acordadas en la Sección 3):
    - Tamaño de entrada: 128x128.
    - Desbalanceo: class weights + augmentation.
    - Descriptores ML: color (HSV) + HOG.
    - El split es 70/15/15 estratificado por (fruta x calidad).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.data.dataset import QUALITY_ORDER, load_rgb

# --------------------------------------------------------------------------- #
# Constantes de preparación
# --------------------------------------------------------------------------- #
IMG_SIZE = (128, 128)  # (ancho, alto) de entrada al modelo
SEED = 42

# Parámetros de extracción de características (ML tradicional)
HSV_BINS = 32          # bins por canal H, S, V -> 96 dims de color
HOG_ORIENT = 9
HOG_PPC = (16, 16)     # pixels_per_cell
HOG_CPB = (2, 2)       # cells_per_block


# --------------------------------------------------------------------------- #
# 1. Limpieza
# --------------------------------------------------------------------------- #
def clean_index(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve el índice sin imágenes corruptas ni diminutas.

    El EDA mostró que las imágenes diminutas (<64 px) están correlacionadas con
    la clase (0 en Good, muchas en Bad/Regular) y que al reescalarlas producen
    artefactos; por eso se descartan.

    Args:
        df: índice crudo (salida de `dataset.build_index`).

    Returns:
        Nuevo DataFrame filtrado, con índice reseteado.
    """
    clean = df[(~df["corrupt"]) & (~df["tiny"])].reset_index(drop=True)
    return clean


# --------------------------------------------------------------------------- #
# 2. Splits estratificados y reproducibles
# --------------------------------------------------------------------------- #
def make_splits(
    df: pd.DataFrame,
    seed: int = SEED,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> pd.DataFrame:
    """Asigna a cada fila un split train/val/test estratificado.

    La estratificación se hace por la etiqueta combinada (fruta x calidad) para
    preservar tanto el balance de calidad (objetivo del modelo) como el de fruta
    en las tres particiones. Es determinista dado `seed`.

    Args:
        df: índice limpio.
        seed: semilla para reproducibilidad.
        val_size, test_size: proporciones del total para validación y prueba.

    Returns:
        Copia del DataFrame con una columna nueva `split` in {train, val, test}.
    """
    df = df.copy()
    strat = df["fruit"] + "_" + df["quality"]

    # 1) Separar test del resto.
    train_val_idx, test_idx = train_test_split(
        df.index, test_size=test_size, random_state=seed, stratify=strat
    )
    # 2) Separar val de lo restante (ajustando la proporción relativa).
    rel_val = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=rel_val,
        random_state=seed,
        stratify=strat.loc[train_val_idx],
    )

    df["split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"
    return df


# --------------------------------------------------------------------------- #
# 3. Carga y normalización de imágenes
# --------------------------------------------------------------------------- #
def load_image_array(path: str | Path, size: tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """Carga una imagen como array RGB float32 normalizado en [0, 1].

    Reusa `load_rgb` (que aplana RGBA->RGB) y redimensiona a `size`.

    Returns:
        Array de forma (alto, ancho, 3), dtype float32, valores en [0, 1].
    """
    im = load_rgb(path, size=size)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr


# --------------------------------------------------------------------------- #
# 4. Data augmentation ligera (para la CNN)
# --------------------------------------------------------------------------- #
def augment_image(im: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Aplica una augmentación aleatoria ligera a una imagen PIL.

    Transformaciones: flip horizontal, rotación pequeña y jitter de
    brillo/contraste. Pensada para enriquecer las clases sin alterar la calidad
    aparente de la fruta (no se usan distorsiones agresivas de color que
    cambiarían la madurez percibida).

    Args:
        im: imagen PIL (RGB).
        rng: generador aleatorio de numpy (para reproducibilidad).

    Returns:
        Imagen PIL aumentada.
    """
    if rng.random() < 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    angle = rng.uniform(-15, 15)
    im = im.rotate(angle, resample=Image.BILINEAR, fillcolor=(255, 255, 255))
    # Jitter suave de brillo y contraste (±15 %).
    im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.85, 1.15))
    im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.85, 1.15))
    return im


# --------------------------------------------------------------------------- #
# 5. Pesos de clase (desbalanceo)
# --------------------------------------------------------------------------- #
def class_weights(labels: pd.Series) -> dict[str, float]:
    """Calcula pesos balanceados por clase para usar en el entrenamiento.

    Args:
        labels: serie con las etiquetas de calidad (texto).

    Returns:
        Diccionario {clase: peso}, con más peso a las clases minoritarias.
    """
    classes = np.array(QUALITY_ORDER)
    weights = compute_class_weight("balanced", classes=classes, y=labels.values)
    return {c: float(w) for c, w in zip(classes, weights)}


# --------------------------------------------------------------------------- #
# 6. Extracción de características (ML tradicional)
# --------------------------------------------------------------------------- #
def color_hist_hsv(arr: np.ndarray, bins: int = HSV_BINS) -> np.ndarray:
    """Histograma de color en espacio HSV (concatenado H, S, V).

    El espacio HSV separa el matiz (madurez/color) de la iluminación, lo que lo
    hace más robusto que RGB para juzgar calidad/madurez.

    Args:
        arr: imagen RGB float [0, 1], forma (H, W, 3).
        bins: número de bins por canal.

    Returns:
        Vector normalizado de longitud 3*bins.
    """
    hsv = rgb2hsv(arr)
    feats = []
    for ch in range(3):
        h, _ = np.histogram(hsv[:, :, ch], bins=bins, range=(0, 1))
        feats.append(h)
    feats = np.concatenate(feats).astype(np.float32)
    total = feats.sum()
    return feats / total if total > 0 else feats


def hog_features(arr: np.ndarray, visualize: bool = False):
    """Descriptor HOG (Histogram of Oriented Gradients) sobre la imagen en gris.

    Captura forma, bordes y defectos (golpes, deformaciones).

    Args:
        arr: imagen RGB float [0, 1], forma (H, W, 3).
        visualize: si True, devuelve también la imagen HOG para graficar.

    Returns:
        Vector HOG, o (vector, imagen_hog) si visualize=True.
    """
    gray = rgb2gray(arr)
    return hog(
        gray,
        orientations=HOG_ORIENT,
        pixels_per_cell=HOG_PPC,
        cells_per_block=HOG_CPB,
        block_norm="L2-Hys",
        visualize=visualize,
        feature_vector=True,
    )


def load_images_uint8(
    df: pd.DataFrame, size: tuple[int, int] = IMG_SIZE, verbose: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Carga un conjunto de imágenes como tensor uint8 (para la CNN).

    Devuelve las imágenes SIN normalizar (uint8 [0, 255]); la normalización la
    hace la propia CNN con una capa Rescaling. Aplana RGBA->RGB vía `load_rgb`.

    Args:
        df: subconjunto del índice (columnas `path` y `quality`).
        size: tamaño de salida (ancho, alto).
        verbose: imprime progreso cada 1000 imágenes.

    Returns:
        (X, y): X de forma (n, alto, ancho, 3) uint8; y array de etiquetas.
    """
    n = len(df)
    X = np.empty((n, size[1], size[0], 3), dtype=np.uint8)
    y = np.empty(n, dtype=object)
    for i, (_, row) in enumerate(df.iterrows()):
        X[i] = np.asarray(load_rgb(row["path"], size=size), dtype=np.uint8)
        y[i] = row["quality"]
        if verbose and (i + 1) % 1000 == 0:
            print(f"  cargadas {i + 1}/{n} imágenes...")
    return X, y.astype(str)


def extract_features(
    df: pd.DataFrame, size: tuple[int, int] = IMG_SIZE, verbose: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Construye la matriz de características (color HSV + HOG) para un conjunto.

    Recorre las imágenes de `df`, las carga normalizadas y concatena el
    histograma HSV con el descriptor HOG.

    Args:
        df: subconjunto del índice (con columnas `path` y `quality`).
        size: tamaño al que se redimensionan las imágenes.
        verbose: imprime progreso cada 1000 imágenes.

    Returns:
        (X, y): X de forma (n, d) float32; y array de etiquetas de calidad.
    """
    feats, labels = [], []
    n = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        arr = load_image_array(row["path"], size=size)
        vec = np.concatenate([color_hist_hsv(arr), hog_features(arr)])
        feats.append(vec)
        labels.append(row["quality"])
        if verbose and (i + 1) % 1000 == 0:
            print(f"  procesadas {i + 1}/{n} imágenes...")
    return np.asarray(feats, dtype=np.float32), np.asarray(labels)
