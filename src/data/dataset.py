"""
dataset.py — Utilidades reutilizables de carga e indexado del dataset de frutas.

Este módulo contiene SOLO lógica reutilizable (sin gráficas, sin guardar nada a
disco). Se importa desde los notebooks de cada fase de CRISP-DM, que son los que
muestran los resultados (tablas, figuras) de forma inline.

Función principal:
    build_index(dataset_dir) -> pandas.DataFrame con una fila por imagen y
    metadatos (fruta, calidad, resolución, modo, formato, flags de calidad).

La construcción del índice no decodifica las imágenes completas (PIL es perezoso
con `.size`/`.mode`), por lo que el escaneo de ~9.5k imágenes es rápido.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# --------------------------------------------------------------------------- #
# Rutas y constantes del dataset (compartidas por todas las fases)
# --------------------------------------------------------------------------- #
# parents[2] = raíz del proyecto (src/data/dataset.py -> src/data -> src -> raíz)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "Fruits"

# Mapeo carpeta de calidad -> etiqueta corta
QUALITY_FOLDERS = {
    "Bad Quality_Fruits": "Bad",
    "Good Quality_Fruits": "Good",
    "Regular Quality_Fruits": "Regular",
}
# Orden lógico (peor -> mejor) para ejes y tablas
QUALITY_ORDER = ["Bad", "Regular", "Good"]
FRUITS = ["Apple", "Banana", "Guava", "Lime", "Orange", "Pomegranate"]
VALID_EXTS = {".jpg", ".jpeg", ".png"}

# Paleta consistente para las clases de calidad (reutilizable en notebooks)
QUALITY_PALETTE = {"Bad": "#d62728", "Regular": "#ff7f0e", "Good": "#2ca02c"}

# Umbral mínimo de lado para considerar una imagen "usable". Por debajo se marca
# como diminuta (`tiny`): al reescalar produce artefactos inservibles.
MIN_SIDE = 64


def build_index(dataset_dir: Path | str = DATASET_DIR) -> pd.DataFrame:
    """Recorre el dataset y devuelve un DataFrame con metadatos por imagen.

    Args:
        dataset_dir: ruta a la carpeta `Fruits/`. Por defecto, la del proyecto.

    Returns:
        DataFrame con columnas:
            path (relativa a la raíz), fruit, quality, width, height, area,
            mode, format, corrupt (bool), tiny (bool).
    """
    dataset_dir = Path(dataset_dir)
    rows = []
    for quality_folder, quality_label in QUALITY_FOLDERS.items():
        qpath = dataset_dir / quality_folder
        if not qpath.is_dir():
            continue
        # Cada subcarpeta es "<Fruta>_<Calidad>", p.ej. "Apple_Good".
        for fruit_dir in sorted(qpath.iterdir()):
            if not fruit_dir.is_dir():
                continue
            fruit = fruit_dir.name.split("_")[0]
            for img_path in fruit_dir.iterdir():
                if img_path.suffix.lower() not in VALID_EXTS:
                    continue
                rows.append(_read_meta(img_path, fruit, quality_label))
    return pd.DataFrame(rows)


def _read_meta(img_path: Path, fruit: str, quality: str) -> dict:
    """Lee metadatos de una imagen sin decodificarla completamente.

    Marca `corrupt=True` si no puede abrirse, y `tiny=True` si algún lado es
    menor que MIN_SIDE.
    """
    meta = {
        "path": str(img_path.relative_to(PROJECT_ROOT)),
        "fruit": fruit,
        "quality": quality,
        "width": np.nan,
        "height": np.nan,
        "area": np.nan,
        "mode": None,
        "format": None,
        "corrupt": False,
        "tiny": False,
    }
    try:
        with Image.open(img_path) as im:
            meta["width"], meta["height"] = im.size
            meta["mode"] = im.mode
            meta["format"] = im.format
        meta["area"] = meta["width"] * meta["height"]
        meta["tiny"] = (meta["width"] < MIN_SIDE) or (meta["height"] < MIN_SIDE)
    except Exception:
        meta["corrupt"] = True
    return meta


def load_rgb(path: Path | str, size: tuple[int, int] | None = None) -> Image.Image:
    """Abre una imagen y la devuelve en RGB (aplana RGBA sobre fondo blanco).

    Útil tanto para el EDA (mostrar muestras) como para la preparación de datos.

    Args:
        path: ruta de la imagen (relativa a la raíz o absoluta).
        size: si se indica (w, h), redimensiona a ese tamaño.

    Returns:
        Imagen PIL en modo RGB.
    """
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    im = Image.open(p)
    if im.mode == "RGBA":
        # Aplanar el canal alfa sobre fondo blanco para no introducir un atajo
        # espurio (todas las RGBA del dataset son de la clase Regular).
        background = Image.new("RGB", im.size, (255, 255, 255))
        background.paste(im, mask=im.split()[3])
        im = background
    else:
        im = im.convert("RGB")
    if size is not None:
        im = im.resize(size)
    return im
