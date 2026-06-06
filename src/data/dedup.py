"""
dedup.py — Detección de casi-duplicados y split consciente de grupos (Fase 5).

El dataset contiene fotos en ráfaga del mismo ejemplar físico (casi idénticas).
Un split aleatorio puede repartir esos casi-duplicados entre train y test,
provocando **fuga de información** y métricas optimistas.

Este módulo:
    1. Calcula un *difference hash* (dHash) de 256 bits por imagen (huella
       perceptual robusta a cambios de iluminación).
    2. Agrupa imágenes casi-idénticas por componentes conexas (distancia de
       Hamming <= umbral) -> cada grupo ≈ un "ejemplar" físico.
    3. Construye un split train/val/test **consciente de grupos**: cada grupo
       queda íntegro en una sola partición (sin fuga entre splits).

Se usa dHash de 16x16 (256 bits) en vez de aHash de 64 bits porque este último
es demasiado grueso: distintas frutas con fondo/brillo similar colisionan y, por
encadenamiento (single-linkage), se fusionan en clústeres enormes irreales. Con
dHash-256 y umbral 10 los grupos quedan en tamaños realistas (ráfagas).

Solo numpy/scipy/PIL (sin dependencias nuevas). Los hashes se cachean.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import GroupShuffleSplit

from src.data.dataset import load_rgb

SEED = 42
HASH_SIZE = 16          # dHash 16x16 -> 256 bits
GROUP_THRESHOLD = 10    # distancia de Hamming máxima para casi-duplicados


def dhash(path: str | Path, hash_size: int = HASH_SIZE) -> np.ndarray:
    """Difference hash (dHash) de una imagen como vector de bits (uint8 0/1).

    Reduce a (hash_size+1, hash_size) en gris y compara píxeles adyacentes
    horizontalmente. Robusto a brillo/contraste; discrimina mejor que aHash.

    Returns:
        Array uint8 de longitud hash_size*hash_size con valores en {0,1}.
    """
    im = load_rgb(path).convert("L").resize((hash_size + 1, hash_size))
    a = np.asarray(im, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten().astype(np.uint8)


def compute_hash_bits(df: pd.DataFrame, cache_path: Path | None = None,
                      verbose: bool = True) -> np.ndarray:
    """Calcula la matriz de bits dHash (n, 256) alineada al orden de `df`.

    Cachea por ruta en un .npz para no recomputar en cada notebook.

    Args:
        df: índice (columna `path`).
        cache_path: ruta .npz de caché {paths, bits}. Si None, no cachea.

    Returns:
        Array uint8 de forma (len(df), HASH_SIZE**2).
    """
    nbits = HASH_SIZE * HASH_SIZE
    cache: dict[str, np.ndarray] = {}
    if cache_path is not None and Path(cache_path).exists():
        d = np.load(cache_path, allow_pickle=True)
        cache = {p: b for p, b in zip(d["paths"], d["bits"])}

    out = np.empty((len(df), nbits), dtype=np.uint8)
    missing = 0
    for i, p in enumerate(df["path"]):
        if p in cache:
            out[i] = cache[p]
        else:
            out[i] = dhash(p)
            cache[p] = out[i]
            missing += 1
            if verbose and missing % 2000 == 0:
                print(f"  hashes calculados: {missing}...")

    if cache_path is not None and missing > 0:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            paths=np.array(list(cache.keys())),
            bits=np.array(list(cache.values()), dtype=np.uint8),
        )
    return out


def assign_groups(bits: np.ndarray, threshold: int = GROUP_THRESHOLD) -> np.ndarray:
    """Agrupa por componentes conexas según distancia de Hamming entre hashes.

    Args:
        bits: matriz (n, nbits) uint8 de hashes.
        threshold: distancia máxima para conectar dos imágenes.

    Returns:
        Array (n,) de etiquetas de grupo (int).
    """
    b = bits.astype(np.float32)
    inv = 1.0 - b
    dist = b @ inv.T + inv @ b.T          # distancia de Hamming por pares
    adj = dist <= threshold
    _, labels = connected_components(csr_matrix(adj), directed=False)
    return labels


def make_group_splits(
    df: pd.DataFrame,
    group_col: str = "group",
    seed: int = SEED,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> pd.DataFrame:
    """Split train/val/test consciente de grupos (sin fuga entre splits).

    Cada grupo queda íntegramente en una sola partición. Requiere que `df`
    tenga la columna `group_col`.

    Returns:
        Copia de df con columna `gsplit` in {train, val, test}.
    """
    df = df.copy()
    idx = np.arange(len(df))
    groups = df[group_col].to_numpy()

    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_i, test_i = next(gss1.split(idx, groups=groups))

    rel_val = val_size / (1.0 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed)
    tr_rel, va_rel = next(gss2.split(trainval_i, groups=groups[trainval_i]))
    train_i, val_i = trainval_i[tr_rel], trainval_i[va_rel]

    gsplit = np.empty(len(df), dtype=object)
    gsplit[train_i] = "train"
    gsplit[val_i] = "val"
    gsplit[test_i] = "test"
    df["gsplit"] = gsplit
    return df
