"""
metrics.py — Métricas y utilidades de evaluación (reutilizables en Fases 4 y 5).

Centraliza el cálculo de las métricas acordadas para el proyecto (métrica
principal = Macro-F1, dado el desbalanceo) y la visualización de matrices de
confusión, para que el modelado (Fase 4) y la evaluación (Fase 5) usen
exactamente los mismos criterios.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)

from src.data.dataset import QUALITY_ORDER


def evaluate(y_true, y_pred, labels: list[str] = QUALITY_ORDER) -> dict:
    """Calcula las métricas clave del proyecto para un conjunto de predicciones.

    Returns:
        dict con: macro_f1, balanced_acc, recall_bad (clase crítica del negocio),
        report (dict de classification_report) y confusion (matriz np.ndarray).
    """
    return {
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro"),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "recall_bad": recall_score(
            y_true, y_pred, labels=["Bad"], average="macro", zero_division=0
        ),
        "report": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels),
    }


def plot_confusion(y_true, y_pred, ax, title: str,
                   labels: list[str] = QUALITY_ORDER, normalize: bool = False):
    """Dibuja una matriz de confusión en el eje `ax` dado.

    Args:
        normalize: si True, normaliza por fila (recall por clase).
    """
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fmt = "d"
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Verdadero")
    ax.set_title(title)


def summary_row(name: str, y_true, y_pred) -> dict:
    """Devuelve una fila resumen (dict) para tablas comparativas de modelos."""
    m = evaluate(y_true, y_pred)
    return {
        "modelo": name,
        "macro_f1": round(m["macro_f1"], 4),
        "balanced_acc": round(m["balanced_acc"], 4),
        "recall_Bad": round(m["recall_bad"], 4),
    }
