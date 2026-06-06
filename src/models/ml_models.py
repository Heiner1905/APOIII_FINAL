"""
ml_models.py — Modelos de ML tradicional y línea base (Fase 4).

Define los estimadores y sus grids de hiperparámetros para `GridSearchCV`.
Modelos elegidos (Sección 4): Random Forest + SVM (RBF). Ambos usan
`class_weight='balanced'` para el desbalanceo. El SVM se encapsula en un
Pipeline con StandardScaler (sensible a la escala); el RF no lo necesita.

La métrica de selección en la búsqueda es `f1_macro` (Macro-F1), coherente con
el criterio del proyecto.
"""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42


def build_baseline() -> DummyClassifier:
    """Línea base trivial: predice siempre la clase mayoritaria."""
    return DummyClassifier(strategy="most_frequent", random_state=SEED)


def rf_grid() -> tuple[RandomForestClassifier, dict]:
    """Random Forest y su grid de hiperparámetros.

    Returns:
        (estimador, param_grid) para GridSearchCV.
    """
    rf = RandomForestClassifier(
        class_weight="balanced", random_state=SEED, n_jobs=-1
    )
    grid = {
        "n_estimators": [200, 400],
        "max_depth": [None, 20],
        "min_samples_leaf": [1, 2],
    }
    return rf, grid


def svm_grid() -> tuple[Pipeline, dict]:
    """SVM (RBF) con escalado, y su grid de hiperparámetros.

    El escalado va dentro del Pipeline para que la validación cruzada no filtre
    información del conjunto de validación (se ajusta solo con el train de cada
    fold).

    Returns:
        (pipeline, param_grid) para GridSearchCV.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", class_weight="balanced", random_state=SEED)),
    ])
    grid = {
        "svc__C": [1, 10],
        "svc__gamma": ["scale"],
    }
    return pipe, grid
