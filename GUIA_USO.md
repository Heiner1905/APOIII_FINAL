# Guía de uso — Clasificador de calidad de frutas

Esta guía explica, paso a paso, cómo **instalar**, **generar los modelos**, **usar
los modelos desde código** y **lanzar la interfaz gráfica (GUI)**.

> Proyecto Final · Algoritmos y Programación III · Universidad ICESI · 2026-1

---

## 1. Requisitos e instalación

Necesitas **Python 3.10–3.12** y el dataset en la carpeta `Fruits/` en la raíz del
proyecto (solo si vas a re-entrenar o re-calibrar; para usar modelos ya entrenados
no hace falta).

```bash
# 1) Crear y activar el entorno virtual
python3 -m venv .venv
source .venv/bin/activate            # Linux/Mac
# .venv\Scripts\activate             # Windows

# 2) Instalar dependencias
pip install -r requirements.txt
```

> En este proyecto el entorno ya está en `.venv/`. Puedes usar directamente
> `.venv/bin/python` y `.venv/bin/streamlit` sin activar nada.

---

## 2. Artefactos necesarios (`artifacts/`)

La GUI y la inferencia requieren estos archivos en `artifacts/` (se generan al
ejecutar los notebooks; **no** se versionan en Git):

| Archivo | Lo genera | Para qué sirve |
|---------|-----------|----------------|
| `features.npz` | `notebooks/04_modelado_ml.ipynb` | Características color+HOG cacheadas. |
| `rf.pkl`, `svm.pkl` | `notebooks/04_modelado_ml.ipynb` | Modelos ML entrenados. |
| `cnn.keras` | `notebooks/04_modelado_cnn.ipynb` | Red neuronal entrenada. |
| `svm_deploy.pkl` | `notebooks/06_despliegue.ipynb` | SVM con probabilidades (GUI). |
| `size_thresholds.json` | `notebooks/06_despliegue.ipynb` | Umbrales de tamaño por fruta. |

### ¿Cómo (re)generarlos?
Ejecuta los notebooks en orden (necesitas la carpeta `Fruits/`):

```bash
# Modelos ML (features + RF + SVM)
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=fruits-venv notebooks/04_modelado_ml.ipynb

# CNN (entrena ~12 min en CPU)
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=fruits-venv notebooks/04_modelado_cnn.ipynb

# Despliegue (SVM con probabilidades + umbrales de tamaño)
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=fruits-venv notebooks/06_despliegue.ipynb
```

> Si solo te pasan los archivos de `artifacts/`, **no** necesitas re-entrenar:
> colócalos en `artifacts/` y pasa directamente a usar la GUI o el código.

---

## 3. Usar la interfaz gráfica (GUI)

```bash
.venv/bin/streamlit run app/streamlit_app.py
```

Se abrirá en el navegador (por defecto http://localhost:8501). En la barra lateral:

1. **Tipo de fruta:** elige la fruta (necesario para estimar el tamaño).
2. **Modelo de calidad:** `SVM (RBF)` (más preciso) o `CNN` (red neuronal).
3. **Fuente de imagen:** *Subir archivo* o *Cámara*.

Luego sube/captura una **foto de una sola fruta sobre fondo simple y uniforme**.
La app mostrará:
- **Calidad predicha** (Bad / Regular / Good) y **probabilidades** por clase.
- **Tamaño estimado** (pequeño / mediano / grande).
- **Segmentación** de la fruta (fondo atenuado).

**Consejo:** para mejores resultados, usa fondo claro y liso, buena iluminación y
la fruta centrada y completa.

---

## 4. Usar los modelos desde código (Python)

La API de inferencia está en `src/deployment/predict.py`. Trabaja con imágenes RGB
como `numpy.ndarray` (uint8).

### Ejemplo: predecir calidad y tamaño de una imagen
```python
import numpy as np
from PIL import Image
from src.deployment.predict import predict_quality, estimate_size

# Cargar una imagen cualquiera como RGB uint8
img = np.asarray(Image.open("mi_fruta.jpg").convert("RGB"), dtype=np.uint8)

# Calidad con el SVM (devuelve probabilidades por clase)
proba_svm = predict_quality(img, kind="svm")
print(proba_svm)         # {'Bad': 0.99, 'Regular': 0.004, 'Good': 0.001}
print(max(proba_svm, key=proba_svm.get))   # clase más probable

# Calidad con la CNN
proba_cnn = predict_quality(img, kind="cnn")

# Tamaño (requiere indicar la fruta)
size = estimate_size(img, fruit="Apple")
print(size["size"], size["area_fraction"])  # p.ej. 'mediano' 0.52
```

### Ejemplo: predecir sobre una imagen del dataset
```python
import numpy as np
from src.data.dataset import build_index, load_rgb
from src.deployment.predict import predict_quality

df = build_index()
ruta = df.iloc[0]["path"]
img = np.asarray(load_rgb(ruta), dtype=np.uint8)
print(predict_quality(img, kind="svm"))
```

> Ejecuta estos scripts desde la **raíz del proyecto** (para que `import src...`
> funcione), por ejemplo: `.venv/bin/python mi_script.py`.

---

## 5. Estructura de módulos reutilizables (`src/`)

| Módulo | Función principal |
|--------|-------------------|
| `src/data/dataset.py` | `build_index()` (índice del dataset), `load_rgb()` (carga RGB). |
| `src/data/preprocess.py` | Limpieza, splits, features (`extract_features`), augmentation. |
| `src/data/dedup.py` | Casi-duplicados (dHash) y split por grupos (auditoría de fuga). |
| `src/models/ml_models.py` | Random Forest, SVM y línea base + sus grids. |
| `src/models/cnn.py` | `build_cnn()` (arquitectura de la CNN). |
| `src/evaluation/metrics.py` | Macro-F1, matrices de confusión, resúmenes. |
| `src/utils/segmentation.py` | Segmentación (Otsu) y clasificación de tamaño. |
| `src/deployment/predict.py` | API de inferencia (calidad + tamaño) para la GUI. |

---

## 6. Solución de problemas

| Síntoma | Causa probable / solución |
|---------|---------------------------|
| `FileNotFoundError: artifacts/...` | Faltan los modelos; ejecuta los notebooks 04 y 06 (sección 2). |
| `ModuleNotFoundError: src...` | Ejecuta desde la **raíz** del proyecto o añade la raíz a `sys.path`. |
| La cámara no aparece en la GUI | El navegador debe tener permiso de cámara; usa `localhost`. |
| Predicción rara con fondo complejo | Usa **fondo simple y uniforme**; la segmentación asume eso. |
| `kernel fruits-venv not found` (notebooks) | Regístralo: `.venv/bin/python -m ipykernel install --user --name=fruits-venv`. |
| TensorFlow imprime warnings de CPU | Normales (optimizaciones CPU); se silencian con `TF_CPP_MIN_LOG_LEVEL=2`. |

---

## 7. Resumen rápido (TL;DR)

```bash
# Instalar
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# (Si faltan modelos) generarlos
.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=fruits-venv notebooks/04_modelado_ml.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=fruits-venv notebooks/04_modelado_cnn.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=fruits-venv notebooks/06_despliegue.ipynb

# Lanzar la GUI
.venv/bin/streamlit run app/streamlit_app.py
```

El mejor modelo es **SVM (RBF)** (Macro-F1 0,957 en prueba). La **CNN** es la
alternativa de Deep Learning. ¡Listo para clasificar frutas! 🍎🍌🍊
