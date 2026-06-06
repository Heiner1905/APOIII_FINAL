# Guía de Entrega y Presentación

Cómo cerrar y entregar el Proyecto Final: qué subir, qué documentos enviar y un
guion para el video.

> Proyecto Final · Algoritmos y Programación III · Universidad ICESI · 2026-1
> Clasificación de calidad de frutas por visión por computadora.

---

## 1. Los tres entregables

| # | Entregable | Archivo / fuente | Estado |
|---|------------|------------------|--------|
| 1 | **Informe IEEE** (≤7 páginas) | `Docs/Informe_IEEE_paper.docx` | ✅ Listo (faltan autores y cita Kaggle) |
| 2 | **Repositorio GitHub** (por fases) | el repo (código + notebooks) | ✅ Listo para commit |
| 3 | **Video** (≤10 min) | grabación tuya | ⏳ Pendiente (guion en §4) |

---

## 2. Qué subir a GitHub (y qué NO)

Tu `.gitignore` deja **fuera** del repo (correcto, son pesados o material de
informe): `Fruits/`, `Docs/`, `reports/`, `artifacts/`, `.venv/`, modelos
(`*.keras`, `*.pkl`, `*.npz`).

### ✅ Sí va al repositorio (evidencia del trabajo por fases)
```
src/                 # código reutilizable (data, models, evaluation, utils, deployment)
app/                 # interfaz Streamlit
notebooks/           # 02..06 con CÓDIGO y RESULTADOS embebidos (clave: muestran la evidencia)
tests/               # (si añades pruebas)
README.md            # presentación del proyecto
GUIA_USO.md          # cómo instalar y usar modelos/GUI
GUIA_ENTREGA.md      # este archivo
requirements.txt
.gitignore  LICENSE  estructura_proyecto.md
```

> **Importante:** los **notebooks** son tu evidencia visible en GitHub (tienen las
> figuras y métricas embebidas). Asegúrate de commitearlos **ejecutados**.

### ❌ No va al repo (se entrega aparte o es local)
- **Informe** (`Docs/Informe_IEEE_paper.docx`) → se entrega por la plataforma del
  curso, no por el repo.
- `reports/*.md` → son tus conclusiones internas (insumo del informe).
- `artifacts/` (modelos) y `Fruits/` (dataset) → pesados; se regeneran con los
  notebooks (documentado en `GUIA_USO.md`).

### Comandos sugeridos (tú manejas Git)
```bash
git add src app notebooks tests README.md GUIA_USO.md GUIA_ENTREGA.md requirements.txt .gitignore
git commit -m "Proyecto final: código, notebooks por fase, app y documentación"
git push
```

---

## 3. Mapa: dónde está cada requisito de la rúbrica

| Requisito | Dónde está |
|-----------|------------|
| CRISP-DM (6 fases) | `notebooks/02..06` + `reports/01..06` + informe §III |
| ≥2 ML tradicional + tuning | RF y SVM con GridSearch — `notebooks/04_modelado_ml.ipynb` |
| 1 CNN desde cero | `notebooks/04_modelado_cnn.ipynb`, `src/models/cnn.py` |
| Línea base + comparación | Tabla I del informe / notebook 05 |
| Métricas adecuadas (Macro-F1, etc.) | `src/evaluation/metrics.py`, notebooks 04–05 |
| Manejo de desbalanceo | class weights + augmentation (notebook 03) |
| Despliegue (GUI) | `app/streamlit_app.py` |
| **PI1 — Ética** | `reports/07_PI1_etica.md` → informe |
| **PI2 — Impactos** | `reports/07_PI2_impactos.md` → informe |
| **PI3 — Solución matemática** | `reports/07_PI3_solucion_matematica.md` → informe |
| Reproducibilidad | semillas fijas, `requirements.txt`, `src/` |

---

## 4. Guion del video (objetivo ~9:00, máx 10:00)

Estructura con tiempos, qué decir y qué mostrar en pantalla.

### 0:00 – 0:45 · Introducción y problema
- **Decir:** "Clasificar calidad de frutas a mano es lento, subjetivo y genera
  pérdidas. Construimos un sistema de visión por computadora que predice calidad
  (mala/regular/buena) y tamaño."
- **Mostrar:** título del informe / una foto de fruta.

### 0:45 – 1:45 · Datos y EDA
- **Decir:** "9.515 imágenes, 6 frutas × 3 calidades. Hallazgo clave del EDA: un
  **sesgo de procedencia** — imágenes diminutas y canal alfa correlacionados con
  la clase."
- **Mostrar:** `notebooks/02_eda.ipynb` (distribución por clase, montaje de
  muestras).

### 1:45 – 3:00 · Metodología y preparación (CRISP-DM)
- **Decir:** "Seguimos CRISP-DM. Limpiamos (descartamos diminutas → 7.803
  imágenes), aplanamos RGBA, redimensionamos a 128×128, split 70/15/15
  estratificado, class weights, y features color HSV + HOG (1.860 dims)."
- **Mostrar:** `notebooks/03_preparacion.ipynb` (pipeline de una imagen,
  augmentation, descriptores).

### 3:00 – 5:00 · Modelado y resultados
- **Decir:** "Comparamos línea base, Random Forest, SVM-RBF y una CNN pequeña
  desde cero. **El mejor fue SVM con Macro-F1 de 0,957** en prueba; recall de
  `Bad` de 0,90 y solo 2 errores del tipo más costoso (Bad→Good)."
- **Mostrar:** Tabla I del informe + matrices de confusión (notebook 05) +
  curvas de la CNN (notebook 04).

### 5:00 – 6:30 · Evaluación honesta: auditoría de fuga (★ diferenciador)
- **Decir:** "Sospechamos fuga por fotos en ráfaga (casi-duplicados). La
  cuantificamos con hashing perceptual: 15 % del test tenía un casi-duplicado en
  train. Rehicimos el split **consciente de grupos** y el desempeño **no baja**
  (incluso sube). Conclusión: el rendimiento es **real**, no inflado."
- **Mostrar:** figura de comparación con/sin fuga (notebook 05).

### 6:30 – 8:00 · Demo en vivo de la GUI
- **Decir:** "Así se usa." Sube/captura una foto de fruta, elige fruta y modelo.
- **Mostrar:** `.venv/bin/streamlit run app/streamlit_app.py` → calidad +
  probabilidades + tamaño + segmentación.
- **Tip:** ten 2–3 fotos listas (una buena, una mala) con fondo simple.

### 8:00 – 9:00 · Competencias y conclusiones
- **Decir:** "Ética (PI1): detectamos y mitigamos sesgo, reportamos honestamente.
  Impactos (PI2): reducción de desperdicio, modelos eficientes en CPU.
  Matemática (PI3): justificamos features, kernel RBF, Macro-F1 y la auditoría de
  fuga. Conclusión: SVM-RBF es el mejor; generaliza de verdad."
- **Mostrar:** las tablas de PI1/PI2 o el informe.

### 9:00 – 9:30 · Cierre y trabajo futuro
- **Decir:** "Trabajo futuro: transfer learning, más especies, tamaño con escala
  física. Gracias."

> **Consejos de grabación:** guion breve a la vista, habla claro, muestra
> pantalla (no leas párrafos completos), y deja la **demo de la GUI** y la
> **auditoría de fuga** como momentos fuertes — son los que más impresionan.

---

## 5. Checklist final antes de entregar

**Informe (`Docs/Informe_IEEE_paper.docx`):**
- [ ] Completar **nombres de los autores**.
- [ ] Completar la **cita exacta del dataset de Kaggle** en Referencias.
- [ ] Verificar que **no excede 7 páginas** (recorta fundamentos teóricos si hace falta).
- [ ] (Si la plantilla lo exige) pasar el cuerpo a **2 columnas** en Word.
- [ ] Revisar que las 5 figuras y las 2 tablas se ven bien.

**Repositorio:**
- [ ] Notebooks **ejecutados** (con figuras/resultados) commiteados.
- [ ] `README.md` con nombres del equipo y descripción.
- [ ] `requirements.txt` actualizado.
- [ ] Que el repo **clona y corre** siguiendo `GUIA_USO.md`.

**Video:**
- [ ] Dura **≤10 min**.
- [ ] Incluye **demo de la GUI** en vivo.
- [ ] Menciona explícitamente **PI1, PI2 y PI3**.

