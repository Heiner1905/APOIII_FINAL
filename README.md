# Clasificación Automática de Calidad de Frutas por Visión por Computadora

Proyecto Final del curso **Algoritmos y Programación III** — Ingeniería de
Sistemas, Universidad ICESI, semestre **2026-1**.

## Estado del proyecto
✅ **Completado** — Todas las fases CRISP-DM implementadas: EDA, preparación, modelado (SVM, Random Forest, CNN), evaluación con auditoría de fuga y despliegue en GUI Streamlit.

## Descripción

Sistema automático que clasifica imágenes de **una fruta individual** (sobre
fondo simple) según su **calidad** y estima su **tamaño**. Busca apoyar a
mercados y agroindustrias, donde la clasificación manual de productos frescos
es lenta, subjetiva y propensa a errores, generando pérdidas económicas y
desperdicio de alimentos.

- **Entrada:** imagen estática de una fruta/verdura sobre fondo uniforme.
- **Salidas:**
  1. **Clase de calidad:** `Bad` / `Regular` / `Good` (3 categorías).
  2. **Estimación de tamaño:** pequeño / mediano / grande.
- **Despliegue:** interfaz gráfica (Streamlit) con carga/captura de imagen.

## Metodología

El proyecto sigue **CRISP-DM**, documentando cada fase: comprensión del negocio
y de los datos, preparación, modelado, evaluación y despliegue.

## Modelos

- **Línea base** (clasificador por clase mayoritaria).
- **Random Forest** y **SVM-RBF** con GridSearchCV sobre características HSV + HOG (1.860 dims).
- **CNN** pequeña entrenada desde cero (TensorFlow/Keras).
- **Mejor modelo:** SVM-RBF con **Macro-F1 = 0,957** en test; recall de clase `Bad` = 0,90.

## Dataset

Basado en *Fruit Quality Classification* (Kaggle) + recolección propia del
grupo. **No se versiona en Git** por su tamaño (~3,3 GB). Estructura esperada:

```
Fruits/
├── Bad Quality_Fruits/
│   ├── Apple_Bad/  Banana_Bad/  Guava_Bad/  Lime_Bad/  Orange_Bad/  Pomegranate_Bad/
├── Good Quality_Fruits/
│   └── {Fruta}_Good/
└── Regular Quality_Fruits/
    └── {Fruta}_Regular/
```

- **6 frutas:** Apple, Banana, Guava, Lime, Orange, Pomegranate.
- **3 niveles de calidad:** Bad, Good, Regular.
- **~9.515 imágenes** (jpg/jpeg/png), resoluciones heterogéneas.

> ⚠️ Coloca la carpeta `Fruits/` en la raíz del proyecto antes de ejecutar.

## Estructura del repositorio

```
.
├── Informe Final/    # Informe IEEE entregable (v2 formato, v3 humanizado)
├── src/              # Código fuente reutilizable (helpers importados por los notebooks)
│   ├── data/         # Carga e indexado del dataset (dataset.py), preprocesamiento
│   ├── models/       # Definición de arquitecturas
│   ├── training/     # Scripts de entrenamiento
│   ├── evaluation/   # Validación y métricas
│   ├── utils/        # Funciones auxiliares
│   └── main.py       # Punto de entrada
├── app/              # Interfaz gráfica Streamlit (streamlit_app.py)
├── notebooks/        # Notebooks por fase: CÓDIGO + figuras/resultados inline
├── reports/          # Conclusiones por fase CRISP-DM (insumo del informe IEEE)
├── artifacts/        # Modelos y caches entrenados (ignorado en Git)
├── tests/            # Pruebas
├── Fruits/           # Dataset (ignorado en Git)
├── requirements.txt
└── README.md
```

> **Flujo de trabajo:** los **notebooks** contienen el código y las
> visualizaciones (resultados renderizados inline, no en carpetas aparte); los
> **reports/** recogen las conclusiones que alimentarán el informe IEEE final; el
> código reutilizable vive en **src/** y se importa desde los notebooks.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/Mac
pip install -r requirements.txt
```

## Uso de la interfaz gráfica

```bash
.venv/bin/streamlit run app/streamlit_app.py
```
Permite cargar/capturar la imagen de una fruta (fondo simple), elegir la fruta y
el modelo (SVM o CNN), y muestra la calidad predicha (con probabilidades), el
tamaño estimado y la segmentación. Requiere los artefactos en `artifacts/`
(generados por los notebooks 04 y 06).

## Equipo

| Nombre | Código |
|--------|--------|
| Heiner Danit Rincón Carrillo | A00402510 |
| Ruben Dario Marquinez Rincón | A00401286 |
| Simon Reyes Riveros | A00400880 |
| Juan David Ocampo Martinez | A00401856 |

Universidad ICESI — Ingeniería de Sistemas — Algoritmos y Programación III, 2026-1.

## Licencia

Ver [LICENSE](LICENSE).
