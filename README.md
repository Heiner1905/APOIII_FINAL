# Clasificación Automática de Calidad de Frutas por Visión por Computadora

Proyecto Final del curso **Algoritmos y Programación III** — Ingeniería de
Sistemas, Universidad ICESI, semestre **2026-1**.

## Estado del proyecto
🟢 **En desarrollo** — Sección 0 (Setup y exploración del dataset) completada.

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

- **≥2 modelos de ML tradicional** (con ajuste de hiperparámetros vía
  validación cruzada / grid search), sobre características extraídas de las
  imágenes (histogramas de color, HOG, momentos, etc.).
- **1 modelo de Deep Learning**: CNN pequeña entrenada preferiblemente desde
  cero (TensorFlow/Keras).
- Comparación contra una **línea base** y entre modelos.

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
├── Docs/             # Lineamientos oficiales, rúbrica, plantilla IEEE (no versionado)
├── src/              # Código fuente reutilizable (helpers importados por los notebooks)
│   ├── data/         # Carga e indexado del dataset (dataset.py), preprocesamiento
│   ├── models/       # Definición de arquitecturas
│   ├── training/     # Scripts de entrenamiento
│   ├── evaluation/   # Validación y métricas
│   ├── utils/        # Funciones auxiliares
│   └── main.py       # Punto de entrada
├── notebooks/        # Notebooks por fase: CÓDIGO + figuras/resultados inline
├── reports/          # Conclusiones por fase CRISP-DM (insumo del informe IEEE)
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

## Equipo

| Nombre | Correo / Usuario |
|--------|------------------|
| _(por completar)_ | _(por completar)_ |

- **Profesor:** _(por completar)_

## Licencia

Ver [LICENSE](LICENSE).
