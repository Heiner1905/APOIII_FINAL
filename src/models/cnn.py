"""
cnn.py — CNN pequeña desde cero para clasificación de calidad (Fase 4).

Arquitectura: 3 bloques convolucionales (Conv-BN-ReLU + MaxPooling) seguidos de
una cabeza densa con dropout. El preprocesamiento (normalización) y la data
augmentation se incluyen como capas del modelo, de modo que:

    - La entrada son imágenes uint8 [0, 255] de 128x128x3 (sin normalizar fuera).
    - La augmentation solo actúa en entrenamiento (Keras la desactiva en
      inferencia automáticamente).
    - El modelo guardado es autocontenido y listo para el despliegue.

Se entrena desde cero (sin transfer learning), como prioriza el enunciado.
"""

from __future__ import annotations


def build_cnn(input_shape=(128, 128, 3), n_classes: int = 3, augment: bool = True):
    """Construye y devuelve la CNN (modelo Keras sin compilar).

    Args:
        input_shape: forma de la entrada (alto, ancho, canales).
        n_classes: número de clases de salida.
        augment: si True, incluye capas de data augmentation (solo activas en
            entrenamiento).

    Returns:
        keras.Model sin compilar.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential(name="cnn_calidad_frutas")
    model.add(keras.Input(shape=input_shape))

    # Data augmentation ligera (coherente con la Fase 3: flip + rotación leve).
    if augment:
        model.add(layers.RandomFlip("horizontal", seed=42))
        model.add(layers.RandomRotation(0.05, fill_mode="constant",
                                        fill_value=255.0, seed=42))

    # Normalización [0, 255] -> [0, 1] dentro del modelo.
    model.add(layers.Rescaling(1.0 / 255))

    # Bloque convolucional 1
    model.add(layers.Conv2D(32, 3, padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D())

    # Bloque convolucional 2
    model.add(layers.Conv2D(64, 3, padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D())

    # Bloque convolucional 3
    model.add(layers.Conv2D(128, 3, padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D())

    # Cabeza densa
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(n_classes, activation="softmax"))

    return model
