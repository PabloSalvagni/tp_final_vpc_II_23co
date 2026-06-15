# Trabajo Final de Vision por Computadora II

Repositorio de Trabajo Final de Vision por Computadora II

- Federica Pavese
- Pablo Salvagni
- Rodrigo Parra

## Título TP

Conteo automático de capítulos (flores) de girasol para asistencia en la estimación de rendimiento agrícola.

## Problema

En la agricultura de precisión, la estimación temprana del rendimiento es vital para la logística y la comercialización. Actualmente, el conteo de capítulos (flores) de girasol se realiza de forma manual y por muestreo estadístico, lo cual es:

- **Lento y costoso:** Requiere personal recorriendo el lote, ahora mejoró con el uso de la tecnología, como drones.
- **Propenso al error humano:** La fatiga visual y la variabilidad en la densidad del cultivo afectan la precisión.
- **No escalable:** Es imposible contar manualmente cada flor en campos de cientos de hectáreas.

**Objetivo:** Desarrollar un prototipo basado en Visión por Computadora que automatice este proceso, proporcionando un mecanismo automatizado de conteo a partir de imágenes aéreas o terrestres.

## Solución

Para resolver este problema, se propone la implementación de un modelo de **Detección de Objetos** basado en la arquitectura **YOLOv8** (You Only Look Once).

**Componentes clave:**

- **Transfer Learning:** Utilizaremos pesos pre-entrenados en el dataset COCO para acelerar la convergencia y mejorar la extracción de características, realizando un fine-tuning específico para la clase "sunflower".
- **Pipeline de Procesamiento:**
  1. Redimensionamiento y normalización de imágenes.
  2. Inferencia mediante el detector para localizar cada capítulo.
  3. Lógica de conteo basada en la cantidad de bounding boxes detectadas con un umbral de confianza específico.

**Métricas de Validación:** La solución se validará mediante el cálculo del **mAP (mean Average Precision)** y la **Latencia de Inferencia**, evaluando la factibilidad de uso en escenarios de tiempo real, como dispositivos móviles y drones.

#### Análisis del Dataset (Roboflow)

El dataset seleccionado para realizar el TP es [Sunflower (ryan 8018)](https://universe.roboflow.com/raiyan8018/sunflower-mn2cr).

**Ficha Técnica para el Paper:**

- **Origen:** Roboflow Universe.
- **Tarea:** Object Detection (Bounding Boxes).
- **Etiquetas:** Identificación de capítulos de girasol (Sunflower heads).
- **Pre-procesamiento sugerido:** Dada la naturaleza del cultivo, aplicaremos Data Augmentation (rotaciones aleatorias y cambios de brillo/saturación) para simular diferentes condiciones de luz solar y ángulos de toma de cámara, evitando el overfitting.
- **División de datos:** Se respetará una estructura de 70% entrenamiento, 20% validación y 10% testeo para asegurar la robustez de los resultados.

---

## Estructura del Repositorio

Para facilitar la lectura, reproducibilidad y corrección del proyecto, el código se encuentra organizado bajo la siguiente estructura:

```text
tp_final_vpc_II_23co/
│
├── README.md                 <- Presentación del TP, integrantes y guía de uso.
├── requirements.txt          <- Dependencias necesarias para ejecutar el proyecto.
├── .gitignore                <- Archivos ignorados por git (pesos intermedios, datasets locales, etc.).
│
├── notebooks/                <- Notebooks finales de investigación y experimentación (ordenadas lógicamente).
│   ├── 01_eda_y_procesamiento.ipynb
│   ├── 02_entrenamiento_yolo.ipynb
│   └── 03_inferencia_y_metricas.ipynb
│   └── colab/                <- Notebooks preliminares de investigación y experimentación para Colab.
│   └── local/                <- Notebooks preliminares de investigación y experimentación para ejecucion local.

│
├── deploy/                   <- Código para el despliegue del modelo y demo de inferencia.
│   └── app.py                <- Script principal para la demostración.
│
├── models/                   <- Modelo entrenado.
│   └── best.pt               <- Mejores pesos del modelo entrenado para la clase "sunflower".
│
├── reports/                  <- Documentación y resultados gráficos del proyecto.
│   ├── figures/              <- Gráficas de evolución, matrices de confusión y métricas.
│   └── TP_VpC2_Presentacion.pdf           <- Documento final en formato paper.
│   └── TP_VpC2.pdf           <- Documento final en formato paper.
│
├── runs/                     <- Logs y salidas automáticas del entrenamiento de YOLO.
│
└── outputs/                  <- Resultados visuales de la inferencia (imágenes/videos de prueba).